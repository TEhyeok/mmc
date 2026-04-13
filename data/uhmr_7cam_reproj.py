"""
U-HMR init (body_pose fixed) + 7-camera 2D reprojection fitting.
Only optimizes global_orient (3) + transl (3) = 6 parameters.
Uses smoothed OpenPose v2 keypoints from masked images.
"""
import os, sys, json, numpy as np, torch, smplx, cv2, time
sys.stdout.reconfigure(line_buffering=True)

device = 'cuda'
BATCH = 50

# ============ Load cameras (VGGT) ============
with open('/home/elicer/vggt_calibration_result.json') as f:
    vggt = json.load(f)
R_fix = np.load('/home/elicer/R_fix_vggt_to_smpl.npy')

cam_K, cam_R, cam_t = [], [], []
for cn in ['cam1','cam2','cam3','cam4','cam5','cam6','cam7']:
    c = vggt[cn]
    cam_K.append(torch.tensor(c['K'], dtype=torch.float32, device=device))
    cam_R.append(torch.tensor(np.array(c['R']) @ R_fix.T, dtype=torch.float32, device=device))
    cam_t.append(torch.tensor(c['t'], dtype=torch.float32, device=device))

# ============ Load U-HMR init ============
uhmr = np.load('/home/elicer/uhmr_smpl_init.npz')
body_pose_init = torch.tensor(uhmr['body_pose'], dtype=torch.float32, device=device)  # (69,)
betas_init = torch.tensor(uhmr['betas'], dtype=torch.float32, device=device)  # (10,)
go_init = uhmr['global_orient']  # (3,)
print('U-HMR body_pose: fixed, |body|=%.2f' % np.linalg.norm(uhmr['body_pose']))
print('U-HMR betas:', uhmr['betas'].round(2))

# ============ Load SMPL ============
body_models = {}
def get_model(bs):
    if bs not in body_models:
        body_models[bs] = smplx.create(model_path='/home/elicer/EasyMocap/data/smplx',
            model_type='smpl', gender='neutral', batch_size=bs).to(device)
    return body_models[bs]

J_reg = torch.tensor(np.load('/home/elicer/EasyMocap/models/J_regressor_body25.npy'),
                     dtype=torch.float32, device=device)

# ============ Load smoothed OpenPose v2 ============
print('Loading smoothed keypoints...')
kp_dir = '/home/elicer/openpose_masked_smoothed_v2'
n_frames = len(os.listdir(os.path.join(kp_dir, 'cam3')))

all_kp2d = np.zeros((n_frames, 7, 25, 2), dtype=np.float32)
all_conf = np.zeros((n_frames, 7, 25), dtype=np.float32)

for ci in range(7):
    cam_dir = os.path.join(kp_dir, 'cam%d' % (ci+1))
    jsons = sorted([f for f in os.listdir(cam_dir) if f.endswith('.json')])
    for fi, fn in enumerate(jsons):
        if fi >= n_frames: break
        with open(os.path.join(cam_dir, fn)) as f:
            data = json.load(f)
        if data['people']:
            kps = np.array(data['people'][0]['pose_keypoints_2d']).reshape(-1, 3)
            all_kp2d[fi, ci, :len(kps)] = kps[:, :2]
            c = kps[:, 2].copy()
            c[c < 0.3] = 0
            all_conf[fi, ci, :len(kps)] = c

kp2d_gpu = torch.tensor(all_kp2d, dtype=torch.float32, device=device)
conf_gpu = torch.tensor(all_conf, dtype=torch.float32, device=device)
print(f'Loaded {n_frames} frames x 7 cams')

# ============ Projection ============
def project_batch_7cam(j25, cam_K, cam_R, cam_t):
    """j25: (B, 25, 3) -> (B, 7, 25, 2)"""
    B = j25.shape[0]
    projs = []
    for ci in range(7):
        p = torch.einsum('ij,bnj->bni', cam_R[ci], j25) + cam_t[ci]
        p2d = torch.einsum('ij,bnj->bni', cam_K[ci], p)
        proj = p2d[:,:,:2] / p2d[:,:,2:3]
        projs.append(proj)
    return torch.stack(projs, dim=1)

def gmof(x, sigma=100.0):
    return x**2 / (x**2 + sigma**2)

# ============ Initial translation from triangulated pelvis ============
# Use existing triangulated data
try:
    with open('/home/elicer/easymocap_data/output/vggt_smpl3_fixed/keypoints3d/000000.json') as f:
        kp3d = np.array(json.load(f)[0]['keypoints3d'])[:, :3]
    Th_init = kp3d[8] * 1.83 / np.linalg.norm(kp3d[0] - (kp3d[11]+kp3d[14])/2)
except:
    Th_init = np.array([0, 0, 3.0])

print('Init Th:', Th_init.round(3))
print('Init global_orient:', go_init.round(3))

# ============ Batch fitting ============
out_dir = '/home/elicer/uhmr_7cam_reproj'
os.makedirs(out_dir + '/smpl', exist_ok=True)

all_Rh = np.zeros((n_frames, 3))
all_Th = np.zeros((n_frames, 3))
all_reproj = []

t0 = time.time()
n_batches = (n_frames + BATCH - 1) // BATCH

for bi in range(n_batches):
    s = bi * BATCH
    e = min(s + BATCH, n_frames)
    B = e - s
    model = get_model(B)

    kp2d = kp2d_gpu[s:e]  # (B, 7, 25, 2)
    conf = conf_gpu[s:e]  # (B, 7, 25)

    # Fixed body_pose and betas (from U-HMR)
    bp_fixed = body_pose_init.unsqueeze(0).expand(B, -1)  # (B, 69)
    betas_fixed = betas_init.unsqueeze(0).expand(B, -1)  # (B, 10)

    # Optimize only global_orient and transl
    Rh = torch.tensor(np.tile(go_init, (B, 1)), dtype=torch.float32, device=device, requires_grad=True)
    Th = torch.tensor(np.tile(Th_init, (B, 1)), dtype=torch.float32, device=device, requires_grad=True)

    # Stage 1: RT only (200 steps)
    opt = torch.optim.Adam([Rh, Th], lr=0.03)
    for step in range(200):
        out = model(global_orient=Rh, body_pose=bp_fixed, betas=betas_fixed, transl=Th)
        j25 = torch.einsum('jv,bvd->bjd', J_reg, out.vertices)
        projs = project_batch_7cam(j25, cam_K, cam_R, cam_t)
        diff = projs - kp2d
        loss = (conf.unsqueeze(-1) * gmof(diff)).sum() / B
        if torch.isnan(loss): break
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([Rh, Th], 1.0)
        opt.step()

    # Stage 2: + body_pose fine-tune (100 steps, small lr)
    bp_tune = bp_fixed.clone().detach().requires_grad_(True)
    opt2 = torch.optim.Adam([Rh, Th, bp_tune], lr=0.005)
    for step in range(100):
        out = model(global_orient=Rh, body_pose=bp_tune, betas=betas_fixed, transl=Th)
        j25 = torch.einsum('jv,bvd->bjd', J_reg, out.vertices)
        projs = project_batch_7cam(j25, cam_K, cam_R, cam_t)
        diff = projs - kp2d
        loss_r = (conf.unsqueeze(-1) * gmof(diff)).sum() / B
        loss_reg = 0.01 * ((bp_tune - bp_fixed)**2).sum() / B  # stay close to U-HMR
        loss = loss_r + loss_reg
        if torch.isnan(loss): break
        opt2.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([Rh, Th, bp_tune], 1.0)
        opt2.step()

    # Evaluate
    with torch.no_grad():
        out = model(global_orient=Rh, body_pose=bp_tune, betas=betas_fixed, transl=Th)
        j25 = torch.einsum('jv,bvd->bjd', J_reg, out.vertices)
        projs = project_batch_7cam(j25, cam_K, cam_R, cam_t)
        diff = projs - kp2d
        errs = []
        for b in range(B):
            mask = conf[b] > 0
            if mask.sum() > 0:
                errs.append(torch.norm(diff[b][mask], dim=1).mean().item())
        avg_err = np.mean(errs) if errs else 999

    rh = Rh.detach().cpu().numpy()
    th = Th.detach().cpu().numpy()
    bp = bp_tune.detach().cpu().numpy()

    for b in range(B):
        if np.any(np.isnan(rh[b])):
            rh[b] = go_init; th[b] = Th_init; bp[b] = body_pose_init.cpu().numpy()

    all_Rh[s:e] = rh
    all_Th[s:e] = th
    all_reproj.extend(errs)

    # Save
    for b in range(B):
        fi = s + b
        result = [{'id': 0,
                   'Rh': [rh[b].tolist()],
                   'Th': [th[b].tolist()],
                   'poses': [np.concatenate([[0,0,0], bp[b]]).tolist()],
                   'shapes': [betas_init.cpu().numpy().tolist()]}]
        with open(os.path.join(out_dir, 'smpl', '%06d.json' % fi), 'w') as f:
            json.dump(result, f)

    elapsed = time.time() - t0
    eta = (n_batches - bi - 1) * elapsed / (bi + 1)
    print(f'Batch {bi+1}/{n_batches} [{s}-{e}]: reproj={avg_err:.1f}px | {elapsed:.0f}s ETA={eta:.0f}s')

total = time.time() - t0
print(f'\nDone: {n_frames} frames in {total:.0f}s')
print(f'Mean reproj: {np.mean(all_reproj):.1f}px')
