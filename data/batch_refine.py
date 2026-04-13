"""
Batch SMPL fitting with SMPLest-X initialization + post-process smoothing.
Pass 1: Batch 2D reprojection fitting (GPU parallel, SMPLest-X init)
Pass 2: Savitzky-Golay temporal smoothing (CPU, instant)
"""
import os, sys, numpy as np, torch, cv2, json
from scipy.signal import savgol_filter
sys.stdout.reconfigure(line_buffering=True)

device = 'cuda'
BATCH = 100

# ============ Load SMPL ============
import smplx
body_models = {}
def get_model(bs):
    if bs not in body_models:
        body_models[bs] = smplx.create(model_path='/home/elicer/EasyMocap/data/smplx',
            model_type='smpl', gender='neutral', batch_size=bs).to(device)
    return body_models[bs]

J_reg = torch.tensor(np.load('/home/elicer/EasyMocap/models/J_regressor_body25.npy'),
                     dtype=torch.float32, device=device)

# ============ Load SMPLest-X results (initialization) ============
smplestx_dir = '/home/elicer/smplestx_sambbox_cam3'
n_frames = len([f for f in os.listdir(smplestx_dir) if f.endswith('.npz')])
print(f'Loading {n_frames} SMPLest-X frames as init...')

init_focals = np.zeros((n_frames, 2))
init_princpts = np.zeros((n_frames, 2))
for fi in range(n_frames):
    d = np.load(os.path.join(smplestx_dir, '%06d.npz' % fi))
    init_focals[fi] = d['focal']
    init_princpts[fi] = d['princpt']

# Use average focal/princpt (same camera)
avg_focal = init_focals.mean(axis=0)
avg_princpt = init_princpts.mean(axis=0)
focal_t = torch.tensor(avg_focal, dtype=torch.float32, device=device)
princpt_t = torch.tensor(avg_princpt, dtype=torch.float32, device=device)
print(f'Focal: {avg_focal}, Princpt: {avg_princpt}')

# ============ Load OpenPose 2D keypoints ============
annot_dir = '/home/elicer/easymocap_data/annots/3'
all_kp2d = np.zeros((n_frames, 25, 2), dtype=np.float32)
all_conf = np.zeros((n_frames, 25), dtype=np.float32)
for fi in range(n_frames):
    with open(os.path.join(annot_dir, '%06d.json' % fi)) as f:
        ann = json.load(f)
    kps = np.array(ann['annots'][0]['keypoints'])
    all_kp2d[fi] = kps[:, :2]
    c = kps[:, 2].copy()
    c[c < 0.3] = 0
    all_conf[fi] = c

kp2d_gpu = torch.tensor(all_kp2d, dtype=torch.float32, device=device)
conf_gpu = torch.tensor(all_conf, dtype=torch.float32, device=device)

# ============ Initial SMPL params from first successful fit ============
# Use SMPLest-X mesh centroid for translation init
mesh0 = np.load(os.path.join(smplestx_dir, '000000.npz'))['mesh']
Th_init = mesh0.mean(axis=0)

# Global orient from earlier analysis
Rh_init = np.array([2.7, 0.5, -0.2])

def gmof(x, sigma=100.0):
    return x**2 / (x**2 + sigma**2)

def project_batch(j25, focal, princpt):
    """j25: (B, 25, 3) -> (B, 25, 2)"""
    x = focal[0] * j25[:, :, 0] / j25[:, :, 2] + princpt[0]
    y = focal[1] * j25[:, :, 1] / j25[:, :, 2] + princpt[1]
    return torch.stack([x, y], dim=2)

# ============ Pass 1: Batch fitting ============
print(f'\n=== Pass 1: Batch fitting, batch={BATCH} ===')
import time
t0 = time.time()

out_dir = '/home/elicer/batch_refined_cam3'
os.makedirs(out_dir + '/smpl', exist_ok=True)

all_Rh = np.zeros((n_frames, 3))
all_Th = np.zeros((n_frames, 3))
all_bp = np.zeros((n_frames, 69))
betas_np = np.zeros(10)

n_batches = (n_frames + BATCH - 1) // BATCH

for bi in range(n_batches):
    s = bi * BATCH
    e = min(s + BATCH, n_frames)
    B = e - s

    model = get_model(B)
    kp2d = kp2d_gpu[s:e]     # (B, 25, 2)
    conf = conf_gpu[s:e]     # (B, 25)

    Rh = torch.tensor(np.tile(Rh_init, (B, 1)), dtype=torch.float32, device=device, requires_grad=True)
    Th = torch.tensor(np.tile(Th_init, (B, 1)), dtype=torch.float32, device=device, requires_grad=True)
    bp = torch.zeros(B, 69, dtype=torch.float32, device=device, requires_grad=True)
    betas = torch.zeros(B, 10, dtype=torch.float32, device=device, requires_grad=True)

    def compute_loss(with_reg=False):
        out = model(global_orient=Rh, body_pose=bp, betas=betas, transl=Th)
        j25 = torch.einsum('jv,bvd->bjd', J_reg, out.vertices)
        proj = project_batch(j25, focal_t, princpt_t)  # (B, 25, 2)
        diff = proj - kp2d
        loss = (conf.unsqueeze(-1) * gmof(diff)).sum() / B
        if with_reg:
            loss = loss + 0.003 * (bp**2).sum() / B + 0.05 * (betas**2).sum() / B
        return loss

    # Stage 1: RT (100 steps)
    opt = torch.optim.Adam([Rh, Th], lr=0.03)
    for step in range(100):
        loss = compute_loss()
        if torch.isnan(loss): break
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([Rh, Th], 1.0)
        opt.step()

    # Stage 2: + body (300 steps)
    opt = torch.optim.Adam([Rh, Th, bp], lr=0.02)
    for step in range(300):
        loss = compute_loss(with_reg=True)
        if torch.isnan(loss): break
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([Rh, Th, bp], 1.0)
        opt.step()

    # Stage 3: + betas (100 steps)
    opt = torch.optim.Adam([Rh, Th, bp, betas], lr=0.005)
    for step in range(100):
        loss = compute_loss(with_reg=True)
        if torch.isnan(loss): break
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([Rh, Th, bp, betas], 1.0)
        opt.step()

    # Evaluate
    with torch.no_grad():
        out = model(global_orient=Rh, body_pose=bp, betas=betas, transl=Th)
        j25 = torch.einsum('jv,bvd->bjd', J_reg, out.vertices)
        proj = project_batch(j25, focal_t, princpt_t)
        diff = (proj - kp2d)
        errs = []
        for b in range(B):
            mask = conf[b] > 0
            if mask.sum() > 0:
                e_px = torch.norm(diff[b][mask], dim=1).mean().item()
                errs.append(e_px)
        avg_err = np.mean(errs) if errs else 999

    rh = Rh.detach().cpu().numpy()
    th = Th.detach().cpu().numpy()
    body = bp.detach().cpu().numpy()
    beta = betas.detach().cpu().numpy()

    # NaN guard
    for b in range(B):
        if np.any(np.isnan(rh[b])):
            rh[b] = Rh_init; th[b] = Th_init; body[b] = 0; beta[b] = 0

    all_Rh[s:e] = rh
    all_Th[s:e] = th
    all_bp[s:e] = body

    elapsed = time.time() - t0
    eta = (n_batches - bi - 1) * elapsed / (bi + 1)
    print(f'Batch {bi+1}/{n_batches} [{s}-{e}]: reproj={avg_err:.1f}px | {elapsed:.0f}s ETA={eta:.0f}s')

# ============ Pass 2: Temporal smoothing ============
print('\n=== Pass 2: Temporal smoothing ===')

window = 11  # must be odd
poly = 3

# Smooth each parameter independently
all_Rh_smooth = np.zeros_like(all_Rh)
all_Th_smooth = np.zeros_like(all_Th)
all_bp_smooth = np.zeros_like(all_bp)

for i in range(3):
    all_Rh_smooth[:, i] = savgol_filter(all_Rh[:, i], window, poly)
    all_Th_smooth[:, i] = savgol_filter(all_Th[:, i], window, poly)

for i in range(69):
    all_bp_smooth[:, i] = savgol_filter(all_bp[:, i], window, poly)

print('Smoothing done.')

# ============ Save ============
betas_avg = betas.detach().cpu().numpy().mean(axis=0)

for fi in range(n_frames):
    result = [{'id': 0,
               'Rh': [all_Rh_smooth[fi].tolist()],
               'Th': [all_Th_smooth[fi].tolist()],
               'poses': [np.concatenate([[0,0,0], all_bp_smooth[fi]]).tolist()],
               'shapes': [betas_avg.tolist()]}]
    with open(os.path.join(out_dir, 'smpl', '%06d.json' % fi), 'w') as f:
        json.dump(result, f)

total = time.time() - t0
print(f'\nDone: {n_frames} frames in {total:.0f}s ({n_frames/total:.1f} fps)')

# Also save raw (unsmoothed) for comparison
np.savez(os.path.join(out_dir, 'raw_params.npz'),
         Rh=all_Rh, Th=all_Th, body_pose=all_bp,
         Rh_smooth=all_Rh_smooth, Th_smooth=all_Th_smooth, body_pose_smooth=all_bp_smooth,
         betas=betas_avg)
