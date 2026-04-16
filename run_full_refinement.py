"""
999-frame gsplat pose refinement — full sequence
서버 실행: conda activate sam3d && python server_scripts/run_full_refinement.py
"""
import numpy as np, torch, smplx, json, os, time
from gsplat import rasterization
from PIL import Image
import torchvision.transforms as T
from scipy.spatial.transform import Rotation as Rsci

# ============ Config ============
N_ITERS = 50         # iterations per frame
LR = 0.005
IMG_SCALE = 0.25     # 480x270
START_FRAME = 0
END_FRAME = 998
FRAME_STEP = 1       # every frame
CAM_NUMS = ['3', '5', '6', '7']  # landscape cameras
SMPL_PATH = '/home/elicer/EasyMocap/data/smplx/'
SAM3D_DIR = '/home/elicer/sam3d_results/'
IMAGES_DIR = '/home/elicer/easymocap_data/images/'
CALIB_FILE = '/home/elicer/vggt_calibration_result.json'
MARKER_MAP = '/home/elicer/server_scripts/smpl_virtual_marker_mapping.json'
OUTPUT_DIR = '/home/elicer/gsplat_refined_smpl/'
# ================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load SMPL
print('Loading SMPL model...')
model = smplx.create(SMPL_PATH, model_type='smpl', gender='neutral', batch_size=1).cuda()

# Load calibration
with open(CALIB_FILE) as f:
    calib = json.load(f)

cameras = []
for cn in CAM_NUMS:
    ck = 'cam' + cn
    if ck in calib:
        c = calib[ck]
        cameras.append({
            'K': torch.tensor(c['K'], dtype=torch.float32).cuda(),
            'R': torch.tensor(c['R'], dtype=torch.float32).cuda(),
            't': torch.tensor(c['t'], dtype=torch.float32).cuda().squeeze(),
            'img_dir': os.path.join(IMAGES_DIR, cn),
            'name': ck
        })
print(f'Loaded {len(cameras)} cameras: {[c["name"] for c in cameras]}')

# Load marker mapping
with open(MARKER_MAP) as f:
    marker_mapping = json.load(f)

# Image transform
transform = T.Compose([T.Resize((int(1080*IMG_SCALE), int(1920*IMG_SCALE))), T.ToTensor()])

# Gaussian fixed params
N_V = 6890
scales = 0.008 * torch.ones(N_V, 3, device='cuda')
quats = torch.zeros(N_V, 4, device='cuda')
quats[:, 0] = 1.0

# Results tracking
all_losses = []
prev_pose = None
t_start = time.time()

print(f'\n=== Starting Full Sequence Refinement ===')
print(f'Frames: {START_FRAME}-{END_FRAME} (step={FRAME_STEP})')
print(f'Iters/frame: {N_ITERS}, LR: {LR}')

for fi in range(START_FRAME, END_FRAME + 1, FRAME_STEP):
    # Check which camera has this frame
    npz_path = os.path.join(SAM3D_DIR, 'cam3', f'{fi:06d}.npz')
    if not os.path.exists(npz_path):
        # Try cam1/cam2 (999 frames, 0-998)
        npz_path = os.path.join(SAM3D_DIR, 'cam1', f'{fi:06d}.npz')
        if not os.path.exists(npz_path):
            continue

    # Load SAM3D params
    d = np.load(npz_path)
    bp_init = d['body_pose_params'][:69]
    root_rot = Rsci.from_matrix(d['pred_global_rots'][0]).as_rotvec()
    sp = d['shape_params'][:10]

    # Load GT images
    gt_images = []
    for cam in cameras:
        img_path = os.path.join(cam['img_dir'], f'{fi:06d}.jpg')
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            gt_images.append(transform(img).permute(1, 2, 0).cuda())
        else:
            gt_images.append(None)

    n_valid = sum(1 for g in gt_images if g is not None)
    if n_valid == 0:
        continue

    # Setup params
    body_pose = torch.tensor(bp_init, dtype=torch.float32, device='cuda', requires_grad=True)
    go = torch.tensor(root_rot, dtype=torch.float32, device='cuda', requires_grad=True)
    betas = torch.tensor(sp, dtype=torch.float32).unsqueeze(0).cuda()
    init_pose = body_pose.clone().detach()

    # Appearance params (reset per frame for now)
    colors_raw = torch.nn.Parameter(torch.zeros(N_V, 3, device='cuda'))
    opacities_raw = torch.nn.Parameter(2.0 * torch.ones(N_V, device='cuda'))

    optimizer = torch.optim.Adam([
        {'params': [body_pose], 'lr': LR},
        {'params': [go], 'lr': LR * 0.5},
        {'params': [colors_raw, opacities_raw], 'lr': LR * 2},
    ])

    best_loss = float('inf')
    best_pose = body_pose.clone().detach()

    for it in range(N_ITERS):
        optimizer.zero_grad()
        out = model(body_pose=body_pose.unsqueeze(0), global_orient=go.unsqueeze(0),
                    betas=betas, transl=torch.zeros(1, 3).cuda())
        verts = out.vertices[0]
        colors = torch.sigmoid(colors_raw)
        opacities = torch.sigmoid(opacities_raw)

        total_loss = torch.tensor(0.0, device='cuda')
        nv = 0
        for ci, cam in enumerate(cameras):
            gt = gt_images[ci]
            if gt is None:
                continue
            H, W = gt.shape[:2]
            K_s = cam['K'].clone()
            K_s[0] *= IMG_SCALE
            K_s[1] *= IMG_SCALE
            vm = torch.eye(4, device='cuda')
            vm[:3, :3] = cam['R']
            vm[:3, 3] = cam['t']
            rendered = rasterization(verts.contiguous(), quats, scales, opacities, colors,
                                     vm.unsqueeze(0), K_s.unsqueeze(0), W, H)[0][0]
            total_loss = total_loss + (rendered - gt).abs().mean()
            nv += 1

        if nv > 0:
            reg = 0.0001 * ((body_pose - init_pose) ** 2).mean()
            smooth = 0.001 * ((body_pose - prev_pose) ** 2).mean() if prev_pose is not None else 0.0
            total_loss = total_loss / nv + reg + smooth
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([body_pose, go], 1.0)
            optimizer.step()

        lv = total_loss.item()
        if lv < best_loss:
            best_loss = lv
            best_pose = body_pose.clone().detach()

    # Extract virtual markers from refined pose
    with torch.no_grad():
        out_final = model(body_pose=best_pose.unsqueeze(0), global_orient=go.detach().unsqueeze(0),
                          betas=betas, transl=torch.zeros(1, 3).cuda())
        verts_final = out_final.vertices[0].cpu().numpy()
        joints_final = out_final.joints[0].cpu().numpy()

    # Virtual markers
    markers = {}
    for name, indices in marker_mapping.items():
        if not name.endswith('_center'):
            markers[name] = verts_final[indices].mean(axis=0)

    # Save
    np.savez(os.path.join(OUTPUT_DIR, f'{fi:06d}.npz'),
             body_pose_init=init_pose.cpu().numpy(),
             body_pose_refined=best_pose.cpu().numpy(),
             global_orient=go.detach().cpu().numpy(),
             betas=betas[0].cpu().numpy(),
             vertices=verts_final,
             joints=joints_final,
             virtual_markers=markers,
             loss=best_loss)

    prev_pose = best_pose.detach()
    all_losses.append(best_loss)

    pc = np.degrees((best_pose - init_pose).abs().mean().item())
    elapsed = time.time() - t_start
    eta = elapsed / (fi - START_FRAME + 1) * (END_FRAME - fi) if fi > START_FRAME else 0

    if fi % 50 == 0 or fi == END_FRAME:
        print(f'  Frame {fi:4d}/{END_FRAME}: loss={best_loss:.5f}, '
              f'pose_delta={pc:.2f}deg, elapsed={elapsed:.0f}s, ETA={eta:.0f}s')

# Summary
elapsed_total = time.time() - t_start
summary = {
    'n_frames': len(all_losses),
    'mean_loss': float(np.mean(all_losses)),
    'min_loss': float(np.min(all_losses)),
    'max_loss': float(np.max(all_losses)),
    'elapsed_seconds': elapsed_total,
    'config': {
        'n_iters': N_ITERS, 'lr': LR, 'img_scale': IMG_SCALE,
        'cameras': CAM_NUMS, 'frame_range': [START_FRAME, END_FRAME, FRAME_STEP],
    }
}
with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f'\n=== COMPLETE ===')
print(f'Frames: {len(all_losses)}, Mean loss: {np.mean(all_losses):.5f}')
print(f'Time: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)')
print(f'Output: {OUTPUT_DIR}')
