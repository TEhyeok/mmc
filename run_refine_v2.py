"""
gsplat pose refinement v2 — reproj fitting 결과를 초기값으로 사용
변경점:
  1. SAM-3D body_pose_params → reproj fitting의 poses/Rh/Th/shapes 사용
  2. regularization 0.0001 → 0.01 (100x 강화)
  3. per-joint gradient clipping + pose change 제한
  4. 222프레임 (reproj fitting 결과 범위)
"""
import numpy as np, torch, smplx, json, os, time
from gsplat import rasterization
from PIL import Image
import torchvision.transforms as T

# ============ Config ============
N_ITERS = 50
LR = 0.003           # slightly lower
IMG_SCALE = 0.25
LAMBDA_REG = 0.01    # 100x stronger than v1
LAMBDA_SMOOTH = 0.005
MAX_POSE_DELTA = 0.15  # radians (~8.6 deg) max change per joint
CAM_NUMS = ['3', '5', '6', '7']
SMPL_PATH = '/home/elicer/EasyMocap/data/smplx/'
REPROJ_DIR = '/home/elicer/reproj_sam3d_7view/smpl/'
IMAGES_DIR = '/home/elicer/easymocap_data/images/'
CALIB_FILE = '/home/elicer/vggt_calibration_result.json'
MARKER_MAP = '/home/elicer/server_scripts/smpl_virtual_marker_mapping.json'
OUTPUT_DIR = '/home/elicer/gsplat_refined_v2/'
# ================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print('Loading SMPL...')
model = smplx.create(SMPL_PATH, model_type='smpl', gender='neutral', batch_size=1).cuda()

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

with open(MARKER_MAP) as f:
    marker_mapping = json.load(f)

transform = T.Compose([T.Resize((int(1080*IMG_SCALE), int(1920*IMG_SCALE))), T.ToTensor()])

N_V = 6890
scales = 0.008 * torch.ones(N_V, 3, device='cuda')
quats = torch.zeros(N_V, 4, device='cuda')
quats[:, 0] = 1.0

# Find available reproj frames
reproj_files = sorted([f for f in os.listdir(REPROJ_DIR) if f.endswith('.json')])
frame_indices = [int(f.replace('.json', '')) for f in reproj_files]
print('Reproj frames: %d (%d to %d)' % (len(frame_indices), frame_indices[0], frame_indices[-1]))
print('Cameras: %s' % [c['name'] for c in cameras])

all_losses_init = []
all_losses_refined = []
prev_pose = None
t_start = time.time()

for fi_idx, fi in enumerate(frame_indices):
    # Load reproj SMPL params (already fitted to 7-view geometry)
    reproj_path = os.path.join(REPROJ_DIR, '%06d.json' % fi)
    reproj_data = json.load(open(reproj_path))
    if isinstance(reproj_data, list):
        reproj_data = reproj_data[0]

    poses_72 = np.array(reproj_data['poses'][0])  # (72,)
    body_pose_init = poses_72[3:72]  # skip global orient placeholder (0,0,0)
    global_orient_init = np.array(reproj_data['Rh'][0])  # (3,)
    transl_init = np.array(reproj_data['Th'][0])  # (3,)
    betas_init = np.array(reproj_data['shapes'][0])  # (10,)

    # Load GT images
    gt_images = []
    for cam in cameras:
        img_path = os.path.join(cam['img_dir'], '%06d.jpg' % fi)
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            gt_images.append(transform(img).permute(1, 2, 0).cuda())
        else:
            gt_images.append(None)

    n_valid = sum(1 for g in gt_images if g is not None)
    if n_valid == 0:
        continue

    # Setup differentiable params
    body_pose = torch.tensor(body_pose_init, dtype=torch.float32, device='cuda', requires_grad=True)
    go = torch.tensor(global_orient_init, dtype=torch.float32, device='cuda', requires_grad=True)
    betas = torch.tensor(betas_init, dtype=torch.float32).unsqueeze(0).cuda()
    transl = torch.tensor(transl_init, dtype=torch.float32).unsqueeze(0).cuda()
    init_pose = body_pose.clone().detach()
    init_go = go.clone().detach()

    colors_raw = torch.nn.Parameter(torch.zeros(N_V, 3, device='cuda'))
    opacities_raw = torch.nn.Parameter(2.0 * torch.ones(N_V, device='cuda'))

    optimizer = torch.optim.Adam([
        {'params': [body_pose], 'lr': LR},
        {'params': [go], 'lr': LR * 0.3},
        {'params': [colors_raw, opacities_raw], 'lr': LR * 3},
    ])

    # Compute initial loss (before refinement)
    with torch.no_grad():
        out_init = model(body_pose=init_pose.unsqueeze(0), global_orient=init_go.unsqueeze(0),
                         betas=betas, transl=transl)
        verts_init = out_init.vertices[0]
        colors_init = torch.sigmoid(colors_raw)
        ops_init = torch.sigmoid(opacities_raw)
        init_loss = 0.0
        nv0 = 0
        for ci, cam in enumerate(cameras):
            gt = gt_images[ci]
            if gt is None: continue
            H, W = gt.shape[:2]
            K_s = cam['K'].clone(); K_s[0] *= IMG_SCALE; K_s[1] *= IMG_SCALE
            vm = torch.eye(4, device='cuda'); vm[:3,:3] = cam['R']; vm[:3,3] = cam['t']
            r = rasterization(verts_init.contiguous(), quats, scales, ops_init, colors_init,
                              vm.unsqueeze(0), K_s.unsqueeze(0), W, H)[0][0]
            init_loss += (r - gt).abs().mean().item()
            nv0 += 1
        if nv0 > 0: init_loss /= nv0

    best_loss = float('inf')
    best_pose = body_pose.clone().detach()
    best_go = go.clone().detach()

    for it in range(N_ITERS):
        optimizer.zero_grad()
        out = model(body_pose=body_pose.unsqueeze(0), global_orient=go.unsqueeze(0),
                    betas=betas, transl=transl)
        verts = out.vertices[0]
        colors = torch.sigmoid(colors_raw)
        opacities = torch.sigmoid(opacities_raw)

        total_loss = torch.tensor(0.0, device='cuda')
        nv = 0
        for ci, cam in enumerate(cameras):
            gt = gt_images[ci]
            if gt is None: continue
            H, W = gt.shape[:2]
            K_s = cam['K'].clone(); K_s[0] *= IMG_SCALE; K_s[1] *= IMG_SCALE
            vm = torch.eye(4, device='cuda'); vm[:3,:3] = cam['R']; vm[:3,3] = cam['t']
            rendered = rasterization(verts.contiguous(), quats, scales, opacities, colors,
                                     vm.unsqueeze(0), K_s.unsqueeze(0), W, H)[0][0]
            total_loss = total_loss + (rendered - gt).abs().mean()
            nv += 1

        if nv > 0:
            total_loss = total_loss / nv
            # Strong regularization
            reg = LAMBDA_REG * ((body_pose - init_pose) ** 2).mean()
            reg_go = LAMBDA_REG * 0.5 * ((go - init_go) ** 2).mean()
            smooth = LAMBDA_SMOOTH * ((body_pose - prev_pose) ** 2).mean() if prev_pose is not None else 0.0
            total_loss = total_loss + reg + reg_go + smooth
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([body_pose], max_norm=0.5)
            torch.nn.utils.clip_grad_norm_([go], max_norm=0.3)
            optimizer.step()

            # Clamp pose change
            with torch.no_grad():
                delta = body_pose - init_pose
                delta_clamped = delta.clamp(-MAX_POSE_DELTA, MAX_POSE_DELTA)
                body_pose.copy_(init_pose + delta_clamped)

        lv = total_loss.item()
        if lv < best_loss:
            best_loss = lv
            best_pose = body_pose.clone().detach()
            best_go = go.clone().detach()

    # Extract virtual markers
    with torch.no_grad():
        out_f = model(body_pose=best_pose.unsqueeze(0), global_orient=best_go.unsqueeze(0),
                      betas=betas, transl=transl)
        verts_f = out_f.vertices[0].cpu().numpy()
        joints_f = out_f.joints[0].cpu().numpy()

    markers = {}
    for name, indices in marker_mapping.items():
        if not name.endswith('_center'):
            markers[name] = verts_f[indices].mean(axis=0)

    np.savez(os.path.join(OUTPUT_DIR, '%06d.npz' % fi),
             body_pose_init=init_pose.cpu().numpy(),
             body_pose_refined=best_pose.cpu().numpy(),
             global_orient_init=init_go.cpu().numpy(),
             global_orient_refined=best_go.cpu().numpy(),
             betas=betas[0].cpu().numpy(),
             transl=transl[0].cpu().numpy(),
             vertices=verts_f, joints=joints_f,
             virtual_markers=markers,
             loss_init=init_loss, loss_refined=best_loss)

    prev_pose = best_pose.detach()
    all_losses_init.append(init_loss)
    all_losses_refined.append(best_loss)

    pc = np.degrees((best_pose - init_pose).abs().mean().item())
    elapsed = time.time() - t_start
    eta = elapsed / (fi_idx + 1) * (len(frame_indices) - fi_idx - 1)

    if fi_idx % 50 == 0 or fi_idx == len(frame_indices) - 1:
        print('  Frame %4d [%3d/%3d]: init_loss=%.5f refined_loss=%.5f delta=%.2fdeg elapsed=%ds ETA=%ds' % (
            fi, fi_idx+1, len(frame_indices), init_loss, best_loss, pc, elapsed, eta))

elapsed_total = time.time() - t_start
summary = {
    'n_frames': len(all_losses_init),
    'mean_loss_init': float(np.mean(all_losses_init)),
    'mean_loss_refined': float(np.mean(all_losses_refined)),
    'loss_improvement': float(np.mean(all_losses_init) - np.mean(all_losses_refined)),
    'elapsed_seconds': elapsed_total,
}
with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print('')
print('=' * 70)
print('COMPLETE: %d frames in %.0fs (%.1f min)' % (len(all_losses_init), elapsed_total, elapsed_total/60))
print('Loss: init=%.5f -> refined=%.5f (improvement=%.5f)' % (
    np.mean(all_losses_init), np.mean(all_losses_refined),
    np.mean(all_losses_init) - np.mean(all_losses_refined)))
print('Output: %s' % OUTPUT_DIR)
