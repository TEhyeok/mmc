#!/usr/bin/env python3
"""
Direct SMPL fitting to triangulated 3D keypoints.
- Proper initialization from skeleton orientation
- Scale correction using subject height (183cm)
- 3-stage optimization: RT → body pose → betas
- Temporal continuity via previous frame initialization

Usage (on server):
  nohup python3 direct_smpl_fit.py > direct_smpl_fit.log 2>&1 &
"""

import json, numpy as np, torch, smplx, cv2, os, time

# ======== Config ========
SUBJECT_HEIGHT_M = 1.83  # 피사체 키
KP3D_DIR = '/home/elicer/easymocap_data/output/vggt_smpl3_fixed/keypoints3d'
OUT_DIR = '/home/elicer/easymocap_data/output/vggt_direct_smpl'
SMPL_MODEL_PATH = '/home/elicer/EasyMocap/data/smplx'
J_REG_PATH = '/home/elicer/EasyMocap/models/J_regressor_body25.npy'

STAGE1_STEPS = 100   # RT only
STAGE2_STEPS = 300   # + body pose
STAGE3_STEPS = 150   # + betas (every 10th frame)
LAMBDA_POSE = 0.001  # pose regularization
LAMBDA_BETA = 0.01   # shape regularization

device = 'cuda'

# ======== Setup ========
body_model = smplx.create(
    model_path=SMPL_MODEL_PATH,
    model_type='smpl', gender='neutral', batch_size=1
).to(device)

J_reg = torch.tensor(
    np.load(J_REG_PATH), dtype=torch.float32, device=device
)

os.makedirs(OUT_DIR + '/smpl', exist_ok=True)

# ======== Compute scale factor ========
# Load frame 0 to get triangulated body height
with open(os.path.join(KP3D_DIR, '000000.json')) as f:
    kp0 = np.array(json.load(f)[0]['keypoints3d'])[:, :3]

# Height = nose to average ankle
tri_height = np.linalg.norm(kp0[0] - (kp0[11] + kp0[14]) / 2)
scale_factor = SUBJECT_HEIGHT_M / tri_height

print(f'Subject height: {SUBJECT_HEIGHT_M}m')
print(f'Triangulated height: {tri_height:.4f}')
print(f'Scale factor: {scale_factor:.4f}')
print(f'After scaling: {tri_height * scale_factor:.4f}m')
print()

# ======== Compute initial orientation from frame 0 ========
kp0_scaled = kp0 * scale_factor
nose, midhip = kp0_scaled[0], kp0_scaled[8]
rsho, lsho = kp0_scaled[2], kp0_scaled[5]

up = nose - midhip
up = up / np.linalg.norm(up)
right = rsho - lsho
right = right - np.dot(right, up) * up
right = right / np.linalg.norm(right)
forward = np.cross(right, up)
R_person = np.column_stack([right, up, forward])
rvec_init, _ = cv2.Rodrigues(R_person)
rvec_init = rvec_init.flatten()

print(f'Initial Rh: {rvec_init.round(3)} (|Rh|={np.linalg.norm(rvec_init)*180/np.pi:.1f} deg)')
print(f'Initial Th: {midhip.round(3)}')
print()

# ======== Count frames ========
n_frames = len([f for f in os.listdir(KP3D_DIR) if f.endswith('.json')])
print(f'Total frames: {n_frames}')
print()

# ======== Fit all frames ========
prev_Rh = rvec_init.copy()
prev_Th = midhip.copy()
prev_body = np.zeros(69)
betas_shared = torch.zeros(1, 10, dtype=torch.float32, device=device, requires_grad=True)

all_mpjpe = []
t_start = time.time()

for frame_i in range(n_frames):
    # Load keypoints and apply scale
    with open(os.path.join(KP3D_DIR, '%06d.json' % frame_i)) as f:
        kp_raw = np.array(json.load(f)[0]['keypoints3d'])

    kp_scaled = kp_raw.copy()
    kp_scaled[:, :3] *= scale_factor

    kp_data = torch.tensor(kp_scaled, dtype=torch.float32, device=device)
    kp_pos = kp_data[:15, :3]
    kp_conf = kp_data[:15, 3]

    # Initialize from previous frame
    Rh = torch.tensor(prev_Rh, dtype=torch.float32, device=device, requires_grad=True)
    Th = torch.tensor(prev_Th, dtype=torch.float32, device=device, requires_grad=True)
    body_pose = torch.tensor(prev_body, dtype=torch.float32, device=device).unsqueeze(0).requires_grad_(True)

    # Stage 1: RT only
    opt1 = torch.optim.Adam([Rh, Th], lr=0.01)
    for s in range(STAGE1_STEPS):
        out = body_model(global_orient=Rh.unsqueeze(0), body_pose=body_pose,
                         betas=betas_shared, transl=Th.unsqueeze(0))
        j25 = J_reg @ out.vertices[0]
        loss = (kp_conf[:, None] * (j25[:15] - kp_pos) ** 2).sum()
        opt1.zero_grad(); loss.backward(); opt1.step()

    # Stage 2: + body pose
    opt2 = torch.optim.Adam([Rh, Th, body_pose], lr=0.005)
    for s in range(STAGE2_STEPS):
        out = body_model(global_orient=Rh.unsqueeze(0), body_pose=body_pose,
                         betas=betas_shared, transl=Th.unsqueeze(0))
        j25 = J_reg @ out.vertices[0]
        loss_kp = (kp_conf[:, None] * (j25[:15] - kp_pos) ** 2).sum()
        loss_reg = LAMBDA_POSE * (body_pose ** 2).sum()
        loss = loss_kp + loss_reg
        opt2.zero_grad(); loss.backward(); opt2.step()

    # Stage 3: + betas (every 10th frame)
    if frame_i % 10 == 0:
        opt3 = torch.optim.Adam([Rh, Th, body_pose, betas_shared], lr=0.002)
        for s in range(STAGE3_STEPS):
            out = body_model(global_orient=Rh.unsqueeze(0), body_pose=body_pose,
                             betas=betas_shared, transl=Th.unsqueeze(0))
            j25 = J_reg @ out.vertices[0]
            loss_kp = (kp_conf[:, None] * (j25[:15] - kp_pos) ** 2).sum()
            loss_reg = LAMBDA_POSE * (body_pose ** 2).sum() + LAMBDA_BETA * (betas_shared ** 2).sum()
            loss = loss_kp + loss_reg
            opt3.zero_grad(); loss.backward(); opt3.step()

    # Compute MPJPE
    with torch.no_grad():
        out = body_model(global_orient=Rh.unsqueeze(0), body_pose=body_pose,
                         betas=betas_shared, transl=Th.unsqueeze(0))
        j25 = (J_reg @ out.vertices[0]).cpu().numpy()
        kp_np = kp_pos.cpu().numpy()
        errs = [np.linalg.norm(j25[i] - kp_np[i]) for i in range(15)]
        mpjpe = np.mean(errs) * 1000  # convert to mm
        all_mpjpe.append(mpjpe)

    # Save previous frame for next init
    prev_Rh = Rh.detach().cpu().numpy()
    prev_Th = Th.detach().cpu().numpy()
    prev_body = body_pose.detach().cpu().numpy().flatten()

    # Save result
    result = [{
        'id': 0,
        'Rh': [prev_Rh.tolist()],
        'Th': [prev_Th.tolist()],
        'poses': [np.concatenate([[0, 0, 0], prev_body]).tolist()],
        'shapes': [betas_shared.detach().cpu().numpy().flatten().tolist()]
    }]
    with open(os.path.join(OUT_DIR, 'smpl', '%06d.json' % frame_i), 'w') as f:
        json.dump(result, f)

    if frame_i % 50 == 0 or frame_i == n_frames - 1:
        elapsed = time.time() - t_start
        fps = (frame_i + 1) / elapsed
        eta = (n_frames - frame_i - 1) / fps if fps > 0 else 0
        print(f'Frame {frame_i:4d}/{n_frames}: MPJPE={mpjpe:6.1f}mm | '
              f'elapsed={elapsed:.0f}s | ETA={eta:.0f}s | {fps:.1f} fps')

# ======== Summary ========
elapsed = time.time() - t_start
print()
print('=' * 60)
print(f'Done: {n_frames} frames in {elapsed:.0f}s ({n_frames/elapsed:.1f} fps)')
print(f'MPJPE: mean={np.mean(all_mpjpe):.1f}mm, median={np.median(all_mpjpe):.1f}mm')
print(f'       min={np.min(all_mpjpe):.1f}mm, max={np.max(all_mpjpe):.1f}mm')
print(f'Betas: {betas_shared.detach().cpu().numpy().flatten().round(2).tolist()}')
print(f'Scale factor: {scale_factor:.4f} (subject height: {SUBJECT_HEIGHT_M}m)')

# Save metadata
meta = {
    'subject_height_m': SUBJECT_HEIGHT_M,
    'scale_factor': float(scale_factor),
    'n_frames': n_frames,
    'mean_mpjpe_mm': float(np.mean(all_mpjpe)),
    'median_mpjpe_mm': float(np.median(all_mpjpe)),
    'betas': betas_shared.detach().cpu().numpy().flatten().tolist(),
    'mpjpe_per_frame': [float(x) for x in all_mpjpe]
}
with open(os.path.join(OUT_DIR, 'fitting_meta.json'), 'w') as f:
    json.dump(meta, f, indent=2)
print(f'Metadata saved to {OUT_DIR}/fitting_meta.json')
