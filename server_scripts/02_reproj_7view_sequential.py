"""
SKILL-4: 7-View Sequential Reprojection Fitting (핵심 스크립트)
NOTE: SAM 3D Body 파이프라인에서는 이 스크립트 대신
      SAM 3D Body 출력을 초기값으로 사용합니다.

변경사항 vs uhmr_7cam_reproj.py:
  1. 7개 카메라 전체 사용 (cam1 회전 포함)
  2. body_pose 정규화 λ = 0.01 → 0.001
  3. Stage 1: 200스텝, lr=0.03
  4. Stage 2: 300스텝, lr=0.01
  5. 진짜 Sequential: Rh, Th, body_pose 모두 전달
  6. 관절각 클램핑 (팔꿈치 0~150도)
  7. GMoF sigma=50 (이상치 강하게 억제)
  8. 왜곡 보정 (cam6 k1=-0.346)
  9. 후처리 Savitzky-Golay smoothing

서버 실행:
  conda activate gart  (또는 uhmr)
  python 02_reproj_7view_sequential.py
"""
import os, sys, json, time
import numpy as np
import torch
import cv2

sys.stdout.reconfigure(line_buffering=True)
device = 'cuda'

# ============================================================
# 설정
# ============================================================
N_FRAMES = 999
STAGE1_STEPS = 200     # RT only
STAGE1_LR = 0.03
STAGE2_STEPS = 500     # + body_pose (SAM init이 좋으므로 deeper refinement)
STAGE2_LR = 0.01
LAMBDA_REG = 0.00005   # SAM init이 정확하므로 정규화 약하게
GMOF_SIGMA = 50.0
CONF_THRESH = 0.3

# 관절각 클램핑 (SMPL joint indices)
# SMPL: 0=pelvis, 1=L_hip, 2=R_hip, ..., 18=L_elbow, 19=R_elbow, ...
ELBOW_L_IDX = 18   # 왼쪽 팔꿈치 (axis-angle index: 18*3 ~ 18*3+2)
ELBOW_R_IDX = 19   # 오른쪽 팔꿈치
ELBOW_FLEX_MIN = -2.6   # ~150도 굴곡
ELBOW_FLEX_MAX = 0.1    # 약간의 과신전 허용

# ============================================================
# 경로 (서버에 맞게 수정)
# ============================================================
# ★ 중요: 기존 성공 파이프라인(4-view 153.5px)은 VGGT 사용.
# COLMAP은 초점거리 편차 531~1367px (신뢰 낮음).
# VGGT는 ~734px 일관적. 기본값은 VGGT.
USE_VGGT = True  # True: VGGT (검증됨), False: COLMAP (실험적)

if USE_VGGT:
    CALIB_JSON = '/home/elicer/vggt_calibration_result.json'
    R_FIX_PATH = '/home/elicer/R_fix_vggt_to_smpl.npy'
else:
    CALIB_JSON = '/home/elicer/server_scripts/colmap_7view_for_reproj.json'
    R_FIX_PATH = None  # COLMAP은 R_fix 불필요 (직접 좌표계)

SMPL_MODEL_PATH = '/home/elicer/EasyMocap/data/smplx'
J_REG_PATH = '/home/elicer/EasyMocap/models/J_regressor_body25.npy'
KP_DIR = '/home/elicer/sam3d_kp_body25'  # SAM 3D Body → Body25 (Phase B 출력)
SAM3D_INIT = '/home/elicer/sam3d_triangulated/sam3d_smpl_init.npz'  # Phase D 출력
OUT_DIR = '/home/elicer/reproj_sam3d_7view'

CAM_NAMES = ['cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'cam6', 'cam7']

# ============================================================
# 로드
# ============================================================
import smplx

print("Loading calibration...")
with open(CALIB_JSON) as f:
    calib = json.load(f)

# R_fix: VGGT 좌표계 → SMPL 좌표계 변환 (VGGT 사용 시 필수)
R_fix = None
if USE_VGGT and R_FIX_PATH:
    R_fix = np.load(R_FIX_PATH)
    print(f"  R_fix loaded from {R_FIX_PATH}")

cam_K, cam_R, cam_t, cam_D = [], [], [], []
for cn in CAM_NAMES:
    if USE_VGGT:
        # VGGT JSON: calib[cn] = {'K': 3x3, 'R': 3x3, 't': 3x1, ...}
        c = calib[cn]
        K = np.array(c['K'])
        R = np.array(c['R'])
        t = np.array(c['t'])
        # R_fix 적용 (기존 uhmr_7cam_reproj.py line 21과 동일)
        if R_fix is not None:
            R = R @ R_fix.T
        D = np.zeros(5)
    else:
        # COLMAP JSON: calib[cn] = {'K': 3x3, 'R': 3x3, 't': 3x1, 'D': 5x1}
        c = calib[cn]
        K = np.array(c['K'])
        R = np.array(c['R'])
        t = np.array(c['t'])
        D = np.array(c.get('D', [0]*5))

    cam_K.append(torch.tensor(K, dtype=torch.float32, device=device))
    cam_R.append(torch.tensor(R, dtype=torch.float32, device=device))
    cam_t.append(torch.tensor(t, dtype=torch.float32, device=device))
    cam_D.append(D)
    print(f"  {cn}: f={K[0,0]:.1f}, |k1|={abs(D[0]):.4f}")

N_CAMS = len(CAM_NAMES)

print("Loading SMPL model...")
body_models = {}
def get_model(bs=1):
    if bs not in body_models:
        body_models[bs] = smplx.create(
            model_path=SMPL_MODEL_PATH,
            model_type='smpl', gender='neutral', batch_size=bs
        ).to(device)
    return body_models[bs]

J_reg = torch.tensor(
    np.load(J_REG_PATH), dtype=torch.float32, device=device
)  # (25, 6890)

print("Loading SAM 3D Body SMPL init (Phase D)...")
sam3d_init = np.load(SAM3D_INIT)
# Per-frame init: (N, 69), (N, 3), (N, 3)
all_go_init = torch.tensor(sam3d_init['global_orient'], dtype=torch.float32, device=device)
all_bp_init = torch.tensor(sam3d_init['body_pose'], dtype=torch.float32, device=device)
all_tr_init = torch.tensor(sam3d_init['transl'], dtype=torch.float32, device=device)
betas_init = torch.tensor(sam3d_init['betas'], dtype=torch.float32, device=device)

# 첫 프레임 초기값
go_init = all_go_init[0]
body_pose_init = all_bp_init[0]
Th_init = all_tr_init[0]

print(f"Init: {len(all_bp_init)} frames, betas={betas_init.cpu().numpy().round(3)}")

# ============================================================
# 7-cam 키포인트 로드
# ============================================================
print("Loading keypoints...")
all_kp2d = np.zeros((N_FRAMES, N_CAMS, 25, 2), dtype=np.float32)
all_conf = np.zeros((N_FRAMES, N_CAMS, 25), dtype=np.float32)

for ci, cn in enumerate(CAM_NAMES):
    cam_dir = os.path.join(KP_DIR, cn)
    jsons = sorted([f for f in os.listdir(cam_dir) if f.endswith('.json')])
    for fi, fn in enumerate(jsons[:N_FRAMES]):
        with open(os.path.join(cam_dir, fn)) as f:
            data = json.load(f)
        if data.get('people'):
            kps = np.array(data['people'][0]['pose_keypoints_2d']).reshape(-1, 3)
            all_kp2d[fi, ci, :len(kps)] = kps[:, :2]
            c = kps[:, 2].copy()
            c[c < CONF_THRESH] = 0
            all_conf[fi, ci, :len(kps)] = c

            # cam6 왜곡 보정 (k1=-0.346, 큰 왜곡)
            if abs(cam_D[ci][0]) > 0.05:
                K_np = np.array(calib[cn]['K'])
                D_np = cam_D[ci][:4].reshape(1, 4)
                pts = kps[:, :2].reshape(-1, 1, 2).astype(np.float64)
                pts_undist = cv2.undistortPoints(pts, K_np, D_np, P=K_np)
                all_kp2d[fi, ci, :len(kps)] = pts_undist.reshape(-1, 2).astype(np.float32)

kp2d_gpu = torch.tensor(all_kp2d, dtype=torch.float32, device=device)
conf_gpu = torch.tensor(all_conf, dtype=torch.float32, device=device)
print(f"Loaded {N_FRAMES} frames x {N_CAMS} cams")

# ============================================================
# 투영 + 손실 함수
# ============================================================
def project_7cam(j25):
    """j25: (25, 3) -> (7, 25, 2)"""
    projs = []
    for ci in range(N_CAMS):
        p = cam_R[ci] @ j25.T + cam_t[ci].unsqueeze(1)  # (3, 25)
        p2d = cam_K[ci] @ p  # (3, 25)
        proj = (p2d[:2] / p2d[2:3]).T  # (25, 2)
        projs.append(proj)
    return torch.stack(projs, dim=0)  # (7, 25, 2)

def gmof(x, sigma=GMOF_SIGMA):
    return x**2 / (x**2 + sigma**2)

def clamp_joints(bp):
    """관절각 클램핑 (in-place, no_grad)."""
    with torch.no_grad():
        # 팔꿈치 굴곡 제한 (첫 번째 축만, 대략적)
        for idx in [ELBOW_L_IDX, ELBOW_R_IDX]:
            s = idx * 3
            bp[s].clamp_(ELBOW_FLEX_MIN, ELBOW_FLEX_MAX)

# ============================================================
# Sequential Fitting
# ============================================================
os.makedirs(os.path.join(OUT_DIR, 'smpl'), exist_ok=True)
model = get_model(1)

prev_Rh = go_init.clone()
prev_Th = Th_init.clone()
prev_bp = body_pose_init.clone()

all_results = {
    'Rh': np.zeros((N_FRAMES, 3)),
    'Th': np.zeros((N_FRAMES, 3)),
    'body_pose': np.zeros((N_FRAMES, 69)),
    'betas': betas_init.cpu().numpy(),
    'reproj_errors': np.zeros(N_FRAMES),
}

t0 = time.time()

for fi in range(N_FRAMES):
    kp2d = kp2d_gpu[fi]  # (7, 25, 2)
    conf = conf_gpu[fi]  # (7, 25)

    # 초기화: 이전 프레임 결과 vs SAM init 블렌딩
    # 첫 프레임이면 SAM init 사용, 이후는 이전 프레임 결과 사용 (sequential)
    if fi == 0:
        Rh = all_go_init[fi].clone().detach().requires_grad_(True)
        Th = all_tr_init[fi].clone().detach().requires_grad_(True)
        bp = all_bp_init[fi].clone().detach().requires_grad_(True)
    else:
        Rh = prev_Rh.clone().detach().requires_grad_(True)
        Th = prev_Th.clone().detach().requires_grad_(True)
        bp = prev_bp.clone().detach().requires_grad_(True)
    betas = betas_init.unsqueeze(0)  # 고정

    # --- Stage 1: RT only ---
    opt1 = torch.optim.Adam([Rh, Th], lr=STAGE1_LR)
    for step in range(STAGE1_STEPS):
        out = model(
            global_orient=Rh.unsqueeze(0),
            body_pose=bp.unsqueeze(0),
            betas=betas,
            transl=Th.unsqueeze(0)
        )
        j25 = (J_reg @ out.vertices[0])  # (25, 3)
        projs = project_7cam(j25)  # (7, 25, 2)
        diff = projs - kp2d
        loss = (conf.unsqueeze(-1) * gmof(diff)).sum()
        if torch.isnan(loss):
            break
        opt1.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([Rh, Th], 1.0)
        opt1.step()

    # --- Stage 2: + body_pose ---
    opt2 = torch.optim.Adam([Rh, Th, bp], lr=STAGE2_LR)
    for step in range(STAGE2_STEPS):
        out = model(
            global_orient=Rh.unsqueeze(0),
            body_pose=bp.unsqueeze(0),
            betas=betas,
            transl=Th.unsqueeze(0)
        )
        j25 = (J_reg @ out.vertices[0])
        projs = project_7cam(j25)
        diff = projs - kp2d
        loss_r = (conf.unsqueeze(-1) * gmof(diff)).sum()
        # 정규화: SAM per-frame init 기준 (fi 범위 내이면)
        bp_ref = all_bp_init[fi] if fi < len(all_bp_init) else body_pose_init
        loss_reg = LAMBDA_REG * ((bp - bp_ref) ** 2).sum()
        loss = loss_r + loss_reg
        if torch.isnan(loss):
            break
        opt2.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([Rh, Th, bp], 1.0)
        opt2.step()
        clamp_joints(bp)

    # --- 평가 ---
    with torch.no_grad():
        out = model(
            global_orient=Rh.unsqueeze(0),
            body_pose=bp.unsqueeze(0),
            betas=betas,
            transl=Th.unsqueeze(0)
        )
        j25 = (J_reg @ out.vertices[0])
        projs = project_7cam(j25)
        diff = projs - kp2d
        mask = conf > 0
        if mask.sum() > 0:
            err = torch.norm(diff[mask], dim=1).mean().item()
        else:
            err = 999.0

    # NaN 가드
    rh_np = Rh.detach().cpu().numpy()
    th_np = Th.detach().cpu().numpy()
    bp_np = bp.detach().cpu().numpy()
    if np.any(np.isnan(rh_np)) or np.any(np.isnan(th_np)):
        rh_np = prev_Rh.cpu().numpy()
        th_np = prev_Th.cpu().numpy()
        bp_np = prev_bp.cpu().numpy()
        print(f"  Frame {fi}: NaN detected, using previous frame")

    # Sequential 전달
    prev_Rh = torch.tensor(rh_np, dtype=torch.float32, device=device)
    prev_Th = torch.tensor(th_np, dtype=torch.float32, device=device)
    prev_bp = torch.tensor(bp_np, dtype=torch.float32, device=device)

    # 저장
    all_results['Rh'][fi] = rh_np
    all_results['Th'][fi] = th_np
    all_results['body_pose'][fi] = bp_np
    all_results['reproj_errors'][fi] = err

    # JSON (EasyMocap 형식)
    result = [{'id': 0,
               'Rh': [rh_np.tolist()],
               'Th': [th_np.tolist()],
               'poses': [np.concatenate([[0, 0, 0], bp_np]).tolist()],
               'shapes': [betas_init.cpu().numpy().tolist()]}]
    with open(os.path.join(OUT_DIR, 'smpl', f'{fi:06d}.json'), 'w') as f:
        json.dump(result, f)

    # 로깅
    elapsed = time.time() - t0
    eta = (N_FRAMES - fi - 1) * elapsed / (fi + 1) if fi > 0 else 0
    if fi % 50 == 0 or fi == N_FRAMES - 1:
        print(f"Frame {fi:4d}/{N_FRAMES}: reproj={err:.1f}px | "
              f"elapsed={elapsed:.0f}s ETA={eta:.0f}s")

# ============================================================
# 후처리: Savitzky-Golay Smoothing
# ============================================================
print("\n=== Post-processing: Savitzky-Golay smoothing ===")
from scipy.signal import savgol_filter

WINDOW = 15  # 홀수
POLY = 3

Rh_smooth = np.zeros_like(all_results['Rh'])
Th_smooth = np.zeros_like(all_results['Th'])
bp_smooth = np.zeros_like(all_results['body_pose'])

for i in range(3):
    Rh_smooth[:, i] = savgol_filter(all_results['Rh'][:, i], WINDOW, POLY)
    Th_smooth[:, i] = savgol_filter(all_results['Th'][:, i], WINDOW, POLY)
for i in range(69):
    bp_smooth[:, i] = savgol_filter(all_results['body_pose'][:, i], WINDOW, POLY)

# 스무딩된 결과 저장
for fi in range(N_FRAMES):
    result = [{'id': 0,
               'Rh': [Rh_smooth[fi].tolist()],
               'Th': [Th_smooth[fi].tolist()],
               'poses': [np.concatenate([[0, 0, 0], bp_smooth[fi]]).tolist()],
               'shapes': [betas_init.cpu().numpy().tolist()]}]
    with open(os.path.join(OUT_DIR, 'smpl', f'{fi:06d}.json'), 'w') as f:
        json.dump(result, f)

# NPZ 아카이브
np.savez(os.path.join(OUT_DIR, 'all_params_7view.npz'),
         Rh=all_results['Rh'], Th=all_results['Th'],
         body_pose=all_results['body_pose'],
         Rh_smooth=Rh_smooth, Th_smooth=Th_smooth,
         body_pose_smooth=bp_smooth,
         betas=all_results['betas'],
         reproj_errors=all_results['reproj_errors'])

total = time.time() - t0
mean_err = all_results['reproj_errors'].mean()
median_err = np.median(all_results['reproj_errors'])
under_100 = (all_results['reproj_errors'] < 100).sum()

print(f"\n{'='*60}")
print(f"DONE: {N_FRAMES} frames in {total:.0f}s ({N_FRAMES/total:.1f} fps)")
print(f"Reproj error: mean={mean_err:.1f}px, median={median_err:.1f}px")
print(f"Frames <100px: {under_100}/{N_FRAMES} ({100*under_100/N_FRAMES:.1f}%)")
print(f"Frames <50px:  {(all_results['reproj_errors']<50).sum()}/{N_FRAMES}")
print(f"Min/Max: {all_results['reproj_errors'].min():.1f} / {all_results['reproj_errors'].max():.1f}px")
print(f"Output: {OUT_DIR}")
print(f"{'='*60}")

# 품질 게이트 G3 확인
if mean_err < 80:
    print("✓ G3 PASS: mean reproj < 80px → GART 학습 진행 가능")
elif mean_err < 100:
    print("△ G3 MARGINAL: mean reproj < 100px → 주의하며 진행")
else:
    print("✗ G3 FAIL: mean reproj >= 100px → λ 재조정 필요")
