"""
Phase C: 7-View DLT 삼각측량.

Phase B에서 생성한 Body25 키포인트를 7개 카메라에서 삼각측량하여
월드 프레임 3D 관절 좌표를 생성합니다.

서버 실행:
  python ~/server_scripts/02b_triangulate_sam3d.py

출력: /home/elicer/sam3d_triangulated/triangulated_3d.npz
"""
import os
import json
import numpy as np
import cv2

# ============================================================
# 설정
# ============================================================
USE_VGGT = True
CALIB_JSON = '/home/elicer/vggt_calibration_result.json'
R_FIX_PATH = '/home/elicer/R_fix_vggt_to_smpl.npy'

KP_DIR = '/home/elicer/sam3d_kp_body25'  # Phase B 출력
OUT_DIR = '/home/elicer/sam3d_triangulated'

CAM_NAMES = ['cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'cam6', 'cam7']
N_FRAMES = 999  # min(cam1=999, cam3-7=1000)
N_JOINTS = 25   # Body25
MIN_VIEWS = 3
CONF_THRESH = 0.3
MAX_REPROJ = 50.0  # px

# cam6 왜곡 (COLMAP에서 가져옴)
CAM6_K1 = -0.346

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 캘리브레이션 로드
# ============================================================
print("Loading calibration...")
with open(CALIB_JSON) as f:
    calib = json.load(f)

R_fix = None
if USE_VGGT and R_FIX_PATH:
    R_fix = np.load(R_FIX_PATH)

cameras = []
for ci, cn in enumerate(CAM_NAMES):
    c = calib[cn]
    K = np.array(c['K'], dtype=np.float64)
    R = np.array(c['R'], dtype=np.float64)
    t = np.array(c['t'], dtype=np.float64).reshape(3, 1)

    if R_fix is not None:
        R = R @ R_fix.T

    # Projection matrix P = K @ [R|t]
    Rt = np.hstack([R, t])
    P = K @ Rt

    # cam6 왜곡 파라미터
    D = np.zeros(5)
    if cn == 'cam6':
        D[0] = CAM6_K1

    cameras.append({
        'name': cn,
        'K': K, 'R': R, 't': t, 'P': P, 'D': D,
    })
    print(f"  {cn}: f={K[0,0]:.1f}")


# ============================================================
# DLT 삼각측량 함수
# ============================================================
def triangulate_dlt(projections, points_2d, confidences=None):
    """N-view DLT triangulation."""
    n = len(projections)
    if n < 2:
        return None
    if confidences is None:
        confidences = [1.0] * n

    A = np.zeros((2 * n, 4))
    for i, (P, uv, w) in enumerate(zip(projections, points_2d, confidences)):
        A[2*i]     = w * (uv[0] * P[2] - P[0])
        A[2*i + 1] = w * (uv[1] * P[2] - P[1])

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-10:
        return None
    return X[:3] / X[3]


def reproject(X, cam):
    """3D point → 2D pixel."""
    x_cam = cam['K'] @ (cam['R'] @ X.reshape(3, 1) + cam['t'])
    return np.array([x_cam[0, 0] / x_cam[2, 0], x_cam[1, 0] / x_cam[2, 0]])


# ============================================================
# 키포인트 로드
# ============================================================
print("\nLoading keypoints...")
all_kp2d = np.zeros((N_FRAMES, len(CAM_NAMES), N_JOINTS, 3), dtype=np.float32)

for ci, cn in enumerate(CAM_NAMES):
    cam_dir = os.path.join(KP_DIR, cn)
    for fi in range(N_FRAMES):
        json_path = os.path.join(cam_dir, f'frame_{fi:06d}.json')
        if not os.path.exists(json_path):
            continue
        with open(json_path) as f:
            data = json.load(f)
        if data.get('people'):
            kps = np.array(data['people'][0]['pose_keypoints_2d']).reshape(-1, 3)
            all_kp2d[fi, ci, :len(kps)] = kps

            # cam6 왜곡 보정
            if abs(cameras[ci]['D'][0]) > 0.05:
                K_np = cameras[ci]['K']
                D_np = cameras[ci]['D'][:4].reshape(1, 4)
                pts = kps[:, :2].reshape(-1, 1, 2).astype(np.float64)
                pts_undist = cv2.undistortPoints(pts, K_np, D_np, P=K_np)
                all_kp2d[fi, ci, :len(kps), :2] = pts_undist.reshape(-1, 2).astype(np.float32)

print(f"  Loaded: {N_FRAMES} frames x {len(CAM_NAMES)} cams")

# ============================================================
# 삼각측량
# ============================================================
print("\nTriangulating...")
joints_3d = np.zeros((N_FRAMES, N_JOINTS, 3), dtype=np.float32)
reproj_errors = np.full((N_FRAMES, N_JOINTS), np.inf, dtype=np.float32)
n_valid = np.zeros((N_FRAMES, N_JOINTS), dtype=np.int32)

for fi in range(N_FRAMES):
    for ji in range(N_JOINTS):
        # 유효 뷰 수집
        valid_P, valid_uv, valid_conf, valid_cams = [], [], [], []
        for ci in range(len(CAM_NAMES)):
            conf = all_kp2d[fi, ci, ji, 2]
            if conf < CONF_THRESH:
                continue
            valid_P.append(cameras[ci]['P'])
            valid_uv.append(all_kp2d[fi, ci, ji, :2])
            valid_conf.append(float(conf))
            valid_cams.append(cameras[ci])

        if len(valid_P) < MIN_VIEWS:
            continue

        # 1차 삼각측량
        X = triangulate_dlt(valid_P, valid_uv, valid_conf)
        if X is None:
            continue

        # 리프로젝션 에러 계산 + 이상치 제거
        errors = []
        for cam, uv in zip(valid_cams, valid_uv):
            u_proj = reproject(X, cam)
            errors.append(np.linalg.norm(u_proj - uv))
        errors = np.array(errors)

        # 2σ 이상치 제거 후 재삼각측량
        if len(errors) > 3:
            mean_e, std_e = errors.mean(), errors.std()
            inlier = errors < (mean_e + 2 * std_e)
            if inlier.sum() >= MIN_VIEWS and inlier.sum() < len(errors):
                P2 = [p for p, ok in zip(valid_P, inlier) if ok]
                uv2 = [u for u, ok in zip(valid_uv, inlier) if ok]
                c2 = [c for c, ok in zip(valid_conf, inlier) if ok]
                cams2 = [cm for cm, ok in zip(valid_cams, inlier) if ok]
                X2 = triangulate_dlt(P2, uv2, c2)
                if X2 is not None:
                    X = X2
                    valid_cams = cams2
                    valid_uv = uv2
                    errors = []
                    for cam, uv in zip(valid_cams, valid_uv):
                        u_proj = reproject(X, cam)
                        errors.append(np.linalg.norm(u_proj - uv))
                    errors = np.array(errors)

        mean_err = errors.mean() if len(errors) > 0 else float('inf')

        if mean_err < MAX_REPROJ:
            joints_3d[fi, ji] = X
            reproj_errors[fi, ji] = mean_err
            n_valid[fi, ji] = len(errors)

    if (fi + 1) % 100 == 0 or fi == 0:
        valid_mask = reproj_errors[fi] < MAX_REPROJ
        n_ok = valid_mask.sum()
        mean_e = reproj_errors[fi][valid_mask].mean() if n_ok > 0 else float('inf')
        print(f"  [{fi+1}/{N_FRAMES}] joints={n_ok}/25, mean_reproj={mean_e:.1f}px")

# ============================================================
# 결과 저장
# ============================================================
out_path = os.path.join(OUT_DIR, 'triangulated_3d.npz')
np.savez_compressed(
    out_path,
    joints_3d=joints_3d,            # (N, 25, 3)
    reproj_errors=reproj_errors,     # (N, 25)
    n_valid_views=n_valid,           # (N, 25)
)

# 통계
body_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
valid_mask = reproj_errors[:, body_joints] < MAX_REPROJ
mean_reproj = reproj_errors[:, body_joints][valid_mask].mean()
coverage = valid_mask.mean() * 100

print(f"\n{'='*60}")
print(f"Results saved: {out_path}")
print(f"  Body joints (0-14) mean reproj: {mean_reproj:.1f}px")
print(f"  Body joints coverage: {coverage:.1f}%")
print(f"{'='*60}")

# G1 게이트 확인
if mean_reproj < 30:
    print(f"\nG1 PASS: mean reproj {mean_reproj:.1f}px < 30px")
else:
    print(f"\nG1 WARNING: mean reproj {mean_reproj:.1f}px >= 30px")
    print("  → 캘리브레이션 확인 필요")
