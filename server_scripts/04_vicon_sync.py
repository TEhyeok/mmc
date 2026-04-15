"""
SKILL-10: Vicon C3D 파싱 + 시간/공간 동기화

로컬 실행 가능 (GPU 불필요):
  python 04_vicon_sync.py

입력:
  - Vicon C3D: data/markerbased/subject_1/set_002/set_002.c3d
  - OpenPose 3D 또는 SMPL joints
출력:
  - 시간 오프셋 (프레임 수)
  - 공간 정렬 (R, t, s)
  - 프레임 대응 테이블
"""
import numpy as np
import json
import os

# ============================================================
# 경로
# ============================================================
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C3D_PATH = os.path.join(BASE, 'data', 'markerbased', 'subject_1', 'set_002', 'set_002.c3d')
CSV_PATH = os.path.join(BASE, 'data', 'markerbased', 'subject_1', 'set_002', 'set_002.csv')
SMPL_JOINTS_PATH = os.path.join(BASE, 'data', 'smpl_fitting', 'smpl_joints.npy')
KP_DIR = os.path.join(BASE, 'data', 'openpose_smoothed_v2')
OUT_DIR = os.path.join(BASE, 'data', 'vicon_sync')

FPS = 240.0

# ============================================================
# Step 1: C3D 파싱
# ============================================================
print("=== Step 1: Parse Vicon C3D ===")

try:
    import ezc3d
    c = ezc3d.c3d(C3D_PATH)
    markers_data = c['data']['points']  # (4, n_markers, n_frames)
    marker_labels = c['parameters']['POINT']['LABELS']['value']
    n_markers = markers_data.shape[1]
    n_frames_vicon = markers_data.shape[2]
    vicon_fps = c['parameters']['POINT']['RATE']['value'][0]

    # (n_frames, n_markers, 3) — mm 단위
    vicon_markers = markers_data[:3].transpose(2, 1, 0)  # (n_frames, n_markers, 3)

    print(f"  Markers: {n_markers} ({', '.join(marker_labels[:5])}...)")
    print(f"  Frames: {n_frames_vicon}")
    print(f"  FPS: {vicon_fps}")
    print(f"  Duration: {n_frames_vicon/vicon_fps:.2f}s")

except ImportError:
    try:
        import c3d as c3d_lib
        reader = c3d_lib.Reader(open(C3D_PATH, 'rb'))
        frames = []
        for i, pts, analog in reader.read_frames():
            frames.append(pts[:, :3])  # (n_markers, 3)
        vicon_markers = np.array(frames)  # (n_frames, n_markers, 3)
        n_frames_vicon = len(frames)
        marker_labels = [l.strip() for l in reader.point_labels]
        n_markers = vicon_markers.shape[1]
        vicon_fps = reader.point_rate
        print(f"  Markers: {n_markers}")
        print(f"  Frames: {n_frames_vicon}")
        print(f"  FPS: {vicon_fps}")
    except Exception as e:
        print(f"  Error: {e}")
        print("  C3D 파싱 실패. ezc3d 또는 c3d 패키지를 설치하세요.")
        exit(1)

# ============================================================
# Step 2: 핵심 마커 추출
# ============================================================
print("\n=== Step 2: Extract key markers ===")

# Plug-in Gait 마커 → 인덱스 매핑
marker_idx = {}
for i, label in enumerate(marker_labels):
    marker_idx[label.strip().upper()] = i

# 핵심 마커 (시간 동기화에 사용)
KEY_MARKERS = {
    'RWRA': 'Right wrist A',
    'RWRB': 'Right wrist B',
    'RELB': 'Right elbow',
    'RSHO': 'Right shoulder',
    'LSHO': 'Left shoulder',
    'RANK': 'Right ankle',
    'LANK': 'Left ankle',
}

for name, desc in KEY_MARKERS.items():
    if name in marker_idx:
        idx = marker_idx[name]
        traj = vicon_markers[:, idx, :]
        valid = np.sum(~np.isnan(traj[:, 0]))
        print(f"  {name} ({desc}): idx={idx}, valid={valid}/{n_frames_vicon}")
    else:
        print(f"  {name} ({desc}): NOT FOUND in labels")

# 손목 중점 (동기화 신호)
if 'RWRA' in marker_idx and 'RWRB' in marker_idx:
    rwrist_vicon = (vicon_markers[:, marker_idx['RWRA'], :] +
                    vicon_markers[:, marker_idx['RWRB'], :]) / 2
else:
    rwrist_vicon = vicon_markers[:, marker_idx.get('RWRA', 0), :]

print(f"\n  Right wrist trajectory: shape={rwrist_vicon.shape}")

# ============================================================
# Step 3: RGB 신호 추출 (OpenPose 손목 궤적)
# ============================================================
print("\n=== Step 3: Extract RGB wrist signal ===")

# OpenPose Body25: joint 4 = RWrist
# 여러 카메라에서 삼각측량하거나, 단일 카메라의 2D 궤적 사용
# 여기서는 SMPL joints를 사용 (3D)
if os.path.exists(SMPL_JOINTS_PATH):
    smpl_joints = np.load(SMPL_JOINTS_PATH)  # (N, 24, 3)
    # SMPL joint 21 = right wrist (SMPL convention)
    rwrist_rgb = smpl_joints[:, 21, :]  # (N, 3)
    n_frames_rgb = len(rwrist_rgb)
    print(f"  SMPL right wrist: {n_frames_rgb} frames")
else:
    # 폴백: OpenPose 2D 사용 (y좌표만)
    print("  SMPL joints not found, using OpenPose 2D (cam3)")
    kp_files = sorted(os.listdir(os.path.join(KP_DIR, 'cam3')))
    rwrist_y = []
    for fn in kp_files[:999]:
        with open(os.path.join(KP_DIR, 'cam3', fn)) as f:
            d = json.load(f)
        if d.get('people'):
            kps = np.array(d['people'][0]['pose_keypoints_2d']).reshape(-1, 3)
            rwrist_y.append(kps[4, 1])  # Body25 joint 4 = RWrist, y좌표
        else:
            rwrist_y.append(np.nan)
    rwrist_rgb = np.array(rwrist_y)
    n_frames_rgb = len(rwrist_rgb)
    print(f"  OpenPose RWrist Y: {n_frames_rgb} frames")

# ============================================================
# Step 4: 교차상관 시간 동기화
# ============================================================
print("\n=== Step 4: Cross-correlation time sync ===")
from scipy.signal import correlate

# 동기화 신호 선택: Y축 (수직 방향) — 투구 동작에서 가장 뚜렷
if rwrist_rgb.ndim == 2:
    sig_rgb = rwrist_rgb[:, 1]  # Y축
else:
    sig_rgb = rwrist_rgb  # 이미 1D

sig_vicon = rwrist_vicon[:, 2]  # Z축 (Vicon Y-up → 보통 Z가 수직)

# NaN 제거
sig_rgb = np.nan_to_num(sig_rgb, nan=np.nanmean(sig_rgb))
sig_vicon = np.nan_to_num(sig_vicon, nan=np.nanmean(sig_vicon))

# 정규화
s1 = (sig_rgb - sig_rgb.mean()) / (sig_rgb.std() + 1e-8)
s2 = (sig_vicon - sig_vicon.mean()) / (sig_vicon.std() + 1e-8)

# 교차상관
corr = correlate(s1, s2, mode='full')
corr /= max(len(s1), len(s2))
lags = np.arange(-len(s2) + 1, len(s1))

peak_idx = np.argmax(np.abs(corr))
offset_frames = int(lags[peak_idx])
offset_sec = offset_frames / FPS
peak_corr = float(corr[peak_idx])

print(f"  RGB frames: {n_frames_rgb}")
print(f"  Vicon frames: {n_frames_vicon}")
print(f"  Offset: {offset_frames} frames ({offset_sec:.4f}s)")
print(f"  Peak correlation: {peak_corr:.4f}")
print(f"  → Vicon frame = RGB frame + {offset_frames}")

if abs(peak_corr) < 0.3:
    print("  ⚠ 낮은 상관 — 신호 축(Y/Z) 재확인 필요")

# ============================================================
# Step 5: 프레임 대응 테이블
# ============================================================
print("\n=== Step 5: Frame correspondence ===")
os.makedirs(OUT_DIR, exist_ok=True)

# RGB frame i → Vicon frame i + offset
correspondence = []
for rgb_frame in range(n_frames_rgb):
    vicon_frame = rgb_frame + offset_frames
    if 0 <= vicon_frame < n_frames_vicon:
        correspondence.append({
            'rgb_frame': int(rgb_frame),
            'vicon_frame': int(vicon_frame),
        })

n_overlap = len(correspondence)
print(f"  Overlapping frames: {n_overlap}")
if n_overlap > 0:
    print(f"  RGB range: {correspondence[0]['rgb_frame']} - {correspondence[-1]['rgb_frame']}")
    print(f"  Vicon range: {correspondence[0]['vicon_frame']} - {correspondence[-1]['vicon_frame']}")

# 저장
sync_result = {
    'offset_frames': offset_frames,
    'offset_sec': offset_sec,
    'peak_correlation': peak_corr,
    'n_overlap_frames': n_overlap,
    'correspondence': correspondence,
    'rgb_signal': 'smpl_joints[:, 21, 1]' if os.path.exists(SMPL_JOINTS_PATH) else 'openpose_cam3_rwrist_y',
    'vicon_signal': 'rwrist_midpoint_z',
}

with open(os.path.join(OUT_DIR, 'sync_result.json'), 'w') as f:
    json.dump(sync_result, f, indent=2)

np.savez(os.path.join(OUT_DIR, 'vicon_markers.npz'),
         markers=vicon_markers,
         labels=marker_labels,
         fps=vicon_fps,
         n_frames=n_frames_vicon)

print(f"\n✓ Saved sync result to {OUT_DIR}/sync_result.json")
print(f"✓ Saved Vicon markers to {OUT_DIR}/vicon_markers.npz")
