"""
SKILL-5+6: GART 7-View 데이터 준비
- ZJU-MoCap 형식 이미지 디렉토리 구성
- SMPL 파라미터 변환 (JSON → NPY)
- annots.npy 생성
- cam1 이미지 90도 회전

서버 실행: python 03_prepare_gart_data.py
"""
import os, json, glob, shutil
import numpy as np
import cv2

# ============================================================
# 경로 (서버에 맞게 수정)
# ============================================================
IMAGE_BASE = '/home/elicer/images'        # 원본 이미지
REPROJ_DIR = '/home/elicer/reproj_7view_result'  # 02 스크립트 출력
CALIB_JSON = '/home/elicer/server_scripts/colmap_7view_for_reproj.json'
GART_DATA  = '/home/elicer/gart_7view_data'

CAM_NAMES = ['cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'cam6', 'cam7']
N_FRAMES = 999

# ============================================================
# Step 1: 이미지 디렉토리 구성
# ============================================================
print("=== Step 1: Image directory setup ===")

for ci, cn in enumerate(CAM_NAMES):
    dst_dir = os.path.join(GART_DATA, 'images', str(ci))
    os.makedirs(dst_dir, exist_ok=True)

    src_dir = os.path.join(IMAGE_BASE, cn)
    src_files = sorted(glob.glob(os.path.join(src_dir, '*.jpg')))

    if not src_files:
        src_files = sorted(glob.glob(os.path.join(src_dir, '*.png')))

    for fi, src_path in enumerate(src_files[:N_FRAMES]):
        dst_path = os.path.join(dst_dir, f'{fi:06d}.jpg')

        if cn == 'cam1':
            # cam1: 세로→가로 90도 CW 회전
            img = cv2.imread(src_path)
            if img is not None:
                rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                cv2.imwrite(dst_path, rot)
            else:
                print(f"  Warning: cannot read {src_path}")
        else:
            # cam2-7: 심볼릭 링크 (디스크 절약)
            if not os.path.exists(dst_path):
                try:
                    os.symlink(os.path.abspath(src_path), dst_path)
                except:
                    shutil.copy2(src_path, dst_path)

    print(f"  {cn} → {dst_dir}: {min(len(src_files), N_FRAMES)} images")

# ============================================================
# Step 2: SMPL 파라미터 변환 (JSON → NPY)
# ============================================================
print("\n=== Step 2: SMPL params conversion ===")
smpl_dir = os.path.join(GART_DATA, 'smpl_params')
os.makedirs(smpl_dir, exist_ok=True)

src_smpl = os.path.join(REPROJ_DIR, 'smpl')
n_converted = 0

for fi in range(N_FRAMES):
    src_path = os.path.join(src_smpl, f'{fi:06d}.json')
    if not os.path.exists(src_path):
        print(f"  Warning: {src_path} not found")
        continue

    with open(src_path) as f:
        d = json.load(f)[0]

    poses = np.array(d['poses'][0])  # (72,) = global(3) + body(69)
    np.save(os.path.join(smpl_dir, f'{fi:03d}.npy'), {
        'global_orient': np.array(d['Rh'][0], dtype=np.float32),
        'body_pose': poses[3:].astype(np.float32),
        'transl': np.array(d['Th'][0], dtype=np.float32),
        'betas': np.array(d['shapes'][0], dtype=np.float32),
    })
    n_converted += 1

print(f"  Converted {n_converted} frames to {smpl_dir}")

# ============================================================
# Step 3: annots.npy 생성
# ============================================================
print("\n=== Step 3: Generate annots.npy ===")

with open(CALIB_JSON) as f:
    calib = json.load(f)

K_list, R_list, T_list, D_list = [], [], [], []
for cn in CAM_NAMES:
    c = calib[cn]
    K_list.append(np.array(c['K'], dtype=np.float32))
    R_list.append(np.array(c['R'], dtype=np.float32))
    # GART expects T in millimeters? Check GART convention
    T_list.append(np.array(c['t'], dtype=np.float32).reshape(3, 1))
    D_list.append(np.zeros(5, dtype=np.float32))  # 왜곡은 이미 보정됨

# 이미지 경로 리스트
ims_list = []
for fi in range(N_FRAMES):
    frame_ims = [f'{ci}/{fi:06d}.jpg' for ci in range(len(CAM_NAMES))]
    ims_list.append({'ims': frame_ims})

annots = {
    'cams': {
        'K': K_list,
        'R': R_list,
        'T': T_list,
        'D': D_list,
    },
    'ims': ims_list,
}

annots_path = os.path.join(GART_DATA, 'annots.npy')
np.save(annots_path, annots, allow_pickle=True)
print(f"  Saved annots.npy to {annots_path}")

# ============================================================
# 검증
# ============================================================
print("\n=== Verification ===")
for ci, cn in enumerate(CAM_NAMES):
    img_dir = os.path.join(GART_DATA, 'images', str(ci))
    n_imgs = len(os.listdir(img_dir))
    print(f"  {cn} (idx {ci}): {n_imgs} images")

n_smpl = len(os.listdir(smpl_dir))
print(f"  SMPL params: {n_smpl} files")
print(f"  annots.npy: {os.path.getsize(annots_path)} bytes")

# cam1 회전 확인
cam1_img = os.path.join(GART_DATA, 'images', '0', '000000.jpg')
if os.path.exists(cam1_img):
    img = cv2.imread(cam1_img)
    if img is not None:
        h, w = img.shape[:2]
        print(f"  cam1 rotated: {w}x{h} {'✓ (landscape)' if w > h else '✗ (still portrait!)'}")

print(f"\n✓ GART data ready at: {GART_DATA}")
