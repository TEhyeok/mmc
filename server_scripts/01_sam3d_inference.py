"""
SAM 3D Body 7-camera 배치 추론 (서버용).

사전 추출된 SAM3 bbox를 사용하여 투구자만 정확히 추론합니다.
각 카메라의 원본 이미지 + bbox → MHR 메시 (18,439 vtx) 출력.

서버 실행:
  conda activate sam3d
  cd ~/sam-3d-body
  export PYOPENGL_PLATFORM=egl
  python ~/server_scripts/01_sam3d_inference.py

사전 조건:
  - SAM 3D Body 설치 완료 (conda env: sam3d)
  - 체크포인트 다운로드 완료 (~/sam-3d-body/checkpoints/)
  - bbox 파일: data/sam3d_bbox/cam{1-7}_bboxes.npy (로컬에서 생성 완료)
"""
import os
import sys
import time
import numpy as np
import cv2

# ============================================================
# 경로 (서버에 맞게 수정)
# ============================================================
SAM3D_DIR = os.path.expanduser('~/sam-3d-body')
sys.path.insert(0, SAM3D_DIR)

CHECKPOINT = os.path.join(SAM3D_DIR, 'checkpoints', 'sam-3d-body-dinov3', 'model.ckpt')
MHR_MODEL = os.path.join(SAM3D_DIR, 'checkpoints', 'sam-3d-body-dinov3', 'assets', 'mhr_model.pt')

# 이미지 디렉토리 (서버: easymocap_data/images/{1-7})
IMAGE_DIRS = {
    'cam1': '/home/elicer/easymocap_data/images/1',  # 세로 1080x1920 → 그대로 입력!
    'cam2': '/home/elicer/easymocap_data/images/2',
    'cam3': '/home/elicer/easymocap_data/images/3',
    'cam4': '/home/elicer/easymocap_data/images/4',
    'cam5': '/home/elicer/easymocap_data/images/5',
    'cam6': '/home/elicer/easymocap_data/images/6',
    'cam7': '/home/elicer/easymocap_data/images/7',
}

# bbox (로컬에서 추출, 서버로 scp)
BBOX_DIR = '/home/elicer/data/sam3d_bbox'

# 출력
OUT_DIR = '/home/elicer/sam3d_results'

# 설정
N_FRAMES = 999  # cam1,2는 999, cam3-7은 1000
TEST_MODE = False  # True: cam3 100프레임만 테스트
TEST_CAM = 'cam3'
TEST_FRAMES = 100

# ============================================================
# SAM 3D Body 로드
# ============================================================
print("Loading SAM 3D Body model...")
os.environ['PYOPENGL_PLATFORM'] = 'egl'  # headless 서버

from notebook.utils import setup_sam_3d_body

estimator = setup_sam_3d_body(
    hf_repo_id='facebook/sam-3d-body-dinov3',
    detector_name='',  # bbox 직접 제공 → 검출기 불필요
    segmentor_path='',  # 세그멘터 불필요
)
print("Model loaded.")

# ============================================================
# 추론 루프
# ============================================================
if TEST_MODE:
    cams_to_run = {TEST_CAM: IMAGE_DIRS[TEST_CAM]}
    max_frames = TEST_FRAMES
    print(f"\n*** TEST MODE: {TEST_CAM} x {TEST_FRAMES} frames ***\n")
else:
    cams_to_run = IMAGE_DIRS
    max_frames = N_FRAMES

total_t0 = time.time()
total_processed = 0

for cam_name, cam_dir in cams_to_run.items():
    print(f"\n{'='*50}")
    print(f"Processing {cam_name}...")
    print(f"{'='*50}")

    # bbox 로드 (tight: 패딩 없음, SAM 3D Body가 내부적으로 1.25x 적용)
    bbox_path = os.path.join(BBOX_DIR, f'{cam_name}_bboxes_tight.npy')
    if os.path.exists(bbox_path):
        all_bboxes = np.load(bbox_path)  # (N, 4) [x1, y1, x2, y2]
        print(f"  Loaded {len(all_bboxes)} bboxes from {bbox_path}")
        use_bbox = True
    else:
        print(f"  ⚠ bbox file not found: {bbox_path}")
        print(f"    → ViTDet 자동 검출 사용")
        all_bboxes = None
        use_bbox = False

    # 이미지 파일 목록
    img_files = sorted([f for f in os.listdir(cam_dir)
                        if f.endswith(('.jpg', '.png'))])[:max_frames]

    # 출력 디렉토리
    out_cam = os.path.join(OUT_DIR, cam_name)
    os.makedirs(out_cam, exist_ok=True)

    # 이미 처리된 프레임 스킵
    existing = set(f.replace('.npz', '') for f in os.listdir(out_cam) if f.endswith('.npz'))

    cam_t0 = time.time()
    processed = 0
    errors = 0

    for fi, fn in enumerate(img_files):
        frame_id = f'{fi:06d}'
        if frame_id in existing:
            continue

        # 이미지 로드
        img_path = os.path.join(cam_dir, fn)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  ⚠ Cannot read: {img_path}")
            errors += 1
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 추론
        try:
            if use_bbox and fi < len(all_bboxes):
                bbox = all_bboxes[fi].reshape(1, 4)  # (1, 4)
                results = estimator.process_one_image(img_rgb, bboxes=bbox)
            else:
                results = estimator.process_one_image(img_rgb)

            if not results or len(results) == 0:
                print(f"  ⚠ Frame {fi}: no detection")
                errors += 1
                continue

            outputs = results[0]  # 첫 번째 사람 (투구자)

            # 결과 저장
            save_dict = {
                'pred_vertices': outputs['pred_vertices'].astype(np.float32),  # (18439, 3)
            }
            # 선택적 필드
            for key in ['focal_length', 'pred_cam_t', 'pred_keypoints_3d',
                        'pred_keypoints_2d', 'pred_body_pose', 'pred_betas',
                        'pred_global_orient', 'bbox']:
                if key in outputs:
                    val = outputs[key]
                    if hasattr(val, 'numpy'):
                        val = val.numpy()
                    save_dict[key] = np.array(val, dtype=np.float32)

            np.savez_compressed(os.path.join(out_cam, f'{frame_id}.npz'), **save_dict)
            processed += 1

        except Exception as e:
            print(f"  ✗ Frame {fi} error: {e}")
            errors += 1

        # 로깅
        if (fi + 1) % 50 == 0 or fi == 0:
            elapsed = time.time() - cam_t0
            fps = processed / elapsed if elapsed > 0 else 0
            remaining = len(img_files) - fi - 1
            eta = remaining / fps if fps > 0 else 0
            print(f"  {cam_name} [{fi+1}/{len(img_files)}]: "
                  f"{fps:.2f} fps, ETA {eta:.0f}s, errors={errors}")

    cam_elapsed = time.time() - cam_t0
    total_processed += processed
    print(f"  {cam_name} DONE: {processed} frames in {cam_elapsed:.0f}s "
          f"({processed/cam_elapsed:.2f} fps), errors={errors}")

# ============================================================
# 완료
# ============================================================
total_elapsed = time.time() - total_t0
print(f"\n{'='*60}")
print(f"TOTAL: {total_processed} frames in {total_elapsed:.0f}s "
      f"({total_processed/total_elapsed:.2f} fps)")
print(f"Output: {OUT_DIR}")
print(f"{'='*60}")

# 결과 확인
print("\n--- Result check ---")
for cam in cams_to_run:
    out_cam = os.path.join(OUT_DIR, cam)
    n_files = len([f for f in os.listdir(out_cam) if f.endswith('.npz')])
    # 첫 파일 확인
    first = sorted([f for f in os.listdir(out_cam) if f.endswith('.npz')])
    if first:
        d = np.load(os.path.join(out_cam, first[0]))
        vtx_shape = d['pred_vertices'].shape
        keys = list(d.keys())
        print(f"  {cam}: {n_files} files, vtx={vtx_shape}, keys={keys}")
    else:
        print(f"  {cam}: EMPTY")

print("\n✓ G0 게이트: pred_vertices shape이 (18439, 3)이면 통과")
