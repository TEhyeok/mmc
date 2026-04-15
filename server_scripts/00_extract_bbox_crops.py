"""
모든 뷰에서 SAM3 마스크 → 투구자 bbox 추출 + bbox crop 이미지 생성.

입력: data/segments/masks/cam{1-7}/frame_*.png (이진 마스크)
출력:
  - data/sam3d_bbox/cam{1-7}/frame_*.json (bbox 좌표)
  - data/sam3d_bbox/bbox_summary.json (전체 통계)

서버에서는 이 bbox를 SAM 3D Body의 bboxes 인자로 전달합니다.
로컬/서버 모두 실행 가능 (GPU 불필요).
"""
import os
import json
import glob
import numpy as np
import cv2

# ============================================================
# 경로 설정
# ============================================================
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASK_DIR = os.path.join(BASE, 'data', 'segments', 'masks')
OUT_DIR = os.path.join(BASE, 'data', 'sam3d_bbox')

CAM_NAMES = ['cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'cam6', 'cam7']
PADDING = 1.25  # SAM 3D Body 기본 패딩과 동일

# ============================================================
# bbox 추출 함수
# ============================================================
def mask_to_bbox(mask, padding=1.25):
    """이진 마스크에서 패딩된 bbox 추출.

    Args:
        mask: (H, W) uint8, 0 or 255
        padding: bbox 확장 비율 (1.25 = 25% 패딩)

    Returns:
        [x1, y1, x2, y2] 또는 None (마스크가 비어있으면)
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    # 패딩 적용
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    new_w = w * padding
    new_h = h * padding

    H, W = mask.shape
    x1_pad = max(0, int(cx - new_w / 2))
    y1_pad = max(0, int(cy - new_h / 2))
    x2_pad = min(W, int(cx + new_w / 2))
    y2_pad = min(H, int(cy + new_h / 2))

    return [x1_pad, y1_pad, x2_pad, y2_pad]


# ============================================================
# 메인
# ============================================================
print("=" * 60)
print("SAM3 마스크 → 투구자 bbox 추출 (7-cam)")
print("=" * 60)

summary = {}

for cam in CAM_NAMES:
    cam_mask_dir = os.path.join(MASK_DIR, cam)
    cam_out_dir = os.path.join(OUT_DIR, cam)
    os.makedirs(cam_out_dir, exist_ok=True)

    mask_files = sorted(glob.glob(os.path.join(cam_mask_dir, 'frame_*.png')))

    if not mask_files:
        print(f"  {cam}: NO MASKS FOUND")
        continue

    bboxes = []
    empty_frames = []

    for fi, mf in enumerate(mask_files):
        mask = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
        bbox = mask_to_bbox(mask, padding=PADDING)

        if bbox is None:
            empty_frames.append(fi)
            # 이전 프레임 bbox 사용 (temporal propagation)
            if bboxes:
                bbox = bboxes[-1]['bbox']
            else:
                bbox = [0, 0, mask.shape[1], mask.shape[0]]  # 전체 이미지

        frame_name = os.path.basename(mf).replace('.png', '')
        bbox_data = {
            'frame': frame_name,
            'frame_idx': fi,
            'bbox': bbox,  # [x1, y1, x2, y2]
            'bbox_width': bbox[2] - bbox[0],
            'bbox_height': bbox[3] - bbox[1],
            'empty_mask': fi in empty_frames,
        }
        bboxes.append(bbox_data)

        # 개별 JSON 저장
        with open(os.path.join(cam_out_dir, f'{frame_name}.json'), 'w') as f:
            json.dump(bbox_data, f)

    # 카메라 통계
    widths = [b['bbox_width'] for b in bboxes]
    heights = [b['bbox_height'] for b in bboxes]

    # 첫 프레임 이미지 해상도
    sample_mask = cv2.imread(mask_files[0], cv2.IMREAD_GRAYSCALE)
    img_h, img_w = sample_mask.shape

    cam_summary = {
        'cam': cam,
        'n_frames': len(bboxes),
        'n_empty': len(empty_frames),
        'image_size': [img_w, img_h],
        'bbox_width_mean': int(np.mean(widths)),
        'bbox_height_mean': int(np.mean(heights)),
        'bbox_width_range': [int(np.min(widths)), int(np.max(widths))],
        'bbox_height_range': [int(np.min(heights)), int(np.max(heights))],
    }
    summary[cam] = cam_summary

    print(f"  {cam}: {len(bboxes)} frames, {len(empty_frames)} empty, "
          f"img={img_w}x{img_h}, "
          f"bbox_avg={int(np.mean(widths))}x{int(np.mean(heights))}")

# 전체 bbox 배열 저장 (서버에서 빠르게 로드)
print("\n--- Saving bbox arrays ---")
for cam in CAM_NAMES:
    cam_out_dir = os.path.join(OUT_DIR, cam)
    json_files = sorted(glob.glob(os.path.join(cam_out_dir, '*.json')))

    all_bboxes = []
    for jf in json_files:
        with open(jf) as f:
            d = json.load(f)
        all_bboxes.append(d['bbox'])

    np.save(os.path.join(OUT_DIR, f'{cam}_bboxes.npy'),
            np.array(all_bboxes, dtype=np.float32))
    print(f"  {cam}: saved {cam}_bboxes.npy ({len(all_bboxes)} frames)")

# 전체 요약 저장
with open(os.path.join(OUT_DIR, 'bbox_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✓ 전체 결과: {OUT_DIR}")
print(f"✓ 요약: {OUT_DIR}/bbox_summary.json")
print(f"✓ 서버 사용: np.load('{cam}_bboxes.npy') → SAM 3D Body bboxes 인자")
