"""
Phase B: SAM 3D Body NPZ → OpenPose Body25 JSON 변환.

SAM 3D Body의 pred_keypoints_2d (MHR 70-joint) 를
OpenPose Body25 포맷 JSON으로 변환합니다.
02_reproj_7view_sequential.py가 이 JSON을 바로 소비 가능합니다.

서버 실행:
  python ~/server_scripts/02a_sam3d_to_keypoints.py

출력: /home/elicer/sam3d_kp_body25/{cam1..cam7}/{frame_name}.json
"""
import os
import json
import numpy as np

# ============================================================
# MHR 70-joint → OpenPose Body25 매핑
# ============================================================
# MHR indices (from sam_3d_body/metadata/mhr70.py):
#   0:nose 1:L_eye 2:R_eye 3:L_ear 4:R_ear
#   5:L_shoulder 6:R_shoulder 7:L_elbow 8:R_elbow
#   9:L_hip 10:R_hip 11:L_knee 12:R_knee 13:L_ankle 14:R_ankle
#   15:L_big_toe 16:L_small_toe 17:L_heel
#   18:R_big_toe 19:R_small_toe 20:R_heel
#   41:R_wrist 62:L_wrist 69:neck

MHR_TO_BODY25 = {
    0: 0,    # Nose
    1: 69,   # Neck
    2: 6,    # R_Shoulder
    3: 8,    # R_Elbow
    4: 41,   # R_Wrist
    5: 5,    # L_Shoulder
    6: 7,    # L_Elbow
    7: 62,   # L_Wrist
    8: None, # MidHip = avg(L_hip, R_hip)
    9: 10,   # R_Hip
    10: 12,  # R_Knee
    11: 14,  # R_Ankle
    12: 9,   # L_Hip
    13: 11,  # L_Knee
    14: 13,  # L_Ankle
    15: 2,   # R_Eye
    16: 5,   # L_Eye  (approximation, Body25 idx 16)
    17: 4,   # R_Ear
    18: 3,   # L_Ear
    # 19-24: L_BigToe, L_SmallToe, L_Heel, R_BigToe, R_SmallToe, R_Heel
    19: 15,  # L_BigToe
    20: 16,  # L_SmallToe
    21: 17,  # L_Heel
    22: 18,  # R_BigToe
    23: 19,  # R_SmallToe
    24: 20,  # R_Heel
}

# ============================================================
# 경로
# ============================================================
SAM3D_DIR = '/home/elicer/sam3d_results'
OUT_DIR = '/home/elicer/sam3d_kp_body25'

CAM_CONFIG = {
    'cam1': {'n': 999},
    'cam2': {'n': 999},
    'cam3': {'n': 1000},
    'cam4': {'n': 1000},
    'cam5': {'n': 1000},
    'cam6': {'n': 1000},
    'cam7': {'n': 1000},
}

# ============================================================
# 변환
# ============================================================
print("=" * 60)
print("Phase B: SAM 3D Body MHR70 → OpenPose Body25 변환")
print("=" * 60)

for cam, cfg in CAM_CONFIG.items():
    res_dir = os.path.join(SAM3D_DIR, cam)
    out_cam = os.path.join(OUT_DIR, cam)
    os.makedirs(out_cam, exist_ok=True)

    n = cfg['n']
    converted = 0
    missing = 0

    for fi in range(n):
        npz_path = os.path.join(res_dir, f'{fi:06d}.npz')
        if not os.path.exists(npz_path):
            missing += 1
            continue

        d = np.load(npz_path)
        kp2d_mhr = d['pred_keypoints_2d']  # (70, 2)

        # Body25 변환: (25, 3) [x, y, confidence]
        body25 = np.zeros((25, 3), dtype=np.float32)

        for b25_idx, mhr_idx in MHR_TO_BODY25.items():
            if mhr_idx is None:
                # MidHip = avg(L_hip=9, R_hip=10)
                body25[b25_idx, :2] = (kp2d_mhr[9] + kp2d_mhr[10]) / 2
                body25[b25_idx, 2] = 1.0
            else:
                if mhr_idx < len(kp2d_mhr):
                    body25[b25_idx, :2] = kp2d_mhr[mhr_idx]
                    body25[b25_idx, 2] = 1.0

        # OpenPose JSON 포맷으로 저장
        out_data = {
            'people': [{
                'pose_keypoints_2d': body25.flatten().tolist()
            }]
        }

        out_path = os.path.join(out_cam, f'frame_{fi:06d}.json')
        with open(out_path, 'w') as f:
            json.dump(out_data, f)

        converted += 1

    print(f"  {cam}: {converted} frames converted, {missing} missing")

print(f"\nOutput: {OUT_DIR}")
print("Done.")
