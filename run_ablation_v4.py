"""
Ablation v4: SMPL body_pose axis-angle → 직접 오일러 분해
JCS 구축 불필요 — SMPL 킨매틱 트리가 이미 부모-자식 상대 회전

SMPL 23 joints kinematic tree:
  0: Pelvis → children: L_Hip(1), R_Hip(2), Spine1(3)
  3: Spine1 → Spine2(6)
  6: Spine2 → Spine3(9)
  9: Spine3 → Neck(12), L_Collar(13), R_Collar(14)
  13: L_Collar → L_Shoulder(16)
  14: R_Collar → R_Shoulder(17)  (투구팔)
  17: R_Shoulder → R_Elbow(19)  (투구팔)  ← 0-indexed: joint 17
  19: R_Elbow → R_Wrist(21)
  1: L_Hip → L_Knee(4)  (리드 다리)
  4: L_Knee → L_Ankle(7)

논문 8개 변수 매핑:
  Variable              SMPL relative rotation      Euler order
  Lead knee flexion     L_Knee (joint 4)            XYZ → [0]
  Trunk forward tilt    Spine1 (joint 3) composite  XZY → [0]
  Trunk lateral tilt    Spine1 (joint 3) composite  XZY → [1]
  Trunk axial rotation  Spine1 (joint 3) composite  XZY → [2]
  Shoulder abduction    R_Shoulder (joint 17)       YZX → [0]
  Shoulder horiz. abd.  R_Shoulder (joint 17)       YZX → [1]
  Shoulder rotation     R_Shoulder (joint 17)       ZYX → [0]
  Elbow flexion         R_Elbow (joint 19)          XYZ → [0]
"""
import numpy as np, torch, smplx, json, os
from scipy.spatial.transform import Rotation

def axis_angle_to_euler(aa, order='XYZ'):
    """axis-angle (3,) → euler angles (degrees)"""
    R = Rotation.from_rotvec(aa)
    return R.as_euler(order.lower(), degrees=True)

def compose_rotations(aa_list):
    """여러 관절의 axis-angle을 순차 합성 (부모→자식 체인)"""
    R_total = Rotation.identity()
    for aa in aa_list:
        R_total = R_total * Rotation.from_rotvec(aa)
    return R_total

def compute_8vars_v4(body_pose_69):
    """
    SMPL body_pose (69,) → 8개 운동학 변수
    body_pose = 23 joints × 3 axis-angle (부모 대비 상대 회전)
    """
    bp = body_pose_69.reshape(23, 3)

    # SMPL joint indices (0-indexed within body_pose, NOT global joint index)
    # body_pose[i] = joint i+1 in SMPL (joint 0 = pelvis = global_orient)
    # So body_pose[0] = L_Hip (joint 1), body_pose[2] = Spine1 (joint 3), etc.
    # Mapping: body_pose index = SMPL_joint_index - 1
    IDX_L_HIP = 0      # joint 1
    IDX_R_HIP = 1      # joint 2
    IDX_SPINE1 = 2      # joint 3
    IDX_L_KNEE = 3      # joint 4
    IDX_R_KNEE = 4      # joint 5
    IDX_SPINE2 = 5      # joint 6
    IDX_L_ANKLE = 6     # joint 7
    IDX_R_ANKLE = 7     # joint 8
    IDX_SPINE3 = 8      # joint 9
    IDX_NECK = 11       # joint 12
    IDX_L_COLLAR = 12   # joint 13
    IDX_R_COLLAR = 13   # joint 14
    IDX_L_SHOULDER = 15 # joint 16
    IDX_R_SHOULDER = 16 # joint 17
    IDX_L_ELBOW = 17    # joint 18
    IDX_R_ELBOW = 18    # joint 19

    result = {}

    # 1. Lead knee flexion: L_Knee 상대 회전 (L_Hip 대비)
    # SMPL에서 L_Knee의 body_pose는 L_Hip 좌표계 기준 상대 회전
    a = axis_angle_to_euler(bp[IDX_L_KNEE], 'XYZ')
    result['lead_knee_flexion'] = a[0]

    # 2-4. Trunk: Spine1 + Spine2 + Spine3 합성 (골반 대비 체간 전체 회전)
    R_trunk = compose_rotations([bp[IDX_SPINE1], bp[IDX_SPINE2], bp[IDX_SPINE3]])
    a_trunk = R_trunk.as_euler('xzy', degrees=True)
    result['trunk_forward_tilt'] = a_trunk[0]
    result['trunk_lateral_tilt'] = a_trunk[1]
    result['trunk_axial_rotation'] = a_trunk[2]

    # 5-6. Shoulder abduction & horizontal abd: R_Shoulder (체간 대비)
    # R_Collar → R_Shoulder 합성
    R_shoulder = compose_rotations([bp[IDX_R_COLLAR], bp[IDX_R_SHOULDER]])
    a_sho = R_shoulder.as_euler('yzx', degrees=True)
    result['shoulder_abduction'] = a_sho[0]
    result['shoulder_horizontal_abd'] = a_sho[1]

    # 7. Shoulder rotation: 장축 회전
    a_sho2 = R_shoulder.as_euler('zyx', degrees=True)
    result['shoulder_rotation'] = a_sho2[0]

    # 8. Elbow flexion: R_Elbow (R_Shoulder 대비)
    a_elb = axis_angle_to_euler(bp[IDX_R_ELBOW], 'XYZ')
    result['elbow_flexion'] = a_elb[0]

    return result

# ============ Main ============
REFINED_DIR = '/home/elicer/gsplat_refined_v2/'

var_names = ['lead_knee_flexion', 'trunk_forward_tilt', 'trunk_lateral_tilt',
             'trunk_axial_rotation', 'shoulder_abduction', 'shoulder_horizontal_abd',
             'shoulder_rotation', 'elbow_flexion']

results_A = {k: [] for k in var_names}
results_B = {k: [] for k in var_names}
frames = []

npz_files = sorted([f for f in os.listdir(REFINED_DIR) if f.endswith('.npz')])
print('Processing %d frames (body_pose direct Euler v4)...' % len(npz_files))

for fi_idx, npz_file in enumerate(npz_files):
    frame_idx = int(npz_file.replace('.npz', ''))
    data = np.load(os.path.join(REFINED_DIR, npz_file), allow_pickle=True)

    # A: init (before gsplat)
    try:
        angles_A = compute_8vars_v4(data['body_pose_init'])
        for k in var_names:
            results_A[k].append(angles_A[k])
    except:
        for k in var_names:
            results_A[k].append(float('nan'))

    # B: refined (after gsplat)
    try:
        angles_B = compute_8vars_v4(data['body_pose_refined'])
        for k in var_names:
            results_B[k].append(angles_B[k])
    except:
        for k in var_names:
            results_B[k].append(float('nan'))

    frames.append(frame_idx)

for k in var_names:
    results_A[k] = np.array(results_A[k])
    results_B[k] = np.array(results_B[k])

OUT = '/home/elicer/kinematic_results_v4/'
os.makedirs(OUT, exist_ok=True)
np.savez(os.path.join(OUT, 'ablation_8vars_v4.npz'), frames=np.array(frames),
         **{'A_' + k: results_A[k] for k in var_names},
         **{'B_' + k: results_B[k] for k in var_names})

print('')
print('=' * 95)
print('Ablation v4: body_pose direct Euler (n=%d frames)' % len(frames))
print('=' * 95)
print('  %-28s %11s %7s %14s %7s %7s %7s' % (
    'Variable', 'A(init)', 'A_SD', 'B(refined)', 'B_SD', 'Delta', 'r(A,B)'))
print('-' * 95)
for k in var_names:
    am = np.nanmean(results_A[k])
    asd = np.nanstd(results_A[k])
    bm = np.nanmean(results_B[k])
    bsd = np.nanstd(results_B[k])
    d = bm - am
    mask = ~(np.isnan(results_A[k]) | np.isnan(results_B[k]))
    r = np.corrcoef(results_A[k][mask], results_B[k][mask])[0, 1] if mask.sum() > 10 else float('nan')
    print('  %-28s %10.1f %6.1f %13.1f %6.1f %+6.1f  %.3f' % (k, am, asd, bm, bsd, d, r))

# CSV
with open(os.path.join(OUT, 'ablation_v4.csv'), 'w') as f:
    f.write('frame,' + ','.join(['A_' + k for k in var_names]) + ',' +
            ','.join(['B_' + k for k in var_names]) + '\n')
    for i in range(len(frames)):
        vals = [str(frames[i])]
        for k in var_names:
            vals.append('%.2f' % results_A[k][i])
        for k in var_names:
            vals.append('%.2f' % results_B[k][i])
        f.write(','.join(vals) + '\n')

print('')
print('Saved: %s' % OUT)
print('v4 COMPLETE!')
