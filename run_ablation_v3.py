"""
Ablation v3: SMPL 관절 위치 직접 사용 + 정점은 JCS 축 방향에만 활용
핵심 변경: 마커 기반 → 관절 기반 JCS
"""
import numpy as np, torch, smplx, json, os
from scipy.spatial.transform import Rotation

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v

# ============ JCS using SMPL joints directly ============

def build_pelvis_jcs_from_joints(j):
    """골반 JCS: SMPL joints 기반
    원점: (L_Hip + R_Hip) / 2
    Y축: Pelvis → Spine1 (상방)
    X축: R_Hip → L_Hip 방향에서 직교 (우측→좌측 = SMPL X)
    Z축: X × Y (전방)
    """
    origin = (j[1] + j[2]) / 2  # L_Hip + R_Hip midpoint
    y = normalize(j[3] - origin)  # Spine1 - hip_mid (superior)
    lat = normalize(j[1] - j[2])  # L_Hip - R_Hip (left direction in SMPL)
    z = normalize(np.cross(lat, y))  # anterior
    x = normalize(np.cross(y, z))    # right
    return np.column_stack([x, y, z]), origin

def build_thorax_jcs_from_joints(j):
    """체간 JCS: SMPL joints 기반
    원점: Spine3
    Y축: Spine2 → Neck (상방)
    Z축: (L_Collar+R_Collar)/2 - Spine3 방향에서 직교 (전방)
    """
    origin = j[9]  # Spine3
    y = normalize(j[12] - j[6])  # Neck - Spine2 (superior)
    collar_mid = (j[13] + j[14]) / 2  # L_Collar + R_Collar
    fwd = collar_mid - j[9]
    z = normalize(fwd - np.dot(fwd, y) * y)  # anterior (orthogonalized)
    x = normalize(np.cross(y, z))  # right
    return np.column_stack([x, y, z]), origin

def build_humerus_jcs_from_joints(j, side='R'):
    """상완 JCS: SMPL joints 기반
    원점: Shoulder
    Y축: Elbow → Shoulder (근위)
    X축: 체간 전방에서 유도
    """
    sho_idx = 17 if side == 'R' else 16
    elb_idx = 19 if side == 'R' else 18

    origin = j[sho_idx]
    y = normalize(j[sho_idx] - j[elb_idx])  # proximal

    # 체간 전방 방향을 참조 (Spine3 → collar midpoint)
    collar_mid = (j[13] + j[14]) / 2
    fwd_ref = normalize(collar_mid - j[9])

    z = normalize(np.cross(fwd_ref, y))  # ~lateral
    # 측면에 따라 부호 조정
    if side == 'L':
        z = -z
    x = normalize(np.cross(y, z))  # anterior
    # 재정렬: x=lateral, y=proximal, z=anterior
    return np.column_stack([z, y, x]), origin

def build_forearm_jcs_from_joints(j, side='R'):
    """전완 JCS"""
    elb_idx = 19 if side == 'R' else 18
    wri_idx = 21 if side == 'R' else 20

    origin = j[elb_idx]
    y = normalize(j[elb_idx] - j[wri_idx])  # proximal

    collar_mid = (j[13] + j[14]) / 2
    fwd_ref = normalize(collar_mid - j[9])

    z = normalize(np.cross(fwd_ref, y))
    if side == 'L':
        z = -z
    x = normalize(np.cross(y, z))
    return np.column_stack([z, y, x]), origin

def build_thigh_jcs_from_joints(j, side='L'):
    """대퇴 JCS"""
    hip_idx = 1 if side == 'L' else 2
    knee_idx = 4 if side == 'L' else 5

    origin = j[hip_idx]
    y = normalize(j[hip_idx] - j[knee_idx])  # proximal

    # 골반 전방 참조
    pelvis_fwd = normalize(j[3] - j[0])  # Spine1 - Pelvis
    lat = normalize(j[1] - j[2])  # L_Hip - R_Hip
    fwd = normalize(np.cross(lat, pelvis_fwd))

    z = normalize(np.cross(fwd, y)) if side == 'R' else normalize(np.cross(y, fwd))
    x = normalize(np.cross(y, z))
    return np.column_stack([x, y, z]), origin

def build_shank_jcs_from_joints(j, side='L'):
    """하퇴 JCS"""
    knee_idx = 4 if side == 'L' else 5
    ankle_idx = 7 if side == 'L' else 8

    origin = j[knee_idx]
    y = normalize(j[knee_idx] - j[ankle_idx])  # proximal

    pelvis_fwd = normalize(j[3] - j[0])
    lat = normalize(j[1] - j[2])
    fwd = normalize(np.cross(lat, pelvis_fwd))

    z = normalize(np.cross(fwd, y)) if side == 'R' else normalize(np.cross(y, fwd))
    x = normalize(np.cross(y, z))
    return np.column_stack([x, y, z]), origin

def ensure_right_handed(R):
    """Ensure rotation matrix is right-handed (det=+1)"""
    if np.linalg.det(R) < 0:
        R[:, 2] = -R[:, 2]  # flip Z axis
    return R

def compute_8vars_v3(joints):
    """SMPL 24 관절에서 직접 8개 변수 산출"""
    def euler(Rc, Rp, order):
        Rc = ensure_right_handed(Rc)
        Rp = ensure_right_handed(Rp)
        Rrel = Rc @ Rp.T
        # Ensure Rrel is also valid
        U, _, Vt = np.linalg.svd(Rrel)
        Rrel = U @ Vt
        if np.linalg.det(Rrel) < 0:
            U[:, -1] *= -1
            Rrel = U @ Vt
        return Rotation.from_matrix(Rrel).as_euler(order.lower(), degrees=True)

    Rp, _ = build_pelvis_jcs_from_joints(joints)
    Rt, _ = build_thorax_jcs_from_joints(joints)
    Rhu, _ = build_humerus_jcs_from_joints(joints, 'R')
    Rfo, _ = build_forearm_jcs_from_joints(joints, 'R')
    Rth, _ = build_thigh_jcs_from_joints(joints, 'L')
    Rsh, _ = build_shank_jcs_from_joints(joints, 'L')

    result = {}

    # 1. Lead knee flexion: Thigh-Shank XYZ
    a = euler(Rsh, Rth, 'XYZ')
    result['lead_knee_flexion'] = a[0]

    # 2-4. Trunk: Pelvis-Thorax XZY
    a = euler(Rt, Rp, 'XZY')
    result['trunk_forward_tilt'] = a[0]
    result['trunk_lateral_tilt'] = a[1]
    result['trunk_axial_rotation'] = a[2]

    # 5-6. Shoulder: Thorax-Humerus YZX
    a = euler(Rhu, Rt, 'YZX')
    result['shoulder_abduction'] = a[0]
    result['shoulder_horizontal_abd'] = a[1]

    # 7. Shoulder rotation: ZYX
    a2 = euler(Rhu, Rt, 'ZYX')
    result['shoulder_rotation'] = a2[0]

    # 8. Elbow flexion: Humerus-Forearm XYZ
    a = euler(Rfo, Rhu, 'XYZ')
    result['elbow_flexion'] = a[0]

    return result

# ============ Main ============
REFINED_DIR = '/home/elicer/gsplat_refined_v2/'
model = smplx.create('/home/elicer/EasyMocap/data/smplx/', model_type='smpl',
                      gender='neutral', batch_size=1).cuda()

var_names = ['lead_knee_flexion','trunk_forward_tilt','trunk_lateral_tilt',
             'trunk_axial_rotation','shoulder_abduction','shoulder_horizontal_abd',
             'shoulder_rotation','elbow_flexion']

results_A = {k: [] for k in var_names}
results_B = {k: [] for k in var_names}
frames = []

npz_files = sorted([f for f in os.listdir(REFINED_DIR) if f.endswith('.npz')])
print('Processing %d frames (joint-based JCS v3)...' % len(npz_files))

for fi_idx, npz_file in enumerate(npz_files):
    frame_idx = int(npz_file.replace('.npz', ''))
    data = np.load(os.path.join(REFINED_DIR, npz_file), allow_pickle=True)

    for cond, bp_key, go_key in [('A', 'body_pose_init', 'global_orient_init'),
                                  ('B', 'body_pose_refined', 'global_orient_refined')]:
        with torch.no_grad():
            out = model(
                body_pose=torch.tensor(data[bp_key], dtype=torch.float32).unsqueeze(0).cuda(),
                global_orient=torch.tensor(data[go_key], dtype=torch.float32).unsqueeze(0).cuda(),
                betas=torch.tensor(data['betas'], dtype=torch.float32).unsqueeze(0).cuda(),
                transl=torch.tensor(data['transl'], dtype=torch.float32).unsqueeze(0).cuda()
            )
            joints = out.joints[0].cpu().numpy()

        try:
            angles = compute_8vars_v3(joints)
            target = results_A if cond == 'A' else results_B
            for k in var_names:
                target[k].append(angles[k])
        except Exception as e:
            target = results_A if cond == 'A' else results_B
            for k in var_names:
                target[k].append(float('nan'))
            if cond == 'A':
                print('Frame %d error: %s' % (frame_idx, e))

    frames.append(frame_idx)
    if fi_idx % 50 == 0:
        print('  %d/%d...' % (fi_idx, len(npz_files)))

for k in var_names:
    results_A[k] = np.array(results_A[k])
    results_B[k] = np.array(results_B[k])

OUT = '/home/elicer/kinematic_results_v3/'
os.makedirs(OUT, exist_ok=True)
np.savez(os.path.join(OUT, 'ablation_8vars_v3.npz'), frames=np.array(frames),
         **{'A_'+k: results_A[k] for k in var_names},
         **{'B_'+k: results_B[k] for k in var_names})

print('')
print('=' * 90)
print('Ablation v3: Joint-based JCS (n=%d frames)' % len(frames))
print('=' * 90)
print('  %-28s %11s %7s %14s %7s %7s' % ('Variable', 'A(init)', 'A_SD', 'B(refined)', 'B_SD', 'Delta'))
print('-' * 90)
for k in var_names:
    am = np.nanmean(results_A[k])
    asd = np.nanstd(results_A[k])
    bm = np.nanmean(results_B[k])
    bsd = np.nanstd(results_B[k])
    d = bm - am
    # Also compute correlation between A and B
    mask = ~(np.isnan(results_A[k]) | np.isnan(results_B[k]))
    if mask.sum() > 10:
        corr = np.corrcoef(results_A[k][mask], results_B[k][mask])[0, 1]
    else:
        corr = float('nan')
    print('  %-28s %10.1f %6.1f %13.1f %6.1f %+6.1f  r=%.3f' % (k, am, asd, bm, bsd, d, corr))

# CSV
with open(os.path.join(OUT, 'ablation_v3.csv'), 'w') as f:
    f.write('frame,' + ','.join(['A_'+k for k in var_names]) + ',' + ','.join(['B_'+k for k in var_names]) + '\n')
    for i in range(len(frames)):
        vals = [str(frames[i])]
        for k in var_names: vals.append('%.2f' % results_A[k][i])
        for k in var_names: vals.append('%.2f' % results_B[k][i])
        f.write(','.join(vals) + '\n')

print('')
print('Saved: %s' % OUT)
print('ABLATION v3 COMPLETE!')
