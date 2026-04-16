"""
Vicon vs gsplat-refined SMPL comparison
- Parse Vicon CSV angles
- Time-align with markerless frames
- Compute CMC, ICC, RMSE, Pearson r
"""
import numpy as np, json, os

# ============ 1. Parse Vicon CSV ============
lines = open('/home/elicer/vicon_set002.csv', encoding='utf-8-sig').readlines()

# Find Model Outputs
mo_start = None
for i, line in enumerate(lines):
    if 'Model Outputs' in line.strip():
        mo_start = i
        break

headers = lines[mo_start + 2].strip().split(',')
subheaders = lines[mo_start + 3].strip().split(',')

# Build column map
col_map = {}
for i, h in enumerate(headers):
    h = h.strip().replace('BJH:', '')
    sh = subheaders[i].strip() if i < len(subheaders) else ''
    if h and sh in ('X', 'Y', 'Z'):
        col_map[h + '_' + sh] = i

# Parse data
data_start = mo_start + 5
vicon_frames = []
vicon_raw = {}
target_cols = {
    # Plug-in Gait: RShoulderAngles.X=flexion/extension, .Y=abduction, .Z=rotation
    # For throwing arm (R), lead leg (L)
    'RShoulderAngles_X': 'shoulder_flexion',
    'RShoulderAngles_Y': 'shoulder_abduction',
    'RShoulderAngles_Z': 'shoulder_rotation',
    'RElbowAngles_X': 'elbow_flexion',
    'LKneeAngles_X': 'lead_knee_flexion',
    # Trunk: use Spine or Thorax angles
    'LSpineAngles_X': 'trunk_forward_tilt',
    'LSpineAngles_Y': 'trunk_lateral_tilt',
    'LSpineAngles_Z': 'trunk_axial_rotation',
}

for k in target_cols.values():
    vicon_raw[k] = []

for i in range(data_start, len(lines)):
    row = lines[i].strip()
    if not row:
        break
    parts = row.split(',')
    frame = int(parts[0])
    vicon_frames.append(frame)

    for col_key, var_name in target_cols.items():
        ci = col_map.get(col_key, -1)
        if ci >= 0 and ci < len(parts):
            val = parts[ci].strip()
            vicon_raw[var_name].append(float(val) if val else float('nan'))
        else:
            vicon_raw[var_name].append(float('nan'))

vicon_frames = np.array(vicon_frames)
for k in vicon_raw:
    vicon_raw[k] = np.array(vicon_raw[k])

print('Vicon: %d frames (%d to %d)' % (len(vicon_frames), vicon_frames[0], vicon_frames[-1]))
for k, v in vicon_raw.items():
    valid = np.sum(~np.isnan(v))
    print('  %s: valid=%d/%d, mean=%.1f, sd=%.1f' % (k, valid, len(v), np.nanmean(v), np.nanstd(v)))

# ============ 2. Load markerless (gsplat-refined) ============
from scipy.spatial.transform import Rotation

def axis_angle_to_euler(aa, order='XYZ'):
    return Rotation.from_rotvec(aa).as_euler(order.lower(), degrees=True)

def compose_rotations(aa_list):
    R_total = Rotation.identity()
    for aa in aa_list:
        R_total = R_total * Rotation.from_rotvec(aa)
    return R_total

def compute_8vars(bp69):
    bp = bp69.reshape(23, 3)
    result = {}
    # Lead knee
    a = axis_angle_to_euler(bp[3], 'XYZ')  # L_Knee
    result['lead_knee_flexion'] = a[0]
    # Trunk (Spine1+2+3)
    R_trunk = compose_rotations([bp[2], bp[5], bp[8]])
    a = R_trunk.as_euler('xzy', degrees=True)
    result['trunk_forward_tilt'] = a[0]
    result['trunk_lateral_tilt'] = a[1]
    result['trunk_axial_rotation'] = a[2]
    # Shoulder (R_Collar + R_Shoulder)
    R_sho = compose_rotations([bp[13], bp[16]])
    a = R_sho.as_euler('yzx', degrees=True)
    result['shoulder_abduction'] = a[0]
    result['shoulder_flexion'] = a[1]  # horizontal abd ≈ flexion
    a2 = R_sho.as_euler('zyx', degrees=True)
    result['shoulder_rotation'] = a2[0]
    # Elbow
    a = axis_angle_to_euler(bp[18], 'XYZ')  # R_Elbow
    result['elbow_flexion'] = a[0]
    return result

REFINED_DIR = '/home/elicer/gsplat_refined_v2/'
npz_files = sorted([f for f in os.listdir(REFINED_DIR) if f.endswith('.npz')])

markerless_A = {k: [] for k in ['lead_knee_flexion','trunk_forward_tilt','trunk_lateral_tilt',
                                  'trunk_axial_rotation','shoulder_abduction','shoulder_flexion',
                                  'shoulder_rotation','elbow_flexion']}
markerless_B = {k: [] for k in markerless_A}
ml_frames = []

for npz_file in npz_files:
    fi = int(npz_file.replace('.npz', ''))
    data = np.load(os.path.join(REFINED_DIR, npz_file), allow_pickle=True)
    try:
        a = compute_8vars(data['body_pose_init'])
        b = compute_8vars(data['body_pose_refined'])
        for k in markerless_A:
            markerless_A[k].append(a[k])
            markerless_B[k].append(b[k])
        ml_frames.append(fi)
    except:
        pass

ml_frames = np.array(ml_frames)
for k in markerless_A:
    markerless_A[k] = np.array(markerless_A[k])
    markerless_B[k] = np.array(markerless_B[k])

print('\nMarkerless: %d frames (%d to %d)' % (len(ml_frames), ml_frames[0], ml_frames[-1]))

# ============ 3. Time alignment ============
# Vicon frames: 512-1386 (875 frames at 240fps)
# Markerless frames: 0-221 (222 frames)
# Need to find offset: markerless frame 0 = vicon frame ???
# Both are 240fps
# Use SFC event to align
# Vicon events: Foot Strike at 2.62s → frame = 2.62*240 = 629 (vicon frame)
# But vicon data starts at frame 512
# SFC in vicon = frame 629 → relative frame 629-512 = 117

# For markerless: SFC should be around similar relative position
# The markerless data covers a subset of the recording
# Assume markerless frame 0 = vicon frame 512 (start of model output)

vicon_start_frame = vicon_frames[0]  # 512
# Align: markerless frame i → vicon index i (1:1 since both 240fps)
# But markerless has only 222 frames, vicon has 875
# markerless frame 0 aligns to vicon frame 512

print('\nAlignment: markerless frame 0 = vicon frame %d' % vicon_start_frame)
print('Overlap: %d frames' % min(len(ml_frames), len(vicon_frames)))

# ============ 4. Compute metrics ============
compare_vars = ['lead_knee_flexion', 'shoulder_abduction', 'shoulder_rotation',
                'elbow_flexion', 'trunk_forward_tilt', 'trunk_lateral_tilt',
                'trunk_axial_rotation', 'shoulder_flexion']

n_overlap = min(len(ml_frames), len(vicon_frames))

print('\n' + '=' * 100)
print('Vicon vs Markerless Comparison (n=%d frames)' % n_overlap)
print('=' * 100)
print('  %-25s %8s %8s | %8s %8s | %8s %8s %8s' % (
    'Variable', 'V_mean', 'V_sd', 'A_mean', 'B_mean', 'r(V,A)', 'r(V,B)', 'RMSE_B'))
print('-' * 100)

for var in compare_vars:
    vicon_key = var
    if var not in vicon_raw:
        print('  %-25s -- not in vicon --' % var)
        continue

    v = vicon_raw[var][:n_overlap]
    a = markerless_A[var][:n_overlap]
    b = markerless_B[var][:n_overlap]

    mask = ~(np.isnan(v) | np.isnan(a) | np.isnan(b))
    if mask.sum() < 10:
        print('  %-25s insufficient valid data (%d)' % (var, mask.sum()))
        continue

    v_m, a_m, b_m = v[mask], a[mask], b[mask]

    r_va = np.corrcoef(v_m, a_m)[0, 1]
    r_vb = np.corrcoef(v_m, b_m)[0, 1]
    rmse_a = np.sqrt(np.mean((v_m - a_m) ** 2))
    rmse_b = np.sqrt(np.mean((v_m - b_m) ** 2))

    print('  %-25s %7.1f %7.1f | %7.1f %7.1f | %7.3f %7.3f %7.1f' % (
        var, np.mean(v_m), np.std(v_m), np.mean(a_m), np.mean(b_m),
        r_va, r_vb, rmse_b))

# Save results
OUT = '/home/elicer/vicon_comparison/'
os.makedirs(OUT, exist_ok=True)
np.savez(os.path.join(OUT, 'comparison_results.npz'),
         vicon_frames=vicon_frames[:n_overlap],
         ml_frames=ml_frames[:n_overlap],
         **{'vicon_' + k: vicon_raw[k][:n_overlap] for k in vicon_raw},
         **{'init_' + k: markerless_A[k][:n_overlap] for k in markerless_A},
         **{'refined_' + k: markerless_B[k][:n_overlap] for k in markerless_B})
print('\nSaved: %s' % OUT)
print('COMPARISON COMPLETE!')
