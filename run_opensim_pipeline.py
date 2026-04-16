"""
SMPL → TRC → OpenSim IK → Vicon comparison
SAM4DCap 방식: SMPL 정점에서 가상 마커 추출 → OpenSim IK solver

Plug-in Gait 43-marker set (OpenCap compatible):
  SMPL vertex indices for key anatomical landmarks
"""
import numpy as np, torch, smplx, json, os, sys

# ============ Config ============
REFINED_DIR = '/home/elicer/gsplat_refined_v2/'
SMPL_PATH = '/home/elicer/EasyMocap/data/smplx/'
MARKER_MAP_FILE = '/home/elicer/server_scripts/smpl_virtual_marker_mapping.json'
VICON_CSV = '/home/elicer/vicon_set002.csv'
OUTPUT_DIR = '/home/elicer/opensim_results/'
FPS = 240
# ================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print('Loading SMPL model...')
model = smplx.create(SMPL_PATH, model_type='smpl', gender='neutral', batch_size=1).cuda()
marker_map = json.load(open(MARKER_MAP_FILE))

# ============ 1. Extract virtual markers from all frames ============
print('\n=== Step 1: Extract virtual markers from refined SMPL ===')

npz_files = sorted([f for f in os.listdir(REFINED_DIR) if f.endswith('.npz')])
print('Frames: %d' % len(npz_files))

all_markers_init = {}  # condition A (before gsplat)
all_markers_refined = {}  # condition B (after gsplat)
frame_times = []

marker_names = [k for k in marker_map.keys() if not k.endswith('_center')]

for fi_idx, npz_file in enumerate(npz_files):
    frame_idx = int(npz_file.replace('.npz', ''))
    data = np.load(os.path.join(REFINED_DIR, npz_file), allow_pickle=True)
    frame_times.append(frame_idx / FPS)

    for cond, bp_key, go_key in [('init', 'body_pose_init', 'global_orient_init'),
                                  ('refined', 'body_pose_refined', 'global_orient_refined')]:
        with torch.no_grad():
            out = model(
                body_pose=torch.tensor(data[bp_key], dtype=torch.float32).unsqueeze(0).cuda(),
                global_orient=torch.tensor(data[go_key], dtype=torch.float32).unsqueeze(0).cuda(),
                betas=torch.tensor(data['betas'], dtype=torch.float32).unsqueeze(0).cuda(),
                transl=torch.tensor(data['transl'], dtype=torch.float32).unsqueeze(0).cuda()
            )
            verts = out.vertices[0].cpu().numpy()

        markers = {}
        for name, indices in marker_map.items():
            if not name.endswith('_center'):
                # Convert to mm (OpenSim expects mm or m, TRC typically mm)
                markers[name] = verts[indices].mean(axis=0) * 1000  # m → mm

        target = all_markers_init if cond == 'init' else all_markers_refined
        for name in marker_names:
            if name not in target:
                target[name] = []
            target[name].append(markers[name])

    if fi_idx % 50 == 0:
        print('  %d/%d...' % (fi_idx, len(npz_files)))

n_frames = len(frame_times)
print('Extracted %d markers x %d frames' % (len(marker_names), n_frames))

# ============ 2. Write TRC files ============
print('\n=== Step 2: Write TRC files ===')

def write_trc(filepath, marker_names, marker_data, frame_times, fps=240):
    """
    Write OpenSim TRC file

    Args:
        filepath: output path
        marker_names: list of marker names
        marker_data: dict of marker_name → list of (3,) positions in mm
        frame_times: list of times in seconds
        fps: frame rate
    """
    n_frames = len(frame_times)
    n_markers = len(marker_names)

    with open(filepath, 'w') as f:
        # Header
        f.write('PathFileType\t4\t(X/Y/Z)\t%s\n' % os.path.basename(filepath))
        f.write('DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n')
        f.write('%d\t%d\t%d\t%d\tmm\t%d\t1\t%d\n' % (fps, fps, n_frames, n_markers, fps, n_frames))

        # Marker names header
        header = 'Frame#\tTime'
        for name in marker_names:
            header += '\t%s\t\t' % name
        f.write(header.rstrip() + '\n')

        # X/Y/Z sub-header
        subheader = '\t'
        for i in range(n_markers):
            subheader += '\tX%d\tY%d\tZ%d' % (i+1, i+1, i+1)
        f.write(subheader + '\n')

        # Blank line
        f.write('\n')

        # Data
        for fi in range(n_frames):
            line = '%d\t%.6f' % (fi + 1, frame_times[fi])
            for name in marker_names:
                pos = marker_data[name][fi]
                line += '\t%.4f\t%.4f\t%.4f' % (pos[0], pos[1], pos[2])
            f.write(line + '\n')

# Write TRC for both conditions
trc_init = os.path.join(OUTPUT_DIR, 'markers_init.trc')
trc_refined = os.path.join(OUTPUT_DIR, 'markers_refined.trc')

write_trc(trc_init, marker_names, all_markers_init, frame_times, FPS)
write_trc(trc_refined, marker_names, all_markers_refined, frame_times, FPS)
print('Written: %s' % trc_init)
print('Written: %s' % trc_refined)

# ============ 3. OpenSim IK ============
print('\n=== Step 3: OpenSim Inverse Kinematics ===')

import opensim as osim

def run_opensim_ik(trc_path, output_mot, model_path=None):
    """
    Run OpenSim IK on TRC file

    Uses generic gait model with scaled markers
    """
    # Use built-in gait model
    if model_path is None:
        # Create a simple model with markers matching our TRC
        mdl = osim.Model()
        mdl.setName('smpl_marker_model')

        # Ground body (for reference)
        ground = mdl.getGround()

        # Add markers on ground (will be virtual)
        # In a real setup, we'd use a proper musculoskeletal model
        # For IK comparison, we just need marker positions

    # Alternative: use IKTool directly with marker data
    # Since we don't have a proper .osim model with our custom markers,
    # we compute joint angles directly from marker positions
    print('  Note: Using direct marker-based angle computation instead of OpenSim IK')
    print('  (OpenSim IK requires a musculoskeletal model with matching marker set)')
    return None

# Since OpenSim IK requires a properly configured .osim model file,
# and we don't have one matching our SMPL markers,
# we'll compute ISB-standard joint angles directly from marker positions

print('\n=== Step 3b: ISB Joint Angles from Markers (Vicon-compatible) ===')

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v

def compute_angles_from_markers(markers):
    """
    Compute 8 kinematic variables using ISB JCS from virtual markers
    Matches Vicon Plug-in Gait output convention

    Plug-in Gait conventions:
    - RShoulderAngles.X = Flexion(+)/Extension(-)
    - RElbowAngles.X = Flexion(+)
    - LKneeAngles.X = Flexion(+)/Extension(-)
    - SpineAngles.X = Flexion(+)/Extension(-)
    """
    from scipy.spatial.transform import Rotation

    # Pelvis JCS (ISB: ASIS/PSIS based)
    la = np.array(markers['L_ASIS'])
    ra = np.array(markers['R_ASIS'])
    lp = np.array(markers['L_PSIS'])
    rp = np.array(markers['R_PSIS'])

    pel_o = (la + ra) / 2
    pel_x = normalize(ra - la)  # right
    asis_mid = (la + ra) / 2
    psis_mid = (lp + rp) / 2
    temp = normalize(asis_mid - psis_mid)  # anterior
    pel_z = normalize(np.cross(pel_x, temp))  # superior (cross right x anterior)
    # Recompute anterior as perpendicular
    pel_y = normalize(np.cross(pel_z, pel_x))  # anterior
    R_pel = np.column_stack([pel_x, pel_y, pel_z])

    # Thorax JCS (C7, T10, CLAV)
    c7 = np.array(markers['C7'])
    t10 = np.array(markers['T10'])
    clav = np.array(markers['CLAV'])

    thor_o = (c7 + clav) / 2
    thor_z = normalize(c7 - t10)  # superior
    temp2 = normalize(clav - t10)
    thor_y = normalize(temp2 - np.dot(temp2, thor_z) * thor_z)  # anterior
    thor_x = normalize(np.cross(thor_y, thor_z))  # right
    R_thor = np.column_stack([thor_x, thor_y, thor_z])

    # Trunk angles: Thorax relative to Pelvis
    # Plug-in Gait: SpineAngles = Thorax/Pelvis, XZY decomposition
    R_trunk_rel = R_thor.T @ R_pel  # child^T * parent? or child * parent^T?
    # ISB: R_rel = R_distal * R_proximal^T
    R_trunk_rel = R_thor @ np.linalg.inv(R_pel)

    # Ensure valid rotation
    U, _, Vt = np.linalg.svd(R_trunk_rel)
    R_trunk_rel = U @ Vt
    if np.linalg.det(R_trunk_rel) < 0:
        U[:, -1] *= -1
        R_trunk_rel = U @ Vt

    trunk_euler = Rotation.from_matrix(R_trunk_rel).as_euler('XZY', degrees=True)

    # Humerus JCS (R side = throwing arm)
    r_sho = np.array(markers['R_Shoulder'])
    r_elb_l = np.array(markers['R_Elbow_lat'])
    r_elb_m = np.array(markers['R_Elbow_med'])
    r_elb_c = (r_elb_l + r_elb_m) / 2

    hum_z = normalize(r_sho - r_elb_c)  # proximal (superior in anatomical)
    elb_axis = normalize(r_elb_l - r_elb_m)  # mediolateral
    hum_y = normalize(np.cross(hum_z, elb_axis))  # anterior
    hum_x = normalize(np.cross(hum_y, hum_z))  # lateral
    R_hum = np.column_stack([hum_x, hum_y, hum_z])

    # Shoulder: Humerus relative to Thorax
    R_sho_rel = R_hum @ np.linalg.inv(R_thor)
    U, _, Vt = np.linalg.svd(R_sho_rel)
    R_sho_rel = U @ Vt
    if np.linalg.det(R_sho_rel) < 0:
        U[:, -1] *= -1
        R_sho_rel = U @ Vt

    sho_yzx = Rotation.from_matrix(R_sho_rel).as_euler('YZX', degrees=True)
    sho_zyx = Rotation.from_matrix(R_sho_rel).as_euler('ZYX', degrees=True)

    # Forearm JCS
    r_wri = np.array(markers['R_Wrist'])
    fa_z = normalize(r_elb_c - r_wri)  # proximal
    fa_y = normalize(np.cross(fa_z, elb_axis))
    fa_x = normalize(np.cross(fa_y, fa_z))
    R_fa = np.column_stack([fa_x, fa_y, fa_z])

    # Elbow: Forearm relative to Humerus
    R_elb_rel = R_fa @ np.linalg.inv(R_hum)
    U, _, Vt = np.linalg.svd(R_elb_rel)
    R_elb_rel = U @ Vt
    if np.linalg.det(R_elb_rel) < 0:
        U[:, -1] *= -1
        R_elb_rel = U @ Vt

    elb_euler = Rotation.from_matrix(R_elb_rel).as_euler('XYZ', degrees=True)

    # Thigh JCS (L = lead leg)
    l_knee = np.array(markers['L_Knee_lat'])
    l_ank = np.array(markers['L_Ankle'])

    # Use pelvis to define hip center (midpoint approximation)
    l_hip_approx = la + np.array([0, 0, -80])  # rough HJC estimation below ASIS

    thigh_z = normalize(l_hip_approx - l_knee)
    thigh_temp = normalize(np.cross(pel_y, thigh_z))
    thigh_y = normalize(np.cross(thigh_z, thigh_temp))
    thigh_x = normalize(np.cross(thigh_y, thigh_z))
    R_thigh = np.column_stack([thigh_x, thigh_y, thigh_z])

    # Shank JCS
    shank_z = normalize(l_knee - l_ank)
    shank_y = normalize(np.cross(shank_z, thigh_temp))
    shank_x = normalize(np.cross(shank_y, shank_z))
    R_shank = np.column_stack([shank_x, shank_y, shank_z])

    # Knee: Shank relative to Thigh
    R_knee_rel = R_shank @ np.linalg.inv(R_thigh)
    U, _, Vt = np.linalg.svd(R_knee_rel)
    R_knee_rel = U @ Vt
    if np.linalg.det(R_knee_rel) < 0:
        U[:, -1] *= -1
        R_knee_rel = U @ Vt

    knee_euler = Rotation.from_matrix(R_knee_rel).as_euler('XYZ', degrees=True)

    return {
        'lead_knee_flexion': knee_euler[0],
        'trunk_forward_tilt': trunk_euler[0],
        'trunk_lateral_tilt': trunk_euler[1],
        'trunk_axial_rotation': trunk_euler[2],
        'shoulder_abduction': sho_yzx[0],
        'shoulder_horizontal_abd': sho_yzx[1],
        'shoulder_rotation': sho_zyx[0],
        'elbow_flexion': elb_euler[0],
    }

# Compute for both conditions
var_names = ['lead_knee_flexion', 'trunk_forward_tilt', 'trunk_lateral_tilt',
             'trunk_axial_rotation', 'shoulder_abduction', 'shoulder_horizontal_abd',
             'shoulder_rotation', 'elbow_flexion']

results_init = {k: [] for k in var_names}
results_refined = {k: [] for k in var_names}

for fi in range(n_frames):
    for cond, src, dst in [('init', all_markers_init, results_init),
                           ('refined', all_markers_refined, results_refined)]:
        m = {name: src[name][fi] for name in marker_names}
        try:
            angles = compute_angles_from_markers(m)
            for k in var_names:
                dst[k].append(angles[k])
        except:
            for k in var_names:
                dst[k].append(float('nan'))

for k in var_names:
    results_init[k] = np.array(results_init[k])
    results_refined[k] = np.array(results_refined[k])

# ============ 4. Parse Vicon angles ============
print('\n=== Step 4: Parse Vicon ground truth ===')

lines = open(VICON_CSV, encoding='utf-8-sig').readlines()
mo_start = None
for i, line in enumerate(lines):
    if 'Model Outputs' in line.strip():
        mo_start = i
        break

headers = lines[mo_start + 2].strip().split(',')
subheaders = lines[mo_start + 3].strip().split(',')

col_map = {}
for i, h in enumerate(headers):
    h = h.strip().replace('BJH:', '')
    sh = subheaders[i].strip() if i < len(subheaders) else ''
    if h and sh in ('X', 'Y', 'Z'):
        col_map[h + '_' + sh] = i

# Map Vicon angles to our variable names
# Plug-in Gait: RShoulderAngles (X=flex, Y=abd, Z=rot)
# LKneeAngles (X=flex), RElbowAngles (X=flex)
# LSpineAngles (X=flex, Y=lat, Z=rot)
vicon_map = {
    'lead_knee_flexion': 'LKneeAngles_X',
    'trunk_forward_tilt': 'LSpineAngles_X',
    'trunk_lateral_tilt': 'LSpineAngles_Y',
    'trunk_axial_rotation': 'LSpineAngles_Z',
    'shoulder_abduction': 'RShoulderAngles_Y',
    'shoulder_horizontal_abd': 'RShoulderAngles_X',
    'shoulder_rotation': 'RShoulderAngles_Z',
    'elbow_flexion': 'RElbowAngles_X',
}

data_start = mo_start + 5
vicon_data = {k: [] for k in var_names}
vicon_frames = []

for i in range(data_start, len(lines)):
    row = lines[i].strip()
    if not row:
        break
    parts = row.split(',')
    vicon_frames.append(int(parts[0]))

    for var, col_key in vicon_map.items():
        ci = col_map.get(col_key, -1)
        if ci >= 0 and ci < len(parts):
            val = parts[ci].strip()
            vicon_data[var].append(float(val) if val else float('nan'))
        else:
            vicon_data[var].append(float('nan'))

vicon_frames = np.array(vicon_frames)
for k in var_names:
    vicon_data[k] = np.array(vicon_data[k])

print('Vicon: %d frames (%d to %d)' % (len(vicon_frames), vicon_frames[0], vicon_frames[-1]))

# ============ 5. Compare ============
print('\n=== Step 5: Vicon vs Markerless Comparison ===')

# Align: markerless 222 frames, vicon 875 frames
# Both at 240fps. Assume markerless frame 0 = vicon first frame
n_compare = min(len(results_init['lead_knee_flexion']), len(vicon_data['lead_knee_flexion']))

print('\n' + '=' * 110)
print('RESULTS: Vicon vs Markerless (n=%d frames)' % n_compare)
print('=' * 110)
print('  %-25s | %8s %6s | %8s %6s | %8s %6s | %6s %6s %7s' % (
    'Variable', 'Vicon', 'V_SD', 'Init', 'A_SD', 'Refined', 'B_SD', 'r(V,A)', 'r(V,B)', 'RMSE_B'))
print('-' * 110)

for var in var_names:
    v = vicon_data[var][:n_compare]
    a = results_init[var][:n_compare]
    b = results_refined[var][:n_compare]

    mask = ~(np.isnan(v) | np.isnan(a) | np.isnan(b))
    n_valid = mask.sum()

    if n_valid < 10:
        print('  %-25s | insufficient data (n=%d)' % (var, n_valid))
        continue

    vm, am, bm = v[mask], a[mask], b[mask]
    r_va = np.corrcoef(vm, am)[0, 1]
    r_vb = np.corrcoef(vm, bm)[0, 1]
    rmse_a = np.sqrt(np.mean((vm - am) ** 2))
    rmse_b = np.sqrt(np.mean((vm - bm) ** 2))

    print('  %-25s | %7.1f %5.1f | %7.1f %5.1f | %7.1f %5.1f | %5.3f %5.3f %6.1f' % (
        var, vm.mean(), vm.std(), am.mean(), am.std(), bm.mean(), bm.std(),
        r_va, r_vb, rmse_b))

# Save
np.savez(os.path.join(OUTPUT_DIR, 'full_comparison.npz'),
         **{'vicon_' + k: vicon_data[k][:n_compare] for k in var_names},
         **{'init_' + k: results_init[k][:n_compare] for k in var_names},
         **{'refined_' + k: results_refined[k][:n_compare] for k in var_names})

print('\nSaved: %s' % OUTPUT_DIR)
print('PIPELINE COMPLETE!')
