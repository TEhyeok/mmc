"""Compute 8 kinematic variables for ablation: init vs refined"""
import numpy as np, os, json, torch, smplx
from scipy.spatial.transform import Rotation

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v

def build_pelvis_jcs(m):
    x = normalize(m['R_ASIS'] - m['L_ASIS'])
    post = normalize(((m['L_PSIS']+m['R_PSIS'])/2) - ((m['L_ASIS']+m['R_ASIS'])/2))
    z = normalize(np.cross(x, post))
    y = normalize(np.cross(z, x))
    return np.column_stack([x, y, z])

def build_thorax_jcs(m):
    z = normalize(m['C7'] - m['T10'])
    fwd = m['CLAV'] - m['T10']
    y = normalize(fwd - np.dot(fwd, z) * z)
    x = normalize(np.cross(y, z))
    return np.column_stack([x, y, z])

def build_humerus_jcs(m, s='R'):
    elb_c = (m[s+'_Elbow_lat'] + m[s+'_Elbow_med']) / 2
    y = normalize(m[s+'_Shoulder'] - elb_c)
    lat = normalize(m[s+'_Elbow_lat'] - m[s+'_Elbow_med'])
    x = normalize(lat - np.dot(lat, y) * y)
    z = normalize(np.cross(x, y))
    return np.column_stack([x, y, z])

def build_forearm_jcs(m, s='R'):
    elb_c = (m[s+'_Elbow_lat'] + m[s+'_Elbow_med']) / 2
    y = normalize(elb_c - m[s+'_Wrist'])
    lat = normalize(m[s+'_Elbow_lat'] - m[s+'_Elbow_med'])
    x = normalize(lat - np.dot(lat, y) * y)
    z = normalize(np.cross(x, y))
    return np.column_stack([x, y, z])

def build_thigh_jcs(m, j, s='L'):
    hip = j[1] if s == 'L' else j[2]
    y = normalize(hip - m[s+'_Knee_lat'])
    xt = np.array([1,0,0]) if s == 'R' else np.array([-1,0,0])
    z = normalize(np.cross(xt, y))
    x = normalize(np.cross(y, z))
    return np.column_stack([x, y, z])

def build_shank_jcs(m, s='L'):
    y = normalize(m[s+'_Knee_lat'] - m[s+'_Ankle'])
    xt = np.array([1,0,0]) if s == 'R' else np.array([-1,0,0])
    z = normalize(np.cross(xt, y))
    x = normalize(np.cross(y, z))
    return np.column_stack([x, y, z])

def compute_8vars(markers, joints):
    def euler(Rc, Rp, order):
        return Rotation.from_matrix(Rc @ Rp.T).as_euler(order.lower(), degrees=True)
    Rp = build_pelvis_jcs(markers)
    Rt = build_thorax_jcs(markers)
    Rhu = build_humerus_jcs(markers, 'R')
    Rfo = build_forearm_jcs(markers, 'R')
    Rth = build_thigh_jcs(markers, joints, 'L')
    Rsh = build_shank_jcs(markers, 'L')

    a_knee = euler(Rsh, Rth, 'XYZ')
    a_trunk = euler(Rt, Rp, 'XZY')
    a_sho = euler(Rhu, Rt, 'YZX')
    a_sho2 = euler(Rhu, Rt, 'ZYX')
    a_elb = euler(Rfo, Rhu, 'XYZ')

    return {
        'lead_knee_flexion': a_knee[0],
        'trunk_forward_tilt': a_trunk[0],
        'trunk_lateral_tilt': a_trunk[1],
        'trunk_axial_rotation': a_trunk[2],
        'shoulder_abduction': a_sho[0],
        'shoulder_horizontal_abd': a_sho[1],
        'shoulder_rotation': a_sho2[0],
        'elbow_flexion': a_elb[0],
    }

# === Main ===
REFINED_DIR = '/home/elicer/gsplat_refined_smpl/'
marker_map = json.load(open('/home/elicer/server_scripts/smpl_virtual_marker_mapping.json'))
model = smplx.create('/home/elicer/EasyMocap/data/smplx/', model_type='smpl',
                      gender='neutral', batch_size=1).cuda()

var_names = ['lead_knee_flexion','trunk_forward_tilt','trunk_lateral_tilt',
             'trunk_axial_rotation','shoulder_abduction','shoulder_horizontal_abd',
             'shoulder_rotation','elbow_flexion']

results_A = {k: [] for k in var_names}
results_B = {k: [] for k in var_names}
frames = []

npz_files = sorted([f for f in os.listdir(REFINED_DIR) if f.endswith('.npz')])
print(f'Processing {len(npz_files)} frames...')

for fi_idx, npz_file in enumerate(npz_files):
    frame_idx = int(npz_file.replace('.npz', ''))
    data = np.load(os.path.join(REFINED_DIR, npz_file), allow_pickle=True)

    for cond, bp_key in [('A', 'body_pose_init'), ('B', 'body_pose_refined')]:
        bp = data[bp_key]
        go = data['global_orient']
        betas = data['betas']

        with torch.no_grad():
            out = model(
                body_pose=torch.tensor(bp, dtype=torch.float32).unsqueeze(0).cuda(),
                global_orient=torch.tensor(go, dtype=torch.float32).unsqueeze(0).cuda(),
                betas=torch.tensor(betas, dtype=torch.float32).unsqueeze(0).cuda(),
                transl=torch.zeros(1, 3).cuda()
            )
            verts = out.vertices[0].cpu().numpy()
            joints = out.joints[0].cpu().numpy()

        markers = {}
        for name, indices in marker_map.items():
            if not name.endswith('_center'):
                markers[name] = verts[indices].mean(axis=0)

        try:
            angles = compute_8vars(markers, joints)
            target = results_A if cond == 'A' else results_B
            for k in var_names:
                target[k].append(angles[k])
        except:
            target = results_A if cond == 'A' else results_B
            for k in var_names:
                target[k].append(np.nan)

    frames.append(frame_idx)
    if fi_idx % 200 == 0:
        print(f'  {fi_idx}/{len(npz_files)}...')

for k in var_names:
    results_A[k] = np.array(results_A[k])
    results_B[k] = np.array(results_B[k])
frames = np.array(frames)

OUT = '/home/elicer/kinematic_results/'
os.makedirs(OUT, exist_ok=True)
np.savez(os.path.join(OUT, 'ablation_8vars.npz'), frames=frames,
         **{f'A_{k}': results_A[k] for k in var_names},
         **{f'B_{k}': results_B[k] for k in var_names})

print('\n' + '='*85)
print('Ablation: 8 Kinematic Variables (n=%d frames)' % len(frames))
print('='*85)
header = '  %-28s %11s %7s %14s %7s %7s' % ('Variable', 'A(init)', 'A_SD', 'B(refined)', 'B_SD', 'Delta')
print(header)
print('-'*85)
for k in var_names:
    am, asd = np.nanmean(results_A[k]), np.nanstd(results_A[k])
    bm, bsd = np.nanmean(results_B[k]), np.nanstd(results_B[k])
    d = bm - am
    print('  %-28s %10.1f° %6.1f° %13.1f° %6.1f° %+6.1f°' % (k, am, asd, bm, bsd, d))

# CSV
with open(os.path.join(OUT, 'ablation_summary.csv'), 'w') as f:
    f.write('variable,A_mean,A_sd,B_mean,B_sd,delta\n')
    for k in var_names:
        f.write('%s,%.2f,%.2f,%.2f,%.2f,%.2f\n' % (
            k, np.nanmean(results_A[k]), np.nanstd(results_A[k]),
            np.nanmean(results_B[k]), np.nanstd(results_B[k]),
            np.nanmean(results_B[k]) - np.nanmean(results_A[k])))

print('\nSaved: %s' % OUT)
print('ABLATION COMPLETE!')
