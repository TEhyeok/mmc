"""
SMPL → BSM 105-marker TRC → OpenSim IK
SAM4DCap/SMPL2AddBiomechanics 방식 구현

1. bsm_markers.yaml에서 SMPL 정점 → 105개 마커 매핑 로드
2. gsplat 정제된 SMPL 정점에서 마커 위치 추출
3. TRC 파일 생성
4. OpenSim IK 실행 → .mot 파일 (관절각도)
5. Vicon 비교
"""
import numpy as np, torch, smplx, json, os, yaml
import opensim as osim

# ============ Config ============
REFINED_DIR = '/home/elicer/gsplat_refined_v2/'
SMPL_PATH = '/home/elicer/EasyMocap/data/smplx/'
BSM_YAML = '/home/elicer/SMPL2AddBiomechanics/smpl2ab/data/bsm_markers.yaml'
OUTPUT_DIR = '/home/elicer/opensim_ik_results/'
FPS = 240
# ================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load BSM marker mapping
print('Loading BSM 105-marker mapping...')
with open(BSM_YAML) as f:
    bsm_markers = yaml.safe_load(f)
marker_names = list(bsm_markers.keys())
marker_indices = list(bsm_markers.values())
print('  %d markers loaded' % len(marker_names))

# Load SMPL model
print('Loading SMPL...')
model = smplx.create(SMPL_PATH, model_type='smpl', gender='neutral', batch_size=1).cuda()

# ============ 1. Extract markers ============
print('\n=== Step 1: Extract 105 markers from SMPL vertices ===')

npz_files = sorted([f for f in os.listdir(REFINED_DIR) if f.endswith('.npz')])
n_frames = len(npz_files)
print('Frames: %d' % n_frames)

# Arrays: (n_frames, n_markers, 3) in meters
markers_init = np.zeros((n_frames, len(marker_names), 3))
markers_refined = np.zeros((n_frames, len(marker_names), 3))
frame_indices = []

for fi_idx, npz_file in enumerate(npz_files):
    frame_idx = int(npz_file.replace('.npz', ''))
    data = np.load(os.path.join(REFINED_DIR, npz_file), allow_pickle=True)
    frame_indices.append(frame_idx)

    for cond, bp_key, go_key, out_arr in [
        ('init', 'body_pose_init', 'global_orient_init', markers_init),
        ('refined', 'body_pose_refined', 'global_orient_refined', markers_refined)]:

        with torch.no_grad():
            out = model(
                body_pose=torch.tensor(data[bp_key], dtype=torch.float32).unsqueeze(0).cuda(),
                global_orient=torch.tensor(data[go_key], dtype=torch.float32).unsqueeze(0).cuda(),
                betas=torch.tensor(data['betas'], dtype=torch.float32).unsqueeze(0).cuda(),
                transl=torch.tensor(data['transl'], dtype=torch.float32).unsqueeze(0).cuda()
            )
            verts = out.vertices[0].cpu().numpy()  # (6890, 3) in meters

        # Extract markers at vertex indices
        out_arr[fi_idx] = verts[marker_indices]  # (n_markers, 3)

    if fi_idx % 50 == 0:
        print('  %d/%d...' % (fi_idx, n_frames))

frame_indices = np.array(frame_indices)
print('Extracted: init %s, refined %s' % (markers_init.shape, markers_refined.shape))

# ============ 2. Write TRC ============
print('\n=== Step 2: Write TRC files ===')

def write_trc(filepath, names, data, fps=240):
    """
    Write OpenSim-compatible TRC file
    names: list of marker names
    data: (n_frames, n_markers, 3) in meters
    """
    n_frames, n_markers, _ = data.shape

    with open(filepath, 'w') as f:
        f.write('PathFileType\t4\t(X/Y/Z)\t%s\n' % os.path.basename(filepath))
        f.write('DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n')
        f.write('%d\t%d\t%d\t%d\tm\t%d\t1\t%d\n' % (fps, fps, n_frames, n_markers, fps, n_frames))

        # Marker names
        line = 'Frame#\tTime'
        for name in names:
            line += '\t%s\t\t' % name
        f.write(line.rstrip() + '\n')

        # X/Y/Z
        line = '\t'
        for i in range(n_markers):
            line += '\tX%d\tY%d\tZ%d' % (i+1, i+1, i+1)
        f.write(line + '\n')

        f.write('\n')

        # Data
        for fi in range(n_frames):
            t = fi / fps
            line = '%d\t%.6f' % (fi + 1, t)
            for mi in range(n_markers):
                line += '\t%.6f\t%.6f\t%.6f' % (data[fi, mi, 0], data[fi, mi, 1], data[fi, mi, 2])
            f.write(line + '\n')

trc_init = os.path.join(OUTPUT_DIR, 'bsm105_init.trc')
trc_refined = os.path.join(OUTPUT_DIR, 'bsm105_refined.trc')
write_trc(trc_init, marker_names, markers_init, FPS)
write_trc(trc_refined, marker_names, markers_refined, FPS)
print('Written: %s' % trc_init)
print('Written: %s' % trc_refined)

# ============ 3. OpenSim IK ============
print('\n=== Step 3: OpenSim IK ===')

def run_ik(trc_path, output_mot, model_file=None):
    """Run OpenSim IK using built-in gait model"""

    # Use generic gait2392 model (comes with OpenSim)
    # Or create a simple model
    if model_file and os.path.exists(model_file):
        mdl = osim.Model(model_file)
    else:
        # Use OpenSim's built-in gait model
        # Find the default model
        install_dir = os.path.dirname(osim.__file__)
        # Common OpenSim model paths
        possible_models = [
            os.path.join(install_dir, 'Models', 'Gait2354_Simbody', 'gait2354_simbody.osim'),
            os.path.join(install_dir, 'Models', 'gait2392_simbody.osim'),
        ]
        mdl = None
        for mp in possible_models:
            if os.path.exists(mp):
                mdl = osim.Model(mp)
                print('  Using model: %s' % mp)
                break

        if mdl is None:
            # Create minimal model for IK
            print('  No built-in model found. Creating marker-based analysis...')
            return None

    # Setup IK tool
    ik_tool = osim.InverseKinematicsTool()
    ik_tool.setModel(mdl)
    ik_tool.setMarkerDataFileName(trc_path)
    ik_tool.setOutputMotionFileName(output_mot)
    ik_tool.setStartTime(0)
    ik_tool.setEndTime((len(frame_indices) - 1) / FPS)

    # Run
    print('  Running IK...')
    ik_tool.run()
    print('  Output: %s' % output_mot)
    return output_mot

# Try running IK
mot_init = os.path.join(OUTPUT_DIR, 'ik_init.mot')
mot_refined = os.path.join(OUTPUT_DIR, 'ik_refined.mot')

# First check available OpenSim models
print('Searching for OpenSim models...')
osim_dir = os.path.dirname(osim.__file__)
print('  OpenSim package dir: %s' % osim_dir)

# List all .osim files nearby
for root, dirs, files in os.walk(osim_dir):
    for f in files:
        if f.endswith('.osim'):
            print('  Found: %s' % os.path.join(root, f))
    if root.count(os.sep) - osim_dir.count(os.sep) > 3:
        break

# Also check SMPL2AddBiomechanics for BSM model
bsm_model = '/home/elicer/SMPL2AddBiomechanics/models/bsm/bsm.osim'
if os.path.exists(bsm_model):
    print('  BSM model found: %s' % bsm_model)
else:
    print('  BSM model NOT found (needs setup_bsm.py)')

# Alternative: Use AddBiomechanics approach (upload TRC to server)
# For now, compute angles from marker positions directly using same JCS as Vicon

print('\n=== Step 3b: Direct ISB angles from 105-marker positions ===')
print('(Using same marker subset as Vicon Plug-in Gait)')

# Map BSM marker names to Plug-in Gait equivalents
# BSM name → Plug-in Gait name
pgait_map = {
    'LSHO': 'L_Shoulder', 'RSHO': 'R_Shoulder',
    'LELB': 'L_Elbow', 'RELB': 'R_Elbow',
    'LWRA': 'L_Wrist', 'RWRA': 'R_Wrist',
    'LKNE': 'L_Knee', 'RKNE': 'R_Knee',
    'LANK': 'L_Ankle', 'RANK': 'R_Ankle',
    'C7': 'C7', 'CLAV': 'CLAV',
    'LUMB': 'LUMB',
    'LFWT': 'L_ASIS', 'RFWT': 'R_ASIS',  # front waist ≈ ASIS
    'LBWT': 'L_PSIS', 'RBWT': 'R_PSIS',  # back waist ≈ PSIS
}

# Get marker indices for Plug-in Gait subset
pgait_indices = {}
for bsm_name, pgait_name in pgait_map.items():
    if bsm_name in bsm_markers:
        idx = marker_names.index(bsm_name)
        pgait_indices[pgait_name] = idx

print('Plug-in Gait markers mapped: %d' % len(pgait_indices))
for k, v in pgait_indices.items():
    print('  %s → BSM idx %d (vertex %d)' % (k, v, marker_indices[v]))

# Save TRC and marker info
np.savez(os.path.join(OUTPUT_DIR, 'marker_data.npz'),
         marker_names=marker_names,
         marker_indices=np.array(marker_indices),
         markers_init=markers_init,
         markers_refined=markers_refined,
         frame_indices=frame_indices,
         pgait_map=pgait_indices)

print('\nSaved: %s' % OUTPUT_DIR)
print('TRC files ready for OpenSim IK or AddBiomechanics upload')
print('DONE!')
