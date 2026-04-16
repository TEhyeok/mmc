"""OpenSim IK 로컬 실행 — Rajagopal 전신 모델 + BSM→Rajagopal 마커 변환 TRC"""
import opensim as osim
import numpy as np
import os, yaml

OUTPUT_DIR = '/home/elicer/opensim_ik_local/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

RAJ_MODEL = '/home/elicer/miniconda3/envs/sam3d/lib/python3.11/site-packages/nimblephysics/models/rajagopal_data/Rajagopal2015.osim'

BSM_TO_RAJ = {
    'C7':'C7','CLAV':'CLAV','LMT5':'LMT5','RMT5':'RMT5','LTOE':'LTOE','RTOE':'RTOE',
    'LFWT':'LASI','RFWT':'RASI','LBWT':'LPSI','RBWT':'RPSI',
    'LSHO':'LACR','RSHO':'RACR','LELB':'LLEL','RELB':'RLEL',
    'LKNE':'LLFC','RKNE':'RLFC','LKNI':'LMFC','RKNI':'RMFC',
    'LANK':'LLMAL','RANK':'RLMAL','LHEE':'LCAL','RHEE':'RCAL',
    'LWRA':'LFAradius','RWRA':'RFAradius','LWRB':'LFAulna','RWRB':'RFAulna',
    'LFLT':'LTH1','RFLT':'RTH1','LFLB':'LTH2','RFLB':'RTH2',
    'LTIB':'LTB1','RTIB':'RTB1','LTIA':'LTB2','RTIA':'RTB2',
    'LSHN':'LTB3','RSHN':'RTB3','LFRM':'LUA1','RFRM':'RUA1',
}

FPS = 240

def convert_trc(input_trc, output_trc):
    """BSM marker names → Rajagopal marker names in TRC file"""
    with open(input_trc) as f:
        lines = f.readlines()

    # Header: line 3 has marker names
    # Format: Frame#\tTime\tMARKER1\t\t\tMARKER2\t\t\t...
    header_line = lines[2]  # marker names
    parts = header_line.strip().split('\t')

    new_parts = [parts[0], parts[1]]  # Frame#, Time
    marker_count = 0
    for i in range(2, len(parts)):
        p = parts[i].strip()
        if p:  # non-empty = marker name
            raj_name = BSM_TO_RAJ.get(p, None)
            if raj_name:
                new_parts.extend([raj_name, '', ''])
                marker_count += 1
            else:
                new_parts.extend([p, '', ''])
                marker_count += 1

    # Rebuild header
    lines[2] = '\t'.join(new_parts).rstrip() + '\n'

    # Update marker count in line 2 (metadata)
    meta = lines[1].strip().split('\t')
    # NumMarkers is at index 3
    # Keep original count
    lines[1] = '\t'.join(meta) + '\n'

    with open(output_trc, 'w') as f:
        f.writelines(lines)

    print('Converted: %s → %s (%d markers renamed)' % (
        os.path.basename(input_trc), os.path.basename(output_trc), len(BSM_TO_RAJ)))

# Convert TRC files
trc_init_raj = os.path.join(OUTPUT_DIR, 'raj_init.trc')
trc_refined_raj = os.path.join(OUTPUT_DIR, 'raj_refined.trc')
convert_trc('/home/elicer/opensim_ik_results/bsm105_init.trc', trc_init_raj)
convert_trc('/home/elicer/opensim_ik_results/bsm105_refined.trc', trc_refined_raj)

def run_ik(model_path, trc_path, output_mot, label):
    """Run OpenSim IK"""
    print('\n=== OpenSim IK: %s ===' % label)

    mdl = osim.Model(model_path)
    mdl.initSystem()

    # Create IK tool
    ik_tool = osim.InverseKinematicsTool()
    ik_tool.setModel(mdl)
    ik_tool.set_marker_file(trc_path)
    ik_tool.set_output_motion_file(output_mot)

    # Time range
    marker_data = osim.MarkerData(trc_path)
    t_start = marker_data.getStartFrameTime()
    t_end = marker_data.getLastFrameTime()
    ik_tool.setStartTime(t_start)
    ik_tool.setEndTime(t_end)
    print('  Time: %.3f to %.3f s (%d frames)' % (t_start, t_end, marker_data.getNumFrames()))

    # Set accuracy
    ik_tool.set_accuracy(1e-5)

    # Run IK
    print('  Running IK...')
    success = ik_tool.run()
    print('  Success: %s' % success)
    print('  Output: %s' % output_mot)

    # Read results
    if os.path.exists(output_mot):
        storage = osim.Storage(output_mot)
        n_rows = storage.getSize()
        labels = []
        for i in range(storage.getColumnLabels().getSize()):
            labels.append(storage.getColumnLabels().get(i))
        print('  Result: %d frames, %d coordinates' % (n_rows, len(labels)-1))
        print('  Columns: %s' % labels[:10])
        return storage, labels
    return None, None

# Run IK for both conditions
mot_init = os.path.join(OUTPUT_DIR, 'ik_init.mot')
mot_refined = os.path.join(OUTPUT_DIR, 'ik_refined.mot')

s1, labels = run_ik(RAJ_MODEL, trc_init_raj, mot_init, 'init')
s2, _ = run_ik(RAJ_MODEL, trc_refined_raj, mot_refined, 'refined')

if s1 and s2:
    print('\n=== Results ===')

    # Extract key coordinates
    thesis_coords = {
        'lead_knee_flexion': 'knee_angle_l',
        'trunk_forward_tilt': 'lumbar_extension',
        'trunk_lateral_tilt': 'lumbar_bending',
        'trunk_axial_rotation': 'lumbar_rotation',
        'shoulder_flexion': 'arm_flex_r',
        'shoulder_abduction': 'arm_add_r',
        'shoulder_rotation': 'arm_rot_r',
        'elbow_flexion': 'elbow_flex_r',
    }

    for thesis_name, coord_name in thesis_coords.items():
        if coord_name in labels:
            idx = labels.index(coord_name) - 1  # -1 because first column is time
            a_vals = np.array([s1.getStateVector(i).getData().get(idx) for i in range(s1.getSize())])
            b_vals = np.array([s2.getStateVector(i).getData().get(idx) for i in range(s2.getSize())])
            n = min(len(a_vals), len(b_vals))
            r = np.corrcoef(a_vals[:n], b_vals[:n])[0,1]
            print('  %s (%s): init=%.1f+/-%.1f, refined=%.1f+/-%.1f, r=%.3f' % (
                thesis_name, coord_name,
                a_vals.mean(), a_vals.std(), b_vals.mean(), b_vals.std(), r))
        else:
            print('  %s: coord %s NOT FOUND in labels' % (thesis_name, coord_name))

    print('\nDONE!')
else:
    print('IK failed for one or both conditions')
