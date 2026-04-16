"""Fix TRC marker names (BSM→Rajagopal) and run ScaleTool + IK"""
import numpy as np
import os

OUTPUT_DIR = '/home/elicer/opensim_scaled_ik/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def rebuild_trc(input_trc, output_trc):
    """Rebuild TRC with only Rajagopal-matched markers"""
    with open(input_trc) as f:
        lines = f.readlines()

    # Parse header line 3 (marker names): "Frame#\tTime\tM1\t\t\tM2\t\t\t..."
    header = lines[2]
    # Extract marker names (non-empty fields after Frame#, Time)
    parts = header.strip().split('\t')
    old_markers = []
    col_indices = []  # data column index for each marker
    ci = 2  # skip Frame#, Time
    for p in parts[2:]:
        if p.strip():
            old_markers.append(p.strip())
            col_indices.append(ci)
        ci += 1

    # Filter: only keep markers that map to Rajagopal
    new_markers = []
    new_col_offsets = []  # (original data col start for X, marker index)
    for i, bsm_name in enumerate(old_markers):
        raj_name = BSM_TO_RAJ.get(bsm_name)
        if raj_name:
            new_markers.append(raj_name)
            # Each marker has 3 columns (X,Y,Z) starting at col_indices[i] in data
            # In data rows, col 0=Frame, col 1=Time, then 3 per marker
            data_col = 2 + i * 3
            new_col_offsets.append(data_col)

    n_markers = len(new_markers)
    print('Rebuilding TRC: %d BSM markers → %d Rajagopal markers' % (len(old_markers), n_markers))
    print('Markers:', new_markers)

    # Parse data rows (skip header: lines 0-4, data starts at line 5)
    data_start = 5
    # Find actual data start (after blank line)
    for di in range(3, len(lines)):
        line = lines[di].strip()
        if line and line[0].isdigit():
            data_start = di
            break

    data_rows = []
    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        if not line:
            break
        parts = line.split('\t')
        data_rows.append(parts)

    n_frames = len(data_rows)
    fps = 240

    # Write new TRC
    with open(output_trc, 'w') as f:
        f.write('PathFileType\t4\t(X/Y/Z)\t%s\n' % os.path.basename(output_trc))
        f.write('DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n')
        f.write('%d\t%d\t%d\t%d\tm\t%d\t1\t%d\n' % (fps, fps, n_frames, n_markers, fps, n_frames))

        # Marker names
        line = 'Frame#\tTime'
        for name in new_markers:
            line += '\t%s\t\t' % name
        f.write(line.rstrip() + '\n')

        # X/Y/Z sub-header
        line = '\t'
        for i in range(n_markers):
            line += '\tX%d\tY%d\tZ%d' % (i+1, i+1, i+1)
        f.write(line + '\n')

        f.write('\n')

        # Data
        for row in data_rows:
            frame = row[0]
            time = row[1]
            line = '%s\t%s' % (frame, time)
            for col_start in new_col_offsets:
                x = row[col_start] if col_start < len(row) else '0'
                y = row[col_start+1] if col_start+1 < len(row) else '0'
                z = row[col_start+2] if col_start+2 < len(row) else '0'
                line += '\t%s\t%s\t%s' % (x, y, z)
            f.write(line + '\n')

    print('Written: %s (%d frames, %d markers)' % (output_trc, n_frames, n_markers))

# Rebuild TRC files
trc_init = os.path.join(OUTPUT_DIR, 'raj_init.trc')
trc_refined = os.path.join(OUTPUT_DIR, 'raj_refined.trc')
rebuild_trc('/home/elicer/opensim_ik_results/bsm105_init.trc', trc_init)
rebuild_trc('/home/elicer/opensim_ik_results/bsm105_refined.trc', trc_refined)

# Verify
import opensim as osim

RAJ_MODEL = '/home/elicer/miniconda3/envs/sam3d/lib/python3.11/site-packages/nimblephysics/models/rajagopal_data/Rajagopal2015.osim'

print('\n=== Verification ===')
mdl = osim.Model(RAJ_MODEL)
mdl.initSystem()

model_markers = set()
for i in range(mdl.getMarkerSet().getSize()):
    model_markers.add(mdl.getMarkerSet().get(i).getName())

marker_data = osim.MarkerData(trc_init)
trc_names = set()
for i in range(marker_data.getNumMarkers()):
    trc_names.add(marker_data.getMarkerNames().get(i))

matched = model_markers & trc_names
print('Model: %d, TRC: %d, Matched: %d' % (len(model_markers), len(trc_names), len(matched)))
print('Matched:', sorted(matched))

# Run IK directly (skip ScaleTool for now — use uniform scaling)
print('\n=== Running IK with default model (no scaling) ===')

def run_ik(model_path, trc_path, output_mot, label):
    mdl = osim.Model(model_path)
    mdl.initSystem()
    ik = osim.InverseKinematicsTool()
    ik.setModel(mdl)
    ik.set_marker_file(trc_path)
    ik.set_output_motion_file(output_mot)
    md = osim.MarkerData(trc_path)
    ik.setStartTime(md.getStartFrameTime())
    ik.setEndTime(md.getLastFrameTime())
    ik.set_accuracy(1e-5)
    print('  %s: Running (%d frames)...' % (label, md.getNumFrames()))
    ik.run()
    if os.path.exists(output_mot):
        s = osim.Storage(output_mot)
        labs = [s.getColumnLabels().get(i) for i in range(s.getColumnLabels().getSize())]
        print('  Done: %d frames, %d coords' % (s.getSize(), len(labs)-1))
        return s, labs
    return None, None

s1, labels = run_ik(RAJ_MODEL, trc_init, os.path.join(OUTPUT_DIR, 'ik_init.mot'), 'init')
s2, _ = run_ik(RAJ_MODEL, trc_refined, os.path.join(OUTPUT_DIR, 'ik_refined.mot'), 'refined')

if s1 and s2:
    thesis = {
        'lead_knee_flexion': 'knee_angle_l',
        'trunk_forward_tilt': 'lumbar_extension',
        'trunk_lateral_tilt': 'lumbar_bending',
        'trunk_axial_rotation': 'lumbar_rotation',
        'shoulder_flexion': 'arm_flex_r',
        'shoulder_abduction': 'arm_add_r',
        'shoulder_rotation': 'arm_rot_r',
        'elbow_flexion': 'elbow_flex_r',
    }
    print('\n' + '=' * 95)
    print('RESULTS: Init vs Refined (Matched %d markers)' % len(matched))
    print('=' * 95)
    print('  %-28s %10s %7s %10s %7s %7s' % ('Variable', 'Init', 'I_SD', 'Refined', 'R_SD', 'r'))
    print('-' * 85)

    for tn, cn in thesis.items():
        if cn not in labels:
            print('  %-28s NOT FOUND' % tn)
            continue
        idx = labels.index(cn) - 1
        a = np.array([s1.getStateVector(i).getData().get(idx) for i in range(s1.getSize())])
        b = np.array([s2.getStateVector(i).getData().get(idx) for i in range(s2.getSize())])
        n = min(len(a), len(b)); a, b = a[:n], b[:n]
        if a.std() < 0.01 and b.std() < 0.01:
            print('  %-28s CONSTANT' % tn)
            continue
        r = np.corrcoef(a, b)[0, 1] if a.std() > 0.01 and b.std() > 0.01 else float('nan')
        print('  %-28s %9.1f %6.1f %9.1f %6.1f %6.3f' % (tn, a.mean(), a.std(), b.mean(), b.std(), r))

    np.savez(os.path.join(OUTPUT_DIR, 'ik_angles.npz'),
             labels=labels,
             **{'init_'+k: np.array([s1.getStateVector(i).getData().get(labels.index(v)-1)
                for i in range(s1.getSize())]) for k,v in thesis.items() if v in labels},
             **{'refined_'+k: np.array([s2.getStateVector(i).getData().get(labels.index(v)-1)
                for i in range(s2.getSize())]) for k,v in thesis.items() if v in labels})
    print('\nSaved!')
print('DONE!')
