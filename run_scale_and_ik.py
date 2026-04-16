"""
OpenSim ScaleTool + IK 파이프라인
1. 첫 프레임 마커로 모델 스케일링 (body segment 길이 조정)
2. 스케일된 모델로 IK 실행
"""
import opensim as osim
import numpy as np
import os

OUTPUT_DIR = '/home/elicer/opensim_scaled_ik/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

RAJ_MODEL = '/home/elicer/miniconda3/envs/sam3d/lib/python3.11/site-packages/nimblephysics/models/rajagopal_data/Rajagopal2015.osim'
TRC_INIT = '/home/elicer/opensim_ik_local/raj_init.trc'
TRC_REFINED = '/home/elicer/opensim_ik_local/raj_refined.trc'

# Subject info (from thesis Table 7)
SUBJECT_MASS = 82.0  # kg
SUBJECT_HEIGHT = 1.797  # m

def scale_model(model_path, trc_path, output_model_path, mass=82.0):
    """Scale OpenSim model to match marker positions"""
    print('\n=== Scaling Model ===')

    mdl = osim.Model(model_path)
    state = mdl.initSystem()

    # Load marker data (use first frame as static trial)
    marker_data = osim.MarkerData(trc_path)
    print('Marker data: %d markers, %d frames' % (
        marker_data.getNumMarkers(), marker_data.getNumFrames()))

    # Get model marker names
    model_markers = set()
    for i in range(mdl.getMarkerSet().getSize()):
        model_markers.add(mdl.getMarkerSet().get(i).getName())

    # Get TRC marker names
    trc_markers = set()
    for i in range(marker_data.getNumMarkers()):
        trc_markers.add(marker_data.getMarkerNames().get(i))

    matched = model_markers & trc_markers
    print('Model markers: %d, TRC markers: %d, Matched: %d' % (
        len(model_markers), len(trc_markers), len(matched)))
    print('Matched:', sorted(matched)[:15])

    # --- ScaleTool Setup ---
    scale_tool = osim.ScaleTool()
    scale_tool.setSubjectMass(mass)

    # GenericModelMaker
    gmm = scale_tool.getGenericModelMaker()
    gmm.setModelFileName(model_path)

    # ModelScaler - use marker-based scaling
    scaler = scale_tool.getModelScaler()
    scaler.setApply(True)
    scaler.setMarkerFileName(trc_path)

    # Use first frame time range for static pose
    t0 = marker_data.getStartFrameTime()
    time_range = osim.ArrayDouble()
    time_range.append(t0)
    time_range.append(t0 + 0.01)  # tiny window (single frame)
    scaler.setTimeRange(time_range)

    # Preserve mass distribution
    scaler.setPreserveMassDist(True)
    scaler.setOutputModelFileName(output_model_path)
    scaler.setOutputScaleFileName(os.path.join(OUTPUT_DIR, 'scale_factors.xml'))

    # Add measurement sets for each body segment pair
    # Each measurement uses marker pairs to compute scale factor
    mset = scaler.getMeasurementSet()

    # Define marker pair measurements for key segments
    measurements = {
        'pelvis': [('RASI', 'LASI')],
        'femur_r': [('RASI', 'RLFC')],
        'femur_l': [('LASI', 'LLFC')],
        'tibia_r': [('RLFC', 'RLMAL')],
        'tibia_l': [('LLFC', 'LLMAL')],
        'torso': [('C7', 'CLAV')],
        'humerus_r': [('RACR', 'RLEL')],
        'humerus_l': [('LACR', 'LLEL')],
    }

    for body_name, marker_pairs in measurements.items():
        for m1, m2 in marker_pairs:
            if m1 in matched and m2 in matched:
                meas = osim.Measurement()
                meas.setName(body_name + '_scale')
                meas.setApply(True)

                # Add marker pair
                mp = osim.MarkerPair()
                mp.setMarkerName(0, m1)
                mp.setMarkerName(1, m2)
                meas.getMarkerPairSet().adoptAndAppend(mp)

                # Apply to body
                bs = osim.BodyScaleSet()
                bscale = osim.BodyScale()
                bscale.setName(body_name)
                axes = osim.ArrayStr()
                axes.append('X')
                axes.append('Y')
                axes.append('Z')
                bscale.setAxisNames(axes)
                bs.adoptAndAppend(bscale)
                meas.setBodyScaleSet(bs)

                mset.adoptAndAppend(meas)
                print('  Scale %s: %s — %s' % (body_name, m1, m2))

    # MarkerPlacer - adjust marker offsets
    placer = scale_tool.getMarkerPlacer()
    placer.setApply(True)
    placer.setMarkerFileName(trc_path)
    placer.setTimeRange(time_range)
    placer.setOutputModelFileName(output_model_path)
    placer.setOutputMotionFileName(os.path.join(OUTPUT_DIR, 'static_pose.mot'))

    # Save setup
    setup_file = os.path.join(OUTPUT_DIR, 'scale_setup.xml')
    scale_tool.printToXML(setup_file)
    print('Scale setup saved: %s' % setup_file)

    # Run scaling
    print('Running ScaleTool...')
    try:
        scale_tool.run()
        print('Scaling complete! Output: %s' % output_model_path)
        return True
    except Exception as e:
        print('ScaleTool error: %s' % str(e))

        # Fallback: manual uniform scaling
        print('\nFallback: Manual uniform scaling based on height ratio...')
        # Rajagopal default height ~1.70m, our subject ~1.80m
        scale_factor = SUBJECT_HEIGHT / 1.70
        print('Scale factor: %.3f' % scale_factor)

        for i in range(mdl.getBodySet().getSize()):
            body = mdl.getBodySet().get(i)
            body.setMass(body.getMass() * (scale_factor ** 3) * (mass / 75.0))

        mdl.printToXML(output_model_path)
        print('Uniformly scaled model saved: %s' % output_model_path)
        return True


def run_ik(model_path, trc_path, output_mot, label):
    """Run IK with scaled model"""
    print('\n=== IK: %s ===' % label)

    mdl = osim.Model(model_path)
    mdl.initSystem()

    ik_tool = osim.InverseKinematicsTool()
    ik_tool.setModel(mdl)
    ik_tool.set_marker_file(trc_path)
    ik_tool.set_output_motion_file(output_mot)

    marker_data = osim.MarkerData(trc_path)
    ik_tool.setStartTime(marker_data.getStartFrameTime())
    ik_tool.setEndTime(marker_data.getLastFrameTime())
    ik_tool.set_accuracy(1e-5)

    print('  Running IK (%d frames)...' % marker_data.getNumFrames())
    ik_tool.run()

    if os.path.exists(output_mot):
        storage = osim.Storage(output_mot)
        labels = [storage.getColumnLabels().get(i)
                  for i in range(storage.getColumnLabels().getSize())]
        print('  Done: %d frames, %d coordinates' % (storage.getSize(), len(labels) - 1))
        return storage, labels
    return None, None


# === Main ===
# 1. Scale model using init TRC (first frame = most stable)
scaled_model = os.path.join(OUTPUT_DIR, 'scaled_rajagopal.osim')
scale_model(RAJ_MODEL, TRC_INIT, scaled_model, SUBJECT_MASS)

# 2. Run IK with scaled model
s1, labels = run_ik(scaled_model, TRC_INIT, os.path.join(OUTPUT_DIR, 'ik_init.mot'), 'init')
s2, _ = run_ik(scaled_model, TRC_REFINED, os.path.join(OUTPUT_DIR, 'ik_refined.mot'), 'refined')

# 3. Results
if s1 and s2:
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

    print('\n' + '=' * 100)
    print('RESULTS: Scaled Model IK — Init vs Refined')
    print('=' * 100)
    print('  %-28s %10s %7s %10s %7s %7s' % ('Variable', 'Init', 'I_SD', 'Refined', 'R_SD', 'r'))
    print('-' * 85)

    for thesis_name, coord_name in thesis_coords.items():
        if coord_name not in labels:
            print('  %-28s NOT FOUND' % thesis_name)
            continue
        idx = labels.index(coord_name) - 1
        a = np.array([s1.getStateVector(i).getData().get(idx) for i in range(s1.getSize())])
        b = np.array([s2.getStateVector(i).getData().get(idx) for i in range(s2.getSize())])
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]

        # Skip if constant
        if a.std() < 0.01 and b.std() < 0.01:
            print('  %-28s CONSTANT (range limit)' % thesis_name)
            continue

        r = np.corrcoef(a, b)[0, 1] if a.std() > 0.01 and b.std() > 0.01 else float('nan')
        print('  %-28s %9.1f %6.1f %9.1f %6.1f %6.3f' % (
            thesis_name, a.mean(), a.std(), b.mean(), b.std(), r))

    # Save angles as NPZ
    all_init = {}
    all_refined = {}
    for name, coord in thesis_coords.items():
        if coord in labels:
            idx = labels.index(coord) - 1
            all_init[name] = np.array([s1.getStateVector(i).getData().get(idx) for i in range(s1.getSize())])
            all_refined[name] = np.array([s2.getStateVector(i).getData().get(idx) for i in range(s2.getSize())])

    np.savez(os.path.join(OUTPUT_DIR, 'ik_angles.npz'),
             **{'init_' + k: v for k, v in all_init.items()},
             **{'refined_' + k: v for k, v in all_refined.items()},
             labels=list(labels))
    print('\nSaved: %s' % OUTPUT_DIR)
    print('DONE!')
else:
    print('IK FAILED')
