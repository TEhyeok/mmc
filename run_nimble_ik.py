"""
nimblephysics 기반 로컬 IK 파이프라인
BSM 105-marker TRC → MarkerFitter → 스켈레톤 스케일링 + IK → 관절각도

nimblephysics의 MarkerFitter는 AddBiomechanics와 동일한 엔진입니다.
"""
import nimblephysics as nimble
import numpy as np
import os, json

# ============ Config ============
TRC_INIT = '/home/elicer/opensim_ik_results/bsm105_init.trc'
TRC_REFINED = '/home/elicer/opensim_ik_results/bsm105_refined.trc'
OUTPUT_DIR = '/home/elicer/nimble_ik_results/'
SUBJECT_HEIGHT = 1.797  # meters (from thesis Table 7)
SUBJECT_MASS = 82.0     # kg
FPS = 240
# ================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_ik_pipeline(trc_path, label):
    """Run full IK pipeline on a TRC file"""
    print('\n' + '=' * 70)
    print('Running IK: %s (%s)' % (label, os.path.basename(trc_path)))
    print('=' * 70)

    # 1. Load the standard Rajagopal full-body model
    print('Loading skeleton...')
    # nimblephysics has a built-in human skeleton
    osim_path = os.path.join(os.path.dirname(nimble.__file__), 'models', 'rajagopal_data', 'Rajagopal2015.osim')
    skel = nimble.biomechanics.OpenSimParser.parseOsim(osim_path)
    skeleton = skel.skeleton
    print('  Model: %s' % osim_path)
    print('  Skeleton: %d DOFs, %d bodies' % (skeleton.getNumDofs(), skeleton.getNumBodyNodes()))

    # List DOF names
    dof_names = [skeleton.getDof(i).getName() for i in range(skeleton.getNumDofs())]
    print('  DOFs: %s' % dof_names[:15])

    # 2. Load TRC
    print('Loading TRC...')
    trc = nimble.biomechanics.OpenSimParser.loadTRC(trc_path)
    n_frames = len(trc.markerTimesteps)
    print('  Frames: %d, Markers: %d' % (n_frames, len(trc.markerTimesteps[0]) if n_frames > 0 else 0))

    # 3. Setup MarkerFitter (same engine as AddBiomechanics)
    print('Setting up MarkerFitter...')
    fitter = nimble.biomechanics.MarkerFitter(skeleton, skel.markersMap)

    # Set anthropometric constraints
    fitter.setStaticTrial(trc.markerTimesteps[0])
    # fitter.setAnthropometricPrior(SUBJECT_HEIGHT, SUBJECT_MASS)

    # 4. Run IK
    print('Running IK fitting (%d frames)...' % n_frames)
    timestamps = [i / FPS for i in range(n_frames)]

    results = fitter.runKinematicsPipeline(
        trc.markerTimesteps,
        nimble.biomechanics.InitialMarkerFitParams(),
        150,  # iterations
        True   # use analytical IK
    )

    print('  IK complete!')

    # 5. Extract joint angles
    print('Extracting joint angles...')
    n_dofs = skeleton.getNumDofs()
    all_angles = np.zeros((n_frames, n_dofs))

    for fi in range(n_frames):
        skeleton.setPositions(results.poses[:, fi])
        all_angles[fi] = skeleton.getPositions()

    # 6. Map DOF names to our 8 variables
    # Rajagopal model DOF names typically include:
    # hip_flexion_r/l, knee_angle_r/l, ankle_angle_r/l
    # lumbar_extension, lumbar_bending, lumbar_rotation
    # arm_flex_r/l, arm_add_r/l, arm_rot_r/l
    # elbow_flex_r/l

    angle_data = {}
    for i, name in enumerate(dof_names):
        angle_data[name] = np.degrees(all_angles[:, i])

    # Save raw results
    out_path = os.path.join(OUTPUT_DIR, 'ik_%s.npz' % label)
    np.savez(out_path, dof_names=dof_names, angles=all_angles, timestamps=timestamps)
    print('  Saved: %s' % out_path)

    # Print summary of key variables
    print('\n  Key joint angles (mean +/- SD):')
    key_dofs = [n for n in dof_names if any(k in n.lower() for k in
                ['knee', 'hip_flex', 'lumbar', 'arm_flex', 'arm_add', 'arm_rot', 'elbow'])]
    for name in key_dofs:
        vals = angle_data[name]
        print('    %s: %.1f +/- %.1f deg' % (name, vals.mean(), vals.std()))

    return angle_data, dof_names

# Run for both conditions
try:
    angles_init, dof_names = run_ik_pipeline(TRC_INIT, 'init')
    angles_refined, _ = run_ik_pipeline(TRC_REFINED, 'refined')

    # Compare
    print('\n' + '=' * 90)
    print('Ablation: Init vs Refined')
    print('=' * 90)
    print('  %-30s %10s %8s %10s %8s %8s' % ('DOF', 'Init', 'I_SD', 'Refined', 'R_SD', 'r'))
    print('-' * 90)

    key_dofs = [n for n in dof_names if any(k in n.lower() for k in
                ['knee', 'hip_flex', 'lumbar', 'arm_flex', 'arm_add', 'arm_rot', 'elbow'])]

    for name in key_dofs:
        a = angles_init[name]
        b = angles_refined[name]
        mask = ~(np.isnan(a) | np.isnan(b))
        if mask.sum() > 10:
            r = np.corrcoef(a[mask], b[mask])[0, 1]
        else:
            r = float('nan')
        print('  %-30s %9.1f %7.1f %9.1f %7.1f %7.3f' % (
            name, a.mean(), a.std(), b.mean(), b.std(), r))

    print('\nDONE!')

except Exception as e:
    print('ERROR: %s' % str(e))
    import traceback
    traceback.print_exc()

    # Fallback: try simpler skeleton fitting
    print('\n--- Fallback: Direct marker IK ---')
    try:
        osim_path = os.path.join(os.path.dirname(nimble.__file__), 'models', 'rajagopal_data', 'Rajagopal2015.osim')
        skel = nimble.biomechanics.OpenSimParser.parseOsim(osim_path)
        skeleton = skel.skeleton
        trc = nimble.biomechanics.OpenSimParser.loadTRC(TRC_REFINED)

        # Simple frame-by-frame IK
        dof_names = [skeleton.getDof(i).getName() for i in range(skeleton.getNumDofs())]
        print('DOFs (%d): %s' % (len(dof_names), dof_names))

        # Get marker list from osim model
        marker_names_osim = list(skel.markersMap.keys())
        print('Model markers (%d): %s' % (len(marker_names_osim), marker_names_osim[:10]))

        # Get TRC marker names
        trc_marker_names = list(trc.markerTimesteps[0].keys()) if trc.markerTimesteps else []
        print('TRC markers (%d): %s' % (len(trc_marker_names), trc_marker_names[:10]))

        # Find matching markers
        matching = set(marker_names_osim) & set(trc_marker_names)
        print('Matching markers: %d' % len(matching))
        print('Matched: %s' % sorted(matching))

    except Exception as e2:
        print('Fallback ERROR: %s' % str(e2))
        traceback.print_exc()
