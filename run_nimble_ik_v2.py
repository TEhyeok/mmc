"""nimblephysics IK v2: BSM→Rajagopal marker mapping + MarkerFitter"""
import nimblephysics as nimble
import numpy as np
import os, json

OUTPUT_DIR = '/home/elicer/nimble_ik_results/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# BSM → Rajagopal marker name mapping
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

# Load model
osim_path = os.path.join(os.path.dirname(nimble.__file__), 'models', 'rajagopal_data', 'Rajagopal2015.osim')
skel = nimble.biomechanics.OpenSimParser.parseOsim(osim_path)
skeleton = skel.skeleton
print('Model: %d DOFs, %d bodies, %d markers' % (
    skeleton.getNumDofs(), skeleton.getNumBodyNodes(), len(skel.markersMap)))

# DOF names
dof_names = []
for ji in range(skeleton.getNumJoints()):
    j = skeleton.getJoint(ji)
    for di in range(j.getNumDofs()):
        dof_names.append(j.getName() + '_' + str(di))
print('DOFs:', dof_names)

def load_and_rename_trc(trc_path):
    """Load TRC and rename BSM markers to Rajagopal names"""
    trc = nimble.biomechanics.OpenSimParser.loadTRC(trc_path)
    renamed = []
    for frame_markers in trc.markerTimesteps:
        new_frame = {}
        for bsm_name, pos in frame_markers.items():
            raj_name = BSM_TO_RAJ.get(bsm_name)
            if raj_name and raj_name in skel.markersMap:
                new_frame[raj_name] = pos
        renamed.append(new_frame)
    return renamed

def run_ik(trc_path, label):
    """Run IK on a TRC file"""
    print('\n=== IK: %s ===' % label)

    # Load and rename markers
    marker_timesteps = load_and_rename_trc(trc_path)
    n_frames = len(marker_timesteps)
    n_markers = len(marker_timesteps[0]) if n_frames > 0 else 0
    print('Frames: %d, Matched markers: %d' % (n_frames, n_markers))
    print('Marker names:', sorted(marker_timesteps[0].keys()) if n_frames > 0 else [])

    # Setup MarkerFitter
    fitter = nimble.biomechanics.MarkerFitter(skeleton, skel.markersMap)
    fitter.setStaticTrial(marker_timesteps[0])

    # Initial params
    init_params = nimble.biomechanics.InitialMarkerFitParams()
    init_params.numBlocks = 12

    # Run kinematics pipeline
    print('Running IK pipeline...')
    results = fitter.runKinematicsPipeline(
        marker_timesteps,
        init_params,
        150,
        True
    )

    print('IK complete! Poses shape: %s' % str(results.poses.shape))

    # Extract angles
    angles = np.degrees(results.poses.T)  # (n_frames, n_dofs)
    print('Angles shape:', angles.shape)

    # Save
    np.savez(os.path.join(OUTPUT_DIR, 'ik_%s.npz' % label),
             dof_names=dof_names, angles=angles,
             poses_raw=results.poses)

    # Print key DOFs
    key_indices = {}
    for i, name in enumerate(dof_names):
        for key in ['hip_r_0','hip_l_0','walker_knee_r_0','walker_knee_l_0',
                     'ankle_r_0','ankle_l_0','back_0','back_1','back_2',
                     'acromial_r_0','acromial_r_1','acromial_r_2',
                     'acromial_l_0','acromial_l_1','acromial_l_2',
                     'elbow_r_0','elbow_l_0']:
            if name == key:
                key_indices[name] = i

    print('\nKey joint angles (mean +/- SD):')
    for name, idx in sorted(key_indices.items()):
        vals = angles[:, idx]
        print('  %s: %.1f +/- %.1f deg' % (name, vals.mean(), vals.std()))

    return angles

# Run both conditions
try:
    angles_init = run_ik('/home/elicer/opensim_ik_results/bsm105_init.trc', 'init')
    angles_refined = run_ik('/home/elicer/opensim_ik_results/bsm105_refined.trc', 'refined')

    # Ablation comparison
    print('\n' + '=' * 90)
    print('ABLATION: Init vs Refined (Rajagopal IK)')
    print('=' * 90)

    # Map Rajagopal DOFs to thesis variables
    thesis_map = {
        'lead_knee_flexion': 'walker_knee_l_0',
        'trunk_forward_tilt': 'back_0',
        'trunk_lateral_tilt': 'back_1',
        'trunk_axial_rotation': 'back_2',
        'shoulder_abduction': 'acromial_r_1',   # adduction DOF
        'shoulder_flexion': 'acromial_r_0',
        'shoulder_rotation': 'acromial_r_2',
        'elbow_flexion': 'elbow_r_0',
    }

    print('  %-28s %10s %7s %10s %7s %7s' % ('Variable', 'Init', 'I_SD', 'Refined', 'R_SD', 'r(I,R)'))
    print('-' * 85)

    for thesis_name, dof_name in thesis_map.items():
        idx = dof_names.index(dof_name) if dof_name in dof_names else -1
        if idx < 0:
            print('  %-28s NOT FOUND (%s)' % (thesis_name, dof_name))
            continue
        a = angles_init[:, idx]
        b = angles_refined[:, idx]
        r = np.corrcoef(a, b)[0, 1]
        print('  %-28s %9.1f %6.1f %9.1f %6.1f %6.3f' % (
            thesis_name, a.mean(), a.std(), b.mean(), b.std(), r))

    print('\nSaved: %s' % OUTPUT_DIR)
    print('DONE!')

except Exception as e:
    print('ERROR: %s' % str(e))
    import traceback
    traceback.print_exc()
