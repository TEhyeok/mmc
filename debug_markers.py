"""Debug virtual markers: extract positions and JCS axes for visualization"""
import numpy as np, torch, smplx, json, os

model = smplx.create('/home/elicer/EasyMocap/data/smplx/', model_type='smpl',
                      gender='neutral', batch_size=1).cuda()
marker_map = json.load(open('/home/elicer/server_scripts/smpl_virtual_marker_mapping.json'))

# Load 5 representative frames
frames_to_check = [0, 50, 100, 150, 200]
results = []

for fi in frames_to_check:
    npz_path = '/home/elicer/gsplat_refined_v2/%06d.npz' % fi
    if not os.path.exists(npz_path):
        print('Frame %d not found' % fi)
        continue

    data = np.load(npz_path, allow_pickle=True)

    # Compute SMPL for both init and refined
    for label, bp_key, go_key in [('init', 'body_pose_init', 'global_orient_init'),
                                   ('refined', 'body_pose_refined', 'global_orient_refined')]:
        with torch.no_grad():
            out = model(
                body_pose=torch.tensor(data[bp_key], dtype=torch.float32).unsqueeze(0).cuda(),
                global_orient=torch.tensor(data[go_key], dtype=torch.float32).unsqueeze(0).cuda(),
                betas=torch.tensor(data['betas'], dtype=torch.float32).unsqueeze(0).cuda(),
                transl=torch.tensor(data['transl'], dtype=torch.float32).unsqueeze(0).cuda()
            )
            verts = out.vertices[0].cpu().numpy()
            joints = out.joints[0].cpu().numpy()

        # Extract markers
        markers = {}
        for name, indices in marker_map.items():
            if not name.endswith('_center'):
                markers[name] = verts[indices].mean(axis=0).tolist()

        # SMPL joint names
        jnames = ['Pelvis','L_Hip','R_Hip','Spine1','L_Knee','R_Knee',
                  'Spine2','L_Ankle','R_Ankle','Spine3','L_Foot','R_Foot',
                  'Neck','L_Collar','R_Collar','Head','L_Shoulder','R_Shoulder',
                  'L_Elbow','R_Elbow','L_Wrist','R_Wrist','L_Hand','R_Hand']

        # Downsample vertices for visualization (every 10th)
        verts_ds = verts[::10].tolist()

        results.append({
            'frame': fi,
            'label': label,
            'joints': {jnames[i]: joints[i].tolist() for i in range(24)},
            'markers': markers,
            'verts_sample': verts_ds,
            'transl': data['transl'].tolist(),
        })

        if label == 'init':
            # Print key positions
            print('Frame %d %s:' % (fi, label))
            print('  Pelvis:     [%.3f, %.3f, %.3f]' % tuple(joints[0]))
            print('  L_Shoulder: [%.3f, %.3f, %.3f]' % tuple(joints[16]))
            print('  R_Shoulder: [%.3f, %.3f, %.3f]' % tuple(joints[17]))
            print('  R_Elbow:    [%.3f, %.3f, %.3f]' % tuple(joints[19]))
            print('  R_Wrist:    [%.3f, %.3f, %.3f]' % tuple(joints[21]))
            print('  --- Markers ---')
            for mk in ['L_ASIS','R_ASIS','L_PSIS','R_PSIS','C7','T10','CLAV',
                       'R_Shoulder','R_Elbow_lat','R_Elbow_med','R_Wrist']:
                pos = markers[mk]
                print('  %s: [%.3f, %.3f, %.3f]' % (mk, pos[0], pos[1], pos[2]))

            # Check sanity
            print('  --- Sanity checks ---')
            # ASIS should be anterior to PSIS
            asis_z = (np.array(markers['L_ASIS'])[2] + np.array(markers['R_ASIS'])[2]) / 2
            psis_z = (np.array(markers['L_PSIS'])[2] + np.array(markers['R_PSIS'])[2]) / 2
            print('  ASIS z=%.3f vs PSIS z=%.3f (ASIS should be > PSIS if Z=front)' % (asis_z, psis_z))

            # C7 should be posterior to CLAV
            print('  C7 z=%.3f vs CLAV z=%.3f (C7 should be < CLAV if Z=front)' % (
                markers['C7'][2], markers['CLAV'][2]))

            # Shoulder should be lateral
            print('  L_Sho x=%.3f, R_Sho x=%.3f (L should be > R if X=left)' % (
                markers['L_Shoulder'][0] if 'L_Shoulder' in markers else 0,
                markers['R_Shoulder'][0]))

            # C7 should be superior to T10
            print('  C7 y=%.3f vs T10 y=%.3f (C7 should be > T10 if Y=up)' % (
                markers['C7'][1], markers['T10'][1]))
            print()

# Save for HTML viewer
out_path = '/home/elicer/debug_markers.json'
with open(out_path, 'w') as f:
    json.dump(results, f)
print('Saved: %s (%d entries)' % (out_path, len(results)))
print('DONE')
