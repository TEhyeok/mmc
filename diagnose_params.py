"""Diagnose SAM-3D-Body params vs SMPL format"""
import numpy as np, torch, smplx, json, os
from scipy.spatial.transform import Rotation

model = smplx.create('/home/elicer/EasyMocap/data/smplx/', model_type='smpl',
                      gender='neutral', batch_size=1).cuda()
marker_map = json.load(open('/home/elicer/server_scripts/smpl_virtual_marker_mapping.json'))

# Load frame 500
d = np.load('/home/elicer/sam3d_results/cam3/000500.npz')
print('=== SAM-3D-Body NPZ keys ===')
for k in d.keys():
    arr = d[k]
    print('  %s: shape=%s dtype=%s range=[%.3f, %.3f]' % (k, arr.shape, arr.dtype, arr.min(), arr.max()))

bp_133 = d['body_pose_params']
sp_45 = d['shape_params']
gr = d['pred_global_rots']  # (127, 3, 3)
kp3d = d['pred_keypoints_3d']  # (70, 3)
verts_mhr = d['pred_vertices']  # (18439, 3)

print('\n=== Test 1: body_pose_params[:69] as SMPL body_pose ===')
root_rot = Rotation.from_matrix(gr[0]).as_rotvec()
with torch.no_grad():
    out1 = model(
        body_pose=torch.tensor(bp_133[:69], dtype=torch.float32).unsqueeze(0).cuda(),
        global_orient=torch.tensor(root_rot, dtype=torch.float32).unsqueeze(0).cuda(),
        betas=torch.tensor(sp_45[:10], dtype=torch.float32).unsqueeze(0).cuda(),
        transl=torch.zeros(1,3).cuda()
    )
v1 = out1.vertices[0].cpu().numpy()
j1 = out1.joints[0].cpu().numpy()
print('Joints[0] pelvis:    [%.3f, %.3f, %.3f]' % tuple(j1[0]))
print('Joints[16] L_sho:    [%.3f, %.3f, %.3f]' % tuple(j1[16]))
print('Joints[17] R_sho:    [%.3f, %.3f, %.3f]' % tuple(j1[17]))
print('Joints[19] R_elb:    [%.3f, %.3f, %.3f]' % tuple(j1[19]))
print('Verts range: X=[%.3f,%.3f] Y=[%.3f,%.3f] Z=[%.3f,%.3f]' % (
    v1[:,0].min(), v1[:,0].max(), v1[:,1].min(), v1[:,1].max(), v1[:,2].min(), v1[:,2].max()))

print('\n=== Test 2: body_pose_params[3:72] as SMPL body_pose (skip global) ===')
with torch.no_grad():
    out2 = model(
        body_pose=torch.tensor(bp_133[3:72], dtype=torch.float32).unsqueeze(0).cuda(),
        global_orient=torch.tensor(bp_133[:3], dtype=torch.float32).unsqueeze(0).cuda(),
        betas=torch.tensor(sp_45[:10], dtype=torch.float32).unsqueeze(0).cuda(),
        transl=torch.zeros(1,3).cuda()
    )
v2 = out2.vertices[0].cpu().numpy()
j2 = out2.joints[0].cpu().numpy()
print('Joints[0] pelvis:    [%.3f, %.3f, %.3f]' % tuple(j2[0]))
print('Joints[16] L_sho:    [%.3f, %.3f, %.3f]' % tuple(j2[16]))
print('Verts range: X=[%.3f,%.3f] Y=[%.3f,%.3f] Z=[%.3f,%.3f]' % (
    v2[:,0].min(), v2[:,0].max(), v2[:,1].min(), v2[:,1].max(), v2[:,2].min(), v2[:,2].max()))

print('\n=== Test 3: MHR 3D keypoints (ground truth) ===')
print('KP3D[0] (head?):     [%.3f, %.3f, %.3f]' % tuple(kp3d[0]))
print('KP3D[1]:             [%.3f, %.3f, %.3f]' % tuple(kp3d[1]))
print('KP3D range: X=[%.3f,%.3f] Y=[%.3f,%.3f] Z=[%.3f,%.3f]' % (
    kp3d[:,0].min(), kp3d[:,0].max(), kp3d[:,1].min(), kp3d[:,1].max(), kp3d[:,2].min(), kp3d[:,2].max()))

print('\n=== Test 4: MHR vertices ===')
print('MHR verts range: X=[%.3f,%.3f] Y=[%.3f,%.3f] Z=[%.3f,%.3f]' % (
    verts_mhr[:,0].min(), verts_mhr[:,0].max(), verts_mhr[:,1].min(), verts_mhr[:,1].max(),
    verts_mhr[:,2].min(), verts_mhr[:,2].max()))

print('\n=== Test 5: Virtual markers from SMPL (Test 1 config) ===')
for name in ['L_ASIS','R_ASIS','C7','T10','CLAV','R_Shoulder','R_Elbow_lat','R_Wrist']:
    indices = marker_map[name]
    pos = v1[indices].mean(axis=0)
    print('  %s: [%.4f, %.4f, %.4f]' % (name, pos[0], pos[1], pos[2]))

print('\n=== Test 6: Compare multiple frames ===')
for fi in [0, 250, 500, 750, 998]:
    npz_path = '/home/elicer/sam3d_results/cam3/%06d.npz' % fi
    if not os.path.exists(npz_path):
        npz_path = '/home/elicer/sam3d_results/cam1/%06d.npz' % fi
    if not os.path.exists(npz_path):
        print('  Frame %d: NOT FOUND' % fi)
        continue
    dd = np.load(npz_path)
    bp = dd['body_pose_params']
    print('  Frame %d: bp[:6]=[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f] norm=%.3f' % (
        fi, bp[0], bp[1], bp[2], bp[3], bp[4], bp[5], np.linalg.norm(bp[:69])))

print('\n=== Test 7: Refined vs init comparison (gsplat output) ===')
ref = np.load('/home/elicer/gsplat_refined_smpl/000500.npz', allow_pickle=True)
bp_i = ref['body_pose_init']
bp_r = ref['body_pose_refined']
diff = np.abs(bp_r - bp_i)
print('body_pose diff: mean=%.4f rad (%.2f deg), max=%.4f rad (%.2f deg)' % (
    diff.mean(), np.degrees(diff.mean()), diff.max(), np.degrees(diff.max())))

# Check per-joint diff
bp_i_23 = bp_i.reshape(23, 3)
bp_r_23 = bp_r.reshape(23, 3)
jnames = ['L_Hip','R_Hip','Spine1','L_Knee','R_Knee','Spine2',
          'L_Ankle','R_Ankle','Spine3','L_Foot','R_Foot','Neck',
          'L_Collar','R_Collar','Head','L_Shoulder','R_Shoulder',
          'L_Elbow','R_Elbow','L_Wrist','R_Wrist','L_Hand','R_Hand']
for i, jn in enumerate(jnames):
    d_deg = np.degrees(np.linalg.norm(bp_r_23[i] - bp_i_23[i]))
    print('  %s: %.2f deg change' % (jn, d_deg))

print('\nDIAGNOSIS COMPLETE')
