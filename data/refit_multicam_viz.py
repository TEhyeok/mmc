#!/usr/bin/env python3
"""Overlay ReFit results from cam3,5,6,7 on original + masked images."""
import numpy as np, torch, cv2, os, json, smplx

body_model = smplx.create(model_path="/home/elicer/EasyMocap/data/smplx",
                          model_type="smpl", gender="neutral", batch_size=1)

bones = [[0,1],[0,2],[0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],
         [9,12],[9,13],[9,14],[12,15],[13,16],[14,17],[16,18],[17,19]]

# Crop offsets per camera (from OpenPose bbox, frame 0)
crop_info = {
    3: (559, 261),
    5: (856, 216),
    6: (373, 326),
    7: (904, 370),
}

out_dir = "/home/elicer/refit_multicam_overlay"
os.makedirs(out_dir, exist_ok=True)

for cam_id in [3, 5, 6, 7]:
    if cam_id == 3:
        d = np.load("/home/elicer/ReFit/poses_optimized.npz")
        c = np.load("/home/elicer/ReFit/cameras.npz")
    else:
        d = np.load(f"/home/elicer/refit_cam{cam_id}.npz")
        c = np.load(f"/home/elicer/refit_cam{cam_id}_cameras.npz")

    focal = c["intrinsic"][0, 0]
    cx_crop = c["intrinsic"][0, 2]
    cy_crop = c["intrinsic"][1, 2]
    crop_x1, crop_y1 = crop_info[cam_id]

    fi = 0
    out = body_model(
        global_orient=torch.tensor(d["global_orient"][fi:fi+1], dtype=torch.float32),
        body_pose=torch.tensor(d["body_pose"][fi:fi+1], dtype=torch.float32),
        betas=torch.tensor(d["betas"][None], dtype=torch.float32),
        transl=torch.tensor(d["transl"][fi:fi+1], dtype=torch.float32))

    verts = out.vertices[0].detach().numpy()
    joints = out.joints[0].detach().numpy()[:24]

    # Project
    j2d = np.zeros((24, 2))
    j2d[:, 0] = focal * joints[:, 0] / joints[:, 2] + cx_crop + crop_x1
    j2d[:, 1] = focal * joints[:, 1] / joints[:, 2] + cy_crop + crop_y1

    v2d = np.zeros((len(verts), 2))
    v2d[:, 0] = focal * verts[:, 0] / verts[:, 2] + cx_crop + crop_x1
    v2d[:, 1] = focal * verts[:, 1] / verts[:, 2] + cy_crop + crop_y1

    # Original image
    img = cv2.imread(f"/home/elicer/baseball_h36m/images/cam{cam_id}/00001.png")

    # Draw mesh (orange)
    for v in v2d[::5]:
        x, y = int(v[0]), int(v[1])
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(img, (x, y), 1, (0, 140, 255), -1)

    # Draw skeleton (blue)
    for a, b in bones:
        cv2.line(img, (int(j2d[a,0]), int(j2d[a,1])),
                 (int(j2d[b,0]), int(j2d[b,1])), (255, 100, 0), 2)
    for j in range(24):
        cv2.circle(img, (int(j2d[j,0]), int(j2d[j,1])), 5, (255, 100, 0), -1)

    cv2.putText(img, f"ReFit cam{cam_id} frame 0", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.imwrite(os.path.join(out_dir, f"cam{cam_id}_frame0.jpg"), img)
    print(f"cam{cam_id}: focal={focal:.0f}, joints x=[{j2d[:,0].min():.0f},{j2d[:,0].max():.0f}] y=[{j2d[:,1].min():.0f},{j2d[:,1].max():.0f}]")

print("Done")
