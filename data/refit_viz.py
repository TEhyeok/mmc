#!/usr/bin/env python3
"""Visualize ReFit SMPL overlay on cam3 images."""
import numpy as np, torch, cv2, os, smplx

d = np.load("/home/elicer/ReFit/poses_optimized.npz")
body_model = smplx.create(model_path="/home/elicer/EasyMocap/data/smplx",
                          model_type="smpl", gender="neutral", batch_size=1)

img_dir = "/home/elicer/baseball_h36m/images/cam3"
out_dir = "/home/elicer/refit_overlay_cam3"
os.makedirs(out_dir, exist_ok=True)

cam = np.load("/home/elicer/ReFit/cameras.npz")
focal = cam["intrinsic"][0, 0]  # 2203
cx = cam["intrinsic"][0, 2]     # 960
cy = cam["intrinsic"][1, 2]     # 540
print(f"Using focal={focal:.1f}, center=({cx:.0f}, {cy:.0f})")

bones = [[0,1],[0,2],[0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],
         [9,12],[9,13],[9,14],[12,15],[13,16],[14,17],[16,18],[17,19]]

for fi in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]:
    out = body_model(
        global_orient=torch.tensor(d["global_orient"][fi:fi+1], dtype=torch.float32),
        body_pose=torch.tensor(d["body_pose"][fi:fi+1], dtype=torch.float32),
        betas=torch.tensor(d["betas"][None], dtype=torch.float32),
        transl=torch.tensor(d["transl"][fi:fi+1], dtype=torch.float32))

    verts = out.vertices[0].detach().numpy()
    joints = out.joints[0].detach().numpy()[:24]

    # Perspective projection
    j2d = np.zeros((24, 2))
    j2d[:, 0] = focal * joints[:, 0] / joints[:, 2] + cx
    j2d[:, 1] = focal * joints[:, 1] / joints[:, 2] + cy

    v2d = np.zeros((len(verts), 2))
    v2d[:, 0] = focal * verts[:, 0] / verts[:, 2] + cx
    v2d[:, 1] = focal * verts[:, 1] / verts[:, 2] + cy

    img = cv2.imread(os.path.join(img_dir, "%05d.png" % (fi + 1)))
    if img is None:
        continue

    # Mesh points (orange, sparse)
    for v in v2d[::10]:
        x, y = int(v[0]), int(v[1])
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(img, (x, y), 1, (0, 140, 255), -1)

    # Skeleton (blue)
    for a, b in bones:
        cv2.line(img, (int(j2d[a,0]), int(j2d[a,1])),
                 (int(j2d[b,0]), int(j2d[b,1])), (255, 100, 0), 2)
    for j in range(24):
        cv2.circle(img, (int(j2d[j,0]), int(j2d[j,1])), 4, (255, 100, 0), -1)

    cv2.putText(img, f"ReFit SMPL cam3 frame {fi}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.imwrite(os.path.join(out_dir, "frame_%03d.jpg" % fi), img)
    print(f"Frame {fi}: joints x=[{j2d[:,0].min():.0f},{j2d[:,0].max():.0f}] y=[{j2d[:,1].min():.0f},{j2d[:,1].max():.0f}]")

print("Done.")
