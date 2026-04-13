"""Refine SMPLest-X results with 2D reprojection + temporal smoothing (cam3)."""
import os, sys, numpy as np, torch, cv2, json
from tqdm import tqdm
sys.path.insert(0, '/home/elicer/SMPLest-X')

import smplx

device = 'cuda'

# Load SMPL (not SMPL-X, for compatibility with GART later)
body_model = smplx.create(model_path='/home/elicer/EasyMocap/data/smplx',
                          model_type='smpl', gender='neutral', batch_size=1).to(device)
J_reg = torch.tensor(np.load('/home/elicer/EasyMocap/models/J_regressor_body25.npy'),
                     dtype=torch.float32, device=device)

# Load SMPLest-X results as initialization
smplestx_dir = '/home/elicer/smplestx_sambbox_cam3'
n_frames = len([f for f in os.listdir(smplestx_dir) if f.endswith('.npz')])
print(f'{n_frames} frames')

# Load all SMPLest-X meshes + camera params
init_meshes = []
init_focals = []
init_princpts = []
for fi in range(n_frames):
    d = np.load(os.path.join(smplestx_dir, '%06d.npz' % fi))
    init_meshes.append(d['mesh'])  # (10475, 3) SMPL-X mesh
    init_focals.append(d['focal'])
    init_princpts.append(d['princpt'])

# Load OpenPose 2D keypoints for cam3
annot_dir = '/home/elicer/easymocap_data/annots/3'
all_kp2d = []
for fi in range(n_frames):
    with open(os.path.join(annot_dir, '%06d.json' % fi)) as f:
        ann = json.load(f)
    kps = np.array(ann['annots'][0]['keypoints'])  # (25, 3)
    all_kp2d.append(kps)

# We'll use SMPLest-X's camera (focal, princpt) for projection
# Since SMPLest-X outputs camera-relative mesh, we project directly

def project(pts, focal, princpt):
    """Project 3D points using perspective. pts: (N,3)"""
    x = focal[0] * pts[:, 0] / pts[:, 2] + princpt[0]
    y = focal[1] * pts[:, 1] / pts[:, 2] + princpt[1]
    return torch.stack([x, y], dim=1)

def gmof(x, sigma=100.0):
    return x**2 / (x**2 + sigma**2)

# For SMPL fitting, we need to convert SMPLest-X mesh (10475 verts, SMPL-X)
# to SMPL parameters. Instead, we'll fit SMPL directly using 2D keypoints,
# but initialize from SMPLest-X's projected joints.

# Step 1: Get initial SMPL params by fitting to SMPLest-X's 2D projections
# For each frame, project SMPLest-X mesh joints to 2D, then fit SMPL to match

out_dir = '/home/elicer/smplestx_refined_cam3'
os.makedirs(out_dir + '/smpl', exist_ok=True)

# Initialize SMPL from first frame
# Use SMPLest-X mesh centroid as translation init
mesh0 = init_meshes[0]
Th_init = mesh0.mean(axis=0)
print(f'Init Th: {Th_init.round(3)}')

# Initial body pose from previous successful SMPLest-X crop
prev_Rh = np.array([2.7, 0.5, -0.2])  # from ReFit cam3 crop result
prev_Th = Th_init
prev_body = np.zeros(69)
betas = torch.zeros(1, 10, dtype=torch.float32, device=device, requires_grad=True)

all_Rh = []
all_Th = []
all_body = []
all_reproj = []

for fi in tqdm(range(n_frames)):
    focal_np = init_focals[fi]
    princpt_np = init_princpts[fi]
    focal = torch.tensor(focal_np, dtype=torch.float32, device=device)
    princpt = torch.tensor(princpt_np, dtype=torch.float32, device=device)

    # 2D targets: OpenPose keypoints
    kps = all_kp2d[fi]
    kp2d = torch.tensor(kps[:, :2], dtype=torch.float32, device=device)
    conf = torch.tensor(kps[:, 2], dtype=torch.float32, device=device)
    conf[conf < 0.3] = 0

    # Initialize from previous frame
    Rh = torch.tensor(prev_Rh, dtype=torch.float32, device=device, requires_grad=True)
    Th = torch.tensor(prev_Th, dtype=torch.float32, device=device, requires_grad=True)
    bp = torch.tensor(prev_body, dtype=torch.float32, device=device).unsqueeze(0).requires_grad_(True)

    # Stage 1: RT (50 steps - fast, good init already)
    opt1 = torch.optim.Adam([Rh, Th], lr=0.01)
    for s in range(50):
        out = body_model(global_orient=Rh.unsqueeze(0), body_pose=bp,
                         betas=betas, transl=Th.unsqueeze(0))
        j25 = J_reg @ out.vertices[0]
        proj = project(j25[:15], focal, princpt)
        loss = (conf[:15].unsqueeze(1) * gmof(proj - kp2d[:15])).sum()
        opt1.zero_grad(); loss.backward(); opt1.step()

    # Stage 2: + body pose (150 steps)
    opt2 = torch.optim.Adam([Rh, Th, bp], lr=0.005)
    for s in range(150):
        out = body_model(global_orient=Rh.unsqueeze(0), body_pose=bp,
                         betas=betas, transl=Th.unsqueeze(0))
        j25 = J_reg @ out.vertices[0]
        proj = project(j25[:25], focal, princpt)
        loss_reproj = (conf.unsqueeze(1) * gmof(proj - kp2d)).sum()

        # Temporal smoothing: penalize deviation from previous frame
        if fi > 0:
            loss_temp = 0.5 * ((bp - torch.tensor(prev_body, device=device).unsqueeze(0))**2).sum()
            loss_temp += 2.0 * ((Th - torch.tensor(prev_Th, device=device))**2).sum()
        else:
            loss_temp = torch.tensor(0.0, device=device)

        loss_reg = 0.005 * (bp**2).sum()
        loss = loss_reproj + loss_reg + loss_temp
        opt2.zero_grad(); loss.backward(); opt2.step()

    # Evaluate
    with torch.no_grad():
        out = body_model(global_orient=Rh.unsqueeze(0), body_pose=bp,
                         betas=betas, transl=Th.unsqueeze(0))
        j25 = J_reg @ out.vertices[0]
        proj = project(j25[:15], focal, princpt)
        diff = (proj - kp2d[:15]).cpu().numpy()
        c = conf[:15].cpu().numpy()
        errs = [np.linalg.norm(diff[j]) for j in range(15) if c[j] > 0]
        mean_err = np.mean(errs) if errs else 999

    prev_Rh = Rh.detach().cpu().numpy()
    prev_Th = Th.detach().cpu().numpy()
    prev_body = bp.detach().cpu().numpy().flatten()

    all_Rh.append(prev_Rh)
    all_Th.append(prev_Th)
    all_body.append(prev_body)
    all_reproj.append(mean_err)

    # Save
    result = [{'id': 0,
               'Rh': [prev_Rh.tolist()],
               'Th': [prev_Th.tolist()],
               'poses': [np.concatenate([[0,0,0], prev_body]).tolist()],
               'shapes': [betas.detach().cpu().numpy().flatten().tolist()]}]
    with open(os.path.join(out_dir, 'smpl', '%06d.json' % fi), 'w') as f:
        json.dump(result, f)

    if fi % 100 == 0:
        print(f'Frame {fi}: reproj={mean_err:.1f}px')

print(f'\nDone. Mean reproj: {np.mean(all_reproj):.1f}px, Median: {np.median(all_reproj):.1f}px')
