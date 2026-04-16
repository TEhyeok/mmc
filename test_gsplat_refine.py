"""Test single-frame gsplat pose refinement on server"""
import numpy as np, torch, smplx, json, os
from gsplat import rasterization
from PIL import Image
import torchvision.transforms as T

print('=== gsplat Pose Refinement Test (frame 500) ===')
FRAME, N_ITERS, LR, S = 500, 30, 0.005, 0.25

model = smplx.create('/home/elicer/EasyMocap/data/smplx/', model_type='smpl',
                      gender='neutral', batch_size=1).cuda()

d = np.load(f'/home/elicer/sam3d_results/cam3/{FRAME:06d}.npz')
bp_init = d['body_pose_params'][:69]
from scipy.spatial.transform import Rotation as Rsci
root_rot = Rsci.from_matrix(d['pred_global_rots'][0]).as_rotvec()

with open('/home/elicer/vggt_calibration_result.json') as f:
    calib = json.load(f)

cam_nums = ['3', '5', '6', '7']
cameras = []
for cn in cam_nums:
    ck = 'cam' + cn
    if ck in calib:
        c = calib[ck]
        cameras.append({
            'K': torch.tensor(c['K'], dtype=torch.float32).cuda(),
            'R': torch.tensor(c['R'], dtype=torch.float32).cuda(),
            't': torch.tensor(c['t'], dtype=torch.float32).cuda().squeeze(),
            'img_dir': '/home/elicer/easymocap_data/images/' + cn + '/',
            'name': ck
        })

gt_images = []
transform = T.Compose([T.Resize((int(1080*S), int(1920*S))), T.ToTensor()])
for cam in cameras:
    img_path = os.path.join(cam['img_dir'], f'{FRAME:06d}.jpg')
    if os.path.exists(img_path):
        img = Image.open(img_path).convert('RGB')
        gt_images.append(transform(img).permute(1, 2, 0).cuda())
        print(f'  {cam["name"]}: {gt_images[-1].shape}')
    else:
        gt_images.append(None)
        print(f'  {cam["name"]}: NOT FOUND at {img_path}')

body_pose = torch.tensor(bp_init, dtype=torch.float32, device='cuda', requires_grad=True)
go = torch.tensor(root_rot, dtype=torch.float32, device='cuda', requires_grad=True)
betas = torch.tensor(d['shape_params'][:10], dtype=torch.float32).unsqueeze(0).cuda()
init_pose = body_pose.clone().detach()

N_V = 6890
colors_raw = torch.nn.Parameter(torch.zeros(N_V, 3, device='cuda'))
opacities_raw = torch.nn.Parameter(2.0 * torch.ones(N_V, device='cuda'))
scales = 0.008 * torch.ones(N_V, 3, device='cuda')
quats = torch.zeros(N_V, 4, device='cuda')
quats[:, 0] = 1

optimizer = torch.optim.Adam([
    {'params': [body_pose], 'lr': LR},
    {'params': [go], 'lr': LR * 0.5},
    {'params': [colors_raw, opacities_raw], 'lr': LR * 2},
])

n_valid_cams = len([g for g in gt_images if g is not None])
print(f'\nRefining: {N_ITERS} iters, {n_valid_cams} cameras')

for it in range(N_ITERS):
    optimizer.zero_grad()
    out = model(body_pose=body_pose.unsqueeze(0), global_orient=go.unsqueeze(0),
                betas=betas, transl=torch.zeros(1, 3).cuda())
    verts = out.vertices[0]
    colors = torch.sigmoid(colors_raw)
    opacities = torch.sigmoid(opacities_raw)

    total_loss = torch.tensor(0.0, device='cuda')
    nv = 0
    for ci, cam in enumerate(cameras):
        gt = gt_images[ci]
        if gt is None:
            continue
        H, W = gt.shape[:2]
        K_s = cam['K'].clone()
        K_s[0] *= S
        K_s[1] *= S
        vm = torch.eye(4, device='cuda')
        vm[:3, :3] = cam['R']
        vm[:3, 3] = cam['t']
        rendered = rasterization(verts.contiguous(), quats, scales, opacities, colors,
                                 vm.unsqueeze(0), K_s.unsqueeze(0), W, H)[0][0]
        total_loss = total_loss + (rendered - gt).abs().mean()
        nv += 1

    if nv > 0:
        reg = 0.0001 * ((body_pose - init_pose) ** 2).mean()
        total_loss = total_loss / nv + reg
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([body_pose, go], 1.0)
        optimizer.step()

    if it % 5 == 0 or it == N_ITERS - 1:
        pc = (body_pose - init_pose).abs().mean().item()
        print(f'  iter {it:3d}: loss={total_loss.item():.6f}, pose_delta={np.degrees(pc):.4f} deg')

fc = (body_pose.detach() - init_pose).abs()
print(f'\n=== Done! Mean: {np.degrees(fc.mean().item()):.4f} deg, Max: {np.degrees(fc.max().item()):.4f} deg ===')

os.makedirs('/home/elicer/gsplat_refined_test/', exist_ok=True)
np.savez('/home/elicer/gsplat_refined_test/frame_500.npz',
         body_pose_init=init_pose.cpu().numpy(),
         body_pose_refined=body_pose.detach().cpu().numpy(),
         global_orient=go.detach().cpu().numpy())
print('SAVED: /home/elicer/gsplat_refined_test/frame_500.npz')
print('TEST PASSED!')
