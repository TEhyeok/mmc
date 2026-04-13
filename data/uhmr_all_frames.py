"""
Run U-HMR on all 1000 frames × 4 views.
For each frame: SAM bbox crop → U-HMR → SMPL params.
"""
import os, sys, json, numpy as np, torch, cv2, time
from pytorch3d.transforms import matrix_to_axis_angle
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, '/home/elicer/U-HMR')
os.chdir('/home/elicer/U-HMR')

from lib.utils.config import get_config
from lib.models.fusion import Mv_Fusion
from torchvision import transforms
from PIL import Image

# Config
CAMS = [3, 5, 6, 7]  # 4 cameras that work with U-HMR
OUT_DIR = '/home/elicer/uhmr_all_frames'
os.makedirs(OUT_DIR, exist_ok=True)

# Load model once
print('Loading U-HMR model...')
cfg = get_config('experiments/h36m/pitching.yaml')
model = Mv_Fusion(cfg, tensorboard_log_dir=None)
model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
checkpoint = torch.load(cfg.TEST.MODEL_FILE, map_location='cuda', weights_only=False)
model.module.load_state_dict(checkpoint['state_dict'], strict=False)
model.eval()
print('Model loaded.')

transform = transforms.Compose([
    transforms.Resize((cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Count frames
masked_dir = '/home/elicer/masked_images'
orig_dirs = {c: '/home/elicer/easymocap_data/images/%d' % c for c in CAMS}
n_frames = len([f for f in os.listdir(os.path.join(masked_dir, 'cam3')) if f.endswith('.png')])
print(f'{n_frames} frames, {len(CAMS)} cameras')

t0 = time.time()

for fi in range(n_frames):
    inputs = []
    valid = True

    for cam in CAMS:
        # SAM bbox from masked image
        mask_path = os.path.join(masked_dir, 'cam%d' % cam, 'frame_%06d.png' % fi)
        masked = cv2.imread(mask_path)
        if masked is None:
            valid = False
            break

        pitcher_mask = (masked.sum(axis=2) > 30).astype(np.uint8)
        ys, xs = np.where(pitcher_mask > 0)
        if len(xs) < 10:
            valid = False
            break

        # Crop bbox
        cx = (xs.min() + xs.max()) / 2
        cy = (ys.min() + ys.max()) / 2
        size = max(xs.max() - xs.min(), ys.max() - ys.min()) * 1.3
        x1 = int(max(0, cx - size/2))
        y1 = int(max(0, cy - size/2))
        x2 = int(min(1920, cx + size/2))
        y2 = int(min(1080, cy + size/2))

        # Crop original image
        orig_path = os.path.join(orig_dirs[cam], '%06d.jpg' % fi)
        orig = cv2.imread(orig_path)
        if orig is None:
            valid = False
            break

        crop = orig[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        img_tensor = transform(pil_img).unsqueeze(0).cuda()
        inputs.append(img_tensor)

    if not valid or len(inputs) != 4:
        if fi % 100 == 0:
            print(f'Frame {fi}: skipped (invalid)')
        continue

    # Run U-HMR
    with torch.no_grad():
        output = model.module.forward_step(inputs, 4)

    # Extract SMPL params
    sp = output['pred_smpl_params']
    body_pose = sp['body_pose'][0].cpu()      # (23, 3, 3) - same across views
    global_orient = sp['global_orient'][0].cpu()  # (1, 3, 3)
    betas = sp['betas'][0].cpu()              # (10,)
    pred_cam_t = output['pred_cam_t'][0].cpu()  # (3,)

    # Convert rotation matrices to axis-angle
    bp_aa = matrix_to_axis_angle(body_pose).numpy()       # (23, 3)
    go_aa = matrix_to_axis_angle(global_orient).numpy()   # (1, 3)

    # Save
    np.savez(os.path.join(OUT_DIR, '%06d.npz' % fi),
             body_pose=bp_aa.flatten(),      # (69,)
             global_orient=go_aa.flatten(),  # (3,)
             betas=betas.numpy(),            # (10,)
             pred_cam_t=pred_cam_t.numpy())  # (3,)

    if fi % 50 == 0:
        elapsed = time.time() - t0
        fps = (fi + 1) / elapsed
        eta = (n_frames - fi - 1) / fps if fps > 0 else 0
        print(f'Frame {fi}/{n_frames}: {elapsed:.0f}s, {fps:.1f} fps, ETA {eta:.0f}s')

total = time.time() - t0
saved = len(os.listdir(OUT_DIR))
print(f'\nDone: {saved}/{n_frames} frames in {total:.0f}s ({saved/total:.1f} fps)')
