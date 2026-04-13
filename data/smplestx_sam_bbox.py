"""SMPLest-X with SAM-derived bbox (skip YOLO, use original images)."""
import os, sys, numpy as np, torch, cv2
from tqdm import tqdm
sys.path.insert(0, '/home/elicer/SMPLest-X')
os.chdir('/home/elicer/SMPLest-X')

from human_models.human_models import SMPLX
from main.base import Tester
from main.config import Config
from utils.data_utils import load_img, process_bbox, generate_patch_image
import torchvision.transforms as transforms
import datetime

cam_id = 3
orig_dir = '/home/elicer/easymocap_data/images/%d' % cam_id
masked_dir = '/home/elicer/masked_images/cam%d' % cam_id
out_dir = '/home/elicer/smplestx_sambbox_cam%d' % cam_id
os.makedirs(out_dir, exist_ok=True)

config_path = './pretrained_models/smplest_x_h/config_base.py'
cfg = Config.load_config(config_path)
checkpoint_path = './pretrained_models/smplest_x_h/smplest_x_h.pth.tar'
time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
new_config = {
    'model': {'pretrained_model_path': checkpoint_path},
    'log': {'exp_name': 'sambbox_' + time_str,
            'log_dir': './outputs/sambbox_' + time_str + '/log'}
}
cfg.update_config(new_config)
cfg.prepare_log()
smpl_x = SMPLX(cfg.model.human_model_path)
demoer = Tester(cfg)
demoer._make_model()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

n_frames = len([f for f in os.listdir(masked_dir) if f.endswith('.png')])
print('Processing %d frames, cam%d' % (n_frames, cam_id))

for fi in tqdm(range(n_frames)):
    masked = cv2.imread(os.path.join(masked_dir, 'frame_%06d.png' % fi))
    pitcher_mask = (masked.sum(axis=2) > 30).astype(np.uint8)
    ys, xs = np.where(pitcher_mask > 0)
    if len(xs) == 0:
        continue

    sam_bbox_xywh = np.array([
        float(xs.min()), float(ys.min()),
        float(xs.max() - xs.min()), float(ys.max() - ys.min())
    ])

    original_img = load_img(os.path.join(orig_dir, '%06d.jpg' % fi))
    h, w = original_img.shape[:2]

    bbox = process_bbox(
        bbox=sam_bbox_xywh, img_width=w, img_height=h,
        input_img_shape=cfg.model.input_img_shape,
        ratio=getattr(cfg.data, 'bbox_ratio', 1.25)
    )

    img, _, _ = generate_patch_image(
        cvimg=original_img, bbox=bbox,
        scale=1.0, rot=0.0, do_flip=False,
        out_shape=cfg.model.input_img_shape
    )
    img = transform(img.astype(np.float32)) / 255
    img = img.cuda()[None, :, :, :]

    with torch.no_grad():
        out = demoer.model({'img': img}, {}, {}, 'test')

    mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]

    focal = [cfg.model.focal[0] / cfg.model.input_body_shape[1] * bbox[2],
             cfg.model.focal[1] / cfg.model.input_body_shape[0] * bbox[3]]
    princpt = [cfg.model.princpt[0] / cfg.model.input_body_shape[1] * bbox[2] + bbox[0],
               cfg.model.princpt[1] / cfg.model.input_body_shape[0] * bbox[3] + bbox[1]]

    np.savez(os.path.join(out_dir, '%06d.npz' % fi),
             mesh=mesh, focal=np.array(focal), princpt=np.array(princpt),
             bbox=bbox, sam_bbox=sam_bbox_xywh)

print('Done:', out_dir)
