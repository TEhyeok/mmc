#!/usr/bin/env python3
"""
VGGT camera calibration for 7 mixed-orientation cameras.
cam1: portrait (1080x1920) → rotated 90° CW to landscape
cam2-7: landscape (1920x1080) → as-is

Outputs K (intrinsics) and R, t (extrinsics) at ORIGINAL resolution.
"""

import sys
sys.path.insert(0, "/home/elicer/vggt")

import torch
import numpy as np
import json
from pathlib import Path
from PIL import Image


def main():
    # ---- Config ----
    image_dir = Path("/home/elicer/baseball_h36m/images")
    frame_id = "00001"  # static frame (pitcher standing still)
    output_path = Path("/home/elicer/vggt_calibration_result.json")

    cam_names = [f"cam{i}" for i in range(1, 8)]
    portrait_cams = {"cam1"}  # cameras that are portrait orientation

    # ---- Prepare images (rotate portrait, save temp files) ----
    import tempfile, shutil
    tmp_dir = Path(tempfile.mkdtemp(prefix="vggt_"))
    image_path_list = []
    orig_sizes = {}  # cam_name -> (W, H) original

    for cam in cam_names:
        img_path = image_dir / cam / f"{frame_id}.png"
        img = Image.open(img_path).convert("RGB")
        orig_sizes[cam] = img.size  # (W, H)

        if cam in portrait_cams:
            # Rotate portrait 90° CW → landscape
            img_rotated = img.transpose(Image.Transpose.ROTATE_270)
            save_path = tmp_dir / f"{cam}_rotated.png"
            img_rotated.save(save_path)
            image_path_list.append(str(save_path))
            print(f"{cam}: portrait {orig_sizes[cam]} → rotated to {img_rotated.size}")
        else:
            image_path_list.append(str(img_path))
            print(f"{cam}: landscape {img.size}")

    # ---- Preprocess for VGGT ----
    from vggt.utils.load_fn import load_and_preprocess_images
    preprocess_mode = "crop"  # crop avoids padding-induced fov distortion
    images_tensor = load_and_preprocess_images(image_path_list, mode=preprocess_mode)
    print(f"\nPreprocess mode: {preprocess_mode}")
    print(f"Preprocessed tensor shape: {images_tensor.shape}")

    # ---- Load model ----
    device = "cuda"
    dtype = torch.bfloat16  # A100 supports bfloat16

    from vggt.models.vggt import VGGT
    print("\nLoading VGGT-1B model...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    print("Model loaded.")

    # ---- Inference ----
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    images_tensor = images_tensor.to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images_tensor)

    # Decode pose encoding to extrinsics and intrinsics
    pose_enc = predictions["pose_enc"]
    image_hw = images_tensor.shape[-2:]  # (H, W) of preprocessed images
    extrinsic_t, intrinsic_t = pose_encoding_to_extri_intri(pose_enc, image_size_hw=image_hw)

    extrinsic = extrinsic_t.cpu().numpy()[0]  # (7, 3, 4)
    intrinsic = intrinsic_t.cpu().numpy()[0]  # (7, 3, 3)

    print(f"\nExtrinsics shape: {extrinsic.shape}")
    print(f"Intrinsics shape: {intrinsic.shape}")

    # ---- Get VGGT's internal padded size ----
    vggt_size = 518  # always 518x518 with mode="pad"
    print(f"VGGT internal resolution: {vggt_size}x{vggt_size}")

    # ---- Compute resize/crop/pad info for each camera ----
    def compute_preprocess_info(w, h, target=518, mode="crop"):
        """Compute resize dims, crop/pad offsets."""
        if mode == "pad":
            if w >= h:
                new_w = target
                new_h = round(h * (new_w / w) / 14) * 14
            else:
                new_h = target
                new_w = round(w * (new_h / h) / 14) * 14
            pad_h = target - new_h
            pad_w = target - new_w
            pad_top = pad_h // 2
            pad_left = pad_w // 2
            crop_top = 0
            return new_w, new_h, pad_left, pad_top, crop_top
        else:  # crop
            new_w = target
            new_h = round(h * (new_w / w) / 14) * 14
            crop_top = max(0, (new_h - target) // 2)
            final_h = min(new_h, target)
            return new_w, new_h, 0, 0, crop_top

    # ---- Transform K and R,t back to original resolution ----
    results = {}

    for i, cam in enumerate(cam_names):
        K_vggt = intrinsic[i]  # (3, 3) at 518x518 padded resolution
        R = extrinsic[i, :3, :3]  # (3, 3)
        t = extrinsic[i, :3, 3]   # (3,)

        orig_w, orig_h = orig_sizes[cam]

        if cam in portrait_cams:
            # Image was rotated: (1080, 1920) → (1920, 1080)
            feed_w, feed_h = orig_h, orig_w  # 1920, 1080
        else:
            feed_w, feed_h = orig_w, orig_h  # 1920, 1080

        # Compute preprocess info for the image that was fed to VGGT
        resized_w, resized_h, pad_left, pad_top, crop_top = compute_preprocess_info(
            feed_w, feed_h, mode=preprocess_mode)
        print(f"  {cam}: fed {feed_w}x{feed_h} → resized {resized_w}x{resized_h}, "
              f"pad_left={pad_left}, pad_top={pad_top}, crop_top={crop_top}")

        # Step 1: Undo crop (add back cropped offset to cy) or undo padding
        K_uncropped = K_vggt.copy()
        if preprocess_mode == "crop":
            K_uncropped[1, 2] += crop_top  # cy: add back center crop offset
        else:
            K_uncropped[0, 2] -= pad_left   # cx
            K_uncropped[1, 2] -= pad_top    # cy

        # Step 2: Scale from resized to original (fed) resolution
        scale_x = feed_w / resized_w
        scale_y = feed_h / resized_h
        K_feed = K_uncropped.copy()
        K_feed[0, :] *= scale_x  # fx, cx
        K_feed[1, :] *= scale_y  # fy, cy

        if cam in portrait_cams:
            # Step 3: Rotate K back 90° CCW (landscape → portrait)
            fx_r, fy_r = K_feed[0, 0], K_feed[1, 1]
            cx_r, cy_r = K_feed[0, 2], K_feed[1, 2]

            K_orig = np.array([
                [fy_r, 0, cy_r],
                [0, fx_r, feed_w - 1 - cx_r],
                [0, 0, 1]
            ])

            # Rotate extrinsics back
            R_rot90 = np.array([
                [0, 1, 0],
                [-1, 0, 0],
                [0, 0, 1]
            ], dtype=np.float64)
            R_orig = R @ R_rot90
            t_orig = t.copy()
        else:
            K_orig = K_feed
            R_orig = R
            t_orig = t

        results[cam] = {
            "width": int(orig_w),
            "height": int(orig_h),
            "K": K_orig.tolist(),
            "R": R_orig.tolist(),
            "t": t_orig.tolist(),
            "fx": float(K_orig[0, 0]),
            "fy": float(K_orig[1, 1]),
            "cx": float(K_orig[0, 2]),
            "cy": float(K_orig[1, 2]),
        }

        print(f"\n--- {cam} ({orig_w}x{orig_h}) ---")
        print(f"  fx={K_orig[0,0]:.1f}, fy={K_orig[1,1]:.1f}, "
              f"cx={K_orig[0,2]:.1f}, cy={K_orig[1,2]:.1f}")
        print(f"  t = [{t_orig[0]:.4f}, {t_orig[1]:.4f}, {t_orig[2]:.4f}]")

    # ---- Save results ----
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # ---- Sanity checks ----
    print("\n=== Sanity Checks ===")
    fxs = [results[c]["fx"] for c in cam_names]
    print(f"Focal lengths: min={min(fxs):.1f}, max={max(fxs):.1f}, "
          f"spread={max(fxs)-min(fxs):.1f}px")
    print(f"(Same camera model → focal lengths should be similar)")

    # Check camera positions (translation vectors)
    positions = np.array([results[c]["t"] for c in cam_names])
    dists = np.linalg.norm(positions - positions.mean(axis=0), axis=1)
    print(f"Camera distances from centroid: {dists.round(3)}")


if __name__ == "__main__":
    main()
