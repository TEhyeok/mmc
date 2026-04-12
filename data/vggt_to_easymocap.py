#!/usr/bin/env python3
"""
Convert VGGT calibration JSON to EasyMocap intri.yml / extri.yml.

Usage:
  python vggt_to_easymocap.py \
    --input vggt_calibration_result.json \
    --output_dir ~/easymocap_data
"""

import json
import argparse
import numpy as np
import cv2
from pathlib import Path


def write_opencv_matrix(f, name, mat, rows, cols):
    """Write a matrix in OpenCV YAML format."""
    f.write(f"{name}: !!opencv-matrix\n")
    f.write(f"  rows: {rows}\n")
    f.write(f"  cols: {cols}\n")
    f.write(f"  dt: d\n")
    flat = mat.flatten().tolist()
    data_str = ", ".join(f"{v:.10f}" for v in flat)
    f.write(f"  data: [{data_str}]\n")


def rotation_matrix_to_rodrigues(R):
    """Convert 3x3 rotation matrix to Rodrigues vector."""
    rvec, _ = cv2.Rodrigues(np.array(R, dtype=np.float64))
    return rvec.flatten()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="VGGT calibration JSON")
    parser.add_argument("--output_dir", required=True,
                        help="EasyMocap data directory")
    args = parser.parse_args()

    with open(args.input) as f:
        vggt = json.load(f)

    out = Path(args.output_dir)
    cam_names = sorted(vggt.keys(),
                       key=lambda x: int(x.replace("cam", "")))
    names = [c.replace("cam", "") for c in cam_names]

    # ---- Write intri.yml ----
    intri_path = out / "intri.yml"
    with open(intri_path, "w") as f:
        f.write("%YAML:1.0\n---\n")
        f.write("names:\n")
        for n in names:
            f.write(f'  - "{n}"\n')

        for cam_key, name in zip(cam_names, names):
            cam = vggt[cam_key]
            K = np.array(cam["K"], dtype=np.float64)
            write_opencv_matrix(f, f"K_{name}", K, 3, 3)
            dist = np.zeros(5)
            write_opencv_matrix(f, f"dist_{name}", dist, 1, 5)

    print(f"Written: {intri_path}")

    # ---- Write extri.yml ----
    extri_path = out / "extri.yml"
    with open(extri_path, "w") as f:
        f.write("%YAML:1.0\n---\n")
        f.write("names:\n")
        for n in names:
            f.write(f'  - "{n}"\n')

        for cam_key, name in zip(cam_names, names):
            cam = vggt[cam_key]
            R_mat = np.array(cam["R"], dtype=np.float64)
            t_vec = np.array(cam["t"], dtype=np.float64)

            # Rodrigues vector from rotation matrix
            rvec = rotation_matrix_to_rodrigues(R_mat)

            write_opencv_matrix(f, f"R_{name}",
                                rvec.reshape(3, 1), 3, 1)
            write_opencv_matrix(f, f"Rot_{name}",
                                R_mat, 3, 3)
            write_opencv_matrix(f, f"T_{name}",
                                t_vec.reshape(3, 1), 3, 1)

    print(f"Written: {extri_path}")

    # ---- Verification ----
    print("\n=== Verification ===")
    for cam_key, name in zip(cam_names, names):
        cam = vggt[cam_key]
        K = np.array(cam["K"])
        R = np.array(cam["R"])
        t = np.array(cam["t"])

        # Camera center in world coordinates
        center = -R.T @ t
        rvec = rotation_matrix_to_rodrigues(R)

        print(f"cam{name}: fx={K[0,0]:.1f} fy={K[1,1]:.1f} "
              f"| center=[{center[0]:.3f},{center[1]:.3f},"
              f"{center[2]:.3f}] "
              f"| rvec=[{rvec[0]:.4f},{rvec[1]:.4f},"
              f"{rvec[2]:.4f}]")

    # Rodrigues roundtrip check
    R_orig = np.array(vggt["cam2"]["R"], dtype=np.float64)
    rvec = rotation_matrix_to_rodrigues(R_orig)
    R_back, _ = cv2.Rodrigues(rvec)
    err = np.max(np.abs(R_orig - R_back))
    print(f"\nRodrigues roundtrip error: {err:.2e}")


if __name__ == "__main__":
    main()
