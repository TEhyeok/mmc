"""Multi-view 2D reprojection SMPL fitting.

Implements eq. from part3_pitching.tex:
  (θ*, β*) = argmin Σ_j Σ_k w_jk ||u_jk - π_j(J_k(θ,β))||² + λ_θ||θ||² + λ_β||β||²
"""
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path


# OpenPose Body25 → SMPL 24 joint mapping
OPENPOSE_TO_SMPL = {
    1: 12,   # Neck → Neck
    8: 0,    # MidHip → Pelvis
    2: 17,   # RSho → R_Shoulder
    3: 19,   # RElb → R_Elbow
    4: 21,   # RWri → R_Wrist
    5: 16,   # LSho → L_Shoulder
    6: 18,   # LElb → L_Elbow
    7: 20,   # LWri → L_Wrist
    9: 2,    # RHip → R_Hip
    10: 5,   # RKne → R_Knee
    11: 8,   # RAnk → R_Ankle
    12: 1,   # LHip → L_Hip
    13: 4,   # LKne → L_Knee
    14: 7,   # LAnk → L_Ankle
}


def run_easymocap(
    data_dir: Path,
    output_dir: Path,
    body_type: str = "body25",
    model_type: str = "smpl",
    gender: str = "neutral",
    n_cameras: int = 7,
    start: int = 0,
    end: int = 999,
) -> bool:
    """Run EasyMocap multi-view SMPL fitting via subprocess.

    Expects EasyMocap directory structure:
        data_dir/
            images/{0,1,...,6}/  (camera folders)
            annots/{0,1,...,6}/  (OpenPose keypoints per camera)
            intri.yml, extri.yml

    Returns:
        True if successful
    """
    import subprocess

    cmd = [
        "python3", "-m", "easymocap.apps.demo.mv1p",
        str(data_dir),
        "--out", str(output_dir),
        "--body", body_type,
        "--model", model_type,
        "--gender", gender,
        "--sub", *[str(i) for i in range(n_cameras)],
        "--start", str(start),
        "--end", str(end),
        "--thres2d", "0.3",
        "--write_smpl_full",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        return result.returncode == 0
    except Exception as e:
        print(f"EasyMocap failed: {e}")
        return False


def fit_smpl_direct_3d(
    keypoints_3d: np.ndarray,
    smpl_model,
    n_iterations: int = 1400,
    lambda_theta: float = 0.001,
    lambda_beta: float = 0.01,
    lambda_smooth: float = 0.1,
    device: str = "cuda",
):
    """Direct 3D keypoint → SMPL fitting using PyTorch optimization.

    WARNING: This approach has structural limitations due to joint definition
    mismatch between OpenPose and SMPL (see fitting_guide.pdf).
    Prefer multi-view 2D reprojection fitting (EasyMocap) when camera
    parameters are available.

    Args:
        keypoints_3d: (N_frames, 25, 3) OpenPose 3D keypoints
        smpl_model: smplx body model
        n_iterations: total optimization iterations
        lambda_theta: pose regularization weight
        lambda_beta: shape regularization weight
        lambda_smooth: temporal smoothness weight
        device: 'cuda' or 'cpu'

    Returns:
        dict with 'poses', 'betas', 'transl', 'joints' keys
    """
    import torch

    N = len(keypoints_3d)
    op_indices = list(OPENPOSE_TO_SMPL.keys())
    smpl_indices = list(OPENPOSE_TO_SMPL.values())

    target = torch.tensor(
        keypoints_3d[:, op_indices, :], dtype=torch.float32, device=device
    )

    # Initialize parameters
    global_orient = torch.zeros(N, 3, device=device, requires_grad=True)
    body_pose = torch.zeros(N, 69, device=device, requires_grad=True)
    betas = torch.zeros(1, 10, device=device, requires_grad=True)
    transl = torch.tensor(
        keypoints_3d[:, 8, :], dtype=torch.float32, device=device, requires_grad=True
    )

    stages = [
        ("Global RT", [global_orient, transl], 1e-2, 200),
        ("Shape", [betas], 1e-2, 200),
        ("Pose", [body_pose], 1e-3, 500),
        ("Joint", [global_orient, body_pose, betas, transl], 1e-4, 500),
    ]

    for stage_name, params, lr, iters in stages:
        optimizer = torch.optim.Adam(params, lr=lr)
        for i in range(iters):
            optimizer.zero_grad()

            output = smpl_model(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas.expand(N, -1),
                transl=transl,
            )
            joints = output.joints[:, :24, :]
            pred = joints[:, smpl_indices, :]

            loss_j = ((pred - target) ** 2).sum()
            loss_theta = lambda_theta * (body_pose ** 2).sum()
            loss_beta = lambda_beta * (betas ** 2).sum()

            loss = loss_j + loss_theta + loss_beta

            if lambda_smooth > 0 and stage_name == "Joint":
                vel = body_pose[1:] - body_pose[:-1]
                acc = vel[1:] - vel[:-1]
                loss += lambda_smooth * (vel ** 2).sum()
                loss += lambda_smooth * 0.5 * (acc ** 2).sum()

            loss.backward()
            optimizer.step()

    # Extract final results
    with torch.no_grad():
        output = smpl_model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas.expand(N, -1),
            transl=transl,
        )
        joints_final = output.joints[:, :24, :].cpu().numpy()

    return {
        "poses": torch.cat([global_orient, body_pose], dim=1).detach().cpu().numpy(),
        "betas": betas.detach().cpu().numpy(),
        "transl": transl.detach().cpu().numpy(),
        "joints": joints_final,
    }
