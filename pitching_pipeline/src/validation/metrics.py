"""Validation metrics: MPJPE, joint angle RMSE, torque nRMSE."""
import numpy as np
from typing import Dict, Optional


def mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean Per Joint Position Error (mm).

    Args:
        pred: (N, J, 3) predicted joint positions
        gt: (N, J, 3) ground truth positions

    Returns:
        MPJPE in mm
    """
    return float(np.mean(np.linalg.norm(pred - gt, axis=2)) * 1000)


def pa_mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    """Procrustes-Aligned MPJPE (mm).

    Aligns pred to gt via rigid body transform before computing MPJPE.
    """
    from ..utils.transforms import procrustes_align

    aligned = np.zeros_like(pred)
    for t in range(len(pred)):
        R, tvec, s = procrustes_align(pred[t], gt[t])
        aligned[t] = (s * R @ pred[t].T).T + tvec

    return float(np.mean(np.linalg.norm(aligned - gt, axis=2)) * 1000)


def joint_angle_rmse(pred_angles: np.ndarray, gt_angles: np.ndarray) -> float:
    """Joint angle RMSE (degrees).

    Args:
        pred_angles: (N,) predicted angles in degrees
        gt_angles: (N,) ground truth angles in degrees
    """
    return float(np.sqrt(np.mean((pred_angles - gt_angles) ** 2)))


def nrmse_torque(pred_torque: np.ndarray, gt_torque: np.ndarray) -> float:
    """Normalized RMSE for torque (%).

    nRMSE = RMSE / (max - min) × 100
    """
    rmse = np.sqrt(np.mean((pred_torque - gt_torque) ** 2))
    range_gt = gt_torque.max() - gt_torque.min()
    if range_gt < 1e-6:
        return float('inf')
    return float(rmse / range_gt * 100)


def compute_all_metrics(
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    pred_angles: Optional[Dict[str, np.ndarray]] = None,
    gt_angles: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, float]:
    """Compute all validation metrics.

    Returns:
        dict with metric names → values
    """
    results = {
        "mpjpe_mm": mpjpe(pred_joints, gt_joints),
        "pa_mpjpe_mm": pa_mpjpe(pred_joints, gt_joints),
    }

    if pred_angles and gt_angles:
        for key in pred_angles:
            if key in gt_angles:
                results[f"rmse_{key}_deg"] = joint_angle_rmse(
                    pred_angles[key], gt_angles[key]
                )

    return results
