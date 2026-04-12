"""N-view DLT triangulation with confidence weighting."""
import numpy as np
from typing import List, Tuple, Optional


def triangulate_dlt(
    projections: List[np.ndarray],
    points_2d: List[np.ndarray],
    confidences: Optional[List[float]] = None,
) -> Optional[np.ndarray]:
    """DLT triangulation from N views.

    Args:
        projections: List of (3,4) projection matrices P = K @ [R|t]
        points_2d: List of (2,) pixel coordinates
        confidences: Optional confidence weights per view

    Returns:
        (3,) 3D point or None if insufficient views
    """
    if len(projections) < 2:
        return None

    n = len(projections)
    if confidences is None:
        confidences = [1.0] * n

    A = np.zeros((2 * n, 4))
    for i, (P, uv, w) in enumerate(zip(projections, points_2d, confidences)):
        A[2 * i] = w * (uv[0] * P[2] - P[0])
        A[2 * i + 1] = w * (uv[1] * P[2] - P[1])

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]

    if abs(X[3]) < 1e-10:
        return None

    return X[:3] / X[3]


def triangulate_joint(
    cameras: List[dict],
    keypoints_2d: List[np.ndarray],
    joint_idx: int,
    min_views: int = 3,
    conf_threshold: float = 0.3,
    max_reproj_error: float = 80.0,
) -> Tuple[Optional[np.ndarray], float]:
    """Triangulate a single joint from multiple camera views.

    Args:
        cameras: List of camera dicts with 'K', 'R', 't', 'P' keys
        keypoints_2d: List of (N_joints, 3) arrays [x, y, confidence]
        joint_idx: Index of the joint to triangulate
        min_views: Minimum number of views required
        conf_threshold: Minimum confidence to include a view
        max_reproj_error: Maximum allowed reprojection error (pixels)

    Returns:
        (3D point or None, mean reprojection error)
    """
    valid_P = []
    valid_uv = []
    valid_conf = []
    valid_cams = []

    for ci, (cam, kps) in enumerate(zip(cameras, keypoints_2d)):
        if kps is None or kps[joint_idx, 2] < conf_threshold:
            continue
        valid_P.append(cam['P'])
        valid_uv.append(kps[joint_idx, :2])
        valid_conf.append(float(kps[joint_idx, 2]))
        valid_cams.append(cam)

    if len(valid_P) < min_views:
        return None, float('inf')

    X = triangulate_dlt(valid_P, valid_uv, valid_conf)
    if X is None:
        return None, float('inf')

    # Compute reprojection error
    errors = []
    for cam, uv in zip(valid_cams, valid_uv):
        x_cam = cam['K'] @ (cam['R'] @ X.reshape(3, 1) + cam['t'].reshape(3, 1))
        u_proj = np.array([x_cam[0, 0] / x_cam[2, 0], x_cam[1, 0] / x_cam[2, 0]])
        errors.append(float(np.linalg.norm(u_proj - uv)))

    mean_error = np.mean(errors)

    # Filter by reprojection error
    if mean_error > max_reproj_error:
        return None, mean_error

    return X, mean_error


def triangulate_all_joints(
    cameras: List[dict],
    keypoints_2d: List[np.ndarray],
    n_joints: int = 25,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Triangulate all joints for a single frame.

    Returns:
        joints_3d: (N_joints, 3) array
        reproj_errors: (N_joints,) array
    """
    joints_3d = np.zeros((n_joints, 3))
    reproj_errors = np.full(n_joints, float('inf'))

    for j in range(n_joints):
        X, err = triangulate_joint(cameras, keypoints_2d, j, **kwargs)
        if X is not None:
            joints_3d[j] = X
            reproj_errors[j] = err

    return joints_3d, reproj_errors
