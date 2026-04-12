"""Rotation and coordinate transform utilities."""
import numpy as np
from scipy.spatial.transform import Rotation


def axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """Axis-angle (3,) → Rotation matrix (3,3)."""
    return Rotation.from_rotvec(aa).as_matrix()


def matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (3,3) → Axis-angle (3,)."""
    return Rotation.from_matrix(R).as_rotvec()


def matrix_to_euler(R: np.ndarray, seq: str = "YXY") -> np.ndarray:
    """Rotation matrix → Euler angles (degrees)."""
    return Rotation.from_matrix(R).as_euler(seq, degrees=True)


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """Quaternion (w,x,y,z) → Rotation matrix (3,3)."""
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def project_point(X: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Project 3D point to 2D pixel.

    Args:
        X: (3,) 3D point in world coordinates
        K: (3,3) camera intrinsic matrix
        R: (3,3) rotation matrix (world → camera)
        t: (3,) translation vector

    Returns:
        (2,) pixel coordinates (u, v)
    """
    x_cam = R @ X.reshape(3, 1) + t.reshape(3, 1)
    x_proj = K @ x_cam
    return np.array([x_proj[0, 0] / x_proj[2, 0], x_proj[1, 0] / x_proj[2, 0]])


def reprojection_error(X: np.ndarray, u_detected: np.ndarray,
                       K: np.ndarray, R: np.ndarray, t: np.ndarray) -> float:
    """Compute reprojection error in pixels."""
    u_proj = project_point(X, K, R, t)
    return float(np.linalg.norm(u_proj - u_detected))


def procrustes_align(source: np.ndarray, target: np.ndarray):
    """Procrustes alignment: find R, t, s that minimize ||s*R*source + t - target||.

    Args:
        source: (N, 3) source points
        target: (N, 3) target points

    Returns:
        R: (3,3) rotation, t: (3,) translation, s: float scale
    """
    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)
    s_centered = source - mu_s
    t_centered = target - mu_t

    s = np.sqrt(np.sum(t_centered ** 2) / np.sum(s_centered ** 2))

    H = s_centered.T @ t_centered
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.diag([1, 1, d])
    R = Vt.T @ S @ U.T

    t_vec = mu_t - s * R @ mu_s
    return R, t_vec, s
