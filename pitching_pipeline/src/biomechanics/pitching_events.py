"""Automatic detection of 6 pitching phases and key events.

Events:
  FC  = Foot Contact (stride foot strikes ground)
  MER = Maximum External Rotation (peak shoulder ER)
  BR  = Ball Release (peak wrist velocity)
  MIR = Maximum Internal Rotation (peak shoulder IR)

Phases:
  1. Wind-up → FC
  2. Stride / Early Cocking → FC → MER
  3. Late Cocking → approaching MER
  4. Acceleration → MER → BR (~30ms)
  5. Deceleration → BR → MIR
  6. Follow-through → after MIR
"""
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from typing import Dict, Optional, Tuple


def butterworth_filter(data: np.ndarray, cutoff: float = 15.0,
                       fs: float = 240.0, order: int = 4) -> np.ndarray:
    """Apply zero-lag Butterworth low-pass filter."""
    nyq = fs / 2
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, data, axis=0)


def detect_foot_contact(
    ankle_z: np.ndarray,
    fps: float = 240.0,
) -> Optional[int]:
    """Detect foot contact from ankle vertical position.

    FC = frame where stride-foot ankle z-velocity crosses zero (descending).
    """
    vel = np.gradient(ankle_z, 1.0 / fps)
    vel_smooth = butterworth_filter(vel, cutoff=10.0, fs=fps)

    # Find zero crossings (positive → negative = foot landing)
    for i in range(1, len(vel_smooth)):
        if vel_smooth[i - 1] > 0 and vel_smooth[i] <= 0:
            return i
    return None


def detect_mer(
    shoulder_rotation: np.ndarray,
) -> Optional[int]:
    """Detect Maximum External Rotation (MER).

    MER = peak of shoulder external rotation angle.
    """
    peaks, _ = find_peaks(shoulder_rotation, distance=50)
    if len(peaks) == 0:
        return None
    # Return the highest peak
    return int(peaks[np.argmax(shoulder_rotation[peaks])])


def detect_ball_release(
    wrist_velocity: np.ndarray,
) -> Optional[int]:
    """Detect Ball Release from wrist velocity.

    BR = frame just after peak wrist velocity (velocity starts to decrease).
    """
    peaks, _ = find_peaks(wrist_velocity, distance=50)
    if len(peaks) == 0:
        return None
    # Highest velocity peak
    peak_idx = peaks[np.argmax(wrist_velocity[peaks])]
    # BR is slightly after peak velocity
    return int(peak_idx + 1)


def detect_max_internal_rotation(
    shoulder_rotation: np.ndarray,
    br_frame: int,
) -> Optional[int]:
    """Detect Maximum Internal Rotation after ball release."""
    if br_frame is None:
        return None
    # MIR = minimum of shoulder rotation after BR (internal = negative direction)
    post_br = shoulder_rotation[br_frame:]
    if len(post_br) == 0:
        return None
    return int(br_frame + np.argmin(post_br))


def detect_pitching_events(
    joints_3d: np.ndarray,
    fps: float = 240.0,
    throwing_side: str = "right",
) -> Dict[str, Optional[int]]:
    """Detect all pitching events from 3D joint trajectories.

    Args:
        joints_3d: (N_frames, N_joints, 3) joint positions
        fps: frame rate
        throwing_side: 'right' or 'left'

    Returns:
        dict with event names → frame indices
    """
    # Joint indices (OpenPose Body25)
    if throwing_side == "right":
        ankle_idx = 11  # RAnk
        wrist_idx = 4   # RWri
        shoulder_idx = 2  # RSho
        elbow_idx = 3   # RElb
    else:
        ankle_idx = 14  # LAnk
        wrist_idx = 7   # LWri
        shoulder_idx = 5  # LSho
        elbow_idx = 6   # LElb

    # Filter joint positions
    joints_smooth = butterworth_filter(joints_3d, cutoff=15.0, fs=fps)

    # Ankle vertical position for FC
    ankle_z = joints_smooth[:, ankle_idx, 1]  # Y = vertical in most coords
    fc = detect_foot_contact(ankle_z, fps)

    # Wrist velocity for BR
    wrist_pos = joints_smooth[:, wrist_idx]
    wrist_vel = np.linalg.norm(np.gradient(wrist_pos, 1.0 / fps, axis=0), axis=1)
    br = detect_ball_release(wrist_vel)

    # Shoulder rotation for MER/MIR (simplified: angle between upper arm and torso)
    shoulder = joints_smooth[:, shoulder_idx]
    elbow = joints_smooth[:, elbow_idx]
    neck = joints_smooth[:, 1]  # Neck

    upper_arm = elbow - shoulder
    trunk = neck - joints_smooth[:, 8]  # Neck - MidHip

    # Cross product magnitude as rotation proxy
    rotation_proxy = np.array([
        np.arctan2(np.linalg.norm(np.cross(ua, tr)), np.dot(ua, tr))
        for ua, tr in zip(upper_arm, trunk)
    ])
    rotation_deg = np.degrees(rotation_proxy)

    mer = detect_mer(rotation_deg)
    mir = detect_max_internal_rotation(rotation_deg, br)

    events = {
        "foot_contact": fc,
        "max_external_rotation": mer,
        "ball_release": br,
        "max_internal_rotation": mir,
    }

    return events


def get_phase(frame: int, events: Dict[str, Optional[int]]) -> str:
    """Get pitching phase for a given frame."""
    fc = events.get("foot_contact")
    mer = events.get("max_external_rotation")
    br = events.get("ball_release")
    mir = events.get("max_internal_rotation")

    if fc is not None and frame < fc:
        return "windup"
    elif mer is not None and frame < mer:
        return "late_cocking"
    elif br is not None and frame < br:
        return "acceleration"
    elif mir is not None and frame < mir:
        return "deceleration"
    else:
        return "follow_through"
