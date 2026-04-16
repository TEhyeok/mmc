"""
compute_8vars_from_smpl.py — SMPL 가상 마커 → JCS → 8개 운동학 변수

파이프라인:
  gsplat 정제된 SMPL → forward kinematics → 가상 마커 추출
  → ISB 관절좌표계(JCS) 구축 → 오일러 분해 → 8개 변수

논문 Table 6 기준:
  Variable              Parent↔Child       Decomposition         Euler
  Lead knee flexion     Thigh↔Shank        Flexion/extension     XYZ
  Trunk forward tilt    Pelvis↔Thorax      Sagittal plane tilt   XZY
  Trunk lateral tilt    Pelvis↔Thorax      Frontal plane tilt    XZY
  Trunk axial rotation  Pelvis↔Thorax      Transverse plane rot  XZY
  Shoulder abduction    Thorax↔Humerus     Frontal plane abd     YZX
  Shoulder horiz. abd.  Thorax↔Humerus     Transverse plane abd  YZX
  Shoulder rotation     Thorax↔Humerus     Long-axis rotation    ZYX
  Elbow flexion         Humerus↔Forearm    Flexion/extension     XYZ

사용법:
  python compute_8vars_from_smpl.py \
    --refined_dir gsplat_refined_smpl/ \
    --output_dir kinematic_results/ \
    --marker_mapping smpl_virtual_marker_mapping.json
"""

import numpy as np
import json
import os
import argparse
from scipy.spatial.transform import Rotation


# ============================================================
# 1. 가상 마커에서 JCS 구축
# ============================================================

def normalize(v):
    """벡터 정규화"""
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def build_pelvis_jcs(markers: dict) -> np.ndarray:
    """
    골반 JCS (Vicon Plug-in Gait 정의)
    원점: L_ASIS, R_ASIS 중점
    X축: L_ASIS → R_ASIS (우측)
    Z축: ASIS 평면의 법선 (상방)
    Y축: Z × X (전방)
    """
    l_asis = markers['L_ASIS']
    r_asis = markers['R_ASIS']
    l_psis = markers['L_PSIS']
    r_psis = markers['R_PSIS']

    origin = (l_asis + r_asis) / 2.0

    # X: 우측 (SMPL: L_ASIS는 +X, R_ASIS는 -X → 반전 필요)
    x_axis = normalize(r_asis - l_asis)

    # ASIS-PSIS 평면의 법선 → Z (상방)
    asis_mid = (l_asis + r_asis) / 2.0
    psis_mid = (l_psis + r_psis) / 2.0
    posterior = normalize(psis_mid - asis_mid)
    z_axis = normalize(np.cross(x_axis, posterior))

    # Y: Z × X (전방)
    y_axis = normalize(np.cross(z_axis, x_axis))

    R = np.column_stack([x_axis, y_axis, z_axis])
    return R, origin


def build_thorax_jcs(markers: dict) -> np.ndarray:
    """
    체간 JCS
    원점: C7, T10, CLAV 중점
    Z축: T10 → C7 (상방)
    Y축: 쇄골 방향에서 정규직교화 (전방)
    X축: Y × Z (우측)
    """
    c7 = markers['C7']
    t10 = markers['T10']
    clav = markers['CLAV']

    origin = (c7 + t10 + clav) / 3.0

    # Z: T10 → C7 (상방)
    z_axis = normalize(c7 - t10)

    # 전방 방향: T10→CLAV의 Z축 직교 성분
    forward = clav - t10
    y_axis = normalize(forward - np.dot(forward, z_axis) * z_axis)

    # X: Y × Z (우측)
    x_axis = normalize(np.cross(y_axis, z_axis))

    R = np.column_stack([x_axis, y_axis, z_axis])
    return R, origin


def build_humerus_jcs(markers: dict, side: str = 'R') -> np.ndarray:
    """
    상완 JCS
    원점: Shoulder
    Y축: Elbow → Shoulder (근위 방향)
    X축: Elbow 내외측에서 정규직교화 (외측)
    Z축: X × Y (전방)
    """
    sho = markers[f'{side}_Shoulder']
    elb_lat = markers[f'{side}_Elbow_lat']
    elb_med = markers[f'{side}_Elbow_med']

    origin = sho
    elb_center = (elb_lat + elb_med) / 2.0

    # Y: 근위 (Elbow→Shoulder)
    y_axis = normalize(sho - elb_center)

    # 내외측 방향
    if side == 'R':
        lateral = normalize(elb_lat - elb_med)  # R: lat is more lateral (-X)
    else:
        lateral = normalize(elb_lat - elb_med)  # L: lat is more lateral (+X)

    # X: 외측 (Y에 직교)
    x_axis = normalize(lateral - np.dot(lateral, y_axis) * y_axis)

    # Z: X × Y (전방)
    z_axis = normalize(np.cross(x_axis, y_axis))

    R = np.column_stack([x_axis, y_axis, z_axis])
    return R, origin


def build_forearm_jcs(markers: dict, side: str = 'R') -> np.ndarray:
    """
    전완 JCS (팔꿈치 굴곡용)
    원점: Elbow center
    Y축: Wrist → Elbow (근위 방향)
    """
    elb_lat = markers[f'{side}_Elbow_lat']
    elb_med = markers[f'{side}_Elbow_med']
    wrist = markers[f'{side}_Wrist']

    elb_center = (elb_lat + elb_med) / 2.0
    origin = elb_center

    y_axis = normalize(elb_center - wrist)

    # X: 내외측
    if side == 'R':
        lateral = normalize(elb_lat - elb_med)
    else:
        lateral = normalize(elb_lat - elb_med)
    x_axis = normalize(lateral - np.dot(lateral, y_axis) * y_axis)

    z_axis = normalize(np.cross(x_axis, y_axis))

    R = np.column_stack([x_axis, y_axis, z_axis])
    return R, origin


def build_thigh_jcs(markers: dict, joints: np.ndarray, side: str = 'L') -> np.ndarray:
    """
    대퇴 JCS (리드 무릎 굴곡용)
    원점: Hip joint center
    Y축: Knee → Hip (근위 방향)
    """
    hip_idx = 1 if side == 'L' else 2  # SMPL joint index
    hip = joints[hip_idx]
    knee = markers[f'{side}_Knee_lat']

    origin = hip
    y_axis = normalize(hip - knee)

    # 전방 방향 (골반 전방 참조)
    x_temp = np.array([1, 0, 0]) if side == 'R' else np.array([-1, 0, 0])
    z_axis = normalize(np.cross(x_temp, y_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))

    R = np.column_stack([x_axis, y_axis, z_axis])
    return R, origin


def build_shank_jcs(markers: dict, side: str = 'L') -> np.ndarray:
    """
    하퇴 JCS
    원점: Knee
    Y축: Ankle → Knee (근위 방향)
    """
    knee = markers[f'{side}_Knee_lat']
    ankle = markers[f'{side}_Ankle']

    origin = knee
    y_axis = normalize(knee - ankle)

    x_temp = np.array([1, 0, 0]) if side == 'R' else np.array([-1, 0, 0])
    z_axis = normalize(np.cross(x_temp, y_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))

    R = np.column_stack([x_axis, y_axis, z_axis])
    return R, origin


# ============================================================
# 2. 상대 회전 → 오일러 분해 → 8개 변수
# ============================================================

def relative_rotation(R_child, R_parent):
    """인접 분절 간 상대 회전 행렬"""
    return R_child @ R_parent.T


def euler_decompose(R_rel, order: str) -> np.ndarray:
    """
    상대 회전행렬 → 오일러 각도 분해 (도 단위)

    Args:
        R_rel: (3,3) 상대 회전행렬
        order: 'XYZ', 'XZY', 'YZX', 'ZYX' 등

    Returns:
        angles: (3,) 도 단위 오일러 각도
    """
    # scipy uses lowercase intrinsic convention
    rot = Rotation.from_matrix(R_rel)

    # 오일러 분해 (내적 = intrinsic)
    angles_rad = rot.as_euler(order.lower(), degrees=False)
    angles_deg = np.degrees(angles_rad)

    return angles_deg


def compute_8_variables(markers: dict, joints: np.ndarray,
                        throwing_side: str = 'R', lead_side: str = 'L') -> dict:
    """
    8개 운동학 변수 산출

    Args:
        markers: dict of marker_name → (3,) position
        joints: (24, 3) SMPL joint positions
        throwing_side: 투구 팔 ('R' for 우투수)
        lead_side: 리드 다리 ('L' for 우투수)

    Returns:
        dict of variable_name → angle (degrees)
    """
    # JCS 구축
    R_pelvis, _ = build_pelvis_jcs(markers)
    R_thorax, _ = build_thorax_jcs(markers)
    R_humerus, _ = build_humerus_jcs(markers, side=throwing_side)
    R_forearm, _ = build_forearm_jcs(markers, side=throwing_side)
    R_thigh, _ = build_thigh_jcs(markers, joints, side=lead_side)
    R_shank, _ = build_shank_jcs(markers, side=lead_side)

    results = {}

    # 1. Lead knee flexion: Thigh↔Shank, XYZ
    R_rel = relative_rotation(R_shank, R_thigh)
    angles = euler_decompose(R_rel, 'XYZ')
    results['lead_knee_flexion'] = angles[0]

    # 2-4. Trunk: Pelvis↔Thorax, XZY
    R_rel = relative_rotation(R_thorax, R_pelvis)
    angles = euler_decompose(R_rel, 'XZY')
    results['trunk_forward_tilt'] = angles[0]    # X: 시상면
    results['trunk_lateral_tilt'] = angles[1]     # Z: 관상면
    results['trunk_axial_rotation'] = angles[2]   # Y: 수평면

    # 5-6. Shoulder abduction & horizontal abd: Thorax↔Humerus, YZX
    R_rel = relative_rotation(R_humerus, R_thorax)
    angles = euler_decompose(R_rel, 'YZX')
    results['shoulder_abduction'] = angles[0]             # Y: 관상면 외전
    results['shoulder_horizontal_abduction'] = angles[1]  # Z: 수평면 외전

    # 7. Shoulder rotation: Thorax↔Humerus, ZYX
    angles_zyx = euler_decompose(R_rel, 'ZYX')
    results['shoulder_rotation'] = angles_zyx[0]  # Z: 장축 회전

    # 8. Elbow flexion: Humerus↔Forearm, XYZ
    R_rel = relative_rotation(R_forearm, R_humerus)
    angles = euler_decompose(R_rel, 'XYZ')
    results['elbow_flexion'] = angles[0]

    return results


# ============================================================
# 3. 시퀀스 처리
# ============================================================

def process_sequence(refined_dir: str, output_dir: str,
                     throwing_side: str = 'R', lead_side: str = 'L'):
    """
    전체 시퀀스에 대해 8개 변수 산출

    Args:
        refined_dir: gsplat_refined_smpl/ (정제된 NPZ 파일들)
        output_dir: 결과 저장 디렉토리
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find all refined NPZ files
    npz_files = sorted([f for f in os.listdir(refined_dir) if f.endswith('.npz')
                        and f != 'refinement_summary.json'])

    all_results = {}
    all_angles = {
        'lead_knee_flexion': [],
        'trunk_forward_tilt': [],
        'trunk_lateral_tilt': [],
        'trunk_axial_rotation': [],
        'shoulder_abduction': [],
        'shoulder_horizontal_abduction': [],
        'shoulder_rotation': [],
        'elbow_flexion': [],
    }
    frames = []

    for npz_file in npz_files:
        frame_idx = int(npz_file.replace('.npz', ''))
        data = np.load(os.path.join(refined_dir, npz_file), allow_pickle=True)

        markers = data['virtual_markers'].item()
        joints = data['joints']

        try:
            angles = compute_8_variables(markers, joints, throwing_side, lead_side)
            all_results[frame_idx] = angles
            frames.append(frame_idx)
            for k, v in angles.items():
                all_angles[k].append(v)
        except Exception as e:
            print(f"Frame {frame_idx}: {e}")
            continue

    # Convert to arrays
    frames = np.array(frames)
    for k in all_angles:
        all_angles[k] = np.array(all_angles[k])

    # Save results
    np.savez(
        os.path.join(output_dir, 'kinematic_8vars.npz'),
        frames=frames,
        **all_angles
    )

    # Save CSV for easy viewing
    with open(os.path.join(output_dir, 'kinematic_8vars.csv'), 'w') as f:
        headers = ['frame'] + list(all_angles.keys())
        f.write(','.join(headers) + '\n')
        for i, fi in enumerate(frames):
            vals = [str(fi)] + [f'{all_angles[k][i]:.2f}' for k in all_angles]
            f.write(','.join(vals) + '\n')

    # Print summary
    print(f"\n=== 8 Kinematic Variables Summary ({len(frames)} frames) ===")
    print(f"{'Variable':<35} {'Mean':>8} {'SD':>8} {'Min':>8} {'Max':>8}")
    print('-' * 75)
    for k, v in all_angles.items():
        if len(v) > 0:
            print(f"  {k:<33} {v.mean():>7.1f}° {v.std():>7.1f}° "
                  f"{v.min():>7.1f}° {v.max():>7.1f}°")

    print(f"\nSaved to: {output_dir}")
    return all_angles, frames


# ============================================================
# 4. CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Compute 8 kinematic variables from refined SMPL')
    parser.add_argument('--refined_dir', required=True, help='Directory with refined NPZ files')
    parser.add_argument('--output_dir', default='kinematic_results/')
    parser.add_argument('--throwing_side', default='R', choices=['L', 'R'])
    parser.add_argument('--lead_side', default='L', choices=['L', 'R'])
    args = parser.parse_args()

    process_sequence(args.refined_dir, args.output_dir,
                     args.throwing_side, args.lead_side)


if __name__ == '__main__':
    main()
