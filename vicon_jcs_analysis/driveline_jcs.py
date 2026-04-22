"""
Driveline (Visual3D CMO.v3s) kinematic pipeline — ported to Python.

Segment coordinate systems follow Visual3D defaults:
  Z = long axis (proximal → distal of the body)
  Y = anterior
  X = medio-lateral (right positive)

Joint angles are decomposed with Cardan sequence whose FIRST axis is specified
by the corresponding RT_<JOINT>_ANGLE's /AXIS1 line in CMO.v3s:

  Knee     AXIS1=X  -> XYZ Cardan (C1=flex/ext, C2=abd/add, C3=rot)
  Elbow    AXIS1=X  -> XYZ Cardan (C1=flex/ext, C2=carrying, C3=pronation)
  Shoulder AXIS1=Z  -> ZYX Cardan (C1=ER/IR, C2=add/abd, C3=horz abd)
  Pelvis   AXIS1=Z  -> ZYX (vs Lab) — we use vs Pelvis for trunk
  Torso    AXIS1=Z  -> ZYX (vs Lab)
  Torso-Pelvis AXIS1=Z -> ZYX

Sign negations (right-arm pitcher):
  RT_SHOULDER_ANGLE: NEGATEX=TRUE, NEGATEY=TRUE, NEGATEZ=TRUE
  RT_KNEE_ANGLE:     NEGATEX=TRUE, NEGATEY=TRUE
  RT_ELBOW_ANGLE:    FALSE (no negation)
  TORSO_PELVIS:      FALSE
  TORSO/PELVIS:      FALSE

This reproduces Driveline OBP `shoulder_angle_z` etc. on arbitrary marker data.
"""
from __future__ import annotations
import numpy as np


def unit(v, eps=1e-9):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n < eps, 1, n)


# ============ Segment CS (Z=long axis, Y=anterior, X=lateral) ============

def pelvis_cs(LASI, RASI, LPSI, RPSI):
    """Pelvis: origin mid-ASIS, X = right (RASI - LASI), Z = up (upward normal),
    Y = anterior (Z × X), per Vicon Plug-in-Gait / Visual3D default."""
    origin = (LASI + RASI) / 2
    X = unit(RASI - LASI)
    midPSI = (LPSI + RPSI) / 2
    v = origin - midPSI
    Z = unit(np.cross(X, v))
    Y = np.cross(Z, X)
    return np.stack([X, unit(Y), unit(Z)], axis=-1), origin


def thorax_cs(C7, CLAV, T10, STRN):
    """Thorax: origin mid(C7,CLAV), Z = up (C7→CLAV cross reversed), 
    Y = anterior (from midback to midfront)."""
    origin = (C7 + CLAV) / 2
    Z = unit(origin - (T10 + STRN) / 2)            # up
    midAnt = (CLAV + STRN) / 2
    midPos = (C7 + T10) / 2
    Y_hint = midAnt - midPos                          # anterior
    Y_perp = Y_hint - (Y_hint * Z).sum(-1, keepdims=True) * Z
    Y = unit(Y_perp)
    X = np.cross(Y, Z)
    return np.stack([unit(X), Y, Z], axis=-1), origin


def humerus_cs(SHO, ELB_lat, UPA=None, ELB_med=None):
    """Upper arm (RAR): Visual3D convention -- Z = DISTAL → PROXIMAL
    (points up when arm hangs). X = medio-lateral, Y = Z × X (anterior)."""
    origin = SHO
    Z = unit(SHO - ELB_lat)                         # distal→proximal
    if ELB_med is not None:
        X_hint = ELB_lat - ELB_med
    elif UPA is not None:
        X_hint = UPA - ELB_lat
    else:
        X_hint = np.tile(np.array([1.0, 0, 0]), (SHO.shape[0], 1))
    X_perp = X_hint - (X_hint * Z).sum(-1, keepdims=True) * Z
    X = unit(X_perp)
    Y = np.cross(Z, X)
    return np.stack([X, unit(Y), Z], axis=-1), origin, ELB_lat if ELB_med is None else (ELB_lat + ELB_med) / 2


def forearm_cs(EJC, WRA, WRB):
    """Forearm: Z = wrist → elbow (distal→proximal)."""
    WJC = (WRA + WRB) / 2
    Z = unit(EJC - WJC)
    X_hint = WRA - WRB                                # radial lateral
    X_perp = X_hint - (X_hint * Z).sum(-1, keepdims=True) * Z
    X = unit(X_perp)
    Y = np.cross(Z, X)
    return np.stack([X, unit(Y), Z], axis=-1), EJC, WJC


def thigh_cs(HJC, KNE_lat, KNE_med=None):
    """Z = knee → hip (distal→proximal)."""
    KJC = (KNE_lat + KNE_med) / 2 if KNE_med is not None else KNE_lat
    Z = unit(HJC - KJC)
    X_hint = KNE_lat - HJC if KNE_med is None else KNE_lat - KNE_med
    X_perp = X_hint - (X_hint * Z).sum(-1, keepdims=True) * Z
    X = unit(X_perp)
    Y = np.cross(Z, X)
    return np.stack([X, unit(Y), Z], axis=-1), HJC, KJC


def shank_cs(KJC, ANK_lat, ANK_med=None):
    """Z = ankle → knee (distal→proximal)."""
    AJC = (ANK_lat + ANK_med) / 2 if ANK_med is not None else ANK_lat
    Z = unit(KJC - AJC)
    X_hint = ANK_lat - KJC
    X_perp = X_hint - (X_hint * Z).sum(-1, keepdims=True) * Z
    X = unit(X_perp)
    Y = np.cross(Z, X)
    return np.stack([X, unit(Y), Z], axis=-1), KJC, AJC


def hip_jc_newington(LASI, RASI, LPSI, RPSI, side='R'):
    R_pel, origin = pelvis_cs(LASI, RASI, LPSI, RPSI)
    d = np.linalg.norm(RASI - LASI, axis=-1, keepdims=True)
    sign = +1 if side == 'R' else -1
    offset_local = np.concatenate(
        [sign * 0.36 * d, -0.19 * d, -0.30 * d], axis=-1)
    return origin + np.einsum('...ij,...j->...i', R_pel, offset_local)


# ============ Cardan decomposition (Visual3D convention) ============

def _clip(x): return np.clip(x, -1, 1)


def euler_zyx(R):
    """R = Rz(a) * Ry(b) * Rx(c).
    Returns (a, b, c) where a = rotation about Z (first axis),
    b = rotation about Y, c = rotation about X.
    """
    b = np.arcsin(_clip(-R[..., 2, 0]))
    a = np.arctan2(R[..., 1, 0], R[..., 0, 0])
    c = np.arctan2(R[..., 2, 1], R[..., 2, 2])
    return a, b, c


def euler_xyz(R):
    """R = Rx(a) * Ry(b) * Rz(c)."""
    b = np.arcsin(_clip(R[..., 0, 2]))
    a = np.arctan2(-R[..., 1, 2], R[..., 2, 2])
    c = np.arctan2(-R[..., 0, 1], R[..., 0, 0])
    return a, b, c


def Rrel(R_parent, R_child):
    """R_rel = R_parent^T · R_child (column-basis)."""
    return np.einsum('...ji,...jk->...ik', R_parent, R_child)


# ============ Main pipeline (Driveline rules) ============

def compute_driveline_angles(markers, pitch_side='R', lead_side='L'):
    """Returns dict of 8 variables (deg) and events (SFC, MER, BR indices).

    Variables follow Driveline CMO.v3s axis conventions:
      Var 1  = knee  Cardan C1 (X-axis) = flexion
      Var 2  = torso-pelvis C3 (X-axis) = forward flexion
      Var 3  = torso-pelvis C2 (Y-axis) = lateral tilt
      Var 4  = torso-pelvis C1 (Z-axis) = axial rotation (hip-shoulder sep)
      Var 5  = shoulder C2 (Y-axis) flipped = abduction (Add+/Abd− per CMO, we flip for "Abd+")
      Var 6  = shoulder C3 (X-axis) = horizontal abduction
      Var 7  = shoulder C1 (Z-axis) = external rotation (+)/internal (−)
      Var 8  = elbow Cardan C1 (X-axis) = flexion
    All with NEGATEX=Y=Z=TRUE for right arm/knee sign conventions.
    """
    def m(nm): return markers[nm] / 1000.0

    LASI, RASI, LPSI, RPSI = m('LASI'), m('RASI'), m('LPSI'), m('RPSI')
    C7, CLAV, T10, STRN = m('C7'), m('CLAV'), m('T10'), m('STRN')

    SHO_P = m(pitch_side + 'SHO')
    ELB_P_lat = m(pitch_side + 'ELB')
    UPA = m(pitch_side + 'UPA') if (pitch_side + 'UPA') in markers else None
    WRA = m(pitch_side + 'WRA'); WRB = m(pitch_side + 'WRB')
    ELB_P_med = m(pitch_side + 'MELB') if (pitch_side + 'MELB') in markers else None

    KNE_L = m(lead_side + 'KNE')
    ANK_L = m(lead_side + 'ANK')
    KNE_L_med = m(lead_side + 'MKNE') if (lead_side + 'MKNE') in markers else None
    ANK_L_med = m(lead_side + 'MANK') if (lead_side + 'MANK') in markers else None

    # ---- CS ----
    R_pel, _ = pelvis_cs(LASI, RASI, LPSI, RPSI)
    R_tho, _ = thorax_cs(C7, CLAV, T10, STRN)
    R_hum, _, EJC = humerus_cs(SHO_P, ELB_P_lat, UPA, ELB_P_med)
    R_fore, _, WJC = forearm_cs(EJC, WRA, WRB)
    HJC_L = hip_jc_newington(LASI, RASI, LPSI, RPSI, side=lead_side)
    R_th, _, KJC_L = thigh_cs(HJC_L, KNE_L, KNE_L_med)
    R_sh, _, AJC_L = shank_cs(KJC_L, ANK_L, ANK_L_med)

    # ---- Relative rotations ----
    R_shoulder = Rrel(R_tho, R_hum)          # humerus vs thorax
    R_elbow = Rrel(R_hum, R_fore)            # forearm vs humerus
    R_knee = Rrel(R_th, R_sh)                # shank vs thigh
    R_trunk_vs_pel = Rrel(R_pel, R_tho)      # trunk vs pelvis

    # ---- Cardan decomposition (Visual3D /AXIS1) ----
    # Shoulder ZYX:  C1=Z(ER+), C2=Y(Add+/Abd-), C3=X(HorzAbd+/Add-)
    sh_c1_z, sh_c2_y, sh_c3_x = euler_zyx(R_shoulder)
    # Elbow XYZ: C1=X (Flex+), C2=Y (carrying), C3=Z (pronation)
    el_c1_x, el_c2_y, el_c3_z = euler_xyz(R_elbow)
    # Knee XYZ: C1=X (Flex+)
    kn_c1_x, _, _ = euler_xyz(R_knee)
    # Torso-Pelvis ZYX: C1=Z(axial), C2=Y(lat), C3=X(fwd)
    tr_c1_z, tr_c2_y, tr_c3_x = euler_zyx(R_trunk_vs_pel)

    # Apply NEGATEX=Y=Z=TRUE for right-arm shoulder + right-leg knee
    # For right-arm pitcher these flip signs so ER becomes (+), etc.
    neg_sh = (pitch_side == 'R')     # right arm: all three axes flip
    neg_kn = (lead_side == 'R')      # right leg: X, Y flip
    if neg_sh:
        sh_c1_z = -sh_c1_z
        sh_c2_y = -sh_c2_y
        sh_c3_x = -sh_c3_x
    if neg_kn:
        kn_c1_x = -kn_c1_x

    deg = np.rad2deg
    vars_out = {
        1: deg(kn_c1_x),                       # lead knee flexion
        2: deg(tr_c3_x),                       # trunk forward flexion (X, third)
        3: deg(tr_c2_y),                       # trunk lateral tilt (Y, second)
        4: deg(tr_c1_z),                       # trunk axial rotation (Z, first)
        5: -deg(sh_c2_y),                      # shoulder abduction: OBP Y is Add+, we flip to Abd+
        6: deg(sh_c3_x),                       # shoulder horizontal abduction (X, third)
        7: deg(sh_c1_z),                       # shoulder external rotation (Z, first) — MER
        8: deg(el_c1_x),                       # elbow flexion
    }

    # Unwrap each to avoid ±180° jumps
    for k in vars_out:
        vars_out[k] = np.rad2deg(np.unwrap(np.deg2rad(vars_out[k])))

    return vars_out, {
        'R_thorax': R_tho, 'R_humerus': R_hum, 'R_shoulder_rel': R_shoulder,
        'EJC': EJC, 'WJC': WJC, 'ANK_L': ANK_L, 'KJC_L': KJC_L,
    }


def detect_events(vars_out, aux, markers, mrate, force_plates=None, pitch_side='R'):
    """Detect SFC (force plate if available) / MER (Var7 peak) / BR (wrist velo peak)."""
    WJC = aux['WJC']
    n = WJC.shape[0]

    # BR: wrist speed peak, excluding first 25%
    v = np.gradient(WJC, axis=0)
    speed = np.linalg.norm(v, axis=1)
    BR = int(np.argmax(speed))
    if BR < int(0.25 * n):
        BR = int(np.argmax(speed[int(0.25*n):]) + int(0.25*n))

    # SFC from force plates
    SFC = None
    if force_plates is not None and force_plates.get('fz_list'):
        thr = force_plates.get('threshold_N', 20.0)
        arate = force_plates['analog_rate']
        min_dur = int(arate * 0.05)
        BR_analog = int(BR / mrate * arate)
        window_lo = max(0, BR_analog - int(arate * 0.5))
        window_hi = min(BR_analog + int(arate * 0.1),
                        len(force_plates['fz_list'][0]) - min_dur)
        per_plate_first = []
        for fz in force_plates['fz_list']:
            above = fz > thr
            found = None
            for i in range(max(window_lo, min_dur), window_hi):
                fwd = above[i:i+min_dur]; bwd = above[i-min_dur:i]
                if above[i] and fwd.mean() > 0.8 and bwd.mean() < 0.2:
                    found = i; break
            per_plate_first.append(found)
        valid = [c for c in per_plate_first if c is not None]
        if valid:
            sfc_analog = min(valid, key=lambda x: abs(x - BR_analog))
            SFC = int(sfc_analog / arate * mrate)
    if SFC is None:
        # Fallback: ankle Y min in BR-0.5s window
        ank_y = aux['ANK_L'][:, 1]
        lo = max(0, BR - int(0.5 * mrate))
        SFC = lo + int(np.argmin(ank_y[lo:BR]))

    # MER: argmax of Var 7 in BR-0.5s window
    win_frames = int(0.5 * mrate)
    mer_lo = max(0, BR - win_frames)
    MER = mer_lo + int(np.argmax(vars_out[7][mer_lo:BR+1]))

    return {'SFC': SFC, 'MER': MER, 'BR': BR}
