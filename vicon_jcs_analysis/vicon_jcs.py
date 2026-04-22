"""
Vicon marker-based JCS + Euler decomposition pipeline
following cjh thesis Table 6 (Joint Coordinate System approach).

Pipeline:
  Vicon 3D markers (Y-up) -> segment JCS frames -> relative rotation matrix
  -> Euler decomposition per joint per Table 6 -> 8 variables (deg).

Coord convention (this module): X = right, Y = up, Z = forward (toward home plate).
"""
from __future__ import annotations
import numpy as np


def unit(v, eps=1e-9):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n < eps, 1, n)


# ------------------------ Segment JCS (cjh 3.5.8) ------------------------

def pelvis_jcs(LASI, RASI, LPSI, RPSI):
    """Pelvis JCS: origin mid-ASIS, X = L->R ASIS, Z = normal of ASIS-PSIS plane
    (upward), Y = Z x X (anterior). Returns R (N,3,3) with columns [X, Y, Z]."""
    origin = (LASI + RASI) / 2
    X = unit(RASI - LASI)
    midPSIS = (LPSI + RPSI) / 2
    v = origin - midPSIS               # vector from PSIS midpoint to ASIS midpoint (forward-down)
    # Z_up = cross(X, v) normalised
    Z_up = unit(np.cross(X, v))        # upward normal
    Y_fwd = np.cross(Z_up, X)          # anterior
    # Table spec: X=right, Z=up, Y=forward; rename axes to our (X,Y=up,Z=fwd)
    # i.e. in our module convention columns = [X, up, forward] = [X, Z_up, Y_fwd]
    R = np.stack([X, Z_up, Y_fwd], axis=-1)
    return R, origin


def thorax_jcs(C7, CLAV, T10, STRN):
    """Thorax JCS (cjh 3.5.8): origin = mid(C7,T10,CLAV); Z = T10->C7 (up);
    Y (forward) = orthogonalized CLAV direction from origin; X = Y x Z (right)."""
    origin = (C7 + T10 + CLAV) / 3
    Z_up = unit(C7 - T10)                       # up
    # CLAV direction relative to origin gives anterior hint
    a = CLAV - origin
    a_perp = a - (a * Z_up).sum(-1, keepdims=True) * Z_up
    Y_fwd = unit(a_perp)
    X_right = np.cross(Y_fwd, Z_up)             # right = forward x up
    X_right = unit(X_right)
    # re-orthogonalize Y
    Y_fwd = np.cross(Z_up, X_right)
    R = np.stack([X_right, Z_up, Y_fwd], axis=-1)
    return R, origin


def humerus_jcs(SHO, ELB_lat, ELB_med=None, UPA=None):
    """Humerus JCS with long axis = Z (to separate YZX plane-of-elev from
    ZYX long-axis rotation). Column order: [X_right, Y_anterior, Z_long].

    Y is chosen to keep the shoulder YZX decomposition yielding:
        (plane-of-elevation, abduction, rotation) on (Y, Z, X) in this frame.
    """
    if ELB_med is not None:
        EJC = (ELB_lat + ELB_med) / 2
        Xhint = ELB_lat - ELB_med
    else:
        EJC = ELB_lat.copy()
        if UPA is not None:
            Xhint = UPA - ELB_lat
        else:
            Xhint = np.tile(np.array([1.0, 0, 0]), (SHO.shape[0], 1))

    origin = SHO
    Z_long = unit(SHO - EJC)                        # long axis = proximal dir
    Xhint_perp = Xhint - (Xhint * Z_long).sum(-1, keepdims=True) * Z_long
    X_right = unit(Xhint_perp)
    Y_ant = np.cross(Z_long, X_right)
    R = np.stack([X_right, Y_ant, Z_long], axis=-1)
    return R, origin, EJC


def thigh_jcs(HJC, KNE_lat, KNE_med=None):
    """Thigh JCS: origin = Hip JC, Y = Knee->Hip (proximal), X = lateral knee axis."""
    if KNE_med is not None:
        KJC = (KNE_lat + KNE_med) / 2
        Xhint = KNE_lat - KNE_med
    else:
        KJC = KNE_lat.copy()
        Xhint = KNE_lat - HJC  # fallback
    Y_prox = unit(HJC - KJC)
    Xhint_perp = Xhint - (Xhint * Y_prox).sum(-1, keepdims=True) * Y_prox
    X_right = unit(Xhint_perp)
    Z_fwd = np.cross(X_right, Y_prox)
    R = np.stack([X_right, Y_prox, Z_fwd], axis=-1)
    return R, HJC, KJC


def shank_jcs(KJC, ANK_lat, ANK_med=None):
    if ANK_med is not None:
        AJC = (ANK_lat + ANK_med) / 2
    else:
        AJC = ANK_lat.copy()
    Y_prox = unit(KJC - AJC)
    Xhint = ANK_lat - KJC
    Xhint_perp = Xhint - (Xhint * Y_prox).sum(-1, keepdims=True) * Y_prox
    X_right = unit(Xhint_perp)
    Z_fwd = np.cross(X_right, Y_prox)
    R = np.stack([X_right, Y_prox, Z_fwd], axis=-1)
    return R, KJC, AJC


# ------------------------ Hip JC (Newington) ------------------------

def hip_jc_newington(LASI, RASI, LPSI, RPSI, side='R'):
    Rp, origin = pelvis_jcs(LASI, RASI, LPSI, RPSI)
    d = np.linalg.norm(RASI - LASI, axis=-1, keepdims=True)
    sign = +1 if side == 'R' else -1
    # Newington offsets: (lateral, vertical-down, posterior)
    # Pelvis JCS columns: [X(right), Y(up), Z(forward)]; so local offset:
    #   lateral: +X  (sign for R)
    #   vertical down: -Y
    #   posterior: -Z
    offset_local = np.concatenate(
        [sign * 0.36 * d, -0.19 * d, -0.30 * d], axis=-1)
    return origin + np.einsum('...ij,...j->...i', Rp, offset_local)


# ------------------------ Euler decompositions ------------------------

def _clip(x):
    return np.clip(x, -1, 1)


def euler_xzy(R):
    """R = Rx(a) * Rz(b) * Ry(c). Returns (a, b, c) rad.
       Trunk convention per cjh Table 6 for Pelvis->Thorax.
         a = forward tilt (sagittal), b = lateral tilt (frontal), c = axial.
    """
    # R[1,0] = -sin(b)*cos(a)*?  -> derive from product
    # Let Rx(a) Rz(b) Ry(c). Compute symbolic:
    # Row 0: [ cos(b)cos(c),               -sin(b),            cos(b)sin(c) ]
    # Row 1: [ sin(a)sin(c) + cos(a)sin(b)cos(c),  cos(a)cos(b),  -sin(a)cos(c) + cos(a)sin(b)sin(c) ]
    # Row 2: [ -cos(a)sin(c) + sin(a)sin(b)cos(c), sin(a)cos(b),  cos(a)cos(c) + sin(a)sin(b)sin(c) ]
    b = np.arcsin(_clip(-R[..., 0, 1]))
    a = np.arctan2(R[..., 2, 1], R[..., 1, 1])
    c = np.arctan2(R[..., 0, 2], R[..., 0, 0])
    return a, b, c


def euler_yzx(R):
    """R = Ry(a) * Rz(b) * Rx(c).
       Shoulder abduction / horiz abd per Table 6.
         a = elevation plane (YZX first axis), b = abduction, c = rotation-ish.
    """
    # Rows of Ry(a) Rz(b) Rx(c):
    # [ cos(a)cos(b),  -cos(a)sin(b)cos(c) + sin(a)sin(c),  cos(a)sin(b)sin(c) + sin(a)cos(c) ]
    # [ sin(b),         cos(b)cos(c),                        -cos(b)sin(c) ]
    # [-sin(a)cos(b),  sin(a)sin(b)cos(c) + cos(a)sin(c),  -sin(a)sin(b)sin(c) + cos(a)cos(c)]
    b = np.arcsin(_clip(R[..., 1, 0]))
    a = np.arctan2(-R[..., 2, 0], R[..., 0, 0])
    c = np.arctan2(-R[..., 1, 2], R[..., 1, 1])
    return a, b, c


def euler_zyx(R):
    """R = Rz(a) * Ry(b) * Rx(c) — shoulder long-axis rotation (first axis = a).
       cjh Table 6 'ZYX' for shoulder_rotation; we take a as long-axis rotation.
    """
    # Rows of Rz(a) Ry(b) Rx(c):
    # [ cos(a)cos(b),   cos(a)sin(b)sin(c) - sin(a)cos(c),  cos(a)sin(b)cos(c) + sin(a)sin(c) ]
    # [ sin(a)cos(b),   sin(a)sin(b)sin(c) + cos(a)cos(c),  sin(a)sin(b)cos(c) - cos(a)sin(c) ]
    # [-sin(b),         cos(b)sin(c),                       cos(b)cos(c) ]
    b = np.arcsin(_clip(-R[..., 2, 0]))
    a = np.arctan2(R[..., 1, 0], R[..., 0, 0])
    c = np.arctan2(R[..., 2, 1], R[..., 2, 2])
    return a, b, c


def euler_xyz(R):
    """R = Rx(a) * Ry(b) * Rz(c) — knee/elbow flexion via XYZ (a = flex)."""
    b = np.arcsin(_clip(R[..., 0, 2]))
    a = np.arctan2(-R[..., 1, 2], R[..., 2, 2])
    c = np.arctan2(-R[..., 0, 1], R[..., 0, 0])
    return a, b, c


# ------------------------ Main compute ------------------------

def Rrel(Rparent, Rchild):
    """R_rel = R_parent^T * R_child (column-basis convention)."""
    return np.einsum('...ji,...jk->...ik', Rparent, Rchild)


def zero_pose_align(R, zero_idx=0):
    """Remove initial offset: R_adj[t] = R[zero_idx]^T @ R[t].
    This forces R_adj[zero_idx] = I, so joint angles start from 0.
    Needed because different segment-CS conventions give arbitrary rotations
    in anatomical neutral that swamp the motion-relevant signal.
    """
    Rref_T = R[zero_idx].T
    return np.einsum('ij,kjl->kil', Rref_T, R)


def compute_8_variables(markers, pitch_side='R', lead_side='L', zero_window=(0, 30),
                        force_plates=None):
    """
    Input: dict of (F,3) arrays in mm, Y-up world.
    zero_window: frames (start, end) used as "anatomical zero" (averaged rotation).
    force_plates: optional dict {
        'fz_list': list of 1D arrays (|Fz| in N) per plate,
        'analog_rate': float (Hz),
        'marker_rate': float (Hz),
        'threshold_N': 20.0,   # Fleisig 1995 standard
    }
    Output: dict var1..8 in degrees + events dict.
    """
    def m(n): return markers[n] / 1000.0

    LASI, RASI, LPSI, RPSI = m('LASI'), m('RASI'), m('LPSI'), m('RPSI')
    C7, CLAV, T10, STRN = m('C7'), m('CLAV'), m('T10'), m('STRN')
    SHO_P = m(pitch_side + 'SHO')
    ELB_P = m(pitch_side + 'ELB')
    # Wrist JC: midpoint of WRA + WRB
    WRA = m(pitch_side + 'WRA'); WRB = m(pitch_side + 'WRB')
    WJC = (WRA + WRB) / 2
    # Upper arm cluster for humerus X hint when no medial epicondyle
    UPA = m(pitch_side + 'UPA') if (pitch_side + 'UPA') in markers else None

    KNE_L = m(lead_side + 'KNE')
    ANK_L = m(lead_side + 'ANK')

    # --- JCS ---
    R_pel, _ = pelvis_jcs(LASI, RASI, LPSI, RPSI)
    R_tho, _ = thorax_jcs(C7, CLAV, T10, STRN)
    R_hum, _, EJC = humerus_jcs(SHO_P, ELB_P, ELB_med=None, UPA=UPA)
    HJC_L = hip_jc_newington(LASI, RASI, LPSI, RPSI, side=lead_side)
    R_th, HJC_L, KJC_L = thigh_jcs(HJC_L, KNE_L, KNE_med=None)
    R_sh, _, AJC_L = shank_jcs(KJC_L, ANK_L, ANK_med=None)

    # --- Relative rotations ---
    R_trunk = Rrel(R_pel, R_tho)            # pelvis -> thorax (Table 6 vars 2,3,4)
    R_shoulder = Rrel(R_tho, R_hum)         # thorax -> humerus (vars 5,6,7)
    R_knee = Rrel(R_th, R_sh)               # thigh -> shank (var 1)

    # --- Zero-pose alignment ---
    # Find the quietest frame (least marker motion) in first 40% to use as zero
    # Use wrist marker as proxy (stationary in wind-up)
    n_frames = R_trunk.shape[0]
    wrist = (m(pitch_side + 'WRA') + m(pitch_side + 'WRB')) / 2
    # Sliding 30-frame speed; pick the lowest-speed window in first 50%
    speed = np.linalg.norm(np.diff(wrist, axis=0), axis=1)
    # pad
    speed = np.concatenate([speed, speed[-1:]])
    W = 30
    half = n_frames // 2
    roll = np.convolve(speed[:half], np.ones(W)/W, mode='valid')
    zi = int(np.argmin(roll) + W // 2)
    # fallback to first frame if something weird
    if zi < 0 or zi >= n_frames: zi = 0

    R_trunk = zero_pose_align(R_trunk, zero_idx=zi)
    R_shoulder = zero_pose_align(R_shoulder, zero_idx=zi)
    R_knee = zero_pose_align(R_knee, zero_idx=zi)

    # --- Decompositions per Table 6 ---
    # Vars 2/3/4: trunk XZY decomposition
    trunk_fwd, trunk_lat, trunk_axial = euler_xzy(R_trunk)

    # Vars 5, 6, 7 all come from the SAME YZX decomposition of R_shoulder,
    # per cjh Table 6 (Shoulder abd = frontal YZX, Shoulder horiz abd = transverse
    # YZX, Shoulder rotation = long-axis YZX). Taking them from a single
    # decomposition guarantees they are three INDEPENDENT components of the
    # shoulder rotation, not redundant projections.
    #   YZX: R = Ry(plane) Rz(abd) Rx(rot)
    #     Var 6 = plane of elevation   (Y, first)
    #     Var 5 = abduction/elevation  (Z, second)
    #     Var 7 = long-axis rotation   (X, third)
    sh_plane, sh_abd, sh_long_axis = euler_yzx(R_shoulder)

    # Var 1: knee flexion via 3-point vector angle (single-DoF geometric def,
    # equivalent to PIG LKneeAngles.X). Cardan decomposition is unstable for
    # single-DoF joint because secondary axes dominate singular orientations.
    thigh_v = HJC_L - KJC_L   # distal->proximal
    shank_v = AJC_L - KJC_L   # distal->proximal (ankle->knee)
    cos_kn = np.einsum('ij,ij->i', thigh_v, shank_v) / (
        np.linalg.norm(thigh_v, axis=-1) * np.linalg.norm(shank_v, axis=-1) + 1e-9)
    kn_flex = np.pi - np.arccos(np.clip(cos_kn, -1, 1))  # 0 = full extension

    # Var 8 (Elbow flex): 3-point vector angle (equivalent to Plug-in-Gait X)
    hum_v = EJC - SHO_P
    fore_v = WJC - EJC
    cos_el = np.einsum('ij,ij->i', hum_v, fore_v) / (
        np.linalg.norm(hum_v, axis=-1) * np.linalg.norm(fore_v, axis=-1) + 1e-9)
    elbow_flex = np.pi - np.arccos(np.clip(cos_el, -1, 1))  # 0 = extended

    def _unwrap_deg(x):
        return np.rad2deg(np.unwrap(x))

    vars_out = {
        1: np.rad2deg(kn_flex),  # knee: geometric 3-point angle, no unwrap
        2: _unwrap_deg(trunk_fwd),
        3: _unwrap_deg(trunk_lat),
        4: _unwrap_deg(trunk_axial),
        5: _unwrap_deg(sh_abd),
        6: _unwrap_deg(sh_plane),
        7: _unwrap_deg(sh_long_axis),
        8: np.rad2deg(elbow_flex),
    }

    # --- Event detection (robust) ---
    # Strategy: find BR first (wrist peak forward speed), then MER (shoulder
    # long-axis rotation extreme before BR), then SFC (lead ankle landing
    # before MER via fast-descent + settle).
    n = kn_flex.shape[0]
    wrist = (m(pitch_side + 'WRA') + m(pitch_side + 'WRB')) / 2  # meters
    # Wrist speed (3D magnitude of velocity), in m/s.
    # Use central diff; rate passed via markers dict (add if present)
    # Assume uniform sampling; we just use per-frame delta and let argmax work
    v = np.gradient(wrist, axis=0)             # (F,3) m/frame
    speed = np.linalg.norm(v, axis=1)          # m/frame
    # Also compute wrist lateral+forward 2D speed (ignore vertical) — more robust
    # against free-fall after release.
    # BR: global max of wrist speed (the trial is bounded so peak is pitch release).
    BR = int(np.argmax(speed))
    # Sanity: BR must not be in first 25% (that's windup) — if so, use max after 25%
    if BR < int(0.25 * n):
        BR = int(np.argmax(speed[int(0.25*n):]) + int(0.25*n))

    # MER tentative (refined after SFC detection): peak shoulder rot before BR
    baseline = vars_out[7][int(0.1*n):int(0.25*n)].mean()

    # SFC: prefer force-plate Fz > threshold (Fleisig 1995 / Diffendaffer 2023:
    # 20 N sustained on the lead foot's plate). Take the contact closest-before-BR,
    # because the stride (lead) foot always lands just before BR.
    SFC = None
    if force_plates is not None and force_plates.get('fz_list'):
        thr = force_plates.get('threshold_N', 20.0)
        arate = force_plates['analog_rate']
        mrate = force_plates['marker_rate']
        min_dur = int(arate * 0.05)   # 50 ms sustained
        BR_analog = int(BR / mrate * arate)
        # Search only in BR ± 0.5 s window (we want the LEAD foot landing, not
        # the rear-foot push-off that's active from the start of the trial).
        window_lo = max(0, BR_analog - int(arate * 0.5))
        window_hi = min(BR_analog + int(arate * 0.1),
                        len(force_plates['fz_list'][0]) - min_dur)
        # For each plate, find the FIRST rising edge within the window only.
        per_plate_first = []
        for fz in force_plates['fz_list']:
            above = fz > thr
            found = None
            for i in range(max(window_lo, min_dur), window_hi):
                fwd = above[i:i+min_dur]
                bwd = above[i-min_dur:i]
                if above[i] and fwd.mean() > 0.8 and bwd.mean() < 0.2:
                    found = i
                    break
            per_plate_first.append(found)
        valid = [c for c in per_plate_first if c is not None]
        if valid:
            # Lead-foot plate: rising edge closest (in absolute time) to BR,
            # since the stride foot lands ~100-300 ms before BR. This handles
            # both 2-plate (Vicon lab) and 3-plate (OBP mound) setups without
            # hardcoding plate identity.
            sfc_analog = min(valid, key=lambda x: abs(x - BR_analog))
            SFC = int(sfc_analog / arate * mrate)
    if SFC is None:
        # Marker-based fallback (previous behaviour)
        scan_lo = int(0.10 * n); scan_hi = min(BR, n-10)
        ank_y_full = ANK_L[:, 1] * 1000
        floor = ank_y_full[scan_lo:scan_hi].min() + 40
        candidates = []
        for i in range(max(scan_lo, 1), scan_hi):
            if ank_y_full[i] < floor and ank_y_full[i-1] >= floor:
                tail = ank_y_full[i:min(i+40, n)]
                if (tail < floor + 20).mean() > 0.6:
                    candidates.append(i)
        if candidates:
            SFC = candidates[-1]
        else:
            SFC = int(np.argmin(ank_y_full[scan_lo:BR]) + scan_lo)

    # MER = Maximum External Rotation = argmax of Var 7 in BR-0.5s window
    win_frames = max(50, min(int(BR * 0.6), 250))
    mer_lo = max(0, BR - win_frames)
    mer_hi = min(BR + 5, n - 1)
    MER = mer_lo + int(np.argmax(vars_out[7][mer_lo:mer_hi+1]))

    events = {'SFC': int(SFC), 'MER': int(MER), 'BR': int(BR)}
    return vars_out, events
