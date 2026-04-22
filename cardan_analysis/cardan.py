"""
Driveline / ASMI style Y-X-Z Cardan angle computation from Vicon marker data.

Option A (strict): with HJC via Newington regression, uses Rajagopal-style JCs.
Option B (simplified): lateral markers only.

Both use the same angle decomposition. The difference is only in joint-center
computation (affects thigh/shank proximal origin, not axis directions because
we rely on well-defined lateral vectors from neighboring segments).

Output: dict keys 1..8 for the 8 pitching variables, all in degrees.
"""
from __future__ import annotations
import numpy as np


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n < 1e-9, 1, n)


def ortho_frame(Xhint, Yhint):
    """Right-handed rotation matrix (..., 3, 3) with columns = X, Y, Z."""
    X = unit(Xhint)
    Z = unit(np.cross(X, Yhint))
    Y = np.cross(Z, X)
    return np.stack([X, Y, Z], axis=-1)


def yxz_cardan(R):
    """R = Ry(a) Rx(b) Rz(c) → (a, b, c) rad."""
    a = np.arctan2(R[..., 0, 2], R[..., 2, 2])
    b = -np.arcsin(np.clip(R[..., 1, 2], -1, 1))
    c = np.arctan2(R[..., 1, 0], R[..., 1, 1])
    return a, b, c


def zxy_cardan(R):
    """R = Rz(c) Rx(b) Ry(a) → (axial=c, lateral=b, forward=a) rad."""
    beta = np.arcsin(np.clip(R[..., 2, 1], -1, 1))
    alpha = np.arctan2(-R[..., 2, 0], R[..., 2, 2])
    gamma = np.arctan2(-R[..., 0, 1], R[..., 1, 1])
    return gamma, beta, alpha


# -------- Segment frames (convention: X right, Y up, Z forward) --------

def pelvis_frame(LASI, RASI, LPSI, RPSI):
    origin = (LASI + RASI) / 2
    Xhint = RASI - LASI                      # right
    Zhint = origin - (LPSI + RPSI) / 2       # forward
    Yhint = np.cross(Zhint, Xhint)           # up
    return ortho_frame(Xhint, Yhint), origin


def thorax_frame(C7, CLAV, T10, STRN, Xhint_shoulder):
    origin = (C7 + CLAV) / 2
    Yhint = origin - (T10 + STRN) / 2
    return ortho_frame(Xhint_shoulder, Yhint), origin


def _project_perp(Xhint, Yaxis):
    Yunit = unit(Yaxis)
    return Xhint - (Xhint * Yunit).sum(-1, keepdims=True) * Yunit


def humerus_frame_isb(shoulder_JC, elbow_JC, wrist_JC):
    """ISB 2005 humerus frame using elbow flexion axis approximation via forearm.

    NOTE: this creates circular dependency — humerus axes depend on forearm, so
    elbow flexion gets suppressed. Use humerus_frame_technical instead.
    """
    origin = shoulder_JC
    Ylong = elbow_JC - shoulder_JC
    forearm_long = wrist_JC - elbow_JC
    Zflex = np.cross(Ylong, forearm_long)
    Xhint = np.cross(Ylong, Zflex)
    return ortho_frame(Xhint, Ylong), origin


def humerus_frame_technical(shoulder_JC, elbow_JC, upper_arm_cluster):
    """Technical humerus frame using upper-arm cluster marker (e.g. RUPA/LUPA).

    Y = long axis (shoulder → elbow, distal-pointing).
    A lateral-ish vector from shoulder to the cluster marker gives an X hint
    independent of the forearm, so elbow flexion DOF is preserved.
    """
    origin = shoulder_JC
    Ylong = elbow_JC - shoulder_JC
    Xhint = upper_arm_cluster - shoulder_JC    # any non-colinear vector
    Xperp = _project_perp(Xhint, Ylong)
    return ortho_frame(Xperp, Ylong), origin


def humerus_frame(shoulder_JC, elbow_JC, Xhint_lat):
    """Legacy thorax-X based humerus frame (kept as fallback)."""
    origin = shoulder_JC
    Ylong = elbow_JC - shoulder_JC
    Xperp = _project_perp(Xhint_lat, Ylong)
    return ortho_frame(Xperp, Ylong), origin


def forearm_frame(elbow_JC, wrist_JC, radial_styloid, ulnar_styloid):
    origin = elbow_JC
    Ylong = wrist_JC - elbow_JC
    Xhint = radial_styloid - ulnar_styloid
    return ortho_frame(Xhint, Ylong), origin


def thigh_frame(hip_JC, knee_JC, Xhint_lat):
    origin = hip_JC
    Ylong = knee_JC - hip_JC
    Xperp = _project_perp(Xhint_lat, Ylong)
    return ortho_frame(Xperp, Ylong), origin


def shank_frame(knee_JC, ankle_JC, Xhint_lat):
    origin = knee_JC
    Ylong = ankle_JC - knee_JC
    Xperp = _project_perp(Xhint_lat, Ylong)
    return ortho_frame(Xperp, Ylong), origin


# -------- Joint centers --------

def hip_jc_newington(LASI, RASI, LPSI, RPSI, side='R'):
    R_pel, origin = pelvis_frame(LASI, RASI, LPSI, RPSI)
    d = np.linalg.norm(RASI - LASI, axis=-1, keepdims=True)
    sign = +1 if side == 'R' else -1
    offset_local = np.concatenate(
        [sign * 0.36 * d, -0.19 * d, -0.30 * d], axis=-1)
    return origin + np.einsum('...ij,...j->...i', R_pel, offset_local)


# -------- Main --------

def compute_angles(markers, opt='A', pitch_side='R', lead_side='L'):
    """
    markers: dict of (F, 3) arrays in mm, Y-up.
    opt: 'A' (Newington HJC) | 'B' (simplified ASI-PSI mid).
    """
    def m(n): return markers[n] / 1000.0

    LASI, RASI, LPSI, RPSI = m('LASI'), m('RASI'), m('LPSI'), m('RPSI')
    C7, CLAV, T10, STRN = m('C7'), m('CLAV'), m('T10'), m('STRN')
    RSHO, LSHO = m('RSHO'), m('LSHO')
    SHO_P = RSHO if pitch_side == 'R' else LSHO
    ELB_P = m(pitch_side + 'ELB')
    WRA = m(pitch_side + 'WRA'); WRB = m(pitch_side + 'WRB')
    KNE_L = m(lead_side + 'KNE'); ANK_L = m(lead_side + 'ANK')

    if opt == 'A':
        HJC_L = hip_jc_newington(LASI, RASI, LPSI, RPSI, lead_side)
    else:
        ASI = m(lead_side + 'ASI'); PSI = m(lead_side + 'PSI')
        HJC_L = (ASI + PSI) / 2

    R_pel, _ = pelvis_frame(LASI, RASI, LPSI, RPSI)
    Xhint_thorax = RSHO - LSHO
    R_thor, _ = thorax_frame(C7, CLAV, T10, STRN, Xhint_thorax)
    # Technical humerus frame: use upper-arm cluster marker if available
    WRIST_P = (WRA + WRB) / 2
    upa_name = pitch_side + 'UPA'
    if upa_name in markers:
        UPA = m(upa_name)
        R_hum, _ = humerus_frame_technical(SHO_P, ELB_P, UPA)
    else:
        R_hum, _ = humerus_frame_isb(SHO_P, ELB_P, WRIST_P)
    R_fore, _ = forearm_frame(ELB_P, WRIST_P, WRA, WRB)
    pel_X = R_pel[..., :, 0]
    R_thigh, _ = thigh_frame(HJC_L, KNE_L, pel_X)
    thigh_X = R_thigh[..., :, 0]
    R_shank, _ = shank_frame(KNE_L, ANK_L, thigh_X)

    def Rdot(Rpar, Rchi):
        return np.einsum('...ji,...jk->...ik', Rpar, Rchi)

    R_sh = Rdot(R_thor, R_hum)
    R_el = Rdot(R_hum, R_fore)
    R_kn = Rdot(R_thigh, R_shank)
    R_tr = Rdot(R_pel, R_thor)

    sh_elev_plane, sh_abd, sh_rot = yxz_cardan(R_sh)
    _, el_flex, _ = yxz_cardan(R_el)
    _, kn_flex, _ = yxz_cardan(R_kn)
    tr_axial, tr_lat, tr_fwd = zxy_cardan(R_tr)

    d = np.rad2deg
    return {
        1: d(sh_rot),            # shoulder int/ext rotation
        2: d(sh_elev_plane),     # horizontal add/abd (elevation plane)
        3: d(sh_abd),            # abd/add (elevation)
        4: d(kn_flex),           # lead knee flex
        5: d(el_flex),           # elbow flex
        6: d(tr_axial),          # trunk axial
        7: d(tr_lat),            # trunk lateral
        8: d(tr_fwd),            # trunk forward
    }
