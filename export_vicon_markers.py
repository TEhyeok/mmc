"""
Vicon c3d의 정적 프레임 마커 위치를 JSON으로 export.

- 투구 시작 전 quiet standing 구간 (wrist 속도 최소) 평균 사용
- 단위: mm → m 변환 (OpenSim 호환)
- 좌표: 원본 유지 (Y-up Vicon)
- 출력: opensim_viewer/vicon_data/<subject>_<trial>_static.json

결과:
  {
    "meta": { "subject": "subject_1", "trial": "set_002",
              "fps": 240, "static_frame": 23, "n_markers": 39 },
    "markers": {
      "C7":    [x, y, z],   # meters
      "CLAV":  [x, y, z],
      ...
    }
  }
"""
from __future__ import annotations
import json
import glob
import os
import sys
import numpy as np
import ezc3d

# ---------- 필터 ----------
# 우리 Vicon PIG (+ 추가) 해부학적 마커 화이트리스트
PIG_ANATOMICAL_MARKERS = {
    # 머리
    'LFHD', 'RFHD', 'LBHD', 'RBHD',
    # 체간
    'C7', 'T10', 'CLAV', 'STRN', 'RBAK',
    # 어깨/팔
    'LSHO', 'RSHO', 'LUPA', 'RUPA',
    'LELB', 'RELB', 'LFRM', 'RFRM',
    'LWRA', 'RWRA', 'LWRB', 'RWRB',
    'LFIN', 'RFIN',
    # 골반
    'LASI', 'RASI', 'LPSI', 'RPSI',
    # 대전자 (우리 데이터에 추가로 있음)
    'LGT', 'RGT',
    # 다리
    'LTHI', 'RTHI', 'LKNE', 'RKNE',
    'LTIB', 'RTIB', 'LANK', 'RANK',
    'LHEE', 'RHEE', 'LTOE', 'RTOE',
}


def is_marker(label: str) -> bool:
    """Keep only real anatomical PIG markers. Explicitly reject virtual
    segment frames (PELO/PELA/... segment_[O|A|L|P]) and all derived signals
    (Angles/Forces/Moments/Power/CentreOfMass)."""
    return label in PIG_ANATOMICAL_MARKERS


def find_static_frame(markers: dict, n_smooth: int = 30,
                      search_limit: int = 200) -> int:
    """Find the quietest frame in the EARLY portion of the trial (pre-windup).

    Uses combined wrist + knee speed to avoid confusing between-pitch pauses
    (which happen mid-trial) with actual quiet standing.
    """
    # Combine multiple markers for robustness
    keys = ['RWRA', 'LWRA', 'RKNE', 'LKNE']
    speeds = []
    for k in keys:
        if k not in markers:
            continue
        arr = markers[k]
        v = np.diff(arr, axis=0)
        s = np.linalg.norm(v, axis=1)
        s = np.nan_to_num(s, nan=1e9)
        speeds.append(s)
    if not speeds:
        return 0
    total = np.sum(speeds, axis=0)
    n = len(total)
    limit = min(search_limit, n)
    limit = max(limit, n_smooth + 5)
    rolling = np.convolve(total[:limit], np.ones(n_smooth) / n_smooth, mode='valid')
    return int(np.argmin(rolling) + n_smooth // 2)


def export_one_trial(c3d_path: str, out_dir: str,
                     subject: str, trial: str,
                     static_window: int = 20) -> str:
    c = ezc3d.c3d(c3d_path)
    labels = c['parameters']['POINT']['LABELS']['value']
    pts = c['data']['points']  # (4, N, F) mm, Y-up

    # Marker dict (world, mm)
    markers_mm = {lbl: pts[:3, i, :].T for i, lbl in enumerate(labels)
                  if is_marker(lbl)}

    fps = float(c['header']['points']['frame_rate'])
    static_idx = find_static_frame(markers_mm)

    lo = max(0, static_idx - static_window // 2)
    hi = min(pts.shape[2], static_idx + static_window // 2 + 1)

    # 각 마커별 윈도우 평균 (NaN 제외), mm → m
    out_markers = {}
    for name, arr in markers_mm.items():
        win = arr[lo:hi]
        valid = ~np.isnan(win).any(axis=1)
        if valid.sum() == 0:
            continue
        mean_mm = np.nanmean(win, axis=0)
        if np.any(np.isnan(mean_mm)):
            continue
        out_markers[name] = (mean_mm / 1000.0).tolist()

    out = {
        "meta": {
            "subject": subject,
            "trial": trial,
            "c3d_path": c3d_path,
            "fps": fps,
            "static_frame": static_idx,
            "static_window": [lo, hi],
            "units": "m",
            "coordinate": "Vicon world (Y-up)",
            "n_markers": len(out_markers),
        },
        "markers": out_markers,
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{subject}_{trial}_static.json")
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  [{subject}/{trial}] static frame={static_idx} window=[{lo},{hi}] markers={len(out_markers)} → {out_path}")
    return out_path


def main():
    DATA_ROOT = '/Users/choejaehyeog/3dgs_to_gart_textbook/data/markerbased/subject_1'
    OUT_DIR = '/Users/choejaehyeog/3dgs_to_gart_textbook/.claude/worktrees/adoring-curie-b5a7b5/opensim_viewer/vicon_data'
    trials = sorted(glob.glob(os.path.join(DATA_ROOT, 'set_*', 'set_*.c3d')))
    print(f"Found {len(trials)} trials under {DATA_ROOT}")
    for c3d_path in trials:
        trial = os.path.basename(c3d_path).replace('.c3d', '')
        subject = os.path.basename(os.path.dirname(os.path.dirname(c3d_path)))
        export_one_trial(c3d_path, OUT_DIR, subject, trial)


if __name__ == '__main__':
    main()
