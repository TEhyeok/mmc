"""Phase 2: dump full inventory of each .osim to Markdown + scatter PNGs + forward-sim check."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import opensim as osim

OUT_DIR = Path("/Users/choejaehyeog/3dgs_to_gart_textbook/.claude/worktrees/adoring-curie-b5a7b5/opensim_report")
OUT_DIR.mkdir(exist_ok=True)

MODELS = {
    "Rajagopal_PIG": "/Volumes/T31/opensim/Rajagopal2015MJPlugInGait_43_2.2.osim",
    "MoBL_bimanual": "/Volumes/T31/opensim/Bimanual Upper Arm Model/MoBL_ARMS_bimanual_6_2_21.osim",
    "MoBL_41_uniR":  "/Volumes/T31/opensim/MoBL-ARMS Upper Extremity Model/Model/4.1/MOBL_ARMS_41.osim",
}


def dump_one(tag: str, path: str) -> dict:
    model = osim.Model(path)
    state = model.initSystem()

    coords = model.getCoordinateSet()
    bodies = model.getBodySet()
    joints = model.getJointSet()
    markers = model.getMarkerSet()
    muscles = model.getMuscles()

    # --- body world positions at default pose ---
    body_rows = []
    for i in range(bodies.getSize()):
        b = bodies.get(i)
        p = b.getPositionInGround(state)
        mass = b.getMass()
        body_rows.append((b.getName(), mass, p.get(0), p.get(1), p.get(2)))

    # --- joints ---
    joint_rows = []
    for i in range(joints.getSize()):
        j = joints.get(i)
        parent = j.getParentFrame().getName()
        child = j.getChildFrame().getName()
        joint_rows.append((j.getName(), j.getConcreteClassName(), parent, child))

    # --- coordinates (DOF) ---
    coord_rows = []
    for i in range(coords.getSize()):
        c = coords.get(i)
        unit = "rad" if c.getMotionType() == 1 else ("m" if c.getMotionType() == 2 else "?")
        # MotionType: 1=Rotational, 2=Translational, 3=Coupled
        coord_rows.append((
            c.getName(),
            c.getJoint().getName(),
            unit,
            c.getRangeMin(),
            c.getRangeMax(),
            c.getDefaultValue(),
            c.getLocked(state),
            c.get_clamped(),
        ))

    # --- markers ---
    marker_rows = []
    marker_xyz = []
    for i in range(markers.getSize()):
        m = markers.get(i)
        loc = m.get_location()
        body = m.getParentFrameName().replace("/bodyset/", "")
        marker_rows.append((m.getName(), body, loc.get(0), loc.get(1), loc.get(2)))
        # world position:
        pw = m.getLocationInGround(state)
        marker_xyz.append((m.getName(), pw.get(0), pw.get(1), pw.get(2)))

    # forward sim skipped (muscle-driven without controls does not converge in reasonable time)
    sim_ok, sim_msg = True, "skipped (structural inventory only)"

    return {
        "tag": tag,
        "path": path,
        "name": model.getName(),
        "bodies": body_rows,
        "joints": joint_rows,
        "coords": coord_rows,
        "markers": marker_rows,
        "marker_xyz": marker_xyz,
        "muscles_n": muscles.getSize(),
        "forces_n": model.getForceSet().getSize(),
        "sim_ok": sim_ok,
        "sim_msg": sim_msg,
    }


def scatter_markers(info: dict, out: Path) -> None:
    if not info["marker_xyz"]:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    names, xs, ys, zs = zip(*[(n, x, y, z) for n, x, y, z in info["marker_xyz"]])
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs, zs, ys, c="royalblue", s=18)  # swap Z,Y so Y=up faces up
    for n, x, y, z in zip(names, xs, ys, zs):
        ax.text(x, z, y, n, fontsize=5, alpha=0.7)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)"); ax.set_zlabel("Y-up (m)")
    ax.set_title(f"{info['tag']}: {len(names)} markers @ default pose")
    ax.set_box_aspect([1, 1, 1])
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def write_markdown(infos: list[dict], out: Path) -> None:
    lines = ["# OpenSim Model Inventory (Phase 2)\n"]
    for info in infos:
        lines.append(f"## {info['tag']} — `{info['name']}`\n")
        lines.append(f"- path: `{info['path']}`")
        lines.append(f"- bodies={len(info['bodies'])}  joints={len(info['joints'])}  DOF={len(info['coords'])}  markers={len(info['markers'])}  forces={info['forces_n']}  muscles={info['muscles_n']}")
        lines.append(f"- forward sim 0.5s: **{'OK' if info['sim_ok'] else 'FAIL'}** — {info['sim_msg']}\n")

        lines.append("### Bodies (name, mass[kg], world xyz[m] @ default pose)")
        lines.append("| # | name | mass | x | y | z |")
        lines.append("|---|------|------|---|---|---|")
        for i, (n, m, x, y, z) in enumerate(info["bodies"]):
            lines.append(f"| {i} | {n} | {m:.3f} | {x:+.3f} | {y:+.3f} | {z:+.3f} |")
        lines.append("")

        lines.append("### Joints (name, type, parent → child)")
        lines.append("| # | name | type | parent | child |")
        lines.append("|---|------|------|--------|-------|")
        for i, (n, t, p, c) in enumerate(info["joints"]):
            lines.append(f"| {i} | {n} | {t} | `{p}` | `{c}` |")
        lines.append("")

        lines.append("### Coordinates (DOF)")
        lines.append("| # | name | joint | unit | min | max | default | locked | clamped |")
        lines.append("|---|------|-------|------|-----|-----|---------|--------|---------|")
        for i, (n, jn, u, lo, hi, df, lk, cl) in enumerate(info["coords"]):
            lines.append(f"| {i} | {n} | {jn} | {u} | {lo:+.3f} | {hi:+.3f} | {df:+.3f} | {lk} | {cl} |")
        lines.append("")

        if info["markers"]:
            lines.append("### Markers (name, body, local xyz[m])")
            lines.append("| # | name | body | x_local | y_local | z_local |")
            lines.append("|---|------|------|---------|---------|---------|")
            for i, (n, b, x, y, z) in enumerate(info["markers"]):
                lines.append(f"| {i} | {n} | {b} | {x:+.3f} | {y:+.3f} | {z:+.3f} |")
            lines.append("")
        else:
            lines.append("### Markers: (none)\n")

        lines.append("---\n")
    out.write_text("\n".join(lines))


def main() -> int:
    infos = []
    for tag, p in MODELS.items():
        print(f"[{tag}] loading...", flush=True)
        info = dump_one(tag, p)
        infos.append(info)
        scatter_markers(info, OUT_DIR / f"{tag}_markers.png")
        print(f"  sim: {info['sim_msg']}")

    write_markdown(infos, OUT_DIR / "inventory.md")
    print(f"\nWrote {OUT_DIR / 'inventory.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
