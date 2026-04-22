"""Export each OpenSim model as GLB + muscles.json + markers.json for three.js viewer.

Writes to: opensim_viewer/<tag>/{bones.glb, muscles.json, markers.json, meta.json}
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import opensim as osim
import trimesh

ROOT = Path("/Users/choejaehyeog/3dgs_to_gart_textbook/.claude/worktrees/adoring-curie-b5a7b5")
OUT = ROOT / "opensim_viewer"
OUT.mkdir(exist_ok=True)

MODELS = {
    "fbls":           "/Volumes/T31/opensim/fullbodylumbar/FBLSmodel.osim",
    "merged_v0":      "/Users/choejaehyeog/3dgs_to_gart_textbook/.claude/worktrees/adoring-curie-b5a7b5/merged_v0.osim",
    "rajagopal_mj":   "/Volumes/T31/opensim/Rajagopal2015MJPlugInGait_43_2.2.osim",
    "rajagopal_2016": "/Users/choejaehyeog/Documents/OpenSim/4.5/Models/Rajagopal/Rajagopal2016.osim",
    "mobl_bimanual":  "/Volumes/T31/opensim/Bimanual Upper Arm Model/MoBL_ARMS_bimanual_6_2_21.osim",
    "mobl_41":        "/Volumes/T31/opensim/MoBL-ARMS Upper Extremity Model/Model/4.1/MOBL_ARMS_41.osim",
}

# Where to look for geometry when the .osim uses bare filenames
GEOMETRY_SEARCH_DIRS = [
    "/Volumes/T31/opensim/Geometry",
    "/Volumes/T31/opensim/Bimanual Upper Arm Model/Geometry",
    "/Volumes/T31/opensim/MoBL-ARMS Upper Extremity Model/Benchmarking Simulations/4.1 Model with Millard/Geometry",
    "/Volumes/T31/opensim/ULB_Project/Geometry",
    "/Users/choejaehyeog/Documents/OpenSim/4.5/Models/Rajagopal/Geometry",
    "/Users/choejaehyeog/Documents/OpenSim/4.5/Geometry",
    "/Applications/OpenSim 4.5/Geometry",
    "/Applications/OpenSim 4.5/OpenSim 4.5.app/Contents/Resources/opensim/Geometry",
]


def find_mesh(rel: str, model_path: str) -> str | None:
    cand = [
        os.path.join(os.path.dirname(model_path), rel),
        os.path.join(os.path.dirname(model_path), "Geometry", rel),
    ] + [os.path.join(d, rel) for d in GEOMETRY_SEARCH_DIRS]
    for p in cand:
        if os.path.isfile(p):
            return p
    # basename fallback
    base = os.path.basename(rel)
    for d in GEOMETRY_SEARCH_DIRS:
        p = os.path.join(d, base)
        if os.path.isfile(p):
            return p
    return None


def simbody_mat_to_np(T) -> np.ndarray:
    """Simbody Transform -> 4x4 numpy (row-major)."""
    R = T.R()
    p = T.p()
    M = np.eye(4)
    for i in range(3):
        for j in range(3):
            M[i, j] = R.asMat33().get(i, j)
        M[i, 3] = p.get(i)
    return M


def body_world_transform(body: osim.Body, state) -> np.ndarray:
    T = body.getTransformInGround(state)
    return simbody_mat_to_np(T)


def load_vtp_as_trimesh(path: str) -> trimesh.Trimesh | None:
    """Read a VTK PolyData XML file into a trimesh. Uses pyvista for robustness."""
    try:
        import pyvista as pv
        m = pv.read(path)
        # Always triangulate first — handles mixed cell types and the n_faces API change.
        tri = m.triangulate()
        faces_arr = np.asarray(tri.faces)
        if faces_arr.size == 0:
            return None
        faces = faces_arr.reshape(-1, 4)[:, 1:4]
        verts = np.asarray(tri.points)
        return trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    except Exception as e:  # noqa: BLE001
        print(f"   ! vtp read failed ({path}): {e}", flush=True)
        return None


def export_model(tag: str, path: str) -> dict:
    print(f"\n=== {tag} ===")
    model = osim.Model(path)
    state = model.initSystem()
    bodyset = model.getBodySet()

    scene = trimesh.Scene()
    body_info = []

    # Iterate bodies, attach each mesh with world transform
    for bi in range(bodyset.getSize()):
        body = bodyset.get(bi)
        bname = body.getName()
        T_world = body_world_transform(body, state)

        # Iterate attached_geometry property
        ag_prop = body.getPropertyByName("attached_geometry")
        n_attached = ag_prop.size()
        meshes_attached = 0
        for gi in range(n_attached):
            comp = osim.Mesh.safeDownCast(body.get_attached_geometry(gi))
            if comp is None:
                continue
            rel = comp.get_mesh_file()
            if not rel:
                continue
            mesh_path = find_mesh(rel, path)
            if mesh_path is None:
                print(f"   [{bname}] missing mesh: {rel}")
                continue
            tm = load_vtp_as_trimesh(mesh_path)
            if tm is None or len(tm.vertices) == 0:
                continue
            # Apply scale_factors if set
            try:
                sf = comp.get_scale_factors()
                S = np.diag([sf.get(0), sf.get(1), sf.get(2), 1.0])
            except Exception:
                S = np.eye(4)
            # Local frame offset relative to body frame
            try:
                # attached_geometry has its own socket to a frame; if same as body, offset=identity
                offset_frame = comp.getFrame()
                T_offset = simbody_mat_to_np(offset_frame.findTransformBetween(
                    state, body))
            except Exception:
                T_offset = np.eye(4)
            M = T_world @ T_offset @ S
            tm.apply_transform(M)
            node_name = f"{bname}__{gi}"
            scene.add_geometry(tm, node_name=node_name, geom_name=node_name)
            meshes_attached += 1
        body_info.append({"name": bname, "meshes": meshes_attached,
                          "pos": T_world[:3, 3].tolist()})

    # --- muscles (path line at default pose) ---
    muscles_out = []
    forces = model.getForceSet()
    for fi in range(forces.getSize()):
        f = forces.get(fi)
        mus = osim.Muscle.safeDownCast(f)
        if mus is None:
            continue
        gp = mus.getGeometryPath()
        try:
            pts = gp.getCurrentPath(state)
        except Exception:
            continue
        line = []
        for k in range(pts.getSize()):
            pp = pts.get(k)
            loc = pp.getLocation(state)  # in parent frame
            frame = pp.getParentFrame()
            # get world position
            T = simbody_mat_to_np(frame.getTransformInGround(state))
            p_local = np.array([loc.get(0), loc.get(1), loc.get(2), 1.0])
            p_world = (T @ p_local)[:3]
            line.append(p_world.tolist())
        if len(line) >= 2:
            muscles_out.append({"name": mus.getName(), "points": line})

    # --- markers ---
    markers_out = []
    ms = model.getMarkerSet()
    for mi in range(ms.getSize()):
        m = ms.get(mi)
        p = m.getLocationInGround(state)
        markers_out.append({
            "name": m.getName(),
            "pos": [p.get(0), p.get(1), p.get(2)],
            "body": m.getParentFrameName().replace("/bodyset/", ""),
        })

    # --- write ---
    model_out = OUT / tag
    model_out.mkdir(exist_ok=True)
    glb_path = model_out / "bones.glb"
    scene.export(glb_path.as_posix())
    (model_out / "muscles.json").write_text(json.dumps(muscles_out))
    (model_out / "markers.json").write_text(json.dumps(markers_out))
    (model_out / "meta.json").write_text(json.dumps({
        "tag": tag,
        "name": model.getName(),
        "bodies": body_info,
        "n_muscles": len(muscles_out),
        "n_markers": len(markers_out),
    }, indent=2))

    print(f"   bodies meshed: {sum(b['meshes'] for b in body_info)}, muscles: {len(muscles_out)}, markers: {len(markers_out)}")
    print(f"   wrote {glb_path}")
    return {"tag": tag, "n_muscles": len(muscles_out), "n_markers": len(markers_out),
            "n_bodies": len(body_info)}


def main() -> int:
    summary = []
    for tag, p in MODELS.items():
        try:
            summary.append(export_model(tag, p))
        except Exception as e:  # noqa: BLE001
            print(f"   FAILED: {e}")
            import traceback; traceback.print_exc()
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nDone:", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
