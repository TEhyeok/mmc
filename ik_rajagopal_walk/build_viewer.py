"""
Scaled Rajagopal 2016 model + ik.mot → 웹뷰어 애니메이션 산출물

출력: opensim_viewer/rajagopal_2016_walk_anim/
  - bones.glb  (각 body = 하나의 node, name = body name, 내부 mesh = local frame)
  - animation.json  (row-major 4x4 per body per frame)
  - markers_anim.json  (64 marker positions per frame)
  - meta.json
"""
import json
import os
from pathlib import Path

import numpy as np
import opensim as osim
import trimesh

HERE = Path(__file__).parent
MODEL = HERE / 'scaled_model.osim'
MOT   = HERE / 'ik.mot'
OUT   = HERE.parent / 'opensim_viewer' / 'rajagopal_2016_walk_anim'
OUT.mkdir(parents=True, exist_ok=True)

TARGET_FPS = 30

# ---------------- Geometry lookup ----------------
GEOMETRY_DIRS = [
    '/Applications/OpenSim 4.5/Geometry',
    '/Applications/OpenSim 4.5/OpenSim 4.5.app/Contents/Resources/opensim/Geometry',
    '/Users/choejaehyeog/Documents/OpenSim/4.5/Geometry',
    '/Users/choejaehyeog/Documents/OpenSim/4.5/Models/Rajagopal/Geometry',
    '/Users/choejaehyeog/miniconda3/envs/opensim45/share/opensim/Geometry',
    '/Users/choejaehyeog/miniconda3/envs/opensim45/share/OpenSim/Geometry',
]


def find_mesh_path(rel: str) -> str | None:
    base = os.path.basename(rel)
    # 1) model directory + relative
    m_dir = str(MODEL.parent)
    cands = [
        os.path.join(m_dir, rel),
        os.path.join(m_dir, 'Geometry', rel),
        os.path.join(m_dir, 'Geometry', base),
    ]
    cands += [os.path.join(d, rel) for d in GEOMETRY_DIRS]
    cands += [os.path.join(d, base) for d in GEOMETRY_DIRS]
    for p in cands:
        if os.path.isfile(p):
            return p
    return None


def load_mesh(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.vtp':
        try:
            import pyvista as pv
            m = pv.read(path).triangulate()
            faces = np.asarray(m.faces)
            if faces.size == 0:
                return None
            faces = faces.reshape(-1, 4)[:, 1:4]
            verts = np.asarray(m.points)
            return trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        except Exception:
            return None
    try:
        m = trimesh.load_mesh(path, force='mesh')
        return m if not m.is_empty else None
    except Exception:
        return None


def simbody_T_to_np(T) -> np.ndarray:
    R, p = T.R(), T.p()
    M = np.eye(4)
    for i in range(3):
        for j in range(3):
            M[i, j] = R.get(i, j)
        M[i, 3] = p.get(i)
    return M


# ---------------- Load model ----------------
print(f'Loading model: {MODEL}')
model = osim.Model(str(MODEL))
state = model.initSystem()
body_set = model.getBodySet()
coord_set = model.getCoordinateSet()
mk_set = model.getMarkerSet()

body_names = [body_set.get(i).getName() for i in range(body_set.getSize())]
mk_names   = [mk_set.get(i).getName() for i in range(mk_set.getSize())]
print(f'  bodies: {len(body_names)}, markers: {len(mk_names)}, DoF: {model.getNumCoordinates()}')

# ---------------- Load IK .mot ----------------
print(f'Loading IK: {MOT}')
tbl = osim.TimeSeriesTable(str(MOT))
in_degrees = tbl.getTableMetaDataAsString('inDegrees') == 'yes'
col_names = list(tbl.getColumnLabels())
times = np.array([tbl.getIndependentColumn()[i] for i in range(tbl.getNumRows())])
n_in = len(times)
fps_in = (n_in - 1) / (times[-1] - times[0])
stride = max(1, int(round(fps_in / TARGET_FPS)))
frames = list(range(0, n_in, stride))
print(f'  input: {n_in} frames @ {fps_in:.1f} Hz → stride {stride} → {len(frames)} frames')

# Pre-fetch columns for coords that exist in model
model_coord_names = [coord_set.get(i).getName() for i in range(coord_set.getSize())]
active = []
for name in model_coord_names:
    if name in col_names:
        v = tbl.getDependentColumn(name).to_numpy()
        active.append((name, v))
print(f'  matched: {len(active)} / {len(model_coord_names)} coordinates')

# ---------------- Step through frames ----------------
DEG = np.pi / 180.0
transforms = {b: [] for b in body_names}
marker_positions = []

for step, ri in enumerate(frames):
    for name, v in active:
        val = float(v[ri])
        c = coord_set.get(name)
        if in_degrees and c.getMotionType() == osim.Coordinate.Rotational:
            val = val * DEG
        c.setValue(state, val)
    model.realizePosition(state)

    for i, bn in enumerate(body_names):
        T = simbody_T_to_np(body_set.get(i).getTransformInGround(state))
        transforms[bn].append(T.flatten().tolist())

    row = []
    for i in range(mk_set.getSize()):
        p = mk_set.get(i).getLocationInGround(state)
        row.append([p.get(0), p.get(1), p.get(2)])
    marker_positions.append(row)

times_out = [float(times[i]) for i in frames]
duration = times_out[-1] - times_out[0]

# ---------------- Save animation.json ----------------
with open(OUT / 'animation.json', 'w') as f:
    json.dump({
        'n_frames': len(frames),
        'duration': duration,
        'times': times_out,
        'bodies': body_names,
        'transforms': transforms,
    }, f)
print(f'  animation.json: {os.path.getsize(OUT / "animation.json"):,} bytes')

# ---------------- Save markers_anim.json ----------------
with open(OUT / 'markers_anim.json', 'w') as f:
    json.dump({'names': mk_names, 'positions': marker_positions}, f)
print(f'  markers_anim.json: {os.path.getsize(OUT / "markers_anim.json"):,} bytes')

# ---------------- Save meta.json ----------------
with open(OUT / 'meta.json', 'w') as f:
    json.dump({
        'tag': 'rajagopal_2016_walk_anim',
        'name': 'Rajagopal2016 walking example',
        'n_frames': len(frames),
        'duration': duration,
        'fps': len(frames) / duration,
        'n_bodies': len(body_names),
        'n_markers': len(mk_names),
    }, f, indent=2)

# ---------------- Build bones.glb ----------------
# Fresh model at default pose
fresh_model = osim.Model(str(MODEL))
fresh_state = fresh_model.initSystem()
fresh_bodies = fresh_model.getBodySet()

scene = trimesh.Scene()
for i in range(fresh_bodies.getSize()):
    body = fresh_bodies.get(i)
    bname = body.getName()

    pieces = []
    ag = body.getPropertyByName('attached_geometry')
    for c_idx in range(ag.size()):
        comp = body.get_attached_geometry(c_idx)
        mesh_comp = osim.Mesh.safeDownCast(comp)
        if mesh_comp is None:
            continue
        mesh_file = mesh_comp.get_mesh_file()
        path = find_mesh_path(mesh_file)
        if not path:
            continue
        m = load_mesh(path)
        if m is None:
            continue
        # scale factors
        sv = mesh_comp.get_scale_factors()
        sx, sy, sz = sv.get(0), sv.get(1), sv.get(2)
        if (sx, sy, sz) != (1.0, 1.0, 1.0):
            S = np.diag([sx, sy, sz, 1.0])
            m.apply_transform(S)
        # offset from attachment frame to body frame
        try:
            frame = comp.getFrame()
            T_off = simbody_T_to_np(
                frame.findTransformBetween(fresh_state, body)
            )
            m.apply_transform(T_off)
        except Exception:
            pass
        pieces.append(m)

    if not pieces:
        placeholder = trimesh.creation.icosphere(radius=0.001)
        scene.add_geometry(placeholder, node_name=bname, geom_name=bname)
        continue
    merged = trimesh.util.concatenate(pieces) if len(pieces) > 1 else pieces[0]
    scene.add_geometry(merged, node_name=bname, geom_name=bname)

glb_path = OUT / 'bones.glb'
scene.export(str(glb_path))
print(f'  bones.glb: {os.path.getsize(glb_path):,} bytes')

print()
print('=' * 60)
print('✅ Viewer artifacts saved:')
print(f'   {OUT}')
print('=' * 60)
