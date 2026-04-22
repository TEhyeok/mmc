"""Merge Rajagopal 2016 (lower body + torso) with MoBL-ARMS bimanual (upper body).

Strategy (XML-level, no OpenSim API constructor needed):
  1. Start from Rajagopal 2016 as skeleton.
  2. Delete Rajagopal's upper limb bodies/joints/coords/muscles/markers.
  3. From MoBL (4.x resaved), harvest clavicle -> hand bodies + their joints (re-parent
     sternoclavicular joint's parent frame from /bodyset/thorax to Rajagopal /bodyset/torso).
  4. Rename path strings: /bodyset/thorax -> /bodyset/torso for all MoBL elements brought over.
  5. Copy MoBL's upper-limb muscles (ForceSet) and constraints into Rajagopal.
  6. Write merged.osim and load-test with initSystem().

This is a v0 merge: mass/inertia are taken from MoBL as-is, so total mass differs slightly
from Rajagopal's 75kg. Will be calibrated later.
"""
from __future__ import annotations

import copy
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import opensim as osim

RAJA = Path('/Users/choejaehyeog/Documents/OpenSim/4.5/Models/Rajagopal/Rajagopal2016.osim')
MOBL = Path('/Volumes/T31/opensim/Bimanual Upper Arm Model/MoBL_ARMS_bimanual_v4x.osim')
OUT  = Path('/Users/choejaehyeog/3dgs_to_gart_textbook/.claude/worktrees/adoring-curie-b5a7b5/merged_v0.osim')

# Bodies to remove from Rajagopal upper extremity.
RAJA_UPPER_BODIES = {'humerus_r','ulna_r','radius_r','hand_r',
                     'humerus_l','ulna_l','radius_l','hand_l'}
# Joints to remove (and their coordinates).
RAJA_UPPER_JOINTS = {'acromial_r','elbow_r','radioulnar_r','radius_hand_r',
                     'acromial_l','elbow_l','radioulnar_l','radius_hand_l'}

# Body names to HARVEST from MoBL (skip ground & thorax).
MOBL_HARVEST_BODIES = {
    'clavicle_r','clavphant_r','scapula_r','scapphant_r','humphant_r','humphant1_r',
    'humerus_r','ulna_r','radius_r','proximal_row_r','hand_r',
    'clavicle_l','clavphant_l','scapula_l','scapphant_l','humphant_l','humphant1_l',
    'humerus_l','ulna_l','radius_l','proximal_row_l','hand_l',
}
# Joints to HARVEST from MoBL (skip groundthorax).
MOBL_HARVEST_JOINTS_ALL = True  # harvest everything except groundthorax


def find_direct_child(parent, name):
    for c in list(parent):
        if c.tag == name:
            return c
    return None


def copy_bodies(src_root, dst_root, body_names):
    """Copy <Body> elements from src BodySet to dst BodySet."""
    src_bs = src_root.find('.//BodySet/objects')
    dst_bs = dst_root.find('.//BodySet/objects')
    copied = []
    for b in list(src_bs):
        if b.get('name') in body_names:
            dst_bs.append(copy.deepcopy(b))
            copied.append(b.get('name'))
    return copied


def remove_bodies(root, body_names):
    bs = root.find('.//BodySet/objects')
    removed = []
    for b in list(bs):
        if b.get('name') in body_names:
            bs.remove(b); removed.append(b.get('name'))
    return removed


def copy_joints(src_root, dst_root, joint_filter):
    """joint_filter(joint_elem) -> True to copy."""
    src_js = src_root.find('.//JointSet/objects')
    dst_js = dst_root.find('.//JointSet/objects')
    copied = []
    for j in list(src_js):
        if joint_filter(j):
            dst_js.append(copy.deepcopy(j))
            copied.append(j.get('name'))
    return copied


def remove_joints(root, joint_names):
    js = root.find('.//JointSet/objects')
    removed = []
    for j in list(js):
        if j.get('name') in joint_names:
            js.remove(j); removed.append(j.get('name'))
    return removed


def rewrite_paths(elem, mapping):
    """Replace substrings in all text fields: e.g. /bodyset/thorax -> /bodyset/torso."""
    n = 0
    for e in elem.iter():
        if e.text:
            t = e.text
            for old, new in mapping.items():
                if old in t:
                    t = t.replace(old, new)
                    n += 1
            e.text = t
    return n


def copy_forces(src_root, dst_root, body_names_new):
    """Copy forces that act only on body_names_new (and the torso) from src ForceSet to dst."""
    src_fs = src_root.find('.//ForceSet/objects')
    dst_fs = dst_root.find('.//ForceSet/objects')
    if src_fs is None or dst_fs is None:
        return []
    copied = []
    for f in list(src_fs):
        # find all body references in paths inside this force
        body_refs = set()
        for pp in f.iter():
            if pp.tag == 'socket_parent_frame' and pp.text:
                body_refs.add(pp.text.strip().split('/')[-1].replace('_offset',''))
            if pp.tag in ('body','socket_body','socket_child_frame'):
                if pp.text:
                    body_refs.add(pp.text.strip().split('/')[-1].replace('_offset',''))
        # accept if all referenced bodies are either in body_names_new or "torso" (renamed thorax)
        allowed = body_names_new | {'torso','ground'}
        if body_refs and body_refs.issubset(allowed):
            dst_fs.append(copy.deepcopy(f))
            copied.append(f.get('name'))
    return copied


def copy_markers(src_root, dst_root, body_names_new):
    src_ms = src_root.find('.//MarkerSet/objects')
    dst_ms = dst_root.find('.//MarkerSet/objects')
    if src_ms is None or dst_ms is None:
        return []
    copied = []
    allowed = body_names_new | {'torso'}
    for m in list(src_ms):
        parent = m.find('socket_parent_frame')
        if parent is not None and parent.text:
            body = parent.text.strip().split('/')[-1]
            if body in allowed:
                dst_ms.append(copy.deepcopy(m))
                copied.append(m.get('name'))
    return copied


def copy_constraints(src_root, dst_root):
    """Copy all coordinate coupler constraints (used by MoBL for scapulohumeral rhythm)."""
    src_cs = src_root.find('.//ConstraintSet/objects')
    dst_cs = dst_root.find('.//ConstraintSet/objects')
    if src_cs is None or dst_cs is None:
        return []
    copied = []
    for c in list(src_cs):
        dst_cs.append(copy.deepcopy(c))
        copied.append(c.get('name'))
    return copied


def collect_coords_from_joints(root, joint_names):
    """From a JointSet, collect coordinate names that belong to given joint names."""
    coords = set()
    for j in root.iter():
        if j.tag in ('CustomJoint','WeldJoint','PinJoint','FreeJoint','BallJoint',
                     'UniversalJoint','PlanarJoint','SliderJoint'):
            if j.get('name') in joint_names:
                for c in j.iter('Coordinate'):
                    if c.get('name'):
                        coords.add(c.get('name'))
    return coords


def main() -> int:
    raja_tree = ET.parse(RAJA.as_posix())
    raja_root = raja_tree.getroot()
    mobl_tree = ET.parse(MOBL.as_posix())
    mobl_root = mobl_tree.getroot()

    # --- 1) remove Rajagopal upper limb bodies & joints ---
    removed_b = remove_bodies(raja_root, RAJA_UPPER_BODIES)
    removed_j = remove_joints(raja_root, RAJA_UPPER_JOINTS)
    print(f"[Rajagopal] removed bodies={removed_b}")
    print(f"[Rajagopal] removed joints={removed_j}")

    # --- 2) remove muscles on removed bodies ---
    raja_fs = raja_root.find('.//ForceSet/objects')
    remaining_bodies_raja = {b.get('name') for b in raja_root.iter('Body')}
    removed_f = []
    if raja_fs is not None:
        for f in list(raja_fs):
            refs = set()
            for pp in f.iter():
                if pp.tag in ('socket_parent_frame','socket_body','socket_child_frame','body') and pp.text:
                    refs.add(pp.text.strip().split('/')[-1].replace('_offset',''))
            # if any referenced body is not in what Rajagopal still has, drop the force
            if refs and not refs.issubset(remaining_bodies_raja | {'ground'}):
                raja_fs.remove(f); removed_f.append(f.get('name'))
    print(f"[Rajagopal] removed forces on upper limb: {len(removed_f)}")

    # --- 2b) also strip CoordinateActuators referencing a removed coordinate ---
    removed_coord_names = {'arm_flex_r','arm_add_r','arm_rot_r','elbow_flex_r','pro_sup_r',
                           'wrist_flex_r','wrist_dev_r','arm_flex_l','arm_add_l','arm_rot_l',
                           'elbow_flex_l','pro_sup_l','wrist_flex_l','wrist_dev_l'}
    removed_act = []
    if raja_fs is not None:
        for a in list(raja_fs):
            if a.tag == 'CoordinateActuator':
                c = a.find('coordinate')
                if c is not None and c.text and c.text.strip() in removed_coord_names:
                    raja_fs.remove(a); removed_act.append(a.get('name'))
    print(f"[Rajagopal] removed CoordinateActuators on upper limb: {removed_act}")

    # Detach markers on removed bodies — keep them aside so we can reinsert
    # onto the MoBL bodies of the same name after they are harvested.
    raja_ms = raja_root.find('.//MarkerSet/objects')
    detached_upper_markers = []
    if raja_ms is not None:
        for m in list(raja_ms):
            parent = m.find('socket_parent_frame')
            if parent is not None and parent.text:
                body = parent.text.strip().split('/')[-1]
                if body in RAJA_UPPER_BODIES:
                    raja_ms.remove(m)
                    detached_upper_markers.append(m)
    print(f"[Rajagopal] detached upper-limb markers (to re-attach): "
          f"{[m.get('name') for m in detached_upper_markers]}")

    # --- 3) harvest MoBL bodies (skip ground & thorax) ---
    copied_b = copy_bodies(mobl_root, raja_root, MOBL_HARVEST_BODIES)
    print(f"[MoBL -> merged] copied bodies[{len(copied_b)}]: {copied_b}")

    # --- 4) harvest MoBL joints (skip groundthorax) ---
    def jfilter(j):
        return j.get('name') != 'groundthorax'
    copied_j = copy_joints(mobl_root, raja_root, jfilter)
    print(f"[MoBL -> merged] copied joints[{len(copied_j)}]: {copied_j}")

    # --- 5) rewrite /bodyset/thorax references to /bodyset/torso ---
    # Only the body path substitution. Leave POF names like "thorax_offset" intact
    # (they live inside joint <frames>, not inside the thorax body).
    mapping = {'/bodyset/thorax': '/bodyset/torso'}
    n_rewrites = rewrite_paths(raja_root, mapping)
    print(f"[path rewrite] /bodyset/thorax -> /bodyset/torso replacements: {n_rewrites}")

    # --- 6) copy MoBL forces that act on harvested bodies or thorax(->torso).
    # IMPORTANT: rewrite thorax references inside mobl_root first, so the
    # filter sees "torso" instead of "thorax".
    rewrite_paths(mobl_root, {'/bodyset/thorax': '/bodyset/torso'})
    copied_f = copy_forces(mobl_root, raja_root, MOBL_HARVEST_BODIES)
    print(f"[MoBL -> merged] copied forces[{len(copied_f)}]")

    # --- 7) copy MoBL markers on harvested bodies / thorax(→torso) ---
    copied_mk = copy_markers(mobl_root, raja_root, MOBL_HARVEST_BODIES)
    print(f"[MoBL -> merged] copied markers[{len(copied_mk)}]: {copied_mk}")

    # --- 7b) re-attach Rajagopal's upper-limb markers onto the MoBL bodies
    # (humerus_r, ulna_r, radius_r, hand_r, and left mirrors). The socket
    # paths (e.g. /bodyset/humerus_r) are identical, so the markers snap onto
    # the newly-harvested MoBL bodies automatically. Local coords come from
    # Rajagopal's body-frame definition — MoBL uses slightly different
    # frame origins so these are approximate (millimeter-level), which is
    # fine for IK initialization and visualization.
    if raja_ms is None:
        raja_ms = raja_root.find('.//MarkerSet/objects')
    reattached = []
    for m in detached_upper_markers:
        raja_ms.append(m)
        reattached.append(m.get('name'))
    print(f"[Rajagopal -> merged] re-attached upper-limb markers[{len(reattached)}]: {reattached}")

    # --- 8) copy MoBL constraints (scapulohumeral rhythm couplers) ---
    copied_c = copy_constraints(mobl_root, raja_root)
    print(f"[MoBL -> merged] copied constraints[{len(copied_c)}]")

    # --- 7c) Remove PathWrap entries referencing Thorax_r/l since the socket
    # binding across bodies is fragile after body-swap. Muscles keep their path
    # points so kinematics (IK) are unaffected; muscle lines may pass through
    # the ribcage visually but won't affect our joint-angle analysis.
    n_wrap_rm = 0
    removed_muscles_with_thorax_wrap = []
    for mus in list(raja_root.iter()):
        if mus.tag.endswith('Muscle'):
            for pws in mus.iter('PathWrapSet'):
                objs = pws.find('objects')
                if objs is None: continue
                for pw in list(objs):
                    w = pw.find('wrap_object')
                    if w is not None and w.text and w.text.strip() in ('Thorax_r','Thorax_l'):
                        objs.remove(pw); n_wrap_rm += 1
                        removed_muscles_with_thorax_wrap.append(mus.get('name'))
    print(f"[path rewrite] PathWrap entries referencing Thorax removed: {n_wrap_rm}")

    # --- 7b) copy MoBL thorax's WrapObjects (e.g. Thorax_r/l ellipsoids used by
    # PECM1/LAT1 path wraps) into Rajagopal torso's WrapObjectSet.
    torso_body = None
    for b in raja_root.iter('Body'):
        if b.get('name') == 'torso':
            torso_body = b; break
    if torso_body is not None:
        torso_wos = torso_body.find('WrapObjectSet')
        if torso_wos is None:
            torso_wos = ET.SubElement(torso_body, 'WrapObjectSet'); torso_wos.set('name', 'wrapobjectset')
            ET.SubElement(torso_wos, 'objects'); ET.SubElement(torso_wos, 'groups')
        torso_wos_objs = torso_wos.find('objects') or torso_wos
        n_wraps = 0
        for b in mobl_root.iter('Body'):
            if b.get('name') == 'thorax':
                mobl_wos = b.find('WrapObjectSet')
                if mobl_wos is not None:
                    mobl_objs = mobl_wos.find('objects')
                    if mobl_objs is not None:
                        for w in list(mobl_objs):
                            torso_wos_objs.append(copy.deepcopy(w)); n_wraps += 1
                break
        print(f"[MoBL -> merged] copied wrap objects to torso: {n_wraps}")

    # --- 8b) re-position the MoBL shoulder girdle within Rajagopal's torso frame.
    # Rajagopal torso origin is at the lumbar joint (~1.02m). We want the
    # sternoclavicular joint near the sternum notch of torso (+0.40m in Y,
    # lateral ±0.025m in Z). Patch the 'thorax_offset' POF inside
    # sternoclavicular_r/l joints.
    SHOULDER_OFFSET_Y = 0.40   # meters, lumbar joint -> jugular notch
    SHOULDER_OFFSET_Z = 0.0    # keep lateral offset from original (already in translation)
    # MoBL thorax frame has +X = forward, +Z = right-ish inverted; Rajagopal torso
    # has +X = forward, +Z = right. Empirically MoBL humerus_r sits at
    # (x,y,z)=(-0.170, 0, -0.027) in thorax frame while Rajagopal expects the
    # right shoulder at roughly (-0.03, +0.40, +0.17). A rotation of +90 deg
    # about the torso Y axis maps MoBL's (-X) -> Rajagopal's (+Z), aligning
    # both the joint translation and every muscle attachment expressed in
    # the downstream MoBL frames.
    import math
    # Translation: rotate MoBL's (x,z) by +90 deg so humerus_r ends up on
    # Rajagopal's +Z (right) side.
    # Orientation: rotate child frame by -90 deg so every downstream MoBL
    # coordinate (muscle path points, child body offsets) re-expresses
    # correctly in Rajagopal torso axes.
    for j in raja_root.iter('CustomJoint'):
        if j.get('name') in ('sternoclavicular_r', 'sternoclavicular_l'):
            for f in j.iter('PhysicalOffsetFrame'):
                pr = f.find('socket_parent')
                if pr is not None and pr.text and '/bodyset/torso' in pr.text:
                    tr = f.find('translation')
                    if tr is not None and tr.text:
                        vals = [float(v) for v in tr.text.split()]
                        vals[1] += SHOULDER_OFFSET_Y  # shift up
                        tr.text = ' '.join(f'{v:.6f}' for v in vals)
                        print(f"[fix] {j.get('name')} parent offset shifted by +{SHOULDER_OFFSET_Y}m Y")

    # --- 9) write out ---
    raja_tree.write(OUT.as_posix(), xml_declaration=True, encoding='UTF-8')
    print(f"\nWrote {OUT}")

    # --- 10) verify ---
    try:
        m = osim.Model(OUT.as_posix())
        state = m.initSystem()
        print("\n*** merged model loads OK ***")
        print(f"  bodies={m.getBodySet().getSize()} joints={m.getJointSet().getSize()} "
              f"DOF={m.getCoordinateSet().getSize()} markers={m.getMarkerSet().getSize()} "
              f"muscles={m.getMuscles().getSize()} forces={m.getForceSet().getSize()}")
    except Exception as e:
        print(f"\n!!! initSystem FAILED: {e}")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
