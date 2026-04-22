"""Phase 1 smoke test: load each .osim and call initSystem()."""
import os
import traceback
import opensim as osim

MODELS = [
    "/Volumes/T31/opensim/Rajagopal2015MJPlugInGait_43_2.2.osim",
    "/Volumes/T31/opensim/Bimanual Upper Arm Model/MoBL_ARMS_bimanual_6_2_21.osim",
]

# MoBL-ARMS unimanual: find the .osim inside Model/
mobl_dir = "/Volumes/T31/opensim/MoBL-ARMS Upper Extremity Model/Model"
if os.path.isdir(mobl_dir):
    for f in os.listdir(mobl_dir):
        if f.endswith(".osim"):
            MODELS.append(os.path.join(mobl_dir, f))


def summarize(path: str) -> None:
    print(f"\n=== {path} ===")
    try:
        model = osim.Model(path)
        state = model.initSystem()
        print(f"  name          : {model.getName()}")
        print(f"  bodies        : {model.getBodySet().getSize()}")
        print(f"  joints        : {model.getJointSet().getSize()}")
        print(f"  coords (DOF)  : {model.getCoordinateSet().getSize()}")
        print(f"  markers       : {model.getMarkerSet().getSize()}")
        print(f"  forces        : {model.getForceSet().getSize()}")
        print(f"  muscles       : {model.getMuscles().getSize()}")
        print(f"  initSystem OK : time={state.getTime():.3f}")
    except Exception as e:  # noqa: BLE001
        print("  FAILED:", e)
        traceback.print_exc()


for m in MODELS:
    summarize(m)
