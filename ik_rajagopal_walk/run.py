"""
Rajagopal 2016 + opensim-models 공식 보행 TRC → Scale → IK
"""
import os
import xml.etree.ElementTree as ET
import opensim as osim

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

MODEL_FILE    = 'Rajagopal2016.osim'
STATIC_TRC    = 'static_walk_1.trc'
DYNAMIC_TRC   = 'motion_capture_walk.trc'
SCALE_XML_IN  = 'scale_setup_walk.xml'
IK_XML_IN     = 'ik_setup_walk.xml'
MARKERSET_XML = 'markerset_walk_preScale.xml'

SCALED_MODEL   = 'scaled_model.osim'
IK_MOT         = 'ik.mot'
SCALE_XML_OUT  = 'scale_setup_patched.xml'
IK_XML_OUT     = 'ik_setup_patched.xml'


def patch_scale(xml_in, xml_out):
    tree = ET.parse(xml_in)
    for e in tree.iter():
        if   e.tag == 'model_file':         e.text = MODEL_FILE
        elif e.tag == 'marker_set_file':    e.text = MARKERSET_XML
        elif e.tag == 'marker_file':        e.text = STATIC_TRC
        elif e.tag == 'output_model_file':  e.text = SCALED_MODEL
    tree.write(xml_out, xml_declaration=True, encoding='UTF-8')


def patch_ik(xml_in, xml_out):
    tree = ET.parse(xml_in)
    for e in tree.iter():
        if   e.tag == 'model_file':         e.text = SCALED_MODEL
        elif e.tag == 'marker_file':        e.text = DYNAMIC_TRC
        elif e.tag == 'output_motion_file': e.text = IK_MOT
        elif e.tag == 'results_directory':  e.text = '.'
    tree.write(xml_out, xml_declaration=True, encoding='UTF-8')


print('=' * 60)
print('Step 1: ScaleTool (Rajagopal 2016 generic → subject scaled)')
print('=' * 60)
patch_scale(SCALE_XML_IN, SCALE_XML_OUT)
osim.ScaleTool(SCALE_XML_OUT).run()
assert os.path.exists(SCALED_MODEL), 'scale failed'
print(f'  → {SCALED_MODEL} ({os.path.getsize(SCALED_MODEL):,} bytes)')

print()
print('=' * 60)
print('Step 2: Inverse Kinematics (walking trial)')
print('=' * 60)
patch_ik(IK_XML_IN, IK_XML_OUT)
osim.InverseKinematicsTool(IK_XML_OUT).run()
assert os.path.exists(IK_MOT), 'IK failed'
print(f'  → {IK_MOT} ({os.path.getsize(IK_MOT):,} bytes)')

print()
print('=' * 60)
print('Step 3: IK summary')
print('=' * 60)
tbl = osim.TimeSeriesTable(IK_MOT)
labels = list(tbl.getColumnLabels())
print(f'  frames: {tbl.getNumRows()}')
for col in ['knee_angle_r', 'hip_flexion_r', 'ankle_angle_r',
            'lumbar_extension', 'arm_flex_r']:
    if col in labels:
        v = tbl.getDependentColumn(col).to_numpy()
        print(f'  {col:<20} min={v.min():+7.2f}  max={v.max():+7.2f}')

print()
print('Done.')
