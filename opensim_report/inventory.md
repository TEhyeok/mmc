# OpenSim Model Inventory (Phase 2)

## Rajagopal_PIG — `Unscaled_MJ_Rajagopal_PlugInGait_43_2.2`

- path: `/Volumes/T31/opensim/Rajagopal2015MJPlugInGait_43_2.2.osim`
- bodies=26  joints=26  DOF=51  markers=64  forces=109  muscles=80
- forward sim 0.5s: **OK** — skipped (structural inventory only)

### Bodies (name, mass[kg], world xyz[m] @ default pose)
| # | name | mass | x | y | z |
|---|------|------|---|---|---|
| 0 | pelvis | 11.777 | +0.000 | +0.930 | +0.000 |
| 1 | femur_r | 9.301 | -0.056 | +0.852 | +0.077 |
| 2 | tibia_r | 3.708 | -0.056 | +0.447 | +0.076 |
| 3 | patella_r | 0.086 | -0.012 | +0.433 | +0.077 |
| 4 | talus_r | 0.100 | -0.066 | +0.047 | +0.076 |
| 5 | calcn_r | 1.250 | -0.115 | +0.005 | +0.084 |
| 6 | toes_r | 0.217 | +0.064 | +0.003 | +0.085 |
| 7 | femur_l | 9.301 | -0.056 | +0.852 | -0.077 |
| 8 | tibia_l | 3.708 | -0.056 | +0.447 | -0.076 |
| 9 | patella_l | 0.086 | -0.012 | +0.433 | -0.077 |
| 10 | talus_l | 0.100 | -0.066 | +0.047 | -0.076 |
| 11 | calcn_l | 1.250 | -0.115 | +0.005 | -0.084 |
| 12 | toes_l | 0.217 | +0.064 | +0.003 | -0.085 |
| 13 | abdomen | 8.552 | -0.101 | +1.012 | +0.000 |
| 14 | thorax | 12.290 | -0.111 | +1.147 | +0.000 |
| 15 | neckHead | 4.984 | -0.119 | +1.415 | +0.000 |
| 16 | clavScap_r | 0.500 | -0.082 | +1.391 | +0.000 |
| 17 | humerus_r | 2.033 | -0.098 | +1.383 | +0.170 |
| 18 | ulna_r | 0.608 | -0.084 | +1.097 | +0.160 |
| 19 | radius_r | 0.608 | -0.091 | +1.084 | +0.186 |
| 20 | hand_r | 0.458 | -0.100 | +0.848 | +0.200 |
| 21 | clavScap_l | 0.500 | -0.082 | +1.391 | +0.000 |
| 22 | humerus_l | 2.033 | -0.098 | +1.383 | -0.170 |
| 23 | ulna_l | 0.608 | -0.084 | +1.097 | -0.160 |
| 24 | radius_l | 0.608 | -0.091 | +1.084 | -0.186 |
| 25 | hand_l | 0.458 | -0.100 | +0.848 | -0.200 |

### Joints (name, type, parent → child)
| # | name | type | parent | child |
|---|------|------|--------|-------|
| 0 | ground_pelvis | CustomJoint | `ground_offset` | `pelvis_offset` |
| 1 | hip_r | CustomJoint | `pelvis_offset` | `femur_r_offset` |
| 2 | walker_knee_r | CustomJoint | `femur_r_offset` | `tibia_r_offset` |
| 3 | patellofemoral_r | CustomJoint | `femur_r_offset` | `patella_r_offset` |
| 4 | ankle_r | PinJoint | `tibia_r_offset` | `talus_r_offset` |
| 5 | subtalar_r | PinJoint | `talus_r_offset` | `calcn_r_offset` |
| 6 | mtp_r | PinJoint | `calcn_r_offset` | `toes_r_offset` |
| 7 | hip_l | CustomJoint | `pelvis_offset` | `femur_l_offset` |
| 8 | walker_knee_l | CustomJoint | `femur_l_offset` | `tibia_l_offset` |
| 9 | patellofemoral_l | CustomJoint | `femur_l_offset` | `patella_l_offset` |
| 10 | ankle_l | PinJoint | `tibia_l_offset` | `talus_l_offset` |
| 11 | subtalar_l | PinJoint | `talus_l_offset` | `calcn_l_offset` |
| 12 | mtp_l | PinJoint | `calcn_l_offset` | `toes_l_offset` |
| 13 | back | CustomJoint | `pelvis_offset` | `abdomen_offset` |
| 14 | thorax_joint | CustomJoint | `abdomen_offset` | `thorax_offset` |
| 15 | neck | CustomJoint | `thorax_offset` | `neckHead_offset` |
| 16 | clav_r | CustomJoint | `thorax_offset` | `clavScap_r_offset` |
| 17 | acromial_r | CustomJoint | `clavScap_r_offset` | `humerus_r_offset` |
| 18 | elbow_r | PinJoint | `humerus_r_offset` | `ulna_r_offset` |
| 19 | radioulnar_r | PinJoint | `ulna_r_offset` | `radius_r_offset` |
| 20 | radius_hand_r | UniversalJoint | `radius_r_offset` | `hand_r_offset` |
| 21 | clav_l | CustomJoint | `thorax_offset` | `clavScap_l_offset` |
| 22 | acromial_l | CustomJoint | `clavScap_l_offset` | `humerus_l_offset` |
| 23 | elbow_l | PinJoint | `humerus_l_offset` | `ulna_l_offset` |
| 24 | radioulnar_l | PinJoint | `ulna_l_offset` | `radius_l_offset` |
| 25 | radius_hand_l | UniversalJoint | `radius_l_offset` | `hand_l_offset` |

### Coordinates (DOF)
| # | name | joint | unit | min | max | default | locked | clamped |
|---|------|-------|------|-----|-----|---------|--------|---------|
| 0 | pelvis_rotation | ground_pelvis | rad | -2.094 | +2.094 | +0.000 | False | False |
| 1 | pelvis_list | ground_pelvis | rad | -1.571 | +1.571 | +0.000 | False | False |
| 2 | pelvis_tilt | ground_pelvis | rad | -1.571 | +1.571 | +0.000 | False | False |
| 3 | pelvis_tx | ground_pelvis | m | -5.000 | +5.000 | +0.000 | False | True |
| 4 | pelvis_ty | ground_pelvis | m | -1.000 | +2.000 | +0.930 | False | True |
| 5 | pelvis_tz | ground_pelvis | m | -3.000 | +3.000 | +0.000 | False | True |
| 6 | hip_flexion_r | hip_r | rad | -3.142 | +3.142 | +0.000 | False | True |
| 7 | hip_adduction_r | hip_r | rad | -1.571 | +1.571 | +0.000 | False | True |
| 8 | hip_rotation_r | hip_r | rad | -1.571 | +1.571 | +0.000 | False | True |
| 9 | knee_angle_r | walker_knee_r | rad | +0.000 | +3.142 | +0.000 | False | True |
| 10 | knee_angle_r_beta | patellofemoral_r | ? | -99999.900 | +99999.900 | +0.000 | False | False |
| 11 | ankle_angle_r | ankle_r | rad | -0.698 | +0.698 | +0.000 | False | True |
| 12 | subtalar_angle_r | subtalar_r | rad | -0.349 | +0.349 | +0.000 | False | True |
| 13 | mtp_angle_r | mtp_r | rad | -0.524 | +0.524 | +0.000 | False | True |
| 14 | hip_flexion_l | hip_l | rad | -3.142 | +3.142 | +0.000 | False | True |
| 15 | hip_adduction_l | hip_l | rad | -1.571 | +1.571 | +0.000 | False | True |
| 16 | hip_rotation_l | hip_l | rad | -1.571 | +1.571 | +0.000 | False | True |
| 17 | knee_angle_l | walker_knee_l | rad | +0.000 | +3.142 | +0.000 | False | True |
| 18 | knee_angle_l_beta | patellofemoral_l | ? | -99999.900 | +99999.900 | +0.000 | False | False |
| 19 | ankle_angle_l | ankle_l | rad | -0.698 | +0.698 | +0.000 | False | True |
| 20 | subtalar_angle_l | subtalar_l | rad | -0.349 | +0.349 | +0.000 | False | True |
| 21 | mtp_angle_l | mtp_l | rad | -0.524 | +0.524 | +0.000 | False | True |
| 22 | lumbar_extension | back | rad | -1.571 | +1.571 | +0.000 | False | True |
| 23 | lumbar_bending | back | rad | -1.571 | +1.571 | +0.000 | False | True |
| 24 | lumbar_rotation | back | rad | -1.571 | +1.571 | +0.000 | False | True |
| 25 | thorax_extension | thorax_joint | rad | -1.571 | +1.571 | +0.000 | True | True |
| 26 | thorax_bending | thorax_joint | rad | -1.571 | +1.571 | +0.000 | True | True |
| 27 | thorax_rotation | thorax_joint | rad | -1.571 | +1.571 | +0.000 | True | True |
| 28 | neck_extension | neck | rad | -1.571 | +1.571 | +0.000 | False | True |
| 29 | neck_bending | neck | rad | -1.571 | +1.571 | +0.000 | False | True |
| 30 | neck_rotation | neck | rad | -1.571 | +1.571 | +0.000 | False | True |
| 31 | clav_r_ext | clav_r | rad | -0.611 | +0.611 | +0.000 | True | True |
| 32 | clav_r_bend | clav_r | rad | -0.611 | +0.611 | +0.000 | False | True |
| 33 | clav_r_rot | clav_r | rad | -0.611 | +0.611 | +0.000 | False | True |
| 34 | arm_flex_r | acromial_r | rad | -1.571 | +3.142 | +0.000 | False | True |
| 35 | arm_add_r | acromial_r | rad | -2.531 | +1.571 | +0.000 | False | True |
| 36 | arm_rot_r | acromial_r | rad | -3.142 | +3.142 | +0.000 | False | True |
| 37 | elbow_flex_r | elbow_r | rad | +0.000 | +3.142 | +0.000 | False | True |
| 38 | pro_sup_r | radioulnar_r | rad | +0.000 | +3.142 | +0.000 | False | True |
| 39 | wrist_flex_r | radius_hand_r | rad | -2.222 | +1.222 | +0.000 | False | True |
| 40 | wrist_dev_r | radius_hand_r | rad | -0.436 | +0.611 | +0.000 | False | True |
| 41 | clav_l_ext | clav_l | rad | -0.611 | +0.611 | +0.000 | True | True |
| 42 | clav_l_bend | clav_l | rad | -0.611 | +0.611 | +0.000 | False | True |
| 43 | clav_l_rot | clav_l | rad | -0.611 | +0.611 | +0.000 | False | True |
| 44 | arm_flex_l | acromial_l | rad | -1.571 | +3.142 | +0.000 | False | True |
| 45 | arm_add_l | acromial_l | rad | -2.531 | +1.571 | +0.000 | False | True |
| 46 | arm_rot_l | acromial_l | rad | -3.142 | +3.142 | +0.000 | False | True |
| 47 | elbow_flex_l | elbow_l | rad | +0.000 | +3.142 | +0.000 | False | True |
| 48 | pro_sup_l | radioulnar_l | rad | +0.000 | +3.142 | +0.000 | False | True |
| 49 | wrist_flex_l | radius_hand_l | rad | -2.222 | +1.222 | +0.000 | False | True |
| 50 | wrist_dev_l | radius_hand_l | rad | -0.436 | +0.611 | +0.000 | False | True |

### Markers (name, body, local xyz[m])
| # | name | body | x_local | y_local | z_local |
|---|------|------|---------|---------|---------|
| 0 | RSHO | clavScap_r | -0.022 | +0.045 | +0.145 |
| 1 | LSHO | clavScap_l | -0.022 | +0.045 | -0.145 |
| 2 | RFHD | neckHead | +0.080 | +0.180 | +0.070 |
| 3 | LFHD | neckHead | +0.080 | +0.180 | -0.070 |
| 4 | RBHD | neckHead | -0.040 | +0.180 | +0.070 |
| 5 | LBHD | neckHead | -0.040 | +0.180 | -0.070 |
| 6 | C7 | thorax | -0.075 | +0.299 | +0.002 |
| 7 | CLAV | thorax | +0.050 | +0.244 | +0.000 |
| 8 | STRN | thorax | +0.130 | +0.060 | +0.000 |
| 9 | T10 | thorax | -0.090 | +0.030 | +0.000 |
| 10 | RBAK | clavScap_r | -0.100 | -0.060 | +0.120 |
| 11 | RUPA | humerus_r | +0.000 | -0.200 | +0.030 |
| 12 | RELB | humerus_r | +0.015 | -0.280 | +0.040 |
| 13 | RFRM | radius_r | -0.041 | -0.070 | -0.002 |
| 14 | RWRA | radius_r | +0.001 | -0.225 | +0.050 |
| 15 | RWRB | radius_r | -0.022 | -0.225 | -0.022 |
| 16 | LUPA | humerus_l | +0.000 | -0.050 | -0.030 |
| 17 | LELB | humerus_l | +0.015 | -0.280 | -0.040 |
| 18 | LFRM | radius_l | -0.037 | -0.149 | -0.005 |
| 19 | LWRA | radius_l | +0.001 | -0.225 | -0.050 |
| 20 | LWRB | radius_l | -0.022 | -0.225 | +0.022 |
| 21 | RASI | pelvis | +0.010 | +0.018 | +0.128 |
| 22 | LASI | pelvis | +0.010 | +0.018 | -0.128 |
| 23 | RPSI | pelvis | -0.155 | +0.035 | +0.045 |
| 24 | LPSI | pelvis | -0.155 | +0.035 | -0.045 |
| 25 | RTHI | femur_r | +0.018 | -0.150 | +0.064 |
| 26 | RKNE | femur_r | +0.000 | -0.404 | +0.050 |
| 27 | RTIB | tibia_r | -0.002 | -0.189 | +0.053 |
| 28 | RANK | tibia_r | -0.005 | -0.389 | +0.053 |
| 29 | RHEE | calcn_r | -0.025 | +0.020 | -0.005 |
| 30 | RTOE | calcn_r | +0.180 | +0.030 | +0.003 |
| 31 | LTHI | femur_l | +0.018 | -0.255 | -0.059 |
| 32 | LKNE | femur_l | +0.000 | -0.404 | -0.050 |
| 33 | LTIB | tibia_l | -0.004 | -0.263 | -0.053 |
| 34 | LANK | tibia_l | -0.005 | -0.389 | -0.053 |
| 35 | LHEE | calcn_l | -0.025 | +0.020 | +0.005 |
| 36 | LTOE | calcn_l | +0.180 | +0.030 | -0.003 |
| 37 | RFIN | hand_r | -0.010 | -0.090 | +0.020 |
| 38 | LFIN | hand_l | -0.010 | -0.090 | -0.020 |
| 39 | RELB_M | humerus_r | +0.000 | -0.283 | -0.051 |
| 40 | LELB_M | humerus_l | +0.000 | -0.283 | +0.051 |
| 41 | RO_HEE | calcn_r | -0.017 | +0.020 | +0.020 |
| 42 | RO_TOE | calcn_r | +0.200 | +0.020 | +0.057 |
| 43 | RI_TOE | calcn_r | +0.262 | +0.020 | -0.025 |
| 44 | LO_HEE | calcn_l | -0.017 | +0.020 | -0.020 |
| 45 | LI_TOE | calcn_l | +0.262 | +0.020 | +0.025 |
| 46 | LO_TOE | calcn_l | +0.200 | +0.020 | -0.057 |
| 47 | LKNE_M | femur_l | +0.002 | -0.406 | +0.058 |
| 48 | RKNE_M | femur_r | +0.002 | -0.406 | -0.058 |
| 49 | LANK_M | tibia_l | +0.003 | -0.389 | +0.043 |
| 50 | RANK_M | tibia_r | +0.003 | -0.389 | -0.043 |
| 51 | FHD | neckHead | +0.113 | +0.180 | +0.000 |
| 52 | BHD | neckHead | -0.081 | +0.180 | +0.000 |
| 53 | LHD | neckHead | +0.013 | +0.180 | -0.086 |
| 54 | RHD | neckHead | +0.013 | +0.180 | +0.086 |
| 55 | THD | neckHead | +0.015 | +0.259 | +0.000 |
| 56 | RHJC | pelvis | -0.056 | -0.078 | +0.077 |
| 57 | LHJC | pelvis | -0.056 | -0.078 | -0.077 |
| 58 | RSJC | humerus_r | +0.000 | +0.000 | +0.000 |
| 59 | LSJC | humerus_l | +0.000 | +0.000 | +0.000 |
| 60 | LBFT | calcn_l | +0.135 | +0.000 | -0.065 |
| 61 | RBFT | calcn_r | +0.135 | -0.000 | +0.065 |
| 62 | RO_ASI | pelvis | -0.016 | +0.040 | +0.147 |
| 63 | LO_ASI | pelvis | -0.016 | +0.040 | -0.147 |

---

## MoBL_bimanual — `Bimanual`

- path: `/Volumes/T31/opensim/Bimanual Upper Arm Model/MoBL_ARMS_bimanual_6_2_21.osim`
- bodies=23  joints=23  DOF=46  markers=0  forces=136  muscles=100
- forward sim 0.5s: **OK** — skipped (structural inventory only)

### Bodies (name, mass[kg], world xyz[m] @ default pose)
| # | name | mass | x | y | z |
|---|------|------|---|---|---|
| 0 | thorax | 0.000 | +0.000 | +0.000 | +0.000 |
| 1 | clavicle_r | 0.156 | -0.025 | +0.007 | +0.006 |
| 2 | clavphant_r | 0.000 | -0.157 | +0.034 | -0.025 |
| 3 | scapula_r | 0.704 | -0.157 | +0.034 | -0.025 |
| 4 | scapphant_r | 0.000 | -0.170 | +0.000 | -0.027 |
| 5 | humphant_r | 0.000 | -0.170 | +0.000 | -0.027 |
| 6 | humphant1_r | 0.000 | -0.170 | +0.000 | -0.027 |
| 7 | humerus_r | 1.998 | -0.170 | +0.000 | -0.027 |
| 8 | ulna_r | 1.105 | -0.304 | -0.257 | -0.012 |
| 9 | radius_r | 0.234 | -0.320 | -0.247 | +0.001 |
| 10 | proximal_row_r | 0.000 | -0.313 | -0.223 | +0.243 |
| 11 | hand_r | 0.582 | -0.312 | -0.219 | +0.258 |
| 12 | clavicle_l | 0.156 | +0.025 | +0.007 | +0.006 |
| 13 | clavphant_l | 0.000 | +0.157 | +0.034 | -0.025 |
| 14 | scapula_l | 0.704 | +0.157 | +0.034 | -0.025 |
| 15 | scapphant_l | 0.000 | +0.170 | +0.000 | -0.027 |
| 16 | humphant_l | 0.000 | +0.170 | +0.000 | -0.027 |
| 17 | humphant1_l | 0.000 | +0.170 | +0.000 | -0.027 |
| 18 | humerus_l | 1.998 | +0.170 | +0.000 | -0.027 |
| 19 | ulna_l | 1.105 | +0.304 | -0.257 | -0.012 |
| 20 | radius_l | 0.234 | +0.320 | -0.247 | +0.001 |
| 21 | proximal_row_l | 0.000 | +0.313 | -0.223 | +0.243 |
| 22 | hand_l | 0.582 | +0.312 | -0.219 | +0.258 |

### Joints (name, type, parent → child)
| # | name | type | parent | child |
|---|------|------|--------|-------|
| 0 | groundthorax | CustomJoint | `ground_offset` | `thorax_offset` |
| 1 | sternoclavicular_r | CustomJoint | `thorax_offset` | `clavicle_r_offset` |
| 2 | unrotscap_r | CustomJoint | `clavicle_r_offset` | `clavphant_r_offset` |
| 3 | acromioclavicular_r | CustomJoint | `clavphant_r_offset` | `scapula_r_offset` |
| 4 | unrothum_r | CustomJoint | `scapula_r_offset` | `scapphant_r_offset` |
| 5 | shoulder0_r | CustomJoint | `scapphant_r_offset` | `humphant_r_offset` |
| 6 | shoulder1_r | CustomJoint | `humphant_r_offset` | `humphant1_r_offset` |
| 7 | shoulder2_r | CustomJoint | `humphant1_r_offset` | `humerus_r_offset` |
| 8 | elbow_r | CustomJoint | `humerus_r_offset` | `ulna_r_offset` |
| 9 | radioulnar_r | CustomJoint | `ulna_r_offset` | `radius_r_offset` |
| 10 | radiocarpal_r | CustomJoint | `radius_r_offset` | `proximal_row_r_offset` |
| 11 | wrist_hand_r | CustomJoint | `proximal_row_r_offset` | `hand_r_offset` |
| 12 | sternoclavicular_l | CustomJoint | `thorax_offset` | `clavicle_l_offset` |
| 13 | unrotscap_l | CustomJoint | `clavicle_l_offset` | `clavphant_l_offset` |
| 14 | acromioclavicular_l | CustomJoint | `clavphant_l_offset` | `scapula_l_offset` |
| 15 | unrothum_l | CustomJoint | `scapula_l_offset` | `scapphant_l_offset` |
| 16 | shoulder0_l | CustomJoint | `scapphant_l_offset` | `humphant_l_offset` |
| 17 | shoulder1_l | CustomJoint | `humphant_l_offset` | `humphant1_l_offset` |
| 18 | shoulder2_l | CustomJoint | `humphant1_l_offset` | `humerus_l_offset` |
| 19 | elbow_l | CustomJoint | `humerus_l_offset` | `ulna_l_offset` |
| 20 | radioulnar_l | CustomJoint | `ulna_l_offset` | `radius_l_offset` |
| 21 | radiocarpal_l | CustomJoint | `radius_l_offset` | `proximal_row_l_offset` |
| 22 | wrist_hand_l | CustomJoint | `proximal_row_l_offset` | `hand_l_offset` |

### Coordinates (DOF)
| # | name | joint | unit | min | max | default | locked | clamped |
|---|------|-------|------|-----|-----|---------|--------|---------|
| 0 | r_x | groundthorax | rad | -6.283 | +6.283 | +0.000 | False | True |
| 1 | r_y | groundthorax | rad | -6.283 | +6.283 | -1.571 | False | True |
| 2 | r_z | groundthorax | rad | -6.283 | +6.283 | +0.000 | False | True |
| 3 | t_x | groundthorax | m | -100.000 | +100.000 | +0.000 | False | True |
| 4 | t_y | groundthorax | m | -100.000 | +100.000 | +0.000 | False | True |
| 5 | t_z | groundthorax | m | -100.000 | +100.000 | +0.000 | False | True |
| 6 | sternoclavicular_r2_r | sternoclavicular_r | rad | -99999.900 | +99999.900 | -0.127 | False | False |
| 7 | sternoclavicular_r3_r | sternoclavicular_r | rad | -99999.900 | +99999.900 | +0.054 | False | False |
| 8 | unrotscap_r3_r | unrotscap_r | rad | -99999.900 | +99999.900 | -0.054 | False | False |
| 9 | unrotscap_r2_r | unrotscap_r | rad | -99999.900 | +99999.900 | +0.127 | False | False |
| 10 | acromioclavicular_r2_r | acromioclavicular_r | rad | -99999.900 | +99999.900 | -0.026 | False | False |
| 11 | acromioclavicular_r3_r | acromioclavicular_r | rad | -99999.900 | +99999.900 | +0.207 | False | False |
| 12 | acromioclavicular_r1_r | acromioclavicular_r | rad | -99999.900 | +99999.900 | +0.093 | False | False |
| 13 | unrothum_r1_r | unrothum_r | rad | -99999.900 | +99999.900 | -0.093 | False | False |
| 14 | unrothum_r3_r | unrothum_r | rad | -99999.900 | +99999.900 | -0.207 | False | False |
| 15 | unrothum_r2_r | unrothum_r | rad | -99999.900 | +99999.900 | +0.026 | False | False |
| 16 | elv_angle_r | shoulder0_r | rad | -1.658 | +2.269 | +0.000 | False | True |
| 17 | shoulder_elv_r | shoulder1_r | rad | +0.000 | +3.142 | +0.524 | False | True |
| 18 | shoulder1_r2_r | shoulder1_r | rad | -99999.900 | +99999.900 | -0.000 | False | False |
| 19 | shoulder_rot_r | shoulder2_r | rad | -1.571 | +2.094 | -0.000 | False | True |
| 20 | elbow_flexion_r | elbow_r | rad | +0.000 | +2.269 | +1.571 | False | True |
| 21 | pro_sup_r | radioulnar_r | rad | -1.571 | +1.571 | -0.000 | False | True |
| 22 | deviation_r | radiocarpal_r | ? | -0.175 | +0.436 | +0.000 | False | True |
| 23 | flexion_r | radiocarpal_r | ? | -1.222 | +1.222 | +0.000 | False | True |
| 24 | wrist_hand_r1_r | wrist_hand_r | rad | -99999.900 | +99999.900 | +0.000 | False | False |
| 25 | wrist_hand_r3_r | wrist_hand_r | rad | -99999.900 | +99999.900 | +0.000 | False | False |
| 26 | sternoclavicular_r2_l | sternoclavicular_l | rad | -99999.900 | +99999.900 | -0.127 | False | False |
| 27 | sternoclavicular_r3_l | sternoclavicular_l | rad | -99999.900 | +99999.900 | +0.054 | False | False |
| 28 | unrotscap_r3_l | unrotscap_l | rad | -99999.900 | +99999.900 | -0.054 | False | False |
| 29 | unrotscap_r2_l | unrotscap_l | rad | -99999.900 | +99999.900 | +0.127 | False | False |
| 30 | acromioclavicular_r2_l | acromioclavicular_l | rad | -99999.900 | +99999.900 | -0.026 | False | False |
| 31 | acromioclavicular_r3_l | acromioclavicular_l | rad | -99999.900 | +99999.900 | +0.207 | False | False |
| 32 | acromioclavicular_r1_l | acromioclavicular_l | rad | -99999.900 | +99999.900 | +0.093 | False | False |
| 33 | unrothum_r1_l | unrothum_l | rad | -99999.900 | +99999.900 | -0.093 | False | False |
| 34 | unrothum_r3_l | unrothum_l | rad | -99999.900 | +99999.900 | -0.207 | False | False |
| 35 | unrothum_r2_l | unrothum_l | rad | -99999.900 | +99999.900 | +0.026 | False | False |
| 36 | elv_angle_l | shoulder0_l | rad | -1.658 | +2.269 | +0.000 | False | True |
| 37 | shoulder_elv_l | shoulder1_l | rad | +0.000 | +3.142 | +0.524 | False | True |
| 38 | shoulder1_r2_l | shoulder1_l | rad | -99999.900 | +99999.900 | -0.000 | False | False |
| 39 | shoulder_rot_l | shoulder2_l | rad | -1.571 | +2.094 | -0.000 | False | True |
| 40 | elbow_flexion_l | elbow_l | rad | +0.000 | +2.269 | +1.571 | False | True |
| 41 | pro_sup_l | radioulnar_l | rad | -1.571 | +1.571 | -0.000 | False | True |
| 42 | deviation_l | radiocarpal_l | ? | -0.175 | +0.436 | +0.000 | False | True |
| 43 | flexion_l | radiocarpal_l | ? | -1.222 | +1.222 | +0.000 | False | True |
| 44 | wrist_hand_r1_l | wrist_hand_l | rad | -99999.900 | +99999.900 | +0.000 | False | False |
| 45 | wrist_hand_r3_l | wrist_hand_l | rad | -99999.900 | +99999.900 | +0.000 | False | False |

### Markers: (none)

---

## MoBL_41_uniR — `Right`

- path: `/Volumes/T31/opensim/MoBL-ARMS Upper Extremity Model/Model/4.1/MOBL_ARMS_41.osim`
- bodies=12  joints=12  DOF=26  markers=10  forces=68  muscles=50
- forward sim 0.5s: **OK** — skipped (structural inventory only)

### Bodies (name, mass[kg], world xyz[m] @ default pose)
| # | name | mass | x | y | z |
|---|------|------|---|---|---|
| 0 | thorax | 0.000 | +0.000 | +0.000 | +0.000 |
| 1 | clavicle | 0.156 | -0.025 | +0.007 | +0.006 |
| 2 | clavphant | 0.000 | -0.157 | +0.034 | -0.025 |
| 3 | scapula | 0.704 | -0.157 | +0.034 | -0.025 |
| 4 | scapphant | 0.000 | -0.170 | +0.000 | -0.027 |
| 5 | humphant | 0.000 | -0.170 | +0.000 | -0.027 |
| 6 | humphant1 | 0.000 | -0.170 | +0.000 | -0.027 |
| 7 | humerus | 1.998 | -0.170 | +0.000 | -0.027 |
| 8 | ulna | 1.105 | -0.151 | -0.248 | +0.123 |
| 9 | radius | 0.234 | -0.170 | -0.258 | +0.129 |
| 10 | proximal_row | 0.000 | -0.190 | -0.460 | +0.265 |
| 11 | hand | 0.582 | -0.192 | -0.471 | +0.276 |

### Joints (name, type, parent → child)
| # | name | type | parent | child |
|---|------|------|--------|-------|
| 0 | groundthorax | CustomJoint | `ground_offset` | `thorax_offset` |
| 1 | sternoclavicular | CustomJoint | `thorax_offset` | `clavicle_offset` |
| 2 | unrotscap | CustomJoint | `clavicle_offset` | `clavphant_offset` |
| 3 | acromioclavicular | CustomJoint | `clavphant_offset` | `scapula_offset` |
| 4 | unrothum | CustomJoint | `scapula_offset` | `scapphant_offset` |
| 5 | shoulder0 | CustomJoint | `scapphant_offset` | `humphant_offset` |
| 6 | shoulder1 | CustomJoint | `humphant_offset` | `humphant1_offset` |
| 7 | shoulder2 | CustomJoint | `humphant1_offset` | `humerus_offset` |
| 8 | elbow | CustomJoint | `humerus_offset` | `ulna_offset` |
| 9 | radioulnar | CustomJoint | `ulna_offset` | `radius_offset` |
| 10 | radiocarpal | CustomJoint | `radius_offset` | `proximal_row_offset` |
| 11 | wrist_hand | CustomJoint | `proximal_row_offset` | `hand_offset` |

### Coordinates (DOF)
| # | name | joint | unit | min | max | default | locked | clamped |
|---|------|-------|------|-----|-----|---------|--------|---------|
| 0 | r_x | groundthorax | rad | -6.283 | +6.283 | +0.000 | False | False |
| 1 | r_y | groundthorax | rad | -6.283 | +6.283 | -1.571 | False | False |
| 2 | r_z | groundthorax | rad | -6.283 | +6.283 | +0.000 | False | False |
| 3 | t_x | groundthorax | m | -100.000 | +100.000 | +0.000 | False | True |
| 4 | t_y | groundthorax | m | -100.000 | +100.000 | +0.000 | False | True |
| 5 | t_z | groundthorax | m | -100.000 | +100.000 | +0.000 | False | True |
| 6 | sternoclavicular_r2 | sternoclavicular | rad | -99999.900 | +99999.900 | -0.127 | False | False |
| 7 | sternoclavicular_r3 | sternoclavicular | rad | -99999.900 | +99999.900 | +0.054 | False | False |
| 8 | unrotscap_r3 | unrotscap | rad | -99999.900 | +99999.900 | -0.054 | False | False |
| 9 | unrotscap_r2 | unrotscap | rad | -99999.900 | +99999.900 | +0.127 | False | False |
| 10 | acromioclavicular_r2 | acromioclavicular | rad | -99999.900 | +99999.900 | -0.026 | False | False |
| 11 | acromioclavicular_r3 | acromioclavicular | rad | -99999.900 | +99999.900 | +0.207 | False | False |
| 12 | acromioclavicular_r1 | acromioclavicular | rad | -99999.900 | +99999.900 | +0.093 | False | False |
| 13 | unrothum_r1 | unrothum | rad | -99999.900 | +99999.900 | -0.093 | False | False |
| 14 | unrothum_r3 | unrothum | rad | -99999.900 | +99999.900 | -0.207 | False | False |
| 15 | unrothum_r2 | unrothum | rad | -99999.900 | +99999.900 | +0.026 | False | False |
| 16 | elv_angle | shoulder0 | rad | -1.658 | +2.269 | +1.571 | False | True |
| 17 | shoulder_elv | shoulder1 | rad | +0.000 | +3.142 | +0.524 | False | True |
| 18 | shoulder1_r2 | shoulder1 | rad | -99999.900 | +99999.900 | -1.571 | False | False |
| 19 | shoulder_rot | shoulder2 | rad | -1.571 | +2.094 | +0.000 | False | True |
| 20 | elbow_flexion | elbow | rad | +0.000 | +2.269 | +0.000 | False | True |
| 21 | pro_sup | radioulnar | rad | -1.571 | +1.571 | -0.000 | False | True |
| 22 | deviation | radiocarpal | ? | -0.175 | +0.436 | +0.000 | False | True |
| 23 | flexion | radiocarpal | ? | -1.222 | +1.222 | +0.000 | False | True |
| 24 | wrist_hand_r1 | wrist_hand | rad | -99999.900 | +99999.900 | +0.000 | False | False |
| 25 | wrist_hand_r3 | wrist_hand | rad | -99999.900 | +99999.900 | +0.000 | False | False |

### Markers (name, body, local xyz[m])
| # | name | body | x_local | y_local | z_local |
|---|------|------|---------|---------|---------|
| 0 | R.Clavicle | clavicle | +0.018 | -0.005 | -0.000 |
| 1 | C7 | thorax | -0.056 | +0.055 | -0.002 |
| 2 | R.Shoulder | scapula | +0.001 | -0.001 | +0.014 |
| 3 | R.Bicep | humerus | +0.016 | -0.181 | -0.006 |
| 4 | R.Elbow.Lateral | humerus | +0.005 | -0.281 | +0.029 |
| 5 | R.Forearm | radius | +0.038 | -0.113 | +0.017 |
| 6 | R.Radius | radius | +0.057 | -0.227 | +0.027 |
| 7 | Handle | hand | +0.053 | -0.084 | +0.009 |
| 8 | R.Elbow.Medial | humerus | -0.004 | -0.284 | -0.054 |
| 9 | R.Ulna | ulna | -0.017 | -0.242 | +0.048 |

---
