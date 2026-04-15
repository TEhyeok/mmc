"""
Phase D: 삼각측량된 3D 관절 → SMPL 초기화 피팅.

Phase C에서 생성한 월드 프레임 3D 관절에 SMPL 모델을 피팅하여
02_reproj_7view_sequential.py의 초기값을 생성합니다.

서버 실행:
  conda activate gart  (smplx 필요)
  python ~/server_scripts/02c_init_smpl_from_sam3d.py

출력: /home/elicer/sam3d_triangulated/sam3d_smpl_init.npz
"""
import os
import sys
import time
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
device = 'cuda'

# ============================================================
# 경로
# ============================================================
TRIANG_PATH = '/home/elicer/sam3d_triangulated/triangulated_3d.npz'
SMPL_MODEL_PATH = '/home/elicer/EasyMocap/data/smplx'
J_REG_PATH = '/home/elicer/EasyMocap/models/J_regressor_body25.npy'
OUT_PATH = '/home/elicer/sam3d_triangulated/sam3d_smpl_init.npz'

# 최적화 설정
STAGE1_STEPS = 100   # global_orient + transl
STAGE1_LR = 0.02
STAGE2_STEPS = 300   # + body_pose
STAGE2_LR = 0.01
STAGE3_STEPS = 150   # + betas (매 10프레임)
STAGE3_LR = 0.005
LAMBDA_POSE = 0.001  # body_pose L2 정규화
LAMBDA_BETA = 0.01   # betas L2 정규화

# ============================================================
# 로드
# ============================================================
print("Loading triangulated 3D joints...")
data = np.load(TRIANG_PATH)
joints_3d_all = data['joints_3d']       # (N, 25, 3)
reproj_errors = data['reproj_errors']   # (N, 25)

N_FRAMES = joints_3d_all.shape[0]
print(f"  {N_FRAMES} frames loaded")

# 유효 관절 마스크
MAX_REPROJ = 50.0
valid_mask = reproj_errors < MAX_REPROJ  # (N, 25)

print("Loading SMPL model...")
import smplx

body_model = smplx.create(
    model_path=SMPL_MODEL_PATH,
    model_type='smpl', gender='neutral', batch_size=1
).to(device)

J_reg = torch.tensor(
    np.load(J_REG_PATH), dtype=torch.float32, device=device
)  # (25, 6890)

# ============================================================
# Body25 → SMPL 관절 매핑 (가중치)
# ============================================================
# Body25 관절 중 SMPL과 대응되는 것만 사용
# Body25: 0=Nose, 1=Neck, 2=RShoulder, 3=RElbow, 4=RWrist,
#         5=LShoulder, 6=LElbow, 7=LWrist, 8=MidHip,
#         9=RHip, 10=RKnee, 11=RAnkle, 12=LHip, 13=LKnee, 14=LAnkle
BODY_JOINTS = list(range(15))  # Body25의 처음 15개 바디 관절만 사용

# 관절별 가중치 (큰 관절 = 높은 가중치)
JOINT_WEIGHTS = np.ones(25, dtype=np.float32)
JOINT_WEIGHTS[0] = 0.5   # Nose (자주 가려짐)
JOINT_WEIGHTS[1] = 1.5   # Neck
JOINT_WEIGHTS[8] = 2.0   # MidHip (루트, 가장 중요)
JOINT_WEIGHTS[2] = JOINT_WEIGHTS[5] = 1.5   # Shoulders
JOINT_WEIGHTS[9] = JOINT_WEIGHTS[12] = 1.5  # Hips


# ============================================================
# 피팅 루프
# ============================================================
print(f"\nSMPL 초기화 피팅 시작")
print(f"  Stage 1: orient+transl ({STAGE1_STEPS} steps, lr={STAGE1_LR})")
print(f"  Stage 2: +body_pose ({STAGE2_STEPS} steps, lr={STAGE2_LR})")
print(f"  Stage 3: +betas ({STAGE3_STEPS} steps, lr={STAGE3_LR})")

all_global_orient = np.zeros((N_FRAMES, 3), dtype=np.float32)
all_body_pose = np.zeros((N_FRAMES, 69), dtype=np.float32)
all_transl = np.zeros((N_FRAMES, 3), dtype=np.float32)

# 공유 betas (전체 시퀀스에서 동일한 체형)
betas = torch.zeros(1, 10, device=device, requires_grad=True)

# 이전 프레임 결과 (sequential init)
prev_go = torch.zeros(1, 3, device=device)
prev_bp = torch.zeros(1, 69, device=device)
prev_tr = torch.tensor([[0, 0, 3.0]], device=device)

t0 = time.time()

for fi in range(N_FRAMES):
    # 타겟 3D 관절
    target_3d = torch.tensor(
        joints_3d_all[fi], dtype=torch.float32, device=device
    )  # (25, 3)
    mask = torch.tensor(
        valid_mask[fi], dtype=torch.bool, device=device
    )  # (25,)
    weights = torch.tensor(
        JOINT_WEIGHTS, dtype=torch.float32, device=device
    )

    # 유효 바디 관절 수
    body_valid = mask[BODY_JOINTS].sum().item()
    if body_valid < 5:
        # 유효 관절 부족 → 이전 프레임 복사
        all_global_orient[fi] = prev_go.detach().cpu().numpy()
        all_body_pose[fi] = prev_bp.detach().cpu().numpy()
        all_transl[fi] = prev_tr.detach().cpu().numpy()
        continue

    # 초기값: 이전 프레임에서
    go = prev_go.clone().detach().requires_grad_(True)
    bp = prev_bp.clone().detach().requires_grad_(True)
    tr = prev_tr.clone().detach().requires_grad_(True)

    # ---------- Stage 1: global_orient + transl ----------
    optimizer = torch.optim.Adam([go, tr], lr=STAGE1_LR)
    for step in range(STAGE1_STEPS):
        optimizer.zero_grad()
        out = body_model(
            global_orient=go, body_pose=bp.detach(),
            betas=betas.detach(), transl=tr
        )
        j25 = (J_reg @ out.vertices[0])  # (25, 3)

        # 마스크 + 가중치 적용 L2 손실
        diff = (j25 - target_3d) * mask.unsqueeze(1).float() * weights.unsqueeze(1)
        loss = (diff ** 2).sum()
        loss.backward()
        optimizer.step()

    # ---------- Stage 2: + body_pose ----------
    optimizer = torch.optim.Adam([go, tr, bp], lr=STAGE2_LR)
    for step in range(STAGE2_STEPS):
        optimizer.zero_grad()
        out = body_model(
            global_orient=go, body_pose=bp,
            betas=betas.detach(), transl=tr
        )
        j25 = (J_reg @ out.vertices[0])

        diff = (j25 - target_3d) * mask.unsqueeze(1).float() * weights.unsqueeze(1)
        loss_data = (diff ** 2).sum()
        loss_reg = LAMBDA_POSE * (bp ** 2).sum()
        loss = loss_data + loss_reg
        loss.backward()
        optimizer.step()

    # ---------- Stage 3: + betas (매 10프레임) ----------
    if fi % 10 == 0:
        betas.requires_grad_(True)
        optimizer = torch.optim.Adam([betas], lr=STAGE3_LR)
        for step in range(STAGE3_STEPS):
            optimizer.zero_grad()
            out = body_model(
                global_orient=go.detach(), body_pose=bp.detach(),
                betas=betas, transl=tr.detach()
            )
            j25 = (J_reg @ out.vertices[0])

            diff = (j25 - target_3d) * mask.unsqueeze(1).float() * weights.unsqueeze(1)
            loss_data = (diff ** 2).sum()
            loss_reg = LAMBDA_BETA * (betas ** 2).sum()
            loss = loss_data + loss_reg
            loss.backward()
            optimizer.step()
        betas.requires_grad_(False)

    # 결과 저장
    all_global_orient[fi] = go.detach().cpu().numpy()
    all_body_pose[fi] = bp.detach().cpu().numpy()
    all_transl[fi] = tr.detach().cpu().numpy()

    # Sequential: 다음 프레임 초기값
    prev_go = go.detach()
    prev_bp = bp.detach()
    prev_tr = tr.detach()

    if (fi + 1) % 50 == 0 or fi == 0:
        elapsed = time.time() - t0
        fps = (fi + 1) / elapsed
        eta = (N_FRAMES - fi - 1) / fps
        # 현재 프레임 MPJPE
        with torch.no_grad():
            out = body_model(
                global_orient=go, body_pose=bp,
                betas=betas, transl=tr
            )
            j25 = (J_reg @ out.vertices[0])
            err_mm = ((j25 - target_3d)[mask] * 1000).norm(dim=1).mean().item()
        print(f"  [{fi+1}/{N_FRAMES}] MPJPE={err_mm:.1f}mm, "
              f"{fps:.1f} fps, ETA {eta:.0f}s")

# ============================================================
# 저장
# ============================================================
betas_np = betas.detach().cpu().numpy().flatten()

np.savez_compressed(
    OUT_PATH,
    global_orient=all_global_orient,  # (N, 3)
    body_pose=all_body_pose,          # (N, 69)
    transl=all_transl,                # (N, 3)
    betas=betas_np,                   # (10,)
)

elapsed = time.time() - t0
print(f"\n{'='*60}")
print(f"SMPL init saved: {OUT_PATH}")
print(f"  {N_FRAMES} frames in {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"  betas: {betas_np.round(3)}")
print(f"{'='*60}")
