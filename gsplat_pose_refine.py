"""
gsplat_pose_refine.py — Differentiable Rendering 기반 SMPL body_pose 정제

파이프라인:
  MHR-초기화 SMPL → gsplat rendering → photometric loss → ∂L/∂body_pose → 정제

핵심:
  body_pose를 미분 가능 변수로 설정하고,
  SMPL forward kinematics → 정점 → Gaussian 위치 → gsplat 렌더링 → L1+SSIM loss
  역전파로 body_pose를 직접 정제

사용법 (서버):
  python gsplat_pose_refine.py \
    --smpl_params reproj_7view_result/smpl/ \
    --images_dir easymocap_data/images/ \
    --calib_file vggt_calibration_result.json \
    --smpl_model_path /path/to/smpl/models/ \
    --output_dir gsplat_refined_smpl/ \
    --n_iters 100 \
    --lr 0.005
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# ============================================================
# 1. SMPL Forward Pass (정점 + 관절 위치 계산)
# ============================================================

class SMPLForward(nn.Module):
    """SMPL forward kinematics wrapper with differentiable body_pose"""

    def __init__(self, smpl_model_path: str, gender: str = 'neutral', device: str = 'cuda'):
        super().__init__()
        import smplx
        self.model = smplx.create(
            model_path=smpl_model_path,
            model_type='smpl',
            gender=gender,
            batch_size=1
        ).to(device)
        self.device = device

        # Virtual marker vertex indices (from smpl_virtual_marker_mapping.json)
        self.marker_mapping = None

    def load_marker_mapping(self, mapping_path: str):
        """Load Plug-in Gait virtual marker → SMPL vertex mapping"""
        with open(mapping_path) as f:
            self.marker_mapping = json.load(f)

    def forward(self, body_pose, global_orient, betas, transl):
        """
        Args:
            body_pose: (1, 69) axis-angle, requires_grad=True for refinement
            global_orient: (1, 3)
            betas: (1, 10)
            transl: (1, 3)
        Returns:
            vertices: (1, 6890, 3)
            joints: (1, 24, 3)
        """
        output = self.model(
            body_pose=body_pose,
            global_orient=global_orient,
            betas=betas,
            transl=transl,
            return_verts=True
        )
        return output.vertices, output.joints

    def extract_virtual_markers(self, vertices):
        """
        정제된 SMPL 정점에서 Plug-in Gait 가상 마커 위치 추출

        Args:
            vertices: (1, 6890, 3)
        Returns:
            markers: dict of marker_name → (3,) position
        """
        if self.marker_mapping is None:
            raise RuntimeError("Call load_marker_mapping() first")

        markers = {}
        for name, indices in self.marker_mapping.items():
            if name.endswith('_center'):
                # Joint center — use SMPL joint directly
                continue
            # Vertex cluster average
            verts = vertices[0, indices, :]  # (n_verts, 3)
            markers[name] = verts.mean(dim=0)  # (3,)

        return markers


# ============================================================
# 2. SMPL → Gaussian 변환
# ============================================================

class SMPLToGaussians(nn.Module):
    """SMPL 정점을 gsplat Gaussian으로 변환

    각 정점을 하나의 Gaussian으로 매핑:
    - position = vertex position
    - color = 학습 가능 (초기화: 피부색 또는 영상에서 샘플링)
    - scale = 고정 (정점 간 평균 거리 기반)
    - opacity = 학습 가능
    - quaternion = identity (회전 없음)
    """

    def __init__(self, n_verts: int = 6890, init_scale: float = 0.008, device: str = 'cuda'):
        super().__init__()
        self.n_verts = n_verts
        self.device = device

        # 학습 가능 파라미터 (색상, 불투명도)
        # 초기 색상: 피부색 근사 (sigmoid 전)
        self.colors_raw = nn.Parameter(torch.zeros(n_verts, 3, device=device))
        # 초기 불투명도: sigmoid(2.0) ≈ 0.88
        self.opacities_raw = nn.Parameter(2.0 * torch.ones(n_verts, device=device))
        # 스케일: 고정
        self.register_buffer('scales', init_scale * torch.ones(n_verts, 3, device=device))
        # 쿼터니언: identity (w=1, x=y=z=0)
        self.register_buffer('quats', torch.tensor([[1., 0., 0., 0.]], device=device).expand(n_verts, -1))

    def forward(self, vertices):
        """
        Args:
            vertices: (1, 6890, 3) from SMPL forward
        Returns:
            means, quats, scales, opacities, colors — gsplat 입력
        """
        means = vertices[0]  # (6890, 3)
        colors = torch.sigmoid(self.colors_raw)
        opacities = torch.sigmoid(self.opacities_raw)
        return means, self.quats, self.scales, opacities, colors


# ============================================================
# 3. gsplat 렌더링
# ============================================================

class GsplatRenderer:
    """gsplat 기반 differentiable renderer"""

    def __init__(self, device: str = 'cuda'):
        self.device = device

    def render(self, means, quats, scales, opacities, colors,
               K, R, tvec, width, height):
        """
        Single camera rendering

        Args:
            means: (N, 3) Gaussian positions
            quats: (N, 4) quaternions
            scales: (N, 3) scales
            opacities: (N,) opacity
            colors: (N, 3) RGB
            K: (3, 3) intrinsics
            R: (3, 3) rotation
            tvec: (3,) translation
            width, height: image dimensions
        Returns:
            rendered: (H, W, 3) rendered image
        """
        from gsplat import rasterization

        # Construct view matrix (4x4)
        viewmat = torch.eye(4, device=self.device, dtype=torch.float32)
        viewmat[:3, :3] = R
        viewmat[:3, 3] = tvec

        rendered, alpha, meta = rasterization(
            means=means.contiguous(),
            quats=quats.contiguous(),
            scales=scales.contiguous(),
            opacities=opacities.contiguous(),
            colors=colors.contiguous(),
            viewmats=viewmat.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=int(width),
            height=int(height),
            near_plane=0.01,
            far_plane=100.0,
            render_mode="RGB",
        )

        return rendered[0]  # (H, W, 3)


# ============================================================
# 4. 손실 함수
# ============================================================

class PoseRefinementLoss(nn.Module):
    """Photometric + Regularization loss for body_pose refinement"""

    def __init__(self, lambda_reproj=0.1, lambda_smooth=0.001, lambda_reg=0.0001):
        super().__init__()
        self.lambda_reproj = lambda_reproj
        self.lambda_smooth = lambda_smooth
        self.lambda_reg = lambda_reg

    def photometric_loss(self, rendered, gt_image, mask=None):
        """L1 photometric loss with optional mask weighting"""
        diff = (rendered - gt_image).abs()
        if mask is not None:
            diff = diff * mask.unsqueeze(-1)
        return diff.mean()

    def reproj_loss(self, joints_3d, keypoints_2d, K, R, tvec, confidence=None):
        """2D reprojection loss for keypoint alignment"""
        # Project 3D joints to 2D
        pts_cam = (R @ joints_3d.T + tvec.unsqueeze(-1)).T  # (J, 3)
        pts_2d = (K @ pts_cam.T).T  # (J, 3)
        pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3].clamp(min=1e-6)  # (J, 2)

        # L2 distance with robust loss (GMoF)
        diff = pts_2d - keypoints_2d
        sigma = 50.0
        loss = (diff ** 2).sum(dim=-1)
        loss = loss / (loss + sigma ** 2)  # GMoF

        if confidence is not None:
            loss = loss * confidence

        return loss.mean()

    def temporal_smoothness(self, body_pose_prev, body_pose_curr):
        """시간적 일관성: 연속 프레임 간 body_pose 변화 제한"""
        if body_pose_prev is None:
            return torch.tensor(0.0, device=body_pose_curr.device)
        return ((body_pose_curr - body_pose_prev) ** 2).mean()

    def pose_regularization(self, body_pose, init_pose):
        """초기 포즈에서 크게 벗어나지 않도록 정규화"""
        return ((body_pose - init_pose) ** 2).mean()

    def forward(self, rendered, gt_image, body_pose, init_pose,
                joints_3d=None, keypoints_2d=None, K=None, R=None, tvec=None,
                body_pose_prev=None, mask=None, confidence=None):
        """Combined loss"""
        loss = self.photometric_loss(rendered, gt_image, mask)

        if joints_3d is not None and keypoints_2d is not None:
            loss += self.lambda_reproj * self.reproj_loss(
                joints_3d, keypoints_2d, K, R, tvec, confidence)

        loss += self.lambda_smooth * self.temporal_smoothness(body_pose_prev, body_pose)
        loss += self.lambda_reg * self.pose_regularization(body_pose, init_pose)

        return loss


# ============================================================
# 5. 메인 정제 파이프라인
# ============================================================

class PoseRefiner:
    """
    gsplat differentiable rendering 기반 SMPL body_pose 정제 파이프라인

    Pipeline:
        1. SMPL params (from MHR + reproj) 로드
        2. 각 프레임에 대해:
           a. body_pose를 requires_grad=True로 설정
           b. SMPL forward → vertices → Gaussians
           c. 7시점 gsplat rendering
           d. photometric loss 계산
           e. Adam 역전파 → body_pose 업데이트
        3. 정제된 body_pose 저장
    """

    def __init__(self, config: dict):
        self.device = config.get('device', 'cuda')
        self.n_iters = config.get('n_iters', 100)
        self.lr = config.get('lr', 0.005)
        self.output_dir = config.get('output_dir', 'gsplat_refined_smpl')

        # Initialize components
        self.smpl = SMPLForward(config['smpl_model_path'], device=self.device)
        self.smpl.load_marker_mapping(config.get('marker_mapping', 'smpl_virtual_marker_mapping.json'))
        self.gaussians = SMPLToGaussians(device=self.device)
        self.renderer = GsplatRenderer(device=self.device)
        self.loss_fn = PoseRefinementLoss(
            lambda_reproj=config.get('lambda_reproj', 0.1),
            lambda_smooth=config.get('lambda_smooth', 0.001),
            lambda_reg=config.get('lambda_reg', 0.0001),
        )

        # Camera parameters
        self.cameras = None
        self.images = None

    def load_cameras(self, calib_path: str):
        """Load 7-camera calibration (VGGT format)"""
        with open(calib_path) as f:
            calib = json.load(f)

        self.cameras = []
        for cam_key in sorted(calib.keys()):
            cam = calib[cam_key]
            K = torch.tensor(cam['K'], dtype=torch.float32, device=self.device)
            R = torch.tensor(cam['R'], dtype=torch.float32, device=self.device)
            t = torch.tensor(cam['t'], dtype=torch.float32, device=self.device).squeeze()
            self.cameras.append({'K': K, 'R': R, 't': t, 'name': cam_key})

        print(f"Loaded {len(self.cameras)} cameras")

    def load_frame_images(self, images_dir: str, frame_idx: int,
                          width: int = 1920, height: int = 1080):
        """Load GT images for a specific frame from all cameras"""
        from PIL import Image
        import torchvision.transforms as T

        transform = T.Compose([T.Resize((height, width)), T.ToTensor()])

        images = []
        for cam in self.cameras:
            cam_name = cam['name']
            img_path = os.path.join(images_dir, cam_name, f'{frame_idx:06d}.jpg')
            if not os.path.exists(img_path):
                img_path = os.path.join(images_dir, cam_name, f'{frame_idx:06d}.png')

            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).permute(1, 2, 0).to(self.device)  # (H, W, 3)
                images.append(img_tensor)
            else:
                images.append(None)

        return images

    def load_smpl_params(self, params_dir: str, frame_idx: int):
        """Load SMPL parameters for a frame"""
        # Try JSON format (from reproj fitting)
        json_path = os.path.join(params_dir, f'{frame_idx:06d}.json')
        if os.path.exists(json_path):
            with open(json_path) as f:
                params = json.load(f)
            body_pose = torch.tensor(params['poses'][3:72], dtype=torch.float32, device=self.device)
            global_orient = torch.tensor(params['Rh'], dtype=torch.float32, device=self.device)
            transl = torch.tensor(params['Th'], dtype=torch.float32, device=self.device)
            betas = torch.tensor(params['shapes'], dtype=torch.float32, device=self.device)
            return body_pose, global_orient, transl, betas

        # Try NPZ format
        npz_path = os.path.join(params_dir, f'{frame_idx:06d}.npz')
        if os.path.exists(npz_path):
            data = np.load(npz_path)
            body_pose = torch.tensor(data['body_pose'], dtype=torch.float32, device=self.device)
            global_orient = torch.tensor(data['global_orient'], dtype=torch.float32, device=self.device)
            transl = torch.tensor(data['transl'], dtype=torch.float32, device=self.device)
            betas = torch.tensor(data['betas'], dtype=torch.float32, device=self.device)
            return body_pose, global_orient, transl, betas

        raise FileNotFoundError(f"No SMPL params found for frame {frame_idx}")

    def refine_frame(self, frame_idx: int, body_pose_init, global_orient,
                     transl, betas, gt_images, body_pose_prev=None):
        """
        단일 프레임 body_pose 정제

        Args:
            frame_idx: 프레임 번호
            body_pose_init: (69,) 초기 body_pose
            global_orient: (3,) 글로벌 회전
            transl: (3,) 이동
            betas: (10,) 체형
            gt_images: list of (H,W,3) 또는 None per camera
            body_pose_prev: 이전 프레임의 정제된 body_pose (시간적 일관성)

        Returns:
            refined_body_pose: (69,) 정제된 body_pose
            final_loss: 최종 손실값
        """
        # body_pose를 미분 가능 변수로 설정
        body_pose = body_pose_init.clone().detach().requires_grad_(True)
        init_pose = body_pose_init.clone().detach()

        # global_orient도 미분 가능하게 (선택적)
        go = global_orient.clone().detach().requires_grad_(True)

        # Optimizer: body_pose + global_orient
        optimizer = torch.optim.Adam([
            {'params': [body_pose], 'lr': self.lr},
            {'params': [go], 'lr': self.lr * 0.5},  # 글로벌 회전은 더 조심스럽게
        ])

        # Gaussian 색상/불투명도 optimizer (별도)
        gauss_optimizer = torch.optim.Adam(self.gaussians.parameters(), lr=self.lr * 2)

        best_loss = float('inf')
        best_pose = body_pose.clone().detach()

        for it in range(self.n_iters):
            optimizer.zero_grad()
            gauss_optimizer.zero_grad()

            # SMPL forward kinematics
            vertices, joints = self.smpl(
                body_pose.unsqueeze(0),
                go.unsqueeze(0),
                betas.unsqueeze(0),
                transl.unsqueeze(0)
            )

            # SMPL vertices → Gaussians
            means, quats, scales, opacities, colors = self.gaussians(vertices)

            # Multi-view rendering + loss accumulation
            total_loss = torch.tensor(0.0, device=self.device)
            n_valid = 0

            for cam_idx, cam in enumerate(self.cameras):
                gt = gt_images[cam_idx]
                if gt is None:
                    continue

                H, W = gt.shape[:2]

                # Render from this camera
                rendered = self.renderer.render(
                    means, quats, scales, opacities, colors,
                    cam['K'], cam['R'], cam['t'],
                    width=W, height=H
                )

                # Compute loss
                cam_loss = self.loss_fn(
                    rendered, gt,
                    body_pose, init_pose,
                    body_pose_prev=body_pose_prev
                )
                total_loss += cam_loss
                n_valid += 1

            if n_valid > 0:
                total_loss = total_loss / n_valid

            # Backprop
            total_loss.backward()

            # Gradient clipping (안정성)
            torch.nn.utils.clip_grad_norm_([body_pose, go], max_norm=1.0)

            optimizer.step()
            gauss_optimizer.step()

            # Track best
            loss_val = total_loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
                best_pose = body_pose.clone().detach()

            if it % 20 == 0 or it == self.n_iters - 1:
                print(f"  Frame {frame_idx}, iter {it:3d}/{self.n_iters}: "
                      f"loss={loss_val:.6f}, best={best_loss:.6f}")

        return best_pose, best_loss

    def refine_sequence(self, params_dir: str, images_dir: str,
                        start_frame: int = 0, end_frame: int = 999,
                        frame_step: int = 1):
        """
        전체 시퀀스 순차 정제

        Args:
            params_dir: SMPL 파라미터 디렉토리
            images_dir: GT 영상 디렉토리
            start_frame, end_frame: 프레임 범위
            frame_step: 프레임 간격
        """
        os.makedirs(self.output_dir, exist_ok=True)

        prev_pose = None
        results = []

        for fi in range(start_frame, end_frame + 1, frame_step):
            print(f"\n=== Frame {fi} ===")

            try:
                # Load SMPL params
                body_pose, go, transl, betas = self.load_smpl_params(params_dir, fi)

                # Load GT images (downsampled for speed)
                gt_images = self.load_frame_images(images_dir, fi, width=480, height=270)

                # Refine
                refined_pose, loss = self.refine_frame(
                    fi, body_pose, go, transl, betas, gt_images, prev_pose)

                # Extract virtual markers from refined pose
                with torch.no_grad():
                    verts, joints = self.smpl(
                        refined_pose.unsqueeze(0), go.unsqueeze(0),
                        betas.unsqueeze(0), transl.unsqueeze(0))
                    markers = self.smpl.extract_virtual_markers(verts)

                # Save refined parameters
                out = {
                    'body_pose': refined_pose.cpu().numpy(),
                    'global_orient': go.detach().cpu().numpy(),
                    'transl': transl.cpu().numpy(),
                    'betas': betas.cpu().numpy(),
                    'loss': loss,
                    'vertices': verts[0].cpu().numpy(),
                    'joints': joints[0].cpu().numpy(),
                    'virtual_markers': {k: v.cpu().numpy() for k, v in markers.items()},
                }
                np.savez(os.path.join(self.output_dir, f'{fi:06d}.npz'), **out)

                prev_pose = refined_pose.detach()
                results.append({'frame': fi, 'loss': loss})

            except Exception as e:
                print(f"  ERROR frame {fi}: {e}")
                continue

        # Save summary
        summary = {
            'n_frames': len(results),
            'mean_loss': np.mean([r['loss'] for r in results]),
            'config': {
                'n_iters': self.n_iters,
                'lr': self.lr,
            }
        }
        with open(os.path.join(self.output_dir, 'refinement_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n=== Refinement Complete ===")
        print(f"Frames: {len(results)}, Mean loss: {summary['mean_loss']:.6f}")
        print(f"Output: {self.output_dir}")


# ============================================================
# 6. CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='gsplat SMPL body_pose refinement')
    parser.add_argument('--smpl_params', required=True, help='SMPL params directory')
    parser.add_argument('--images_dir', required=True, help='GT images directory')
    parser.add_argument('--calib_file', required=True, help='Camera calibration JSON')
    parser.add_argument('--smpl_model_path', default='smpl/models/', help='SMPL model directory')
    parser.add_argument('--marker_mapping', default='smpl_virtual_marker_mapping.json')
    parser.add_argument('--output_dir', default='gsplat_refined_smpl/')
    parser.add_argument('--n_iters', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--end_frame', type=int, default=999)
    parser.add_argument('--frame_step', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--lambda_reproj', type=float, default=0.1)
    parser.add_argument('--lambda_smooth', type=float, default=0.001)
    parser.add_argument('--lambda_reg', type=float, default=0.0001)
    args = parser.parse_args()

    config = vars(args)

    refiner = PoseRefiner(config)
    refiner.load_cameras(args.calib_file)
    refiner.refine_sequence(
        args.smpl_params, args.images_dir,
        args.start_frame, args.end_frame, args.frame_step
    )


if __name__ == '__main__':
    main()
