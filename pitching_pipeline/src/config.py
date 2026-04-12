"""Configuration loader for pitching pipeline."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import yaml


@dataclass
class CameraConfig:
    n_cameras: int = 7
    fps: int = 240
    layout: Dict = field(default_factory=dict)


@dataclass
class ViconConfig:
    fps: int = 240
    n_markers: int = 42
    marker_set: str = "plug_in_gait"
    force_plates: int = 2
    force_plate_fps: int = 1200


@dataclass
class CalibrationConfig:
    method: str = "colmap_masked"
    camera_model: str = "PINHOLE"
    single_camera_per_folder: bool = True
    frame_step: int = 10


@dataclass
class PoseConfig:
    method: str = "openpose"
    n_joints: int = 25
    confidence_threshold: float = 0.3
    cam7_tracked: bool = True


@dataclass
class SMPLConfig:
    method: str = "easymocap"
    model_type: str = "smpl"
    gender: str = "neutral"
    lambda_theta: float = 0.001
    lambda_beta: float = 0.01
    lambda_smooth: float = 0.1
    n_iterations: int = 300


@dataclass
class ReconstructionConfig:
    method: str = "gart"
    pose_refinement: bool = True
    phase_adaptive: bool = True
    n_iterations: int = 30000


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    name: str = "pitching_subject1_set002"
    data_dir: Path = Path("../data")
    output_dir: Path = Path("../outputs")
    cameras: CameraConfig = field(default_factory=CameraConfig)
    vicon: ViconConfig = field(default_factory=ViconConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    smpl: SMPLConfig = field(default_factory=SMPLConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        config = cls()
        config.name = raw.get("project", {}).get("name", config.name)
        config.data_dir = Path(raw.get("project", {}).get("data_dir", config.data_dir))
        config.output_dir = Path(raw.get("project", {}).get("output_dir", config.output_dir))

        cam = raw.get("cameras", {})
        config.cameras = CameraConfig(
            n_cameras=cam.get("n_cameras", 7),
            fps=cam.get("fps", 240),
            layout=cam.get("layout", {}),
        )

        vic = raw.get("vicon", {})
        config.vicon = ViconConfig(
            fps=vic.get("fps", 240),
            n_markers=vic.get("n_markers", 42),
        )

        cal = raw.get("calibration", {})
        config.calibration = CalibrationConfig(
            method=cal.get("method", "colmap_masked"),
        )

        pose = raw.get("pose_estimation", {})
        config.pose = PoseConfig(
            method=pose.get("method", "openpose"),
            confidence_threshold=pose.get("confidence_threshold", 0.3),
        )

        smpl = raw.get("smpl_fitting", {})
        opt = smpl.get("optimization", {})
        config.smpl = SMPLConfig(
            method=smpl.get("method", "easymocap"),
            model_type=smpl.get("model_type", "smpl"),
            gender=smpl.get("gender", "neutral"),
            lambda_theta=opt.get("lambda_theta", 0.001),
            lambda_beta=opt.get("lambda_beta", 0.01),
            lambda_smooth=opt.get("lambda_smooth", 0.1),
            n_iterations=opt.get("n_iterations", 300),
        )

        recon = raw.get("reconstruction", {})
        pr = recon.get("pose_refinement", {})
        config.reconstruction = ReconstructionConfig(
            method=recon.get("method", "gart"),
            pose_refinement=pr.get("enabled", True),
            phase_adaptive=pr.get("phase_adaptive", True),
            n_iterations=recon.get("gart", {}).get("n_iterations", 30000),
        )

        return config
