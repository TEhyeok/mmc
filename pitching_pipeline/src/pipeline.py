"""Main pipeline orchestrator for pitching biomechanics analysis."""
import logging
from pathlib import Path
from typing import Optional

from .config import PipelineConfig

logger = logging.getLogger(__name__)


class PitchingPipeline:
    """End-to-end pipeline: Calibration → Pose → SMPL → 3DGS → Biomechanics → Validation.

    Each step can be run independently or as part of the full pipeline.
    Intermediate results are saved to output_dir for inspection.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, steps: Optional[list] = None):
        """Run pipeline steps.

        Args:
            steps: List of step numbers to run (0-5), or None for all.
        """
        all_steps = [
            (0, "Calibration", self.step0_calibration),
            (1, "2D Pose Estimation", self.step1_pose_2d),
            (2, "SMPL Fitting", self.step2_smpl_fitting),
            (3, "3DGS Reconstruction", self.step3_reconstruction),
            (4, "Biomechanics", self.step4_biomechanics),
            (5, "Validation", self.step5_validation),
        ]

        for num, name, func in all_steps:
            if steps is not None and num not in steps:
                continue
            logger.info(f"[Step {num}] {name}")
            func()

    def step0_calibration(self):
        """Camera calibration: masked COLMAP or DA3 PnP."""
        method = self.config.calibration.method
        logger.info(f"  Method: {method}")

        if method == "colmap_masked":
            self._run_masked_colmap()
        elif method == "da3_pnp":
            self._load_da3_cameras()
        else:
            raise ValueError(f"Unknown calibration method: {method}")

    def step1_pose_2d(self):
        """2D pose estimation + triangulation."""
        logger.info(f"  Method: {self.config.pose.method}")
        # Load OpenPose 2D keypoints (already extracted)
        # Triangulate to 3D
        # Save to output_dir/pose/

    def step2_smpl_fitting(self):
        """Multi-view SMPL fitting."""
        method = self.config.smpl.method
        logger.info(f"  Method: {method}")

        if method == "easymocap":
            self._run_easymocap()
        elif method == "direct_3d":
            self._run_direct_3d_fitting()
        elif method == "joint_optimization":
            self._run_joint_optimization()

    def step3_reconstruction(self):
        """3DGS reconstruction + pose refinement."""
        method = self.config.reconstruction.method
        logger.info(f"  Method: {method}")
        # Run GART/GauHuman/ExAvatar/3DGS-Avatar
        # Apply phase-adaptive pose refinement

    def step4_biomechanics(self):
        """Virtual markers → OpenSim → clinical metrics."""
        logger.info("  Extracting virtual markers from SMPL mesh")
        logger.info("  Running OpenSim IK/ID")
        logger.info("  Computing pitching events")
        logger.info("  Computing clinical metrics (MER, valgus torque, etc.)")

    def step5_validation(self):
        """Compare with Vicon gold standard."""
        logger.info("  Computing MPJPE, joint angle RMSE")
        logger.info("  Bland-Altman analysis")
        logger.info("  ICC computation")

    # === Private methods ===

    def _run_masked_colmap(self):
        """Run COLMAP on person-masked images."""
        logger.info("  Creating masked images (person → black)")
        logger.info("  Running COLMAP SfM on background only")
        logger.info("  Output K is in original image resolution → no transform needed")

    def _load_da3_cameras(self):
        """Load DA3 PnP refined camera parameters."""
        logger.info("  Loading cameras_da3_pnp_refined.json")
        logger.info("  WARNING: Coordinate transform from 280x280 required")

    def _run_easymocap(self):
        """Run EasyMocap multi-view SMPL fitting."""
        from .smpl.multiview_fit import run_easymocap
        logger.info("  Preparing EasyMocap input (images, annots, intri/extri.yml)")
        logger.info("  Running EasyMocap mv1p")

    def _run_direct_3d_fitting(self):
        """Run direct 3D keypoint → SMPL fitting."""
        logger.info("  WARNING: Structural limitations (joint definition mismatch)")
        logger.info("  Consider using easymocap (2D reprojection) instead")

    def _run_joint_optimization(self):
        """Run proposed joint optimization (photometric + silhouette + keypoint + temporal)."""
        logger.info("  Joint optimization: SMPL + 3DGS simultaneous")
        logger.info("  Losses: photometric(7view) + silhouette(SAM3) + keypoint(OP2D) + temporal")
        logger.info("  Phase-adaptive temporal regularization for pitching")
