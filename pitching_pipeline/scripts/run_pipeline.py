#!/usr/bin/env python3
"""CLI entry point for pitching biomechanics pipeline."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PipelineConfig
from src.pipeline import PitchingPipeline


def main():
    parser = argparse.ArgumentParser(
        description="7-view 240Hz 3DGS pitching biomechanics pipeline"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--steps", type=int, nargs="+", default=None,
        help="Pipeline steps to run (0-5). Default: all"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = PipelineConfig.from_yaml(args.config)
    pipeline = PitchingPipeline(config)

    print("=" * 60)
    print("GART-Pitch: 3DGS Pitching Biomechanics Pipeline")
    print("=" * 60)
    print(f"  Config: {args.config}")
    print(f"  Data: {config.data_dir}")
    print(f"  Output: {config.output_dir}")
    print(f"  Steps: {args.steps or 'all (0-5)'}")
    print(f"  Calibration: {config.calibration.method}")
    print(f"  Pose: {config.pose.method}")
    print(f"  SMPL: {config.smpl.method}")
    print(f"  Reconstruction: {config.reconstruction.method}")
    print("=" * 60)

    pipeline.run(steps=args.steps)


if __name__ == "__main__":
    main()
