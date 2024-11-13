from loguru import logger

import os
import argparse
import pathlib
from tqdm import tqdm

import numpy as np

from pose import PoseInterpolator
from pose.utils import matrix_to_xyz_quat
from camera.utils import load_extrinsics


def get_args():
    parser = argparse.ArgumentParser(description="Synchronize camera pose")
    parser.add_argument("--ref_pose_path", type=str, required=True)
    parser.add_argument("--target_timestamps", type=str, required=True)
    parser.add_argument("--extrinsic", type=str, default=None)
    parser.add_argument("--output_pose_path", type=str, required=True)
    args = parser.parse_args()

    args.output_pose_path = pathlib.Path(args.output_pose_path)
    args.output_pose_path.parent.mkdir(parents=True, exist_ok=True)

    return args


def main(args):
    logger.info("\n\033[1mSynchronize poses with interpolation\033[0m")

    if not os.path.exists(args.ref_pose_path):
        logger.error(f"Reference pose file not found: {args.ref_pose_path}")
        return
    
    if not os.path.exists(args.target_timestamps):
        logger.error(f"Target timestamps file not found: {args.target_timestamps}")
        return

    # Load reference poses and target timestamps
    ref_poses = np.loadtxt(args.ref_pose_path)  # timestamp, x, y, z, qw, qx, qy, qz
    target_timestamps = np.loadtxt(args.target_timestamps)
    logger.info(f"Load {len(ref_poses)} Reference Poses ({args.ref_pose_path})")
    logger.info(f"Load {len(target_timestamps)} timestamps ({args.target_timestamps})")

    # Load the extrinsic
    Hlc = np.eye(4)
    if args.extrinsic:
        Hcl = load_extrinsics(args.extrinsic)  # source: LiDAR, target: camera
        Hlc = np.linalg.inv(Hcl)

    # Create the pose interpolator
    pose_interpolator = PoseInterpolator(ref_poses)

    sync_poses = []
    for ts in tqdm(target_timestamps, desc="Synchronizing poses", leave=False):
        Hwl = pose_interpolator.get_interpolated_transform(ts)
        sync_poses.append([ts, *matrix_to_xyz_quat(Hwl @ Hlc)])

    # Save the synchronized poses
    np.savetxt(
        args.output_pose_path,
        sync_poses,
        fmt="%.6f %.8f %.8f %.8f %.8f %.8f %.8f %.8f",
    )
    logger.success(
        f"Saved {len(sync_poses)} synchronized poses to {args.output_pose_path}"
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
