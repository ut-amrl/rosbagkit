"""
Author: Dongmyeong Lee (domlee[at]utexas.edu)
Date:   Nov 21, 2023
Description: Synchronize the poses to the timestamps of raw LiDAR scans.
             Lie Algebra is used to interpolate the poses
"""
import argparse
from pathlib import Path
from natsort import natsorted
from bisect import bisect_left
from tqdm import tqdm

import numpy as np

from manifpy import SE3, SO3


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--timestamp_dir",
        type=str,
        default="/home/dongmyeong/Projects/AMRL/CODa/timestamps",
        help="Path to the dataset",
    )
    parser.add_argument(
        "-p",
        "--pose_dir",
        type=str,
        default="/home/dongmyeong/Projects/AMRL/CODa/poses/inekfodom",
        help="Path to the dataset",
    )
    return parser


def xyz_quaternion_to_SE3(x, y, z, qw, qx, qy, qz) -> SE3:
    """Convert the [x, y, z, qw, qx, qy, qz] to an SE(3) transformation matrix"""
    position = np.array([x, y, z])
    quaternion = np.array([qx, qy, qz, qw])
    quaternion /= np.linalg.norm(quaternion)
    return SE3(position, quaternion)


def SE3_to_xyz_quaternion(X: SE3) -> np.ndarray:
    """Convert the SE(3) transformation matrix to [x, y, z, qw, qx, qy, qz]"""
    x, y, z, qx, qy, qz, qw = X.coeffs()
    return np.array([x, y, z, qw, qx, qy, qz])


def synchronize_pose(poses, timestamps):
    """
    synchronize the poses to the given timestamps

    Args:
        poses: [N, 7] array of poses in the form [ts, x, y, z, qw, qx, qy, qz]
        timestamps: [M] array of timestamps to synchronize to (M > N)

    Returns:
        sync_poses: [M, 7] array of poses in the form [ts, x, y, z, qw, qx, qy, qz]
    """

    # Interpolate the pose changes
    sync_poses = []
    for i, ts in enumerate(timestamps):
        # Find the surrounding timestamps
        upper_idx = bisect_left(poses[:, 0], ts)

        if upper_idx == 0:
            # ts is less than any timestamp we have, use the first pose
            sync_pose = poses[0]
        elif upper_idx == len(poses):
            # ts is greater than any timestamp we have, use the last pose
            sync_pose = poses[-1]
        elif poses[upper_idx][0] == ts:
            # Exact match found, use the corresponding pose
            sync_pose = poses[upper_idx]
        else:
            # Interpolate between the two surrounding timestamps
            lower_idx = upper_idx - 1

            t = (ts - poses[lower_idx, 0]) / (poses[upper_idx, 0] - poses[lower_idx, 0])

            # Interpolate the pose change
            pose_lower_SE3 = xyz_quaternion_to_SE3(*poses[lower_idx][1:])
            pose_upper_SE3 = xyz_quaternion_to_SE3(*poses[upper_idx][1:])
            # xi = log(pose_upper_SE3.inverse() * pose_lower_SE3)
            xi = pose_upper_SE3 - pose_lower_SE3
            # pose_interp_SE3 = pose_lower_SE3 * exp(t * xi)
            pose_interp_SE3 = pose_lower_SE3 + (t * xi)

            sync_pose = np.hstack([ts, SE3_to_xyz_quaternion(pose_interp_SE3)])

        sync_poses.append(sync_pose)

    return np.array(sync_poses)


if __name__ == "__main__":
    args = get_parser().parse_args()

    # Get the paths to the pose files and the timestamps
    poses_files = natsorted(Path(args.pose_dir).glob("*.txt"))

    for pose_file in tqdm(poses_files, total=len(poses_files)):
        poses = np.genfromtxt(pose_file)  # [x, y, z, qw, qx, qy, qz]
        poses = poses[:, :8]

        ts_file = Path(args.timestamp_dir) / f"{pose_file.stem}.txt"
        timestamps = np.genfromtxt(ts_file)

        sync_poses = synchronize_pose(poses, timestamps)

        # Save the synchronizied poses
        output_dir = Path(args.pose_dir) / "sync" / f"{pose_file.stem}.txt"
        with open(output_dir, "w") as f:
            for p in sync_poses:
                f.write(f"{p[0]:.6f} " + " ".join(f"{x:.8f}" for x in p[1:]) + "\n")
