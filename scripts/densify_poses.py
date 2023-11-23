"""
Author: Dongmyeong Lee (domlee[at]utexas.edu)
Date:   Nov 21, 2023
Description: Densify the poses from LiDAR odometry to the timestamps of raw LiDAR scans.
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
        "-d",
        "--dataset",
        type=str,
        default="/home/dongmyeong/Projects/AMRL/CODa",
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


def densify_pose(poses, timestamps, dense_timestamps):
    """
    Densify the poses to the given timestamps

    Args:
        poses: [N, 7] array of poses in the form [x, y, z, qw, qx, qy, qz]
        timestamps: [N] array of timestamps
        dense_timestamps: [M] array of timestamps to densify to (M > N)

    Returns:
        densified_poses: [M, 7] array of poses in the form [x, y, z, qw, qx, qy, qz]
    """
    assert len(poses) == len(timestamps)

    # Convert poses to SE(3) objects and calculate the pose changes
    poses_SE3 = []
    pose_changes_se3 = []
    for i, pose in enumerate(poses):
        poses_SE3.append(xyz_quaternion_to_SE3(*pose))
        if i > 0:
            xi = poses_SE3[i] - poses_SE3[i - 1]  # log(X_prev^-1 * X_curr)
            pose_changes_se3.append(xi)

    # Interpolate the pose changes
    densified_poses = []
    for i, ts in enumerate(dense_timestamps):
        # Find the surrounding timestamps
        upper_idx = bisect_left(timestamps, ts)

        if upper_idx == 0:
            # ts is less than any timestamp we have, use the first pose
            densified_pose = poses[0]
        elif upper_idx == len(timestamps):
            # ts is greater than any timestamp we have, use the last pose
            densified_pose = poses[-1]
        elif timestamps[upper_idx] == ts:
            # Exact match found, use the corresponding pose
            densified_pose = poses[upper_idx]
        else:
            # Interpolate between the two surrounding timestamps
            lower_idx = upper_idx - 1

            t = (ts - timestamps[lower_idx]) / (
                timestamps[upper_idx] - timestamps[lower_idx]
            )

            # Interpolate the pose change
            xi = pose_changes_se3[lower_idx]
            interpolated_pose = poses_SE3[lower_idx] + (t * xi)
            densified_pose = SE3_to_xyz_quaternion(interpolated_pose)

        densified_poses.append(densified_pose)

    return np.array(densified_poses)


if __name__ == "__main__":
    args = get_parser().parse_args()

    # Get the paths to the pose files and the timestamps
    dataset_dir = Path(args.dataset)
    poses_files = natsorted((dataset_dir / "poses").glob("*.txt"))
    ts_files = natsorted((dataset_dir / "timestamps").glob("*.txt"))
    assert len(poses_files) == len(ts_files)

    for pose_file, ts_file in tqdm(zip(poses_files, ts_files), total=len(poses_files)):
        pose_data = np.genfromtxt(pose_file)

        poses = pose_data[:, 1:]  # [x, y, z, qw, qx, qy, qz]
        timestamps = pose_data[:, 0]
        dense_timestamps = np.genfromtxt(ts_file)

        initial_pose = np.array([0, 0, 0, 1, 0, 0, 0])
        poses = np.vstack([initial_pose, poses])
        timestamps = np.hstack([dense_timestamps[0], timestamps])

        densified_poses = densify_pose(poses, timestamps * 1e6, dense_timestamps * 1e6)

        # Save the densified poses
        output_dir = dataset_dir / "poses" / "dense" / f"{pose_file.stem}.txt"
        with open(output_dir, "w") as f:
            for ts, pose in zip(dense_timestamps, densified_poses):
                f.write(f"{ts:.6f} " + " ".join(f"{x:.8f}" for x in pose) + "\n")
