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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--timestamp_file",
        type=str,
        default="/home/dongmyeong/Projects/AMRL/CODa/timestamps/0.txt",
        help="Path to the timestamps file",
    )
    parser.add_argument(
        "-p",
        "--pose_file",
        type=str,
        default="/home/dongmyeong/Projects/AMRL/CODa/interactive_slam_results/odom_poses/0.txt",
        help="Path to the pose file",
    )
    parser.add_argument(
        "-k",
        "--keep_pose",
        action="store_true",
        help="Keep the pose with unsynced timestamps",
    )
    return parser.parse_args()


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


def synchronize_pose(poses, timestamps, args):
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

        case = -1
        if upper_idx == 0:
            # ts is less than any timestamp we have, use the first pose
            case = 0
            sync_pose = poses[0]
        elif upper_idx == len(poses):
            # ts is greater than any timestamp we have, use the last pose
            case = 1
            sync_pose = poses[-1]
        elif poses[upper_idx][0] == ts:
            # Exact match found, use the corresponding pose
            case = 2
            sync_pose = poses[upper_idx]
        else:
            # Interpolate between the two surrounding timestamps
            case = 3
            lower_idx = upper_idx - 1

            t = (ts - poses[lower_idx, 0]) / (poses[upper_idx, 0] - poses[lower_idx, 0])

            # Interpolate the pose change
            pose_lower_SE3 = xyz_quaternion_to_SE3(*poses[lower_idx][1:])
            pose_upper_SE3 = xyz_quaternion_to_SE3(*poses[upper_idx][1:])
            # xi = (pose_lower_SE3.inverse() * pose_upper_SE3).log()
            xi = pose_upper_SE3 - pose_lower_SE3
            # pose_interp_SE3 = pose_lower_SE3 * (t * xi).exp()
            pose_interp_SE3 = pose_lower_SE3 + (t * xi)

            sync_pose = np.hstack([ts, SE3_to_xyz_quaternion(pose_interp_SE3)])

        sync_pose[0] = ts  # ensure the timestamp is from the timestamps file
        sync_poses.append(sync_pose)
        if args.keep_pose and case == 3:
            sync_poses.append(poses[upper_idx])

    return np.array(sync_poses)


def main():
    args = get_args()

    poses = np.genfromtxt(args.pose_file)
    poses = poses[:, :8]  # [x, y, z, qw, qx, qy, qz]

    timestamps = np.genfromtxt(args.timestamp_file)

    sync_poses = synchronize_pose(poses, timestamps, args)

    output = Path(args.pose_file).parent / "sync" / f"{Path(args.pose_file).stem}.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for p in sync_poses:
            f.write(f"{p[0]:.6f} " + " ".join(f"{x:.8f}" for x in p[1:]) + "\n")


if __name__ == "__main__":
    main()
