"""
Author: Dongmyeong Lee (domlee[at]utexas.edu)
Date:   Apr 23, 2024
Description: Motion compensation for pointclouds (ouster) using high-frequency poses
"""

import sys
import argparse
import pathlib
import warnings
from tqdm import tqdm
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

import rospy
import rosbag
from sensor_msgs.msg import PointCloud2

# https://github.com/eric-wieser/ros_numpy/issues/37
np.float = np.float32
import ros_numpy

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from utils.lie_math import xyz_quat_to_SE3, SE3_to_xyz_quat


def get_closest_poses(poses, ts):
    upper_pose_idx = np.searchsorted(poses[:, 0], ts, side="right")
    lower_pose_idx = upper_pose_idx - 1
    if upper_pose_idx == len(poses) or lower_pose_idx == -1:
        warnings.warn(f"No pose range found for {ts}. Assume there is no motion")
        if upper_pose_idx == len(poses):
            lower_pose_idx = upper_pose_idx = len(poses) - 1
        elif lower_pose_idx == -1:
            lower_pose_idx = upper_pose_idx = 0

    # Get the lower and upper poses
    lower_pose = poses[lower_pose_idx]
    upper_pose = poses[upper_pose_idx]
    return lower_pose, upper_pose


def motion_compensation(pc_msg, poses):
    # Convert the pointcloud message to numpy array
    # NOTE: the unit of the timestamp is in nanoseconds
    scan_ts = pc_msg.header.stamp.to_sec()
    pc_msg.__class__ = PointCloud2
    pc_np = ros_numpy.numpify(pc_msg)

    pc_xyz = np.vstack((pc_np["x"], pc_np["y"], pc_np["z"])).T
    # https://github.com/ouster-lidar/ouster_example/issues/184
    # https://github.com/ouster-lidar/ouster-ros/discussions/206
    timestamps = scan_ts + pc_np["t"] * 1e-9

    # remove nan values
    valid_idx = np.where(~np.isnan(pc_xyz).any(axis=1))[0]
    if len(valid_idx) != len(pc_xyz):
        pc_xyz = pc_xyz[valid_idx]
        timestamps = timestamps[valid_idx]
        print(f"removed {len(valid_idx) - len(pc_xyz)} nan values")

    # Group the pointclouds by timestamps
    unique_timestamps, inverse_indices = np.unique(timestamps, return_inverse=True)
    pc_groups = {
        timestamp: np.where(inverse_indices == idx)[0]
        for idx, timestamp in enumerate(unique_timestamps)
    }

    # Check if the scan_ts is within the pose range
    lower_pose, upper_pose = get_closest_poses(poses, scan_ts)
    if lower_pose[0] == upper_pose[0]:
        base_pose = np.array([scan_ts] + list(lower_pose[1:]))
        return pc_xyz, base_pose

    # pose at the scan_ts
    lower_SE3 = xyz_quat_to_SE3(lower_pose[1:])
    upper_SE3 = xyz_quat_to_SE3(upper_pose[1:])
    t = (scan_ts - lower_pose[0]) / (upper_pose[0] - lower_pose[0])
    base_SE3 = lower_SE3 + t * (upper_SE3 - lower_SE3)

    comp_pc = np.zeros_like(pc_xyz)
    for timestamp in unique_timestamps:
        # Find the closest pose
        lower_pose, upper_pose = get_closest_poses(poses, timestamp)
        lower_SE3 = xyz_quat_to_SE3(lower_pose[1:])
        upper_SE3 = xyz_quat_to_SE3(upper_pose[1:])

        # Interpolate the pose if the timestamp is in between the lower and upper poses
        curr_SE3 = lower_SE3
        if lower_pose[0] != upper_pose[0]:
            t = (timestamp - lower_pose[0]) / (upper_pose[0] - lower_pose[0])
            curr_SE3 = lower_SE3 + t * (upper_SE3 - lower_SE3)

        relative_SE3 = base_SE3.between(curr_SE3)
        relative_matrix = relative_SE3.transform()
        curr_pc = np.hstack(
            [pc_xyz[pc_groups[timestamp]], np.ones((len(pc_groups[timestamp]), 1))]
        )
        comp_pc[pc_groups[timestamp]] = curr_pc @ relative_matrix[:3].T

    # pose at the scan_ts
    base_pose = np.array([scan_ts] + list(SE3_to_xyz_quat(base_SE3)))
    return comp_pc, base_pose


def main(args):
    # Load the point cloud
    print(f"Reading pointclouds from {args.bagfile}...")
    bag = rosbag.Bag(args.bagfile)
    pc_msgs = sorted(
        list(bag.read_messages(topics=[args.pc_topic])),
        key=lambda x: x.message.header.stamp,
    )
    print(f"Loaded {len(pc_msgs)} pointcloud messages")

    # Load the poses
    dense_poses = np.loadtxt(args.dense_posefile)  # timestamp, x, y, z, qw, qx, qy, qz
    print(f"Loaded {len(dense_poses)} poses")

    # Interpolate the poses
    timestamps = []
    poses = []
    frame = 0
    for idx, pc_msg in tqdm(enumerate(pc_msgs), total=len(pc_msgs)):
        pc_timestamp = pc_msg.message.header.stamp.to_sec()
        timestamps.append(pc_timestamp)

        # Motion compensation
        comp_pc, pose = motion_compensation(pc_msg.message, dense_poses)

        if comp_pc.shape[0] == 0:
            warnings.warn(f"Empty pointcloud at {pc_timestamp}")
            continue

        # Save the synchronized pose
        poses.append(pose)

        # Save the compensated pointcloud
        pc_outfile = args.out_pc_dir / f"3d_comp_os1_{frame}.bin"
        comp_pc.astype(np.float32).tofile(pc_outfile)

        frame += 1

    # Save the poses
    np.savetxt(args.out_posefile, poses, fmt="%.6f %.8f %.8f %.8f %.8f %.8f %.8f %.8f")

    # Save the timestamps
    np.savetxt(args.out_timestamps, timestamps, fmt="%.6f")

    print(f"{frame} pointclouds are compensated and saved to {args.out_pc_dir}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bagfile", type=str, required=True)
    parser.add_argument("--pc_topic", type=str, required=True)
    parser.add_argument("--dense_posefile", type=str, required=True)
    parser.add_argument("--out_pc_dir", type=str, required=True)
    parser.add_argument("--out_timestamps", type=str)
    parser.add_argument("--out_posefile", type=str)
    args = parser.parse_args()

    args.out_pc_dir = pathlib.Path(args.out_pc_dir)
    args.out_pc_dir.mkdir(parents=True, exist_ok=True)
    if args.out_timestamps is None:
        args.out_timestamps = str(args.out_pc_dir / "timestamps.txt")
    if args.out_posefile is None:
        args.out_posefile = str(args.out_pc_dir / "poses.txt")

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
