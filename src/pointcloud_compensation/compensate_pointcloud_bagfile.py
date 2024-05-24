"""
Author: Dongmyeong Lee (domlee[at]utexas.edu)
Date:   Apr 23, 2024
Description: Motion compensation for pointclouds (ouster) using high-frequency poses
"""

import argparse
import pathlib
import warnings
from tqdm import tqdm

import numpy as np
from scipy.spatial.transform import Rotation as R

import rosbag
from sensor_msgs.msg import PointCloud2

# https://github.com/eric-wieser/ros_numpy/issues/37
np.float = np.float32
import ros_numpy

from src.utils.pose_interpolator import PoseInterpolator


def motion_compensation(pc_msg, pose_interpolator):
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
    if not pose_interpolator.is_time_in_range(scan_ts):
        return pc_xyz, pose_interpolator.get_interpolated_pose(scan_ts)

    compensated_pc = np.zeros_like(pc_xyz)
    for timestamp in unique_timestamps:
        source_pc = np.hstack(
            [pc_xyz[pc_groups[timestamp]], np.ones((len(pc_groups[timestamp]), 1))]
        )

        relative_transform = pose_interpolator.get_relative_transform(
            source_time=timestamp, target_time=scan_ts
        )

        compensated_pc[pc_groups[timestamp]] = source_pc @ relative_transform[:3].T

    # pose at the scan_ts
    return compensated_pc, pose_interpolator.get_interpolated_pose(scan_ts)


def main(args):
    # Load the point cloud
    print(f"Reading pointclouds from {args.bagfile}...")
    bag = rosbag.Bag(args.bagfile)
    pc_msgs = sorted(
        list(bag.read_messages(topics=[args.pc_topic])),
        key=lambda x: x.message.header.stamp,
    )
    print(f" * Loaded {len(pc_msgs)} pointcloud messages")

    # Load the poses
    ref_poses = np.loadtxt(args.ref_posefile)  # timestamp, x, y, z, qw, qx, qy, qz
    print(f" * Loaded {len(ref_poses)} LiDAR poses")

    # Create the pose interpolator
    pose_interpolator = PoseInterpolator(ref_poses)

    # Interpolate the poses
    timestamps = []
    poses = []
    for frame, pc_msg in tqdm(enumerate(pc_msgs), total=len(pc_msgs)):
        # Motion compensation
        compensated_pc, pose = motion_compensation(pc_msg.message, pose_interpolator)

        if compensated_pc.shape[0] == 0:
            warnings.warn(f"Empty pointcloud at {pc_msg.message.header.stamp.to_sec()}")
            continue

        # Save the pose at the scan timestamp
        timestamps.append(pc_msg.message.header.stamp.to_sec())
        poses.append(pose)
        assert pose[0] == pc_msg.message.header.stamp.to_sec()

        # Save the compensated pointcloud
        pc_outfile = args.out_pc_dir / f"3d_comp_os1_{frame}.bin"
        compensated_pc.astype(np.float32).tofile(pc_outfile)

    # Save the poses and timestamps
    np.savetxt(args.out_posefile, poses, fmt="%.6f %.8f %.8f %.8f %.8f %.8f %.8f %.8f")
    np.savetxt(args.out_timestamps, timestamps, fmt="%.6f")

    print(f"{frame} pointclouds are compensated and saved to {args.out_pc_dir}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bagfile", type=str, required=True)
    parser.add_argument("--pc_topic", type=str, required=True)
    parser.add_argument("--ref_posefile", type=str, required=True)
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
