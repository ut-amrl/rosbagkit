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
from utils.msg_converter import np_to_pointcloud2
from utils.lie_math import xyz_qwxyz_to_SE3, SE3_to_xyz_qwxyz


def motion_compensation(pc_msg, lower_pose, upper_pose):
    # Convert the pointcloud message to numpy array
    scan_ts = pc_msg.header.stamp.to_sec()
    pc_msg.__class__ = PointCloud2
    pc_np = ros_numpy.numpify(pc_msg)

    pc_xyz = np.vstack((pc_np["x"], pc_np["y"], pc_np["z"])).T
    # https://github.com/ouster-lidar/ouster_example/issues/184
    timestamps = scan_ts + pc_np["t"] * 1e-9

    if lower_pose[0] == upper_pose[0]:
        base_pose = np.array([scan_ts] + list(lower_pose[1:]))
        return pc_xyz, base_pose

    # Interpolate the poses in Lie Algebra Manifold
    # Compensate based on the pc_msg timestamp
    lower_SE3 = xyz_qwxyz_to_SE3(lower_pose[1:])
    upper_SE3 = xyz_qwxyz_to_SE3(upper_pose[1:])
    t = (scan_ts - lower_pose[0]) / (upper_pose[0] - lower_pose[0])
    base_SE3 = lower_SE3 + (t * (upper_SE3 - lower_SE3))

    compensated_pc = np.zeros_like(pc_xyz)
    for idx, (xyz, ts) in enumerate(zip(pc_xyz, timestamps)):
        # Aware that timestamp is the last time of the points
        t = (ts - scan_ts) / (upper_pose[0] - scan_ts)
        relative_se3 = (t * (upper_SE3 - base_SE3)).exp()
        compensated_pc[idx] = relative_se3.exp().act(xyz)

    # pose at the scan_ts
    base_pose = np.array([scan_ts] + list(SE3_to_xyz_qwxyz(base_SE3)))
    return (compensated_pc, base_pose)


def main(args):
    rospy.init_node("pointcloud_compensation", anonymous=True)
    pc_pub = rospy.Publisher("/compensated_pointcloud", PointCloud2, queue_size=10)

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
    for idx, pc_msg in tqdm(enumerate(pc_msgs), total=len(pc_msgs)):
        pc_timestamp = pc_msg.message.header.stamp.to_sec()
        timestamps.append(pc_timestamp)

        # Find the lower and upper poses
        upper_pose_idx = np.searchsorted(dense_poses[:, 0], pc_timestamp, side="right")
        lower_pose_idx = upper_pose_idx - 1
        if upper_pose_idx == len(dense_poses) or lower_pose_idx == -1:
            warnings.warn(
                f"No pose range found for {pc_timestamp}. Assume there is no motion"
            )
            if upper_pose_idx == len(dense_poses):
                lower_pose_idx = upper_pose_idx = len(dense_poses) - 1
            elif lower_pose_idx == -1:
                lower_pose_idx = upper_pose_idx = 0

        # Get the lower and upper poses
        upper_pose = dense_poses[upper_pose_idx]
        lower_pose = dense_poses[lower_pose_idx]

        # Motion compensation
        comp_pc, pose = motion_compensation(pc_msg.message, lower_pose, upper_pose)
        poses.append(pose)

        # # Save the compensated pointcloud
        out_file = args.outdir / f"3d_comp_os1_{idx}.bin"
        comp_pc.astype(np.float32).tofile(out_file)

        pc_msg = np_to_pointcloud2(comp_pc, "x y z", "os1", pc_msg.message.header.stamp)
        pc_pub.publish(pc_msg)

    # Save the poses
    pose_outfile = args.pose_outfile
    np.savetxt(pose_outfile, poses, fmt="%.6f %.8f %.8f %.8f %.8f %.8f %.8f %.8f")

    # Save the timestamps
    ts_outfile = args.ts_outfile
    np.savetxt(ts_outfile, timestamps, fmt="%.6f")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bagfile", type=str, required=True)
    parser.add_argument("--pc_topic", type=str, required=True)
    parser.add_argument("--dense_posefile", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--ts_outfile", type=str)
    parser.add_argument("--pose_outfile", type=str)
    args = parser.parse_args()

    args.outdir = pathlib.Path(args.outdir)
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.ts_outfile is None:
        args.ts_outfile = args.outdir / "timestamps.txt"
    if args.pose_outfile is None:
        args.pose_outfile = args.outdir / "poses.txt"

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
