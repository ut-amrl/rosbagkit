"""
Author: Dongmyeong Lee
Date:   Apr 30, 2024
Description: GICP pointcloud registration
"""

import os
import sys
import argparse
import pathlib
from natsort import natsorted
import time
from tqdm import tqdm

import numpy as np
import open3d as o3d
import pygicp  # https://github.com/SMRT-AIST/fast_gicp
import matplotlib.pyplot as pyplot

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from utils.lie_math import xyz_quat_to_matrix, matrix_to_xyz_quat


def main(args):
    # Load pose and pointcloud
    odom_poses = np.loadtxt(args.pose_file)[:, :8]  # timestamp, x, y, z, qw, qx, qy, qz
    pc_files = natsorted(args.pc_dir.glob("*.bin"))
    assert len(odom_poses) == len(pc_files), f"{len(odom_poses)} != {len(pc_files)}"
    print(f"Loaded {len(odom_poses)} poses and pointclouds")

    estimated_poses = odom_poses.copy()

    # You can choose any of FastGICP, FastVGICP, FastVGICPCuda, or NDTCuda
    reg = None
    if args.option == "FastGICP":
        reg = pygicp.FastGICP()
    elif args.option == "FastVGICP":
        reg = pygicp.FastVGICP()
    else:
        raise ValueError(f"Invalid option: {args.option}")

    print(f"Running {args.option} for {args.pc_dir}...")

    reg.set_num_threads(args.threads)
    reg.set_max_correspondence_distance(0.5)

    stamps = [time.time()]  # for FPS calculation
    poses = [np.identity(4)]  # sensor trajectory

    for i, filename in tqdm(enumerate(pc_files), total=len(pc_files)):
        # Load the pointcloud
        points = np.fromfile(filename, dtype=np.float32).reshape(-1, 3)
        # remove points that are too close to the sensor
        points = points[np.linalg.norm(points, axis=1) > args.blind]
        # downsample the pointcloud
        points = pygicp.downsample(points, 0.1)

        if i == 0:
            reg.set_input_target(points)
            delta = np.identity(4)
            H_prev = xyz_quat_to_matrix(odom_poses[i][1:])
        else:
            # compute the relative transformation
            H_curr = xyz_quat_to_matrix(odom_poses[i][1:])
            initial_guess = np.linalg.inv(H_prev).dot(H_curr)
            H_prev = H_curr

            # align the pointcloud
            reg.set_input_source(points)
            delta = reg.align(initial_guess)
            reg.swap_source_and_target()

        # Accumulate the delta to compute the full sensor trajectory
        poses.append(poses[-1].dot(delta))
        estimated_poses[i, 1:] = matrix_to_xyz_quat(poses[-1])

        # FPS calculation for the last ten frames
        stamps = stamps[-9:] + [time.time()]
        # print("fps:%.3f" % (len(stamps) / (stamps[-1] - stamps[0])))

        # Plot the estimated trajectory
        if args.plot and i % 30 == 0:
            pyplot.clf()
            traj = np.array([x[:3, 3] for x in poses])
            pyplot.plot(traj[:, 0], traj[:, 1])
            pyplot.axis("equal")
            pyplot.pause(0.01)

    # Save the estimated trajectory
    np.savetxt(
        args.out_pose_file,
        estimated_poses,
        fmt="%.6f %.8f %.8f %.8f %.8f %.8f %.8f %.8f",
    )
    print(f"Saved the estimated poses to {args.out_pose_file}\n")


def get_args():
    parser = argparse.ArgumentParser(description="GICP pointcloud registration")
    parser.add_argument("--pc_dir", type=str, required=True)
    parser.add_argument("--pose_file", type=str, required=True)
    parser.add_argument("--out_pose_file", type=str, required=True)
    parser.add_argument(
        "--option", type=str, choices=["FastGICP", "FastVGICP"], default="FastVGICP"
    )
    parser.add_argument("--blind", type=float, default=2.0, help="Blind region")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    args.pc_dir = pathlib.Path(args.pc_dir)
    os.makedirs(pathlib.Path(args.out_pose_file).parent, exist_ok=True)

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
