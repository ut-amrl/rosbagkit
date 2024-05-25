"""
Author: Dongmyeong Lee (domlee[at]utexas.edu)
Date:   Apr 22, 2024
Description: Compensate Pointcloud with high-frequency pose
"""

import os
import pathlib
import argparse
from tqdm import tqdm
from natsort import natsorted

import numpy as np

from src.utils.pose_interpolator import PoseInterpolator


def motion_compensation(pc_file, scan_ts, dt, pose_interpolator):
    N_HORIZON = 1024
    N_RING = 128

    try:
        points = np.fromfile(pc_file, dtype=np.float32).reshape(N_RING, N_HORIZON, -1)
        points = points[:, :, :3]  # (N_RING, N_HORIZON, 3)

    except IOError:
        raise IOError("Could not read the file:", pc_file)

    # Check if the scan_ts is within the pose range
    if not pose_interpolator.is_time_in_range(scan_ts):
        return points.reshape(-1, 3), pose_interpolator.get_interpolated_pose(scan_ts)

    # time between horizontal scans
    t_values = np.linspace(0, dt, N_HORIZON, endpoint=False)  # unit: second
    timestamps = scan_ts + t_values

    # Motion compensation
    compensated_pc = np.zeros((N_RING * N_HORIZON, 3), dtype=np.float32)
    for idx, timestamp in enumerate(timestamps):
        pc_packet = np.hstack([points[:, idx, :], np.ones((N_RING, 1))])

        relative_transform = pose_interpolator.get_relative_transform(
            source_time=timestamp, target_time=scan_ts
        )

        compensated_pc[idx * N_RING : (idx + 1) * N_RING] = (
            pc_packet @ relative_transform[:3].T
        )

    return compensated_pc, pose_interpolator.get_interpolated_pose(scan_ts)


def main(args):
    print(f"Compensate Pointcloud for CODa sequence {args.seq}")

    # Pose Data
    ref_poses = np.loadtxt(args.ref_posefile)[:, :8]
    print(f"Total {len(ref_poses)} pose data loaded from {args.ref_posefile}")

    print(f"Loading pointcloud data from {args.pc_dir}")

    # Point Cloud Data and its Timestamp
    pc_files = natsorted(list(args.pc_dir.glob("*.bin")))
    timestamps = np.loadtxt(args.timestamp, dtype=np.float64)
    print(f"Total {len(pc_files)} pointcloud files and timestamps")
    assert len(pc_files) == len(timestamps), f"{len(pc_files)} != {len(timestamps)}"

    # Create the pose interpolator
    pose_interpolator = PoseInterpolator(ref_poses)

    poses = np.zeros((len(timestamps), 8))
    for idx, ts in tqdm(enumerate(timestamps), total=len(timestamps)):
        # motion compensation
        dt = (
            timestamps[idx + 1] - ts
            if idx + 1 < len(timestamps)
            else timestamps[idx] - timestamps[idx - 1]
        )
        compensated_pc, pose = motion_compensation(
            pc_files[idx], ts, dt, pose_interpolator
        )
        poses[idx] = pose

        # Save the compensated pointcloud
        pc_outfile = args.out_dir / f"3d_comp_os1_{args.seq}_{idx}.bin"
        compensated_pc.astype(np.float32).tofile(pc_outfile)

    # Save the synchronized poses
    np.savetxt(args.out_posefile, poses, fmt="%.6f %.8f %.8f %.8f %.8f %.8f %.8f %.8f")
    print(f"Saved the synchronized poses to {args.out_posefile}", end="\n")


def get_args():
    parser = argparse.ArgumentParser(description="Compensate Pointcloud")
    parser.add_argument("--dataset_dir", type=str, help="Path to the dataset directory")
    parser.add_argument("--seq", type=int, help="Sequence number")
    parser.add_argument("--ref_posefile", type=str, help="Path to the dense pose file")
    parser.add_argument("--out_posefile", type=str, help="Path to the output pose file")

    args = parser.parse_args()

    args.dataset_dir = pathlib.Path(args.dataset_dir)
    args.pc_dir = args.dataset_dir / "3d_raw" / "os1" / str(args.seq)
    args.timestamp = args.dataset_dir / "timestamps" / f"{args.seq}.txt"

    args.out_dir = args.dataset_dir / "3d_comp" / "os1" / str(args.seq)
    os.makedirs(args.out_dir, exist_ok=True)

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
