"""
Author: Donmgmyeong Lee (domlee[at]utexas.edu)
Date:   Feb 11, 2024
Description: Get the poses and pointclouds (result of odometry_saver)
"""

import argparse
from pathlib import Path
from natsort import natsorted
from tqdm import tqdm

import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d


def load_pose(pose_file: str) -> np.ndarray:
    """
    Load estimated pose from a odom file (.odom) from the odometry_saver

    Args:
        pose_file: Path to the .odom file
    Returns:
        pose: (7,) estimated pose (x, y, z, qw, qx, qy, qz)
    """
    with open(pose_file, "r") as f:
        lines = f.readlines()

        pose_matrix = np.zeros((4, 4))
        for i, line in enumerate(lines):
            pose_matrix[i] = np.array([float(x) for x in line.split()])

        pose = np.zeros(7)
        pose[:3] = pose_matrix[:3, 3]
        pose[3:] = np.roll(R.from_matrix(pose_matrix[:3, :3]).as_quat(), 1)

    return pose


def pc_to_bin(pcd_file: str, bin_file: str):
    """
    Convert a .pcd file to a .bin file

    Args:
        pcd_file: Path to the .pcd file
        bin_file: Path to the .bin file
    """
    # load the pointcloud
    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(pcd.points)

    # save the pointcloud as a .bin file
    points.astype(np.float32).tofile(bin_file)
    print(f"Saved {bin_file}")


def main(args):
    # Get the pose and pcd files (result from odoemtry_saver)
    odometry_files = natsorted(list(Path(args.odom_dir).glob("[0-9]*.odom")))
    pcd_files = natsorted(list(Path(args.odom_dir).glob("[0-9]*.pcd")))
    assert len(odometry_files) == len(pcd_files)

    if args.timestamp:
        ref_timestamps = np.loadtxt(args.timestamp, delimiter=" ", usecols=0)

    frame = 0
    for odom_file, pcd_file in tqdm(zip(odometry_files, pcd_files)):
        assert odom_file.stem == pcd_file.stem
        sec, nsec = odom_file.stem.split("_")
        ts = float(sec) + float(nsec) * 1e-9

        if args.timestamp:
            # ts of odom is last packet ts, so we deduct 1 to get the ts of the frame
            frame = np.searchsorted(ref_timestamps, ts, side="left") + args.offset
            print(f"frame: {frame}, ts: {ts}, ref_ts: {ref_timestamps[frame]}")

        # Save the pointcloud
        bin_file = args.pc_outdir / f"{args.prefix}{frame}.bin"
        pc_to_bin(str(pcd_file), str(bin_file))

        # Save the pose
        pose = load_pose(odom_file)
        with open(args.pose_outfile, "a") as f:
            f.write(f"{ts:.6f} " + " ".join(f"{p:.8f}" for p in pose) + "\n")

        if args.timestamp is None:
            frame += 1


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--odom_dir",
        type=str,
        required=True,
        help="Path to the map directory (result of odometry_saver)",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        help="Path to the timestamp file to determine the frame number",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset to add to the frame number",
    )
    parser.add_argument(
        "--pc_outdir",
        type=str,
        help="Output directory to save the pointclouds",
    )
    parser.add_argument(
        "--pose_outfile",
        type=str,
        help="Output file to save the poses",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix to add to the pointcloud file",
    )
    args = parser.parse_args()

    args.pc_outdir = Path(args.pc_outdir)
    args.pose_outfile = Path(args.pose_outfile)

    args.pc_outdir.mkdir(parents=True, exist_ok=True)
    args.pose_outfile.parent.mkdir(parents=True, exist_ok=True)
    if args.pose_outfile.exists():
        args.pose_outfile.unlink()  # remove the existing file

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
