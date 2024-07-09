import os
import pathlib
import argparse
from natsort import natsorted
import numpy as np
from tqdm import tqdm

from src.utils.camera import load_extrinsic_matrix
from src.utils.lie_math import matrix_to_xyz_quat
from src.utils.pose_interpolator import PoseInterpolator


def main(args):
    # Load reference poses
    ref_poses = np.loadtxt(args.ref_pose_file)  # timestamp, x, y, z, qw, qx, qy, qz
    print(f"Loaded {len(ref_poses)} poses ({args.ref_pose_file})")

    pose_interpolator = PoseInterpolator(ref_poses)

    # Load the target timestamps
    target_timestamps = np.loadtxt(args.target_timestamps)
    print(f"Loaded {len(target_timestamps)} timestamps ({args.target_timestamps})")

    # Load the extrinsic
    Hcl = load_extrinsic_matrix(args.extrinsic)  # source: LiDAR, target: camera
    Hlc = np.linalg.inv(Hcl)

    cam_poses = []
    for ts in tqdm(target_timestamps):
        Hwl = pose_interpolator.get_interpolated_transform(ts)
        cam_poses.append([ts, *matrix_to_xyz_quat(Hwl @ Hlc)])

    np.savetxt(
        args.out_pose_file,
        cam_poses,
        fmt="%.6f %.8f %.8f %.8f %.8f %.8f %.8f %.8f",
    )

    print(f"Saved the camera poses to {args.out_pose_file}", end="\n")


def get_args():
    parser = argparse.ArgumentParser(description="Synchronize camera pose")
    parser.add_argument("--ref_pose_file", type=str)
    parser.add_argument("--target_timestamps", type=str)
    parser.add_argument("--extrinsic", type=str)
    parser.add_argument("--out_pose_file", type=str)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_pose_file), exist_ok=True)

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
