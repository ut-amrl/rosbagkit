import os
import argparse
import numpy as np
from tqdm import tqdm

from pose import PoseInterpolator
from pose.utils import matrix_to_xyz_quat
from camera.utils import load_extrinsics


def get_args():
    parser = argparse.ArgumentParser(description="Synchronize camera pose")
    parser.add_argument("--ref_pose_file", type=str, required=True)
    parser.add_argument("--target_timestamps", type=str, required=True)
    parser.add_argument("--extrinsic", type=str, default=None)
    parser.add_argument("--out_pose_file", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_pose_file), exist_ok=True)

    return args


def main(args):
    print("\n\033[1mSynchronize poses with interpolation\033[0m")

    # Load reference poses and target timestamps
    ref_poses = np.loadtxt(args.ref_pose_file)  # timestamp, x, y, z, qw, qx, qy, qz
    print(f"* Loaded {len(ref_poses)} poses ({args.ref_pose_file})")
    target_timestamps = np.loadtxt(args.target_timestamps)
    print(f"* Loaded {len(target_timestamps)} timestamps ({args.target_timestamps})")

    # Load the extrinsic
    Hlc = np.eye(4)
    if args.extrinsic:
        Hcl = load_extrinsics(args.extrinsic)  # source: LiDAR, target: camera
        Hlc = np.linalg.inv(Hcl)
        print(f"* Loaded the extrinsic calibration matrix ({args.extrinsic})")

    # Create the pose interpolator
    pose_interpolator = PoseInterpolator(ref_poses)

    sync_poses = []
    for ts in tqdm(target_timestamps):
        Hwl = pose_interpolator.get_interpolated_transform(ts)
        sync_poses.append([ts, *matrix_to_xyz_quat(Hwl @ Hlc)])

    # Save the synchronized poses
    np.savetxt(
        args.out_pose_file,
        sync_poses,
        fmt="%.6f %.8f %.8f %.8f %.8f %.8f %.8f %.8f",
    )
    print(f"Saved the camera poses to {args.out_pose_file}", end="\n")


if __name__ == "__main__":
    args = get_args()
    main(args)
