"""
Description: Visualize the projection of pointcloud to the image plane
"""

import argparse
import pathlib
from natsort import natsorted
import numpy as np
import cv2

from utils.coda_utils import load_camera_params, load_extrinsic_matrix
from utils.pose_interpolator import PoseInterpolator
from utils.projection import project_to_image, project_to_rectified
from utils.visualization import visualize_points_on_image


def main(args):
    print(f"Project pointcloud to image plane for {args.scene} scene of {args.dataset}")

    # Load the pointcloud and poses
    pc_files = list(map(str, natsorted(args.pc_dir.glob("*.bin"))))
    pc_poses = np.loadtxt(args.pc_posefile)[:, :8]
    assert len(pc_files) == len(pc_poses), f"{len(pc_files)} != {len(pc_poses)}"
    print(f" * Loaded {len(pc_files)} pointclouds and poses")

    # Load the images and timestamps
    img_files = list(map(str, natsorted(args.img_dir.glob("*.jpg"))))
    timestamps = np.loadtxt(args.img_timestamps)
    assert len(img_files) == len(timestamps), f"{len(img_files)} != {len(timestamps)}"
    print(f" * Loaded {len(img_files)} images and poses")

    # Load the camera intrinsics and extrinsics
    cam_params = load_camera_params(args.intrinsic_file)
    Hcl = load_extrinsic_matrix(args.extrinsic_file)

    # Create pose interpolator
    pose_interpolator = PoseInterpolator(pc_poses)

    for img_file, time in zip(img_files[:: args.step], timestamps[:: args.step]):
        img = cv2.imread(img_file)

        pc_idx = pose_interpolator.find_closest_index(time)
        ref_pc = np.fromfile(pc_files[pc_idx], dtype=np.float32).reshape(-1, 3)
        relative_transform = pose_interpolator.get_relative_transform(
            source_time=pc_poses[pc_idx, 0], target_time=time
        )
        curr_pc = ref_pc @ relative_transform[:3, :3].T + relative_transform[:3, 3].T

        # Project the pointcloud to the image plane
        if args.img_type == "raw":
            pc_img, pc_depth, valid_indices = project_to_image(
                curr_pc, Hcl, cam_params["K"], cam_params["D"], img.shape[:2]
            )
        elif args.img_type == "rect":
            pc_img, pc_depth, valid_indices = project_to_rectified(
                curr_pc, Hcl, cam_params["R"], cam_params["P"], img.shape[:2]
            )
        else:
            raise NotImplementedError(f"{args.img_type} is not implemented")

        # Draw the projected points
        visualize_points_on_image(img, pc_img, pc_depth)


def get_args():
    parser = argparse.ArgumentParser(description="Project pointcloud to image plane")
    parser.add_argument(
        "--dataset", type=str, default="CODa", choices=["CODa", "Wanda"], help="Dataset"
    )
    parser.add_argument("--dataset_dir", type=str, help="Path to the dataset")
    parser.add_argument("--scene", type=str, help="Scene name")
    parser.add_argument(
        "--img_type",
        type=str,
        default="raw",
        choices=["raw", "rect"],
        help="Image type",
    )
    parser.add_argument("--step", type=int, default=100, help="Step size")
    args = parser.parse_args()

    args.dataset_dir = pathlib.Path(args.dataset_dir)
    if args.dataset == "CODa":
        args.pc_dir = args.dataset_dir / "3d_comp" / "os1" / args.scene
        args.pose_file = args.dataset_dir / "poses" / f"{args.scene}.txt"
        args.img_dir = args.dataset_dir / f"2d_{args.img_type}" / args.scene / "cam0"
        args.calib_dir = (
            args.dataset_dir
            / "calibrations"
            / args.scene
            / "calib_cam0_intrinsics.yaml"
        )
    elif args.dataset == "Wanda":
        args.pc_dir = args.dataset_dir / "3d_comp" / args.scene
        args.pc_posefile = args.dataset_dir / "poses" / args.scene / "os1.txt"
        # args.img_dir = args.dataset_dir / f"2d_{args.img_type}" / args.scene / "left"
        # args.img_timestamps = (
        #     args.dataset_dir / "timestamps" / args.scene / "img_left.txt"
        # )
        # args.intrinsic_file = (
        #     args.dataset_dir / "calibrations" / args.scene / "cam_left_intrinsics.yaml"
        # )
        # args.extrinsic_file = (
        #     args.dataset_dir / "calibrations" / args.scene / "os_to_cam_left.yaml"
        # )
        args.img_dir = args.dataset_dir / f"2d_{args.img_type}" / args.scene / "right"
        args.img_timestamps = (
            args.dataset_dir / "timestamps" / args.scene / "img_right.txt"
        )
        args.intrinsic_file = (
            args.dataset_dir / "calibrations" / args.scene / "cam_right_intrinsics.yaml"
        )
        args.extrinsic_file = (
            args.dataset_dir / "calibrations" / args.scene / "os_to_cam_right.yaml"
        )

    else:
        raise NotImplementedError(f"{args.dataset} is not implemented")

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
