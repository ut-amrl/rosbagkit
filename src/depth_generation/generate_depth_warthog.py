"""
Aurthor: Dongmyeong Lee (domlee[at]cs.utexas.edu)
Date:    Jun 11, 2024
Description: Generate depth images from pointcloud
"""

import argparse
import pathlib
from natsort import natsorted
from tqdm import tqdm
import concurrent.futures
from multiprocessing import Manager

import numpy as np
import cv2

from src.utils.camera import load_extrinsics, load_cam_params
from src.utils.transforms import xyz_quat_to_matrix
from src.utils.pose_interpolator import PoseInterpolator
from src.utils.depth import write_depth

from common import project_volume_to_depth, alternating_indices


def process_frame(idx, data, args, pose_interpolator):
    img_file = data["img_files"][idx]
    img_ts = data["img_timestamps"][idx]
    cam_type = data["cam_types"][idx]

    # accumulate the pointclouds around the image timestams in the world frame
    accumulated_pc = []
    close_pc_idx = np.argmin(np.abs(data["poses"][:, 0] - img_ts))
    for i in alternating_indices(args.window_size):
        pc_idx = close_pc_idx + i
        if pc_idx < 0 or pc_idx >= len(data["pc_files"]):
            continue

        pc = np.fromfile(data["pc_files"][pc_idx], dtype=np.float32).reshape(-1, 3)
        Hwl = xyz_quat_to_matrix(data["poses"][pc_idx, 1:])

        pc_world = pc @ Hwl[:3, :3].T + Hwl[:3, 3].T
        accumulated_pc.append(pc_world)

    # Get the transformation from the world to the rectified image plane
    # Hiw = Rect * Hcl * Hlw
    Hwl = pose_interpolator.get_interpolated_transform(img_ts)
    rect = np.eye(4)
    rect[:3, :3] = data["cam_params"][cam_type]["R"]  # rectification matrix
    Hiw = rect @ data["cam_extrinsics"][cam_type] @ np.linalg.inv(Hwl)

    # compute the depth image by projecting the pointcloud to the image plane
    img = cv2.imread(str(img_file))
    depth = project_volume_to_depth(
        img,
        accumulated_pc,
        Hiw,
        data["cam_params"][cam_type]["P"],
        volume=args.volume,
        visualize=True,
    )

    # save the depth image
    depth_file = img_file.stem.replace("rect", "depth") + ".png"
    write_depth(depth, args.depth_outdirs[cam_type] / depth_file)


def main(args):
    print(f"Generating depth images for {args.scene}")
    data = load_data(args)
    pose_interpolator = PoseInterpolator(data["poses"])

    # Generate the depth images
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        indices = list(range(len(data["img_files"])))
        with tqdm(total=len(indices), desc="Generating depth images") as pbar:
            futures = {
                executor.submit(process_frame, idx, data, args, pose_interpolator): idx
                for idx in indices
            }
            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)


def load_data(args):
    # Load the point cloud files and LiDAR poses
    pc_files = list(map(str, natsorted(args.pc_dir.glob("*.bin"))))
    poses = np.loadtxt(args.pose_file)  # timestamp, x, y, z, qw, qx, qy, qz
    assert len(pc_files) == len(poses)
    print(f"Loaded {len(pc_files)} pointclouds and poses")
    print(f"pc dir: {args.pc_dir}")
    print(f"pose file: {args.pose_file}")

    # Load the images and timestamps (front)
    img_front_files = natsorted(args.img_front_dir.glob("*.png"))
    img_front_timestamps = np.loadtxt(args.img_front_timestamps)
    assert len(img_front_files) == len(img_front_timestamps)
    print(f"Loaded {len(img_front_files)} images")

    # Load the images and timestamps (rear)
    img_rear_files = natsorted(args.img_rear_dir.glob("*.png"))
    img_rear_timestamps = np.loadtxt(args.img_rear_timestamps)
    assert len(img_rear_files) == len(img_rear_timestamps)
    print(f"Loaded {len(img_rear_files)} images")

    # Concatenate the front and rear images
    img_files = img_front_files + img_rear_files
    img_timestamps = np.concatenate([img_front_timestamps, img_rear_timestamps])
    cam_types = [0] * len(img_front_files) + [1] * len(img_rear_files)

    sorted_indices = np.argsort(img_timestamps)
    img_files = [img_files[i] for i in sorted_indices]
    img_timestamps = img_timestamps[sorted_indices]
    cam_types = [cam_types[i] for i in sorted_indices]

    # Load the calibrations
    cam_front_extrinsic = load_extrinsics(args.cam_front_extrinsic)
    cam_front_params = load_cam_params(args.cam_front_intrinsics)
    cam_rear_extrinsic = load_extrinsics(args.cam_rear_extrinsic)
    cam_rear_params = load_cam_params(args.cam_rear_intrinsics)
    cam_extrinsics = [cam_front_extrinsic, cam_rear_extrinsic]
    cam_params = [cam_front_params, cam_rear_params]

    data = {
        "pc_files": pc_files,
        "poses": poses,
        "img_files": img_files,
        "img_timestamps": img_timestamps,
        "cam_types": cam_types,
        "cam_extrinsics": cam_extrinsics,
        "cam_params": cam_params,
    }
    return data


def get_args():
    parser = argparse.ArgumentParser(description="Generate depth images")
    parser.add_argument("--dataset_dir", type=str, help="Path to the dataset")
    parser.add_argument("--scene", type=str, help="Scene name")
    parser.add_argument(
        "--window_size", type=int, default=5, help="window size for pc accumulation"
    )
    parser.add_argument(
        "--volume", type=float, default=0.05, help="point volume for depth estimation"
    )
    parser.add_argument("--workers", type=int, default=20, help="Number of workers")
    args = parser.parse_args()

    args.dataset_dir = pathlib.Path(args.dataset_dir)

    args.pc_dir = args.dataset_dir / "3d_comp" / args.scene
    args.pose_file = args.dataset_dir / "poses" / args.scene / "os1.txt"

    args.img_front_dir = args.dataset_dir / "2d_rect" / "front" / args.scene
    args.img_front_timestamps = (
        args.dataset_dir / "timestamps" / args.scene / "img_aux_front.txt"
    )
    args.img_rear_dir = args.dataset_dir / "2d_rect" / "rear" / args.scene
    args.img_rear_timestamps = (
        args.dataset_dir / "timestamps" / args.scene / "img_aux_rear.txt"
    )

    args.calib_dir = args.dataset_dir / "calibrations" / args.scene
    args.cam_front_intrinsics = args.calib_dir / "cam_aux_front_intrinsics.yaml"
    args.cam_front_extrinsic = args.calib_dir / "os_to_cam_aux_front.yaml"
    args.cam_rear_intrinsics = args.calib_dir / "cam_aux_rear_intrinsics.yaml"
    args.cam_rear_extrinsic = args.calib_dir / "os_to_cam_aux_rear.yaml"

    args.depth_front_outdir = args.dataset_dir / "2d_depth" / "front" / args.scene
    args.depth_rear_outdir = args.dataset_dir / "2d_depth" / "rear" / args.scene
    args.depth_front_outdir.mkdir(parents=True, exist_ok=True)
    args.depth_rear_outdir.mkdir(parents=True, exist_ok=True)
    args.depth_outdirs = [args.depth_front_outdir, args.depth_rear_outdir]

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
