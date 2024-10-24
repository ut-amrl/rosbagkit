"""
Aurthor: Dongmyeong Lee (domlee[at]cs.utexas.edu)
Date:    Jun 11, 2024
Description: Generate depth images from pointcloud
"""

import argparse
import pathlib
import shutil
from natsort import natsorted
from tqdm import tqdm
import concurrent.futures
from multiprocessing import Manager

import numpy as np
import cv2

from camera.utils import load_extrinsics, load_cam_params
from pose import PoseInterpolator
from pose.utils import xyz_quat_to_matrix
from depth.utils import write_depth

from depth_utils import project_volume_to_depth, alternating_indices


def get_args():
    parser = argparse.ArgumentParser(description="Generate depth images")
    parser.add_argument("--dataset_dir", type=str, help="Path to the dataset")
    parser.add_argument("--scene", type=str, help="Scene name")
    parser.add_argument("--cam", type=str, help="Camera name")
    parser.add_argument(
        "--window_size", type=int, default=5, help="window size for pc accumulation"
    )
    parser.add_argument(
        "--volume", type=float, default=0.08, help="point volume for depth estimation"
    )
    parser.add_argument("--workers", type=int, default=20, help="Number of workers")
    args = parser.parse_args()

    args.dataset_dir = pathlib.Path(args.dataset_dir)

    args.pc_dir = args.dataset_dir / "3d_comp" / args.scene
    args.pose_file = args.dataset_dir / "poses" / args.scene / "os1.txt"

    args.img_dir = args.dataset_dir / "2d_rect" / args.cam / args.scene
    args.img_timestamps = (
        args.dataset_dir / "timestamps" / args.scene / f"{args.cam}.txt"
    )

    args.calib_dir = args.dataset_dir / "calibrations" / args.scene
    args.cam_intrinsics = args.calib_dir / f"{args.cam}_intrinsics.yaml"
    args.cam_extrinsic = args.calib_dir / f"os_to_{args.cam}.yaml"

    args.depth_outdir = args.dataset_dir / "2d_depth" / args.cam / args.scene
    args.depth_outdir.mkdir(parents=True, exist_ok=True)

    return args


def main(args):
    print(f"Generating depth images for GQ ({args.scene}), camera: {args.cam}")
    data = load_data(args)
    pose_interpolator = PoseInterpolator(data["poses"])

    shutil.rmtree(args.depth_outdir, ignore_errors=True)

    # Generate the depth images
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        with tqdm(total=len(data["img_files"]), desc="Generating depth images") as pbar:
            futures = {
                executor.submit(process_frame, idx, data, args, pose_interpolator): idx
                for idx in range(len(data["img_files"]))
            }
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)


def load_data(args):
    # Load the point cloud files and LiDAR poses
    pc_files = list(map(str, natsorted(args.pc_dir.glob("*.bin"))))
    poses = np.loadtxt(args.pose_file)  # timestamp, x, y, z, qw, qx, qy, qz
    assert len(pc_files) == len(poses), f"{len(pc_files)} != {len(poses)}"
    print(f"Loaded {len(pc_files)} pointclouds and poses")

    # Load the images and timestamps (front)
    img_files = natsorted(args.img_dir.glob("*.jpg"))
    img_timestamps = np.loadtxt(args.img_timestamps)
    assert len(img_files) == len(img_timestamps), f"{len(img_files)} != {len(img_timestamps)}"
    print(f"Loaded {len(img_files)} images")

    # Load the calibrations
    cam_extrinsic = load_extrinsics(args.cam_extrinsic)
    cam_params = load_cam_params(args.cam_intrinsics)

    data = {
        "pc_files": pc_files,
        "poses": poses,
        "img_files": img_files,
        "img_timestamps": img_timestamps,
        "cam_extrinsic": cam_extrinsic,
        "cam_params": cam_params,
    }
    return data


def process_frame(idx, data, args, pose_interpolator):
    img_file = data["img_files"][idx]
    img_ts = data["img_timestamps"][idx]

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
    Hwl = pose_interpolator.get_interpolated_transform(img_ts)
    Hcw = data["cam_extrinsic"] @ np.linalg.inv(Hwl)  # Hcw = Hcl * Hlw

    # compute the depth image by projecting the pointcloud to the image plane
    img = cv2.imread(str(img_file))
    rect = np.eye(4)
    rect[:3, :3] = data["cam_params"]["R"]  # rectification matrix
    projection = data["cam_params"]["P"] @ rect

    depth = project_volume_to_depth(
        img,
        accumulated_pc,
        Hcw,
        projection,
        volume=args.volume,
        visualize=False,
    )

    # save the depth image
    depth_file = img_file.stem.replace("rect", "depth") + ".png"
    write_depth(depth, args.depth_outdir / depth_file)


if __name__ == "__main__":
    args = get_args()
    main(args)

