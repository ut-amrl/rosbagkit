"""
Aurthor: Dongmyeong Lee (domlee[at]cs.utexas.edu)
Date:    Jun 11, 2024
Description: Generate depth images from pointcloud
"""

import os
import argparse
import pathlib
from natsort import natsorted
from tqdm import tqdm
import concurrent.futures
from multiprocessing import Manager

import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import matplotlib.pyplot as plt

from src.utils.camera import load_extrinsic_matrix, load_camera_params
from src.utils.transforms import xyz_quat_to_matrix
from src.utils.projection import project_to_rectified
from src.utils.pose_interpolator import PoseInterpolator
from src.utils.depth import show_depth, write_depth, read_depth
from src.utils.visualization import visualize_rgbd_image


def compute_depth(img, accumulated_pc, extrinsic, P, voxel_size):
    imgH, imgW = img.shape[:2]
    depth = np.zeros((imgH, imgW), dtype=np.float32)

    # project the pointclouds to the image plane (assume points with volume)
    for pc in accumulated_pc:

        # project the pointcloud to the rectified image plane
        pc_img, pc_depth, _ = project_to_rectified(pc, extrinsic, P, (imgH, imgW))

        # sort the points by depth (from near to far)
        sort_idx = np.argsort(pc_depth)

        # compute the voxel size in the image plane (occlusion-aware depth estimation)
        dx = P[0, 0] * voxel_size / pc_depth
        dy = P[1, 1] * voxel_size / pc_depth

        for i in sort_idx:
            min_x = max(0, round(pc_img[i, 0] - dx[i] / 2))
            min_y = max(0, round(pc_img[i, 1] - dy[i] / 2))
            max_x = min(imgW, round(pc_img[i, 0] + dx[i] / 2))
            max_y = min(imgH, round(pc_img[i, 1] + dy[i] / 2))

            # Use broadcasting to efficiently update the depth map
            mask = depth[min_y:max_y, min_x:max_x] == 0
            depth[min_y:max_y, min_x:max_x][mask] = pc_depth[i]

    return depth


def alternating_indices(window_size):
    """Generate alternating indices from -window_size//2 to window_size//2

    Example: window_size=7 -> [0, -1, 1, -2, 2, -3, 3]
    """
    yield 0
    for i in range(1, window_size // 2 + 1):
        yield -i
        yield i


def process_image(idx, data, args, pose_interpolator):
    img_file = data["img_files"][idx]
    img_ts = data["img_timestamps"][idx]

    Hcw = data["extrinsic"] @ np.linalg.inv(
        pose_interpolator.get_interpolated_transform(img_ts)
    )  # Hcw = Hcl * Hlw

    # accumulate the pointclouds around the image timestamp
    accumulated_pc = []
    close_pc_idx = np.argmin(np.abs(data["poses"][:, 0] - img_ts))
    for i in alternating_indices(args.window_size):
        pc_idx = close_pc_idx + i
        if pc_idx < 0 or pc_idx >= len(data["pc_files"]):
            continue

        pc = np.fromfile(data["pc_files"][pc_idx], dtype=np.float32).reshape(-1, 3)
        hwl = xyz_quat_to_matrix(data["poses"][pc_idx, 1:])

        pc_world = pc @ hwl[:3, :3].T + hwl[:3, 3].T
        accumulated_pc.append(pc_world)

    # compute the depth image
    img = cv2.imread(str(img_file))
    depth = compute_depth(
        img, accumulated_pc, Hcw, data["cam_params"]["P"], voxel_size=args.voxel_size
    )

    # save the depth image
    depth_file = img_file.stem.replace("rect", "depth") + ".png"
    write_depth(depth, args.depth_outdir / depth_file)


def main(args):
    print(f"Generating depth images for {args.scene}")
    data = load_data(args)
    pose_interpolator = PoseInterpolator(data["poses"])

    # Generate the depth images
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        indices = list(range(len(data["img_files"])))
        with tqdm(total=len(indices), desc="Generating depth images") as pbar:
            futures = {
                executor.submit(process_image, idx, data, args, pose_interpolator): idx
                for idx in indices
            }
            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)


def load_data(args):
    # Load the point cloud files and poses
    pc_files = list(map(str, natsorted(args.pc_dir.glob("*.bin"))))
    poses = np.loadtxt(args.pose_file)  # timestamp, x, y, z, qw, qx, qy, qz
    assert len(pc_files) == len(poses)
    print(f"Loaded {len(pc_files)} pointclouds and poses")
    print(f"pc dir: {args.pc_dir}")
    print(f"pose file: {args.pose_file}")

    # Load the images and timestamps
    img_files = natsorted(args.img_dir.glob("*.jpg"))
    img_timestamps = np.loadtxt(args.img_timestamps)
    assert len(img_files) == len(img_timestamps)
    print(f"Loaded {len(img_files)} images")

    # Load the calibrations
    extrinsic = load_extrinsic_matrix(args.cam_extrinsics)
    cam_params = load_camera_params(args.cam_intrinsics)

    data = {
        "pc_files": pc_files,
        "poses": poses,
        "img_files": img_files,
        "img_timestamps": img_timestamps,
        "cam_params": cam_params,
        "extrinsic": extrinsic,
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
        "--voxel_size", type=float, default=0.05, help="voxel size for depth estimation"
    )
    parser.add_argument("--workers", type=int, default=20, help="Number of workers")
    args = parser.parse_args()

    args.dataset_dir = pathlib.Path(args.dataset_dir)

    args.pc_dir = args.dataset_dir / "3d_comp" / args.scene
    args.pose_file = args.dataset_dir / "poses" / args.scene / "os1.txt"

    args.img_dir = args.dataset_dir / "2d_rect" / args.scene
    args.img_timestamps = args.dataset_dir / "timestamps" / args.scene / "img_aux.txt"

    args.calib_dir = args.dataset_dir / "calibrations" / args.scene
    args.cam_intrinsics = args.calib_dir / "cam_aux_intrinsics.yaml"
    args.cam_extrinsics = args.calib_dir / "os_to_cam_aux.yaml"

    args.depth_outdir = args.dataset_dir / "2d_depth" / args.scene
    args.depth_outdir.mkdir(parents=True, exist_ok=True)

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
