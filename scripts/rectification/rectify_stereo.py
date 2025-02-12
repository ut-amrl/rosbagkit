from loguru import logger

import argparse
import pathlib
from tqdm import tqdm
import bisect

import numpy as np
import cv2

from camera.rectification import StereoRectifier
from camera.utils import load_cam_params, load_extrinsics


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--left_calib", type=str, required=True)
    parser.add_argument("--right_calib", type=str, required=True)
    parser.add_argument("--left_timestamps", type=str, required=True)
    parser.add_argument("--right_timestamps", type=str, required=True)
    parser.add_argument("--extrinsics", type=str, required=True)
    args = parser.parse_args()

    args.image_path = pathlib.Path(args.image_path)
    args.image_left_dir = args.image_path / "cam_left"
    args.image_right_dir = args.image_path / "cam_right"

    args.output_path = pathlib.Path(args.output_path)
    args.output_left_dir = args.output_path / "cam_left"
    args.output_right_dir = args.output_path / "cam_right"
    args.output_left_dir.mkdir(parents=True, exist_ok=True)
    args.output_right_dir.mkdir(parents=True, exist_ok=True)
    return args


def sync_indices_closest(left_timestamps, right_timestamps, threshold=0.005):
    """
    Synchronize left and right timestamps by finding, for each left timestamp,
    the closest right timestamp (that hasn't been paired yet) within the threshold.
    """
    left_idx = []
    right_idx = []
    used_right = set()  # To avoid pairing the same right timestamp twice
    timestamps = []

    for i, lt in enumerate(left_timestamps):
        # Find the insertion point for lt in the sorted right_timestamps list.
        pos = bisect.bisect_left(right_timestamps, lt)
        best_j = None
        best_diff = threshold  # Only accept pairs with diff below threshold

        # Check both the candidate at pos-1 and pos, if available.
        for j in [pos - 1, pos]:
            if 0 <= j < len(right_timestamps) and j not in used_right:
                diff = abs(lt - right_timestamps[j])
                if diff < best_diff:
                    best_diff = diff
                    best_j = j
        if best_j is not None:
            left_idx.append(i)
            right_idx.append(best_j)
            used_right.add(best_j)
            timestamps.append(lt)
    return left_idx, right_idx, timestamps


def main(args):
    logger.info(f"Rectifying stereo images from {args.image_path}")

    T_0to1 = load_extrinsics(args.extrinsics)

    left_cam_params = load_cam_params(args.left_calib)
    right_cam_params = load_cam_params(args.right_calib)
    image_size = left_cam_params["img_size"][::-1]
    K_left = left_cam_params["K"]
    D_left = left_cam_params["D"]
    K_right = right_cam_params["K"]
    D_right = right_cam_params["D"]

    stereo_rectifier = StereoRectifier(
        K_left, D_left, K_right, D_right, image_size, T_0to1[:3, :3], T_0to1[:3, 3]
    )

    # Load image files
    image_left_files = list(sorted(args.image_left_dir.glob("*.png")))
    image_right_files = list(sorted(args.image_right_dir.glob("*.png")))

    # sync image timestamps
    left_ts = np.loadtxt(args.left_timestamps)
    right_ts = np.loadtxt(args.right_timestamps)
    left_idx, right_idx, ts = sync_indices_closest(left_ts, right_ts)
    logger.debug(f"timestamp sync ({len(left_idx)}, {len(left_ts)}) -> {len(left_idx)}")
    np.savetxt(args.output_path / "timestamps.txt", ts, fmt="%.6f")

    # Rectify stereo images
    for frame, (li, ri) in tqdm(
        enumerate(zip(left_idx, right_idx), start=1),
        desc="Rectifying Stereo Images",
        total=len(left_idx),
    ):
        img_left = cv2.imread(str(image_left_files[li]))
        img_right = cv2.imread(str(image_right_files[ri]))

        left_rectified = stereo_rectifier.rectify(img_left, left=True)
        right_rectified = stereo_rectifier.rectify(img_right, left=False)

        left_output_file = args.output_left_dir / f"2d_rect_left_{frame:05d}.png"
        right_output_file = args.output_right_dir / f"2d_rect_right_{frame:05d}.png"

        cv2.imwrite(str(left_output_file), left_rectified)
        cv2.imwrite(str(right_output_file), right_rectified)

    logger.success(f"Rectified stereo images saved to {args.output_path}")


if __name__ == "__main__":
    args = get_args()
    main(args)
