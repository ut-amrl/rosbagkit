"""
Author: Donmgmyeong Lee (domlee[at]utexas.edu)
Date:   Jan 12, 2024
Description: Get the keyframe poses from the interactive_slam output
"""
import argparse
from pathlib import Path
from natsort import natsorted
from bisect import bisect_left, bisect_right
from tqdm import tqdm
import time

from scipy.spatial.transform import Rotation as R

import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph_dir",
        type=str,
        default="/home/dongmyeong/Projects/AMRL/CODa/interactive_slam_results/dense_keyframe/0",
        help="Path to the graph directory (output of interactive_slam)",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="/home/dongmyeong/Projects/AMRL/CODa",
        help="Path to the dataset directory",
    )
    args = parser.parse_args()
    args.dataset_dir = Path(args.dataset)
    args.timestamps_dir = args.dataset_dir / "timestamps"
    args.output_dir = Path(args.dataset) / "poses" / "dense_keyframe"
    # args.output_dir = Path(args.dataset) / "poses" / "global_keyframe"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


def load_keyframe_pose(keyframe_pose_file: str) -> np.ndarray:
    """
    Load estimated pose from a keyframe file (.data) from interactive_slam

    Args:
        keyframe_pose_file: Path to the (.data) file containing the estimated pose.

    Returns:
        keyframe_pose: (8,) estimated pose (timestamp, x, y, z, qw, qx, qy, qz)
    """
    with open(keyframe_pose_file, "r") as f:
        lines = f.readlines()

        # timestamp
        timestamp_line = lines[0].strip().split(" ")
        timestamp = float(timestamp_line[1])

        # estimated pose (SE3)
        pose_lines = lines[2:6]
        keyframe_pose_matrix = np.array(
            [list(map(float, line.split())) for line in pose_lines]
        )
        r = R.from_matrix(keyframe_pose_matrix[:3, :3])

        # estimated pose (timestamp, x, y, z, qw, qx, qy, qz)
        keyframe_pose = np.zeros(8)
        keyframe_pose[0] = timestamp
        keyframe_pose[1:4] = keyframe_pose_matrix[:3, 3]
        keyframe_pose[4:] = r.as_quat()[[3, 0, 1, 2]]

    return keyframe_pose


def get_relevant_keyframes(
    keyframes: np.array, timestamps: np.array, max_time_diff: float = 1.0
) -> np.array:
    """
    Get the relevant keyframes that are within the timestamps
    excluding keyframes that differ by more than max_time_diff

    Args:
        keyframes: (K, 8) array of poses [timestamp, x, y, z, qw, qx, qy, qz]
        timestamps: (T,) array of timestamps
        max_time_diff: maximum time difference between the keyframe and the timestamp

    Returns:
        relevant_keyframes: (K', 8) array of poses [timestamp, x, y, z, qw, qx, qy, qz]
    """
    min_ts, max_ts = timestamps[0], timestamps[-1]

    # Find the surrounding timestamps
    l_idx = bisect_left(keyframes[:, 0], min_ts - max_time_diff)
    u_idx = bisect_right(keyframes[:, 0], max_ts + max_time_diff)

    if l_idx < len(keyframes) and (keyframes[l_idx, 0] < min_ts - max_time_diff):
        l_idx += 1
    if u_idx > 0 and (keyframes[u_idx - 1, 0] > max_ts + max_time_diff):
        u_idx -= 1

    return keyframes[l_idx:u_idx]


def main():
    args = get_args()

    # Get the paths to the pose files and the timestamps
    keyframes_files = list(Path(args.graph_dir).glob("[0-9]*/data"))
    print("Load keyframe poses from: ", args.graph_dir)

    # Get the keyframe poses
    keyframes = np.array(
        sorted(
            (load_keyframe_pose(pose_file) for pose_file in keyframes_files),
            key=lambda x: x[0],  # sort by timestamp
        )
    )
    print("Keyframes: ", keyframes.shape[0])

    ts_files = natsorted(args.timestamps_dir.glob("*.txt"))

    i = 0
    for ts_file in ts_files:
        timestamps = np.loadtxt(ts_file)

        # Get the relevant keyframes
        relevant_keyframes = get_relevant_keyframes(keyframes, timestamps)

        if len(relevant_keyframes) == 0:
            continue

        # Save the relevant keyframes
        output_file = args.output_dir / f"{ts_file.stem}.txt"
        times = set()
        with open(output_file, "w") as f:
            for keyframe in relevant_keyframes:
                # ts of odom is last packet ts, so we deduct 1 to get the ts of the frame
                frame = np.searchsorted(timestamps, keyframe[0], side="left") - 1
                ts = timestamps[frame]

                if ts in times:
                    continue # sanity check
                times.add(ts)

                f.write(
                    f"{ts:.6f} " + " ".join(f"{p:.8f}" for p in keyframe[1:]) + "\n"
                )
        print(f"Saved {output_file}")
        i += 1


if __name__ == "__main__":
    main()
