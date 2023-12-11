import argparse
from pathlib import Path
from natsort import natsorted
from bisect import bisect_left, bisect_right
from tqdm import tqdm

import numpy as np

from helpers.coda_utils import load_keyframe_pose


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keyframe_poses_dir",
        type=str,
        default="/home/dongmyeong/Projects/AMRL/CODa/correction/map/ut_campus",
        help="Path to the keyframe poses directory (output of interactive_slam)",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="/home/dongmyeong/Projects/AMRL/CODa",
        help="Path to the dataset directory",
    )
    return parser


def get_relevant_keyframes(
    keyframes: np.array, timestamps: np.array, max_time_diff: int = 100
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


if __name__ == "__main__":
    args = get_parser().parse_args()

    # Get the paths to the pose files and the timestamps
    keyframes_files = list(Path(args.keyframe_poses_dir).glob("[0-9]*/data"))
    print("Load keyframe poses from: ", args.keyframe_poses_dir)

    # Get the keyframe poses
    keyframes = np.array(
        sorted(
            (load_keyframe_pose(pose_file) for pose_file in keyframes_files),
            key=lambda x: x[0],  # sort by timestamp
        )
    )
    print("Keyframes: ", keyframes.shape)

    dataset_path = Path(args.dataset)
    ts_files = natsorted((dataset_path / "timestamps").glob("*.txt"))

    for ts_file in ts_files:
        timestamps = np.loadtxt(ts_file)

        # Get the relevant keyframes
        relevant_keyframes = get_relevant_keyframes(keyframes, timestamps)

        if len(relevant_keyframes) == 0:
            continue

        # Save the relevant keyframes
        output_dir = dataset_path / f"poses/keyframe/{ts_file.stem}.txt"
        with open(output_dir, "w") as f:
            for keyframe in relevant_keyframes:
                ts = keyframe[0]
                pose = keyframe[1:]
                f.write(f"{ts:.6f} " + " ".join(f"{p:.8f}" for p in pose) + "\n")
        print(f"Saved {output_dir}")
