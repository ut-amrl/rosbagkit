"""
Author:      Dongmyeong Lee (domlee[at]utexas.edu)
Date:        Dec 12, 2023
Description: (temp) Add poses to the dataset
"""
import os
import argparse
from pathlib import Path
from natsort import natsorted
import json
from tqdm import tqdm

import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.coda_utils import load_extrinsic_matrix


def get_parser():
    parser = argparse.ArgumentParser(description="Get instance 2D bounding box")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="/home/dongmyeong/Projects/AMRL/CODa",
        help="Path to the dataset",
    )
    return parser


sequence_conditions = {
    0: "cloudy",
    1: "cloudy",
    2: "dark",
    3: "sunny",
    4: "dark",
    5: "dark",
    6: "sunny",
    7: "sunny",
    8: "cloudy",
    9: "cloudy",
    10: "cloudy",
    11: "sunny",
    12: "cloudy",
    13: "rainy",
    14: "dark",
    15: "rainy",
    16: "rainy",
    17: "sunny",
    18: "sunny",
    19: "sunny",
    20: "sunny",
    21: "cloudy",
    22: "sunny",
}


def main(args):
    # Data Path
    dataset_path = Path(args.dataset)

    annotations_dir = dataset_path / "annotations"

    for cam in ["cam0", "cam1"]:
        anno_seq_dirs = natsorted([d for d in annotations_dir.glob(f"{cam}/*")])
        for anno_seq_dir in anno_seq_dirs:
            anno_files = natsorted([f for f in anno_seq_dir.glob("*.json")])

            # pose
            seq = anno_seq_dir.name
            pose_file = dataset_path / "poses" / "correct" / f"{seq}.txt"
            poses = np.loadtxt(pose_file, delimiter=" ")

            # extrinsics
            H_lc = load_extrinsic_matrix(
                dataset_path / "calibrations" / seq / f"calib_os1_to_{cam}.yaml"
            )

            for anno_file in anno_files:
                annotations = json.load(open(anno_file, "r"))
                info = annotations["info"]
                frame = int(info["frame"])

                if "pose" in annotations:
                    continue

                H_lg = np.eye(4)
                H_lg[:3, :3] = R.from_quat(poses[frame, [5, 6, 7, 4]]).as_matrix()
                H_lg[:3, 3] = poses[frame, 1:4]

                H_cg = H_lg @ H_lc

                # camera pose
                x, y, z = H_cg[:3, 3]
                qx, qy, qz, qw = R.from_matrix(H_cg[:3, :3]).as_quat()

                annotations["info"]["weather_condition"] = sequence_conditions[int(seq)]
                annotations["info"]["pose"] = [x, y, z, qw, qx, qy, qz]

                with open(anno_file, "w") as f:
                    json.dump(annotations, f, indent=None)


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
