"""
Author:  Dongmyeong Lee (domlee[at]utexas.edu)
Created: 08/23/2023
"""

import os
import re
import argparse

import numpy as np


def main(dataset_path, sequences):
    for sequence in sequences:
        # dataset directories
        cam0_root_dir    = os.path.join(dataset_path, "2d_raw/cam0", sequence)
        cam1_root_dir    = os.path.join(dataset_path, "2d_raw/cam1", sequence)
        bbox_3d_root_dir = os.path.join(dataset_path, "3d_bbox/os1", sequence)
        pc_root_dir      = os.path.join(dataset_path, "3d_comp/os1", sequence)

        pose_file = os.path.join(dataset_path, f"poses/{sequence}.txt")
        pose_np = np.fromfile(pose_file, sep=' ').reshape(-1, 8)

        for bbox_3d_file in os.listdir(bbox_3d_root_dir):
            frame = re.split('_|\.', bbox_3d_file)[-2]



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CODa path and sequence.")
    
    # Flags for dataset_path and sequence
    parser.add_argument("-d", "--dataset_path", type=str, required=True,
                        help="Path to the dataset")
    parser.add_argument("-s", "--sequence", type=str, required=True, nargs="+",
                        help="Sequence number")
    
    args = parser.parse_args()
    main(args.dataset_path, args.sequence)
