"""
Author:      Dongmyeong Lee (domlee[at]utexas[dot]edu)
Date:        September 16, 2023
Description: Script to generate CODa dataset for YOLO
"""
import os
import argparse
import re
from shutil import rmtree
from typing import Optional, List, Tuple
from tqdm import tqdm

import numpy as np
import cv2

from datasets.CODa.constants import BBOX_CLASS_TO_ID

# Constants
image_folder = "2d_rect"
bbox_2d_folder = "2d_bbox"
out_folder = "YOLO_CODa"
cam_list = ["cam0", "cam1"]
bbox_2d_file_pattern = re.compile(r"2d_bbox_(\w+)_(\d+)_(\d+).txt")


def process_frame(img_file: str, bbox_2d_file: str, out_dir: str,
                  visualize: bool=False) -> None:
    """
    Process a single frame.
    Args:
        img_file:  Camera image file
        bbox_2d_file:  2D bounding box file
        out_dir:    Output directory
    """
    if not os.path.exists(img_file) or not os.path.exists(bbox_2d_file):
        return

    image = cv2.imread(img_file)
    h, w, _ = image.shape

    out_image_folder = os.path.join(out_dir, "images")
    out_label_folder = os.path.join(out_dir, "labels")
    os.makedirs(out_image_folder, exist_ok=True)
    os.makedirs(out_label_folder, exist_ok=True)

    with open(bbox_2d_file, "r") as f:
        annotations = []

        for line in f.readlines():
            uid, class_id, occlusion, x1, y1, x2, y2 = line.split()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            occlusion = float(occlusion)

            if occlusion > 0.8:
                continue

            x_center = (x1 + x2) / (2 * w)
            y_center = (y1 + y2) / (2 * h)
            width = (x2 - x1) / w
            height = (y2 - y1) / h

            annotation = (int(class_id), x_center, y_center, width, height)
            annotations.append(annotation)

    if len(annotations) == 0:
        return

    file_name = os.path.splitext(os.path.basename(img_file))[-2]
    os.symlink(img_file, os.path.join(out_image_folder, f"{file_name}.jpg"))
    label_file_path = os.path.join(out_label_folder, f"{file_name}.txt")
    with open(label_file_path, "w") as f:
        f.write(' '.join(map(str, annotation)) + '\n')

    # Visualize
    if not visualize:
        return
    for annotation in annotations:
        class_id, x_center, y_center, width, height = annotation
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("image", image)
    cv2.waitKey(1)


def main(dataset_path: str, out_path: str, visualize: bool=False) -> None:
    """
    Main function to generate the dataset for YOLOv8.
    Args:
        dataset_path: Path to the dataset
    """
    # Create the output directory
    if os.path.exists(out_path):
        confirm = input(
            f"The folder '{out_path}' already exists. Overwrite? [Y/n] ")
        if confirm.lower() in ['y', 'yes', '']:
            rmtree(out_path)
            print(f"Deleted '{out_path}'")
        else:
            print("Aborted!")
            exit()
    os.makedirs(out_path, exist_ok=True)

    # Write YAML file
    coda_yaml_file = os.path.join(out_path, "CODa.yaml")
    with open(coda_yaml_file, "w") as f:
        f.write("train: ../train/images\n")
        f.write("val: ../val/images\n")
        f.write("names:\n")

        for idx, class_name in enumerate(BBOX_CLASS_TO_ID.keys()):
            f.write(f"  {idx}: {class_name.lower()}\n")

    for sequence in range(23):
        for cam in cam_list:
            img_root_dir = os.path.join(dataset_path, image_folder, cam,
                                        str(sequence))
            bbox_2d_root_dir = os.path.join(dataset_path, bbox_2d_folder, cam,
                                            str(sequence))

            # Skip if the directory does not exist
            if not os.path.exists(bbox_2d_root_dir):
                continue

            # Get the frame numbers
            frames = [
                int(bbox_2d_file_pattern.match(file_name).group(3))
                for file_name in os.listdir(bbox_2d_root_dir)
                if bbox_2d_file_pattern.match(file_name)
            ]
            frames.sort()

            # Process each frame
            for frame in tqdm(frames):
                img_file = os.path.join(
                    img_root_dir,
                    f"{image_folder}_{cam}_{sequence}_{frame}.jpg",
                )
                bbox_2d_file = os.path.join(
                    bbox_2d_root_dir,
                    f"{bbox_2d_folder}_{cam}_{sequence}_{frame}.txt",
                )
                # Process the frame
                split = "train" if (frame % 8 != 0) else "val"
                out_dir = os.path.join(out_path, split)
                process_frame(img_file, bbox_2d_file, out_dir, visualize)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CODa path")
    parser.add_argument("-d", "--dataset_path", type=str,
                        default="/home/dongmyeong/Projects/AMRL/CODa",
                        help="Path to the dataset")
    parser.add_argument("-o", "--out_path", type=str,
                        default="/home/dongmyeong/Projects/AMRL/CAO-SLAM/detector/ultralytics/datasets/CODa",
                        help="Path to the output directory")
    parser.add_argument("-v", "--visualize", action="store_true")
    args = parser.parse_args()

    main(args.dataset_path, args.out_path, args.visualize)
