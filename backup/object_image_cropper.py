"""
Author:      Dongmyeong Lee (domlee[at]utexas[dot]edu)
Date:        September 16, 2023
Description: Script to crop object images from the dataset.
"""
import os
import argparse
import re
from typing import Optional, List, Tuple
from tqdm import tqdm

import numpy as np
import cv2

from helpers.image_utils import save_cropped_image_with_margin

# Constants
image_folder = "2d_rect"
bbox_2d_folder = "2d_bbox"
cropped_folder = "2d_cropped_new"
cam_list = ["cam0", "cam1"]
bbox_2d_file_pattern = re.compile(r"2d_bbox_(\w+)_(\d+)_(\d+).txt")
margin_ratio = 0.1


def process_frame(
    cam_img_file: str,
    bbox_2d_file: str,
    frame: int,
    output_dir: Optional[str] = None,
    size: int = 224,
    visualize: bool = False,
) -> None:
    """
    Process a single frame.
    Args:
        cam_img_file:  Camera image file
        bbox_2d_file:  2D bounding box file
        frame:         Frame number
        output_dir:    Output directory
        size:          Size of the cropped image
        visualize:     Visualize the bounding boxes
    """
    if not os.path.exists(cam_img_file) or not os.path.exists(bbox_2d_file):
        return

    image = cv2.imread(cam_img_file)
    with open(bbox_2d_file, "r") as f:
        uid_list = []
        bbox_list = []
        occlusion_list = []

        for line in f.readlines():
            uid, class_id, occlusion, x1, y1, x2, y2 = line.split()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            occlusion = float(occlusion)

            # Occlusion condition
            if occlusion > 0.3:
                continue

            # Save the cropped image
            if output_dir:
                output_file = os.path.join(output_dir, uid, f"{frame}.jpg")
                save_cropped_image_with_margin(image, (x1, y1, x2, y2),
                                               (size, size), margin_ratio,
                                               output_file)

            # Store for visualization
            bbox_list.append((x1, y1, x2, y2))
            uid_list.append(uid)
            occlusion_list.append(occlusion)

        if visualize:
            for bbox, uid in zip(bbox_list, uid_list):
                cv2.rectangle(image, bbox[:2], bbox[2:], (0, 255, 0), 2)
                cv2.putText(
                    image,
                    f"{uid} {class_id}",
                    (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow(frame, image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def main(
    dataset_path: str,
    sequences: List[str],
    size: int = 224,
    visualize: bool = False,
) -> None:
    """
    Main function to crop object images from the dataset
    Args:
        dataset_path: Path to the dataset
        sequences:    Sequence number(s)
        visualize:    Visualize the bounding boxes on the images
        size:         Size of the cropped image
    """
    for sequence in sequences:
        for cam in cam_list:
            cam_root_dir = os.path.join(dataset_path, image_folder, cam,
                                        sequence)
            bbox_2d_root_dir = os.path.join(dataset_path, bbox_2d_folder, cam,
                                            sequence)

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
            # output_dir = os.path.join(dataset_path, cropped_folder,
            # cam, sequence)
            output_dir = os.path.join(dataset_path, cropped_folder, sequence)
            for frame in tqdm(frames):
                # output_prefix = f"{bbox_2d_folder}_{cam}_{sequence}_{frame}"

                cam_img_file = os.path.join(
                    cam_root_dir,
                    f"{image_folder}_{cam}_{sequence}_{frame}.jpg",
                )
                bbox_2d_file = os.path.join(
                    bbox_2d_root_dir,
                    f"{bbox_2d_folder}_{cam}_{sequence}_{frame}.txt",
                )
                # Process the frame
                process_frame(cam_img_file, bbox_2d_file, frame, output_dir,
                              size, visualize)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CODa path and sequence.")
    parser.add_argument("-d",
                        "--dataset_path",
                        type=str,
                        required=True,
                        help="Path to the dataset")
    parser.add_argument("-s",
                        "--sequences",
                        type=str,
                        required=True,
                        help="Sequence number(s)",
                        nargs="+")
    parser.add_argument("-v", "--visualize", action="store_true")
    parser.add_argument("--size",
                        type=int,
                        default=224,
                        help="Size of the cropped images")

    opt = parser.parse_args()

    # Visualize when the output directory is not specified
    main(opt.dataset_path, opt.sequences, opt.size, opt.visualize)
