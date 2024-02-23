"""
Author:      Dongmyeong Lee (domlee[at]utexas.edu)
Date:        Dec 12, 2023
Description: Segment Image by prompting bbox
"""
import os
import argparse
from pathlib import Path
from natsort import natsorted
import json
from tqdm import tqdm

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamPredictor


def get_parser():
    parser = argparse.ArgumentParser(description="Get instance 2D bounding box")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="/home/dongmyeong/Projects/AMRL/CODa",
        help="Path to the dataset",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="vit_h",
        help="Segment Anything Model",
    )
    parser.add_argument(
        "-ckpt",
        "--checkpoint",
        type=str,
        default="/home/dongmyeong/Projects/others/segment-anything/weights/sam_vit_h_4b8939.pth",
        help="Path to the checkpoint of Segment Anything Model",
    )

    return parser


def show_mask(image, mask, random_color=False, a=0.5):
    mask_bool = mask.astype(bool)

    if random_color:
        color = np.random.randint(0, 256, (3,)).tolist()  # Random color
    else:
        color = (30, 144, 255)  # Dodger blue color

    # Creating a colored mask
    mask_colored = np.zeros_like(image)
    mask_colored[mask_bool] = color

    image[mask_bool] = cv2.addWeighted(image, 1 - a, mask_colored, a, 0)[mask_bool]


def show_box(image, box, color=(0, 255, 0)):  # Green color
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)


def show_text(image, text, position, color=(0, 255, 0)):  # Green color
    cv2.putText(
        image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
    )


def show_contour(image, contours, color=(0, 255, 0)):  # Green color
    for contour in contours:
        cv2.drawContours(image, [contour], -1, color, 2)


def load_bbox(bbox_file):
    with open(bbox_file, "r") as f:
        lines = f.readlines()

    labels = [None] * len(lines)
    bboxes = [[None] * 4] * len(lines)
    for i, line in enumerate(lines):
        parts = line.split()
        labels[i] = (parts[0], parts[1])
        bboxes[i] = [int(p) for p in parts[2:]]
    return bboxes, labels


def main(args):
    # Load model
    sam = sam_model_registry[args.model](checkpoint=args.checkpoint)
    sam.to(device="cuda")
    predictor = SamPredictor(sam)

    classes = ["Tree", "Pole", "Bollard"]

    # Data Path
    dataset_path = Path(args.dataset)
    cams = ["cam0", "cam1"]

    for cam in cams:
        images_dir = dataset_path / "2d_rect" / cam
        bboxes_dir = dataset_path / "2d_bbox" / cam

        bbox_seq_dirs = natsorted([d for d in bboxes_dir.iterdir() if d.is_dir()])

        for bbox_seq_dir in bbox_seq_dirs:
            seq = bbox_seq_dir.name
            print(f"Processing {cam}/{seq}...")
            bbox_files = natsorted(bbox_seq_dir.glob("*.txt"))
            for bbox_file in tqdm(bbox_files, total=len(bbox_files)):
                frame = bbox_file.stem
                bboxes, labels = load_bbox(bbox_file)

                if len(bboxes) < 1:
                    continue

                # Load image
                file_name = f"2d_rect_{cam}_{seq}_{frame}.jpg"
                image_file = images_dir / seq / file_name
                image = cv2.imread(str(image_file))
                predictor.set_image(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

                # Transform bbox
                input_boxes = torch.tensor(bboxes, device=predictor.device)
                transformed_boxes = predictor.transform.apply_boxes_torch(
                    input_boxes, image.shape[:2]
                )

                # Predict
                masks, iou_preds, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )

                # Get instance segmentation
                frame_annotation = {
                    "info": {
                        "img_size": image.shape[:2],
                        "camera": cam,
                        "sequence": int(seq),
                        "frame": int(frame),
                        "image_file": file_name,
                    },
                    "instances": [],
                }

                used_masks = np.zeros_like(masks[0].cpu().numpy().squeeze())
                for mask, iou_pred, bbox, label in zip(
                    masks, iou_preds, bboxes, labels
                ):
                    # Filter out low quality masks
                    iou_pred = iou_pred.cpu().numpy().squeeze()
                    if iou_pred < 0.85:
                        continue

                    mask = mask.cpu().numpy().squeeze().astype(np.uint8)
                    # Filter out small masks
                    if mask.sum() < 2000:
                        continue

                    # Draw Contour
                    contours, _ = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if len(contours) < 1:
                        continue

                    # Get approximated contour
                    approxes = []
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area < 1000:
                            continue
                        epsilon = 0.01 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        approxes.append(approx)
                    if len(approxes) < 1:
                        continue

                    # Get new bbox
                    x, y, w, h = cv2.boundingRect(np.concatenate(approxes, axis=0))
                    new_bbox = [x, y, x + w, y + h]
                    new_area = w * h
                    old_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if new_area < old_area * 0.5:
                        continue

                    # Visualize
                    show_mask(image, mask, random_color=True)
                    show_contour(image, approxes, color=(255, 0, 0))
                    show_box(image, bbox, color=(0, 0, 255))
                    show_box(image, new_bbox)
                    show_text(image, label[0], (new_bbox[0], new_bbox[1] + 10))
                    cv2.imwrite(
                        str(
                            dataset_path
                            / "annotations"
                            / "test"
                            / f"{cam}_{seq}_{frame}.jpg"
                        ),
                        image,
                    )

                    class_name = label[0]
                    instance_id = label[1]

                    # Frame annotation
                    frame_annotation["instances"].append(
                        {
                            "class": class_name,
                            "id": int(instance_id),
                            "bbox": new_bbox,
                            "area": new_area,
                            "segmentation": [c.ravel().tolist() for c in approxes],
                        }
                    )

                if len(frame_annotation["instances"]) < 1:
                    continue

                json_file = dataset_path / "annotations" / cam / seq / f"{frame}.json"
                os.makedirs(json_file.parent, exist_ok=True)
                with open(json_file, "w") as f:
                    json.dump(frame_annotation, f, indent=None)


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
