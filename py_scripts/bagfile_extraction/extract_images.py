import os
import argparse
from tqdm import tqdm
import pathlib
import numpy as np

import rosbag
import cv2
from cv_bridge import CvBridge


def synchronize_images(
    left_msgs,
    right_msgs,
    left_outdir,
    right_outdir,
    left_ts_file,
    right_ts_file,
    prefix="",
):
    print("Synchronizing images and saving them...")
    bridge = CvBridge()

    left_idx, right_idx = 0, 0
    frame = 0

    left_timestamps, right_timestamps = [], []
    while left_idx < len(left_msgs) and right_idx < len(right_msgs):
        left_msg = left_msgs[left_idx]
        right_msg = right_msgs[right_idx]
        left_time = left_msg.timestamp.to_sec()
        right_time = right_msg.timestamp.to_sec()

        if abs(left_time - right_time) < 0.02:  # Allowable time difference
            left_image = bridge.compressed_imgmsg_to_cv2(left_msg.message)
            right_image = bridge.compressed_imgmsg_to_cv2(right_msg.message)

            left_img_file = str(left_outdir / f"{prefix}_left_{frame}.jpg")
            right_img_file = str(right_outdir / f"{prefix}_right_{frame}.jpg")
            cv2.imwrite(left_img_file, left_image)
            cv2.imwrite(right_img_file, right_image)

            left_timestamps.append(left_time)
            right_timestamps.append(right_time)
            print(f"Frame: {frame} TS: {left_time} (left) {right_time} (right)")

            frame += 1
            left_idx += 1
            right_idx += 1
        elif left_time < right_time:
            left_idx += 1
        else:
            right_idx += 1

    np.savetxt(left_ts_file, left_timestamps, fmt="%.6f")
    np.savetxt(right_ts_file, right_timestamps, fmt="%.6f")

    print(f"Total synchronized image pairs saved: {frame}")


def process_messages(msgs, output_dir, ts_file, prefix=""):
    bridge = CvBridge()

    timestamps = []
    for i, msg in tqdm(enumerate(msgs), total=len(msgs), leave=False):
        image = bridge.compressed_imgmsg_to_cv2(msg.message)
        cv2.imwrite(str(output_dir / f"{prefix}{i}.jpg"), image)
        timestamps.append(msg.message.header.stamp.to_sec())

    np.savetxt(ts_file, timestamps, fmt="%.6f")

    print(f"Total images saved: {len(msgs)}")


def main(args):
    # Process the bag file
    bag = rosbag.Bag(args.bagfile, "r")
    print(f"Extracting images from {args.bagfile}...")

    # Sort the messages by timestamp
    left_msgs = sorted(
        list(bag.read_messages(topics=[args.img_left_topic])),
        key=lambda x: x.message.header.stamp,
    )
    right_msgs = sorted(
        list(bag.read_messages(topics=[args.img_right_topic])),
        key=lambda x: x.message.header.stamp,
    )
    print(f"Total images: {len(left_msgs)} (left), {len(right_msgs)} (right)")

    # Extract images
    if args.sync:
        synchronize_images(
            left_msgs,
            right_msgs,
            args.left_outdir,
            args.right_outdir,
            args.left_ts_file,
            args.right_ts_file,
            "2d_rect",
        )
    else:
        process_messages(
            left_msgs, args.left_outdir, args.left_ts_file, "2d_rect_left_"
        )
        process_messages(
            right_msgs, args.right_outdir, args.right_ts_file, "2d_rect_right_"
        )

    bag.close()


def get_args():
    # Specify the bag file and topics
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("--bagfile", type=str, help="Input ROS bag.")
    parser.add_argument("--img_left_topic", type=str, help="Left image topic")
    parser.add_argument("--img_right_topic", type=str, help="Right image topic")
    parser.add_argument("--img_outdir", type=str, help="Output directory for images")
    parser.add_argument("--ts_outdir", type=str, help="Output directory for timestamps")
    parser.add_argument("--sync", action="store_true", help="Synchronize images")
    args = parser.parse_args()

    args.left_outdir = pathlib.Path(args.img_outdir) / "left"
    args.right_outdir = pathlib.Path(args.img_outdir) / "right"
    args.left_ts_file = pathlib.Path(args.ts_outdir) / "img_left.txt"
    args.right_ts_file = pathlib.Path(args.ts_outdir) / "img_right.txt"

    args.left_outdir.mkdir(parents=True, exist_ok=True)
    args.right_outdir.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.ts_outdir).mkdir(parents=True, exist_ok=True)

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
