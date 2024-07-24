import os
import argparse
from tqdm import tqdm
import pathlib
import numpy as np

import rosbag
import cv2
from cv_bridge import CvBridge

from src.utils.depth import write_depth


def convert_msg_to_image(msg, bridge: CvBridge) -> np.ndarray:
    if msg._type == "sensor_msgs/CompressedImage":
        return bridge.compressed_imgmsg_to_cv2(msg)
    elif msg._type == "sensor_msgs/Image":
        return bridge.imgmsg_to_cv2(msg)
    else:
        raise ValueError("Unsupported image message type")


def process_messages(msgs, output_dir, ts_file, prefix="", is_depth=False):
    if not msgs:
        return

    print(f"Saving images to {output_dir}...")
    bridge = CvBridge()

    timestamps = []
    for i, msg in tqdm(enumerate(msgs), total=len(msgs), leave=False):
        try:
            image = convert_msg_to_image(msg, bridge)
            outfile = str(output_dir / f"{prefix}{i}.png")
            if is_depth:
                write_depth(image, outfile)
            else:
                cv2.imwrite(outfile, image)
            timestamps.append(msg.header.stamp.to_sec())
        except Exception as e:
            print(f"Error converting image: {e}")
            continue

    np.savetxt(ts_file, timestamps, fmt="%.6f")
    print(f"Total images saved: {len(msgs)}")


def main(args):
    # Process the bag file
    print(f"Extracting images from {args.bagfile}...")
    bag = rosbag.Bag(args.bagfile, "r")

    topics_to_msgs = {topic: [] for topic in args.img_topics}

    # Read Images and Sort the messages by timestamp
    for topic, msg, t in tqdm(
        bag.read_messages(topics=args.img_topics),
        desc="Reading images",
        total=bag.get_message_count(topic_filters=args.img_topics),
    ):
        topics_to_msgs[topic].append(msg)

    # Extract images
    for topic, msgs in topics_to_msgs.items():
        idx = args.img_topics.index(topic)
        msgs.sort(key=lambda msg: msg.header.stamp)
        print(f"Processing {len(msgs)} messages from topic {topic}...")

        is_depth = "depth" in topic

        process_messages(
            msgs,
            args.img_outdirs[idx],
            args.ts_outfiles[idx],
            args.prefixs[idx],
            is_depth,
        )

    bag.close()


def get_args():
    # Specify the bag file and topics
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("--bagfile", type=str, required=True, help="Input ROS bag.")

    parser.add_argument("--img_topics", nargs="+", required=True, help="Image topics")
    parser.add_argument(
        "--img_outdirs", nargs="+", required=True, help="Output directories"
    )
    parser.add_argument(
        "--ts_outfiles", nargs="+", required=True, help="Output timestamp files"
    )
    parser.add_argument("--prefixs", nargs="+", default="", help="Prefix")
    args = parser.parse_args()

    assert (
        len(args.img_topics)
        == len(args.img_outdirs)
        == len(args.ts_outfiles)
        == len(args.prefixs)
    )

    args.img_outdirs = [pathlib.Path(p) for p in args.img_outdirs]
    args.ts_outfiles = [pathlib.Path(p) for p in args.ts_outfiles]

    # Create the output directories
    for outdir in args.img_outdirs:
        outdir.mkdir(parents=True, exist_ok=True)

    for ts_file in args.ts_outfiles:
        ts_file.parent.mkdir(parents=True, exist_ok=True)

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
