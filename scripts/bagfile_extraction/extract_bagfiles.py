from loguru import logger
import argparse
import pathlib
import yaml
import csv
from tqdm import tqdm

import numpy as np
import cv2

from ros_utils import (
    read_bagfile,
    read_image_msg,
    read_gps_msg,
    read_odometry_msg,
    read_twist_msg,
    read_twist_stamped_msg,
)


def get_args():
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument(
        "--config", type=str, default="config/GQ_bagfiles_config.yaml", required=True
    )
    args = parser.parse_args()
    return args


def process_msgs_csv(msgs, outfile):
    rows = []
    header = None

    for ts, msg in tqdm(msgs, total=len(msgs), leave=False):
        # Skip messages with invalid timestamps
        if hasattr(msg, "header") and ts < 1e-3:
            logger.warning(f"Invalid timestamp {ts} for message {msg}")
            continue

        msg_type = msg.__class__.__name__
        if msg_type == "sensor_msgs__msg__NavSatFix":
            data = read_gps_msg(msg)
        elif msg_type == "nav_msgs__msg__Odometry":
            data = read_odometry_msg(msg)
        elif msg_type == "geometry_msgs__msg__Twist":
            data = read_twist_msg(msg)
        elif msg_type == "geometry_msgs__msg__TwistStamped":
            data = read_twist_stamped_msg(msg)
        else:
            raise NotImplementedError(f"Unsupported message type {msg_type}")

        if header is None:
            header = ["timestamp"] + list(data.keys())

        def format_value(val):
            if isinstance(val, np.ndarray):
                formatted = ";".join(f"{x:.8f}" for x in val.flatten())
                return f"[{formatted}]"
            elif isinstance(val, (int, float)):
                return f"{val:.8f}"
            else:
                return str(val)

        row = [f"{ts:.6f}"] + [format_value(data[k]) for k in header[1:]]
        rows.append(row)

    # Write to CSV file
    with open(outfile, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(rows)

    logger.success(f"Saved {len(rows)} rows to {outfile}")


def process_msgs_image(msgs, outdir, ts_file, prefix):
    """
    Generic function to process messages. The process_fn is a callback to handle
    the individual message conversion and writing.
    """
    timestamps = []
    for frame, (ts, msg) in tqdm(
        enumerate(msgs, 1),
        total=len(msgs),
        desc=f"Processing ({outdir.name})",
        leave=False,
    ):
        # Skip messages with invalid timestamps
        if hasattr(msg, "header") and ts < 1e-3:
            logger.warning(f"Invalid timestamp {ts} for message {msg}")
            continue

        outfile = str(outdir / f"{prefix}{frame:06d}.png")
        image = read_image_msg(msg)
        cv2.imwrite(outfile, image)
        timestamps.append(ts)

    np.savetxt(ts_file, timestamps, fmt="%.6f")
    logger.success(f"Saved {len(timestamps)} files to {outdir}")


def main(args):
    # Read the config file
    with open(args.config) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.error(exc)

    # Extract bag files
    topics_info = config["topics"]
    bag_list = config["bag_list"]
    for idx, bag_info in enumerate(bag_list):
        logger.info(f"[{idx+1}/{len(bag_list)}] Extract {bag_info['file']}...")

        # Read the bag file
        bagfile = pathlib.Path(config["bagfile_root"]) / bag_info["file"]
        bag_name = bag_info["name"]
        topics_to_msgs = read_bagfile(
            bagfile,
            topics_info.keys(),
            bag_info.get("start", 0),
            bag_info.get("end", -1),
        )

        # Process the messages for each topic
        for topic, topic_info in topics_info.items():
            output_dir = pathlib.Path(config["output_root"]) / topic_info["outdir"]
            output_dir.mkdir(parents=True, exist_ok=True)

            if topic not in topics_to_msgs or len(topics_to_msgs[topic]) == 0:
                logger.warning(f"No messages found for topic {topic}")
                continue

            msg = topics_to_msgs[topic]
            fmt = topic_info["format"]

            if fmt == "image":
                sub = topic_info["sub"]
                ts_file = output_dir / bag_name / f"timestamp_{sub}.txt"
                outdir = output_dir / bag_name / topic_info["sub"]
                outdir.mkdir(parents=True, exist_ok=True)
                prefix = f"{topic_info['outdir']}_{sub}_"
                process_msgs_image(msg, outdir, ts_file, prefix)
            elif fmt == "csv":
                outfile = output_dir / f"{bag_name}.csv"
                process_msgs_csv(msg, outfile)
            else:
                raise NotImplementedError(f"Unsupported format {fmt}")


if __name__ == "__main__":
    args = get_args()
    main(args)
