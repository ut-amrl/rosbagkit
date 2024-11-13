from loguru import logger

import argparse
import pathlib
import shutil
from tqdm import tqdm
import math

import numpy as np
import cv2

from ros_utils import read_bagfile, read_image_msg


def get_args():
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("--bagfile", type=str, required=True, help="Input ROS bag.")
    parser.add_argument("--img_topics", nargs="+", required=True, help="Image topics")
    parser.add_argument("--img_outdirs", nargs="+", required=True)
    parser.add_argument("--ts_outfiles", nargs="+", required=True)
    parser.add_argument("--prefixs", nargs="+", default="", help="Prefix for filenames")
    parser.add_argument("--start_time", type=float, default=0.0, help="Start time")
    parser.add_argument("--end_time", type=float, default=float("inf"), help="End time")
    args = parser.parse_args()

    assert len(args.img_topics) == len(args.img_outdirs) == len(args.ts_outfiles)
    assert len(args.prefixs) in [0, len(args.img_topics)]

    # Image output directories
    args.img_outdirs = [pathlib.Path(p) for p in args.img_outdirs]
    for outdir in args.img_outdirs:
        if outdir.exists():
            logger.warning(f"Output directory {outdir} already exists. Overwrite it.")
            shutil.rmtree(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

    # Image Timestamp output files
    args.ts_outfiles = [pathlib.Path(p) for p in args.ts_outfiles]
    for ts_file in args.ts_outfiles:
        if ts_file.exists():
            logger.warning(f"Timestamp file {ts_file} already exists. Overwrite it.")
            ts_file.unlink()
        ts_file.parent.mkdir(parents=True, exist_ok=True)

    return args


def main(args):
    logger.info(f"\n\033[1mExtract images from bagfile {args.bagfile}\033[0m")
    logger.info(f"Image topics: {args.img_topics}")

    # Read the bag file
    topics_to_msgs = read_bagfile(
        args.bagfile, args.img_topics, args.start_time, args.end_time
    )

    # Process the messages
    for topic, img_outdir, ts_outfile, prefix in zip(
        args.img_topics, args.img_outdirs, args.ts_outfiles, args.prefixs
    ):
        if topic not in topics_to_msgs or len(topics_to_msgs[topic]) == 0:
            logger.warning(f"No messages found for topic {topic}")
            continue
        process_image_msgs(topics_to_msgs[topic], img_outdir, ts_outfile, topic, prefix)


def process_image_msgs(msgs, output_dir, ts_file, topic, prefix=""):
    timestamps = []
    fmt_str = f"{prefix}{{:05d}}.jpg"

    for frame, (ts, msg) in enumerate(
        tqdm(msgs, total=len(msgs), desc=f"Processing ({topic}) msgs", leave=False)
    ):
        # Skip messages with invalid timestamps
        if hasattr(msg, "header") and ts < 1e-3:
            logger.warning(f"Invalid timestamp {ts} for message {msg}")
            continue

        image = read_image_msg(msg)
        cv2.imwrite(str(output_dir / fmt_str.format(frame)), image)
        timestamps.append(ts)

    np.savetxt(ts_file, timestamps, fmt="%.6f")
    logger.success(f"Saved {len(timestamps)} images to {output_dir}")



if __name__ == "__main__":
    args = get_args()
    main(args)
