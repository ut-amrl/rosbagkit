from loguru import logger
import argparse
import pathlib
import shutil
from tqdm import tqdm

import numpy as np
import cv2

from ros_utils import read_bagfile, read_image_msg, read_depth_msg
from depth.utils import write_depth


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
    if not args.prefixs:
        args.prefixs = [""] * len(args.img_topics)
    else:
        assert len(args.prefixs) == len(args.img_topics)

    # Image output directories
    args.img_outdirs = [pathlib.Path(p) for p in args.img_outdirs]
    for outdir in args.img_outdirs:
        if outdir.exists():
            logger.warning(f"Output directory {outdir} already exists. Deleting it...")
            shutil.rmtree(outdir, ignore_errors=True)
        outdir.mkdir(parents=True, exist_ok=True)

    # Image Timestamp output files
    args.ts_outfiles = [pathlib.Path(p) for p in args.ts_outfiles]
    for ts_file in args.ts_outfiles:
        if ts_file.exists():
            logger.warning(f"Timestamp file {ts_file} already exists. Deleting it...")
            ts_file.unlink()
        ts_file.parent.mkdir(parents=True, exist_ok=True)

    return args


def process_msgs(msgs, output_dir, ts_file, topic, prefix, process_fn, extension):
    """
    Generic function to process messages. The process_fn is a callback to handle
    the individual message conversion and writing.
    """
    timestamps = []
    for frame, (ts, msg) in tqdm(
        enumerate(msgs, 1), total=len(msgs), desc=f"Processing ({topic})", leave=False
    ):
        # Skip messages with invalid timestamps
        if hasattr(msg, "header") and ts < 1e-3:
            logger.warning(f"Invalid timestamp {ts} for message {msg}")
            continue

        filename = f"{prefix}{frame:05d}.{extension}"
        out_filepath = output_dir / filename
        process_fn(msg, str(out_filepath))
        timestamps.append(ts)

    np.savetxt(ts_file, timestamps, fmt="%.6f")
    logger.success(f"Saved {len(timestamps)} files to {output_dir}")


def process_image_fn(msg, filename):
    image = read_image_msg(msg)
    cv2.imwrite(filename, image)


def process_depth_fn(msg, filename):
    depth = read_depth_msg(msg)
    write_depth(depth, filename)


def main(args):
    logger.info(f"Extract images from bagfile {args.bagfile}")
    logger.info(f"Image topics: {args.img_topics}")

    # Read the bag file
    topics_to_msgs = read_bagfile(
        args.bagfile, args.img_topics, args.start_time, args.end_time
    )

    # Process the messages
    for topic, outdir, ts_file, prefix in zip(
        args.img_topics, args.img_outdirs, args.ts_outfiles, args.prefixs
    ):
        if topic not in topics_to_msgs or len(topics_to_msgs[topic]) == 0:
            logger.warning(f"No messages found for topic {topic}")
            continue

        if "depth" in topic.lower():
            process_fn = process_depth_fn
            ext = "png"
        else:
            process_fn = process_image_fn
            ext = "png"

        msgs = topics_to_msgs[topic]
        process_msgs(msgs, outdir, ts_file, topic, prefix, process_fn, ext)


if __name__ == "__main__":
    args = get_args()
    main(args)
