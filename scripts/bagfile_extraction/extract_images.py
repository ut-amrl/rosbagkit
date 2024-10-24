import argparse
from tqdm import tqdm
import pathlib
import numpy as np

import cv2

from ros_utils import read_bagfile, convert_image


def get_args():
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("--bagfile", type=str, required=True, help="Input ROS bag.")
    parser.add_argument("--img_topics", nargs="+", required=True, help="Image topics")
    parser.add_argument("--img_outdirs", nargs="+", required=True)
    parser.add_argument("--ts_outfiles", nargs="+", required=True)
    parser.add_argument("--prefixs", nargs="+", default="", help="Prefix")
    parser.add_argument("--start_time", type=float, default=0.0, help="Start time")
    parser.add_argument("--end_time", type=float, default=float("inf"), help="End time")
    args = parser.parse_args()

    assert len(args.img_topics) == len(args.img_outdirs) == len(args.ts_outfiles)

    args.img_outdirs = [pathlib.Path(p) for p in args.img_outdirs]
    args.ts_outfiles = [pathlib.Path(p) for p in args.ts_outfiles]

    for outdir in args.img_outdirs:
        outdir.mkdir(parents=True, exist_ok=True)

    for ts_file in args.ts_outfiles:
        ts_file.parent.mkdir(parents=True, exist_ok=True)

    return args


def main(args):
    print(f"\n\033[1mExtract images from bagfile {args.bagfile}\033[0m")

    # Read the bag file
    topics_to_msgs = read_bagfile(
        args.bagfile, args.img_topics, args.start_time, args.end_time
    )

    # Process the messages
    for topic, img_outdir, ts_outfile, prefix in zip(
        args.img_topics, args.img_outdirs, args.ts_outfiles, args.prefixs
    ):
        msgs = topics_to_msgs[topic]
        process_messages(msgs, img_outdir, ts_outfile, prefix)


def process_messages(msgs, output_dir, ts_file, prefix=""):
    if not msgs:
        return

    print(f"Save images to {output_dir}")
    timestamps = []
    frame = 0

    for ts, msg in tqdm(msgs, total=len(msgs), desc=f"Processing messages"):
        # Skip messages with invalid timestamps
        if hasattr(msg, "header") and ts < 1e-3:
            continue

        image = convert_image(msg)
        outfile = str(output_dir / f"{prefix}{frame:05d}.jpg")
        cv2.imwrite(outfile, image)

        timestamps.append(ts)
        frame += 1

    np.savetxt(ts_file, timestamps, fmt="%.6f")
    print(f"Duration: {timestamps[-1] - timestamps[0]:.2f}s")


if __name__ == "__main__":
    args = get_args()
    main(args)
