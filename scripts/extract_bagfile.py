import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from rosbagkit.bagreader import read_bagfile
from rosbagkit.conversions.depth import read_depth_msg, save_depth
from rosbagkit.conversions.geo import read_gps_msg
from rosbagkit.conversions.image import read_image_msg, save_image
from rosbagkit.conversions.motion import (
    read_imu_msg,
    read_odometry_msg,
    read_twist_msg,
    read_twist_stamped_msg,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


MSG_READERS = {
    "sensor_msgs__msg__NavSatFix": read_gps_msg,
    "sensor_msgs__msg__Imu": read_imu_msg,
    "nav_msgs__msg__Odometry": read_odometry_msg,
    "geometry_msgs__msg__Twist": read_twist_msg,
    "geometry_msgs__msg__TwistStamped": read_twist_stamped_msg,
}


def extract_bagfile(config: dict):
    bagfile_root = Path(config["bagfile_root"])
    output_root = Path(config["output_root"])
    topics_info = config["topics"]
    bag_list = config["bag_list"]

    for idx, bag_info in enumerate(bag_list):
        bagfile = bagfile_root / bag_info["file"]
        logger.info(f"[EXTRACT] [{idx + 1}/{len(bag_list)}] {bagfile}")

        output_dir = output_root / bag_info["name"]

        topics_to_msgs = read_bagfile(
            bagfile, list(topics_info.keys()), bag_info.get("start", 0), bag_info.get("end", -1)
        )

        for topic, topic_cfg in topics_info.items():
            msgs = topics_to_msgs.get(topic)
            if not msgs:
                logger.warning(f"[MISSING] No messages found for topic: {topic}")
                continue

            process_topic_msgs(topic_cfg, msgs, output_dir)


def process_topic_msgs(topic_cfg: dict, msgs: list[tuple[float, object]], output_dir: Path):
    fmt = topic_cfg.get("format")

    if fmt == "csv":
        outfile = output_dir / topic_cfg["outpath"]
        outfile.parent.mkdir(parents=True, exist_ok=True)
        process_csv_msgs(msgs, outfile)
    elif fmt == "image" or fmt == "depth":
        subdir = output_dir / topic_cfg["outdir"]
        subdir.mkdir(parents=True, exist_ok=True)
        prefix_base = "_".join(topic_cfg["outdir"].split("/"))
        prefix = f"{prefix_base}_"
        ts_file = output_dir / f"timestamp_{prefix_base}.txt"
        if fmt == "image":
            process_image_msgs(msgs, subdir, ts_file, prefix, read_image_msg, save_image)
        elif fmt == "depth":
            process_image_msgs(msgs, subdir, ts_file, prefix, read_depth_msg, save_depth)
    else:
        raise NotImplementedError(f"Unsupported format: {fmt}")


def process_csv_msgs(msgs: list[tuple[float, object]], outfile: Path):
    records = []
    for ts, msg in tqdm(msgs, leave=False, dynamic_ncols=True):
        if hasattr(msg, "header") and ts < 1e-3:
            logger.warning(f"Invalid timestamp {ts} for message {msg}")
            continue

        msg_type = msg.__class__.__name__
        reader_fn = MSG_READERS.get(msg_type)
        if reader_fn is None:
            raise NotImplementedError(f"Unsupported message type {msg_type}")

        try:
            row = reader_fn(msg)
            row["timestamp"] = ts
            for k, v in row.items():
                if isinstance(v, np.ndarray):
                    row[k] = v.flatten().tolist()
            records.append(row)
        except Exception as e:
            logger.exception(f"Failed to process message {msg_type}: {e}")

    df = pd.DataFrame(records)
    if "timestamp" in df.columns:
        cols = ["timestamp"] + [col for col in df.columns if col != "timestamp"]
        df = df[cols]

    try:
        df.to_csv(outfile, float_format="%.8f", index=False)
        logger.info(f"[SUCCESS] Saved {len(df)} rows to {outfile}")
    except Exception as e:
        logger.exception(f"Failed to write CSV: {e}")


def process_image_msgs(
    msgs: list[tuple[float, object]],
    outdir: Path,
    ts_file: Path,
    prefix: str,
    read_fn: callable,
    save_fn: callable,
):
    timestamps = []
    for frame_idx, (ts, msg) in enumerate(
        tqdm(msgs, desc=f"process {outdir.name}", leave=False, dynamic_ncols=True)
    ):
        if hasattr(msg, "header") and ts < 1e-3:
            logger.warning(f"Invalid timestamp {ts} for message {msg}")
            continue

        outfile = str(outdir / f"{prefix}{frame_idx:06d}.png")

        try:
            image = read_fn(msg)
            save_fn(image, outfile)
        except Exception as e:
            logger.warning(f"Failed to process/save image {outfile}: {e}")
            continue

        timestamps.append(ts)

    np.savetxt(ts_file, np.array(timestamps).reshape(-1, 1), fmt="%.6f", delimiter=",")
    logger.info(f"[SUCCESS] Saved {len(timestamps)} images to: {outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data from ROS bagfiles.")
    parser.add_argument("--config", type=str, default="config/extract/jackal_ahg_courtyard.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            logger.error(exc)

    extract_bagfile(config)
