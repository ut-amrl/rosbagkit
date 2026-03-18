import argparse
from collections import defaultdict
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm

from rosbagkit import export_image_msgs, msgs_to_dataframe
from rosbagkit.bagreader import read_bagfile
from rosbagkit.camera.rectification import build_stereo_rectifier, sync_indices_closest
from rosbagkit.camera.undistortion import build_undistorter
from rosbagkit.conversions.depth import read_depth_msg, read_pointcloud_depth_msg, save_depth
from rosbagkit.conversions.geo import read_gps_msg
from rosbagkit.conversions.image import read_image_msg, save_image
from rosbagkit.conversions.motion import (
    read_imu_msg,
    read_odometry_msg,
    read_pose_with_covariance_stamped_msg,
    read_twist_msg,
    read_twist_stamped_msg,
)

EXTRACTION_REQUIRED_KEYS = {"bagfile_root", "output_root", "topics", "scenes"}
RECTIFICATION_REQUIRED_KEYS = {"left_topic", "right_topic", "left_calib", "right_calib", "extrinsics"}
UNDISTORTION_REQUIRED_KEYS = {"topic", "calib"}

MSG_READERS = {
    "sensor_msgs__msg__NavSatFix": read_gps_msg,
    "sensor_msgs__msg__Imu": read_imu_msg,
    "nav_msgs__msg__Odometry": read_odometry_msg,
    "geometry_msgs__msg__Twist": read_twist_msg,
    "geometry_msgs__msg__TwistStamped": read_twist_stamped_msg,
    "geometry_msgs__msg__PoseWithCovarianceStamped": read_pose_with_covariance_stamped_msg,
}


def _has_invalid_header_timestamp(ts: float, msg: object) -> bool:
    return hasattr(msg, "header") and ts < 1e-3


def extract_bagfile(config: dict[str, Any]) -> None:
    validate_extraction_config(config)

    bagfile_root = Path(config["bagfile_root"])
    output_root = Path(config["output_root"])
    topics_info = config["topics"]
    scenes = config["scenes"]
    rectification = build_rectification_config(config.get("rectification"), topics_info)
    undistortion = build_undistortion_config(config.get("undistortion"), topics_info, rectification)

    skipped_topics = set()
    if rectification is not None:
        skipped_topics = {rectification["left_topic"], rectification["right_topic"]}
    if undistortion is not None:
        skipped_topics.add(undistortion["topic"])

    for idx, (scene_name, scene_cfg) in enumerate(scenes.items(), start=1):
        output_dir = output_root / scene_name
        tqdm.write(f"[EXTRACT] [{idx}/{len(scenes)}] scene={scene_name} output={output_dir}")

        topics_to_msgs = collect_scene_messages(
            scene_name=scene_name, scene_cfg=scene_cfg, bagfile_root=bagfile_root, topics=list(topics_info.keys())
        )

        for topic, topic_cfg in topics_info.items():
            msgs = topics_to_msgs.get(topic)
            if not msgs:
                tqdm.write(f"[MISSING] No messages found for topic: {topic}")
                continue

            if topic in skipped_topics:
                if rectification is not None and topic in {rectification["left_topic"], rectification["right_topic"]}:
                    tqdm.write(f"[RECTIFY] Skipping raw export for stereo topic: {topic}")
                elif undistortion is not None and topic == undistortion["topic"]:
                    tqdm.write(f"[UNDISTORT] Skipping raw export for undistorted topic: {topic}")
                continue

            fmt = topic_cfg["format"]

            if fmt == "csv":
                outfile = output_dir / topic_cfg["outpath"]
                outfile.parent.mkdir(parents=True, exist_ok=True)
                process_csv_msgs(msgs, outfile)
                continue

            if fmt in {"image", "depth", "pointcloud_depth"}:
                subdir = output_dir / topic_cfg["outdir"]
                subdir.mkdir(parents=True, exist_ok=True)

                prefix_base = "_".join(topic_cfg["outdir"].split("/"))
                ts_file = output_dir / f"timestamp_{prefix_base}.txt"
                prefix = f"{prefix_base}_"

                if fmt == "image":
                    export_image_msgs(msgs, subdir, ts_file, prefix)
                    continue

                if fmt == "depth":
                    read_fn = read_depth_msg
                    save_fn = save_depth
                else:
                    read_fn = partial(read_pointcloud_depth_msg, **topic_cfg)
                    save_fn = save_depth

                process_raster_msgs(msgs, subdir, ts_file, prefix, read_fn, save_fn)
                continue

            raise NotImplementedError(f"Unsupported format: {fmt}")

        if rectification is not None:
            process_rectified_stereo_msgs(rectification, topics_to_msgs, output_dir)
        if undistortion is not None:
            process_undistorted_image_msgs(undistortion, topics_to_msgs, output_dir)


def validate_extraction_config(config: dict[str, Any]) -> None:
    missing = sorted(EXTRACTION_REQUIRED_KEYS - config.keys())
    if missing:
        raise KeyError(f"Config missing required keys: {missing}")

    if not isinstance(config["topics"], dict) or not config["topics"]:
        raise ValueError("'topics' must be a non-empty mapping")

    if not isinstance(config["scenes"], dict) or not config["scenes"]:
        raise ValueError("'scenes' must be a non-empty mapping")

    for scene_name, scene_cfg in config["scenes"].items():
        bagfiles = scene_cfg.get("bagfiles")
        if not bagfiles:
            raise ValueError(f"Scene {scene_name} must define non-empty 'bagfiles'")

        start = float(scene_cfg.get("start", 0))
        end = float(scene_cfg.get("end", -1))
        if end >= 0 and start > end:
            raise ValueError(f"Scene {scene_name} has start > end: {start} > {end}")


def build_rectification_config(rect_cfg: dict[str, Any] | None, topics_info: dict[str, dict]) -> dict[str, Any] | None:
    if not rect_cfg or not rect_cfg.get("enabled", False):
        return None

    missing = sorted(RECTIFICATION_REQUIRED_KEYS - rect_cfg.keys())
    if missing:
        raise KeyError(f"Missing rectification config keys: {missing}")

    left_topic = rect_cfg["left_topic"]
    right_topic = rect_cfg["right_topic"]
    if left_topic == right_topic:
        raise ValueError("Rectification left_topic and right_topic must be different")

    for topic in (left_topic, right_topic):
        if topic not in topics_info:
            raise KeyError(f"Rectification topic not found in topics config: {topic}")
        if topics_info[topic].get("format") != "image":
            raise ValueError(f"Rectification topic must use format 'image': {topic}")

    left_calib = Path(rect_cfg["left_calib"])
    right_calib = Path(rect_cfg["right_calib"])
    extrinsics = Path(rect_cfg["extrinsics"])

    for name, path in {"left_calib": left_calib, "right_calib": right_calib, "extrinsics": extrinsics}.items():
        if not path.exists():
            raise FileNotFoundError(f"Rectification file not found for {name}: {path}")

    threshold = float(rect_cfg.get("threshold", 0.005))
    if threshold <= 0:
        raise ValueError("Rectification threshold must be > 0")

    cfg = {
        "left_topic": left_topic,
        "right_topic": right_topic,
        "left_calib": left_calib,
        "right_calib": right_calib,
        "extrinsics": extrinsics,
        "threshold": threshold,
        "output_dir": rect_cfg.get("output_dir", "2d_rect"),
        "left_subdir": rect_cfg.get("left_subdir", "cam_left"),
        "right_subdir": rect_cfg.get("right_subdir", "cam_right"),
        "timestamp_file": rect_cfg.get("timestamp_file", "timestamps.txt"),
        "left_prefix": rect_cfg.get("left_prefix", "2d_rect_left_"),
        "right_prefix": rect_cfg.get("right_prefix", "2d_rect_right_"),
        "rectifier": build_stereo_rectifier(left_calib, right_calib, extrinsics),
    }

    tqdm.write(
        f"[RECTIFY] Enabled export-time stereo rectification for "
        f"left={left_topic} right={right_topic} output={cfg['output_dir']}"
    )
    return cfg


def build_undistortion_config(
    undist_cfg: dict[str, Any] | None,
    topics_info: dict[str, dict],
    rectification: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not undist_cfg or not undist_cfg.get("enabled", False):
        return None

    missing = sorted(UNDISTORTION_REQUIRED_KEYS - undist_cfg.keys())
    if missing:
        raise KeyError(f"Missing undistortion config keys: {missing}")

    topic = undist_cfg["topic"]
    if topic not in topics_info:
        raise KeyError(f"Undistortion topic not found in topics config: {topic}")
    if topics_info[topic].get("format") != "image":
        raise ValueError(f"Undistortion topic must use format 'image': {topic}")

    if rectification is not None and topic in {rectification["left_topic"], rectification["right_topic"]}:
        raise ValueError(f"Undistortion topic conflicts with stereo rectification topic: {topic}")

    calib = Path(undist_cfg["calib"])
    if not calib.exists():
        raise FileNotFoundError(f"Undistortion calibration file not found: {calib}")

    cfg = {
        "topic": topic,
        "calib": calib,
        "output_dir": undist_cfg.get("output_dir", "undistorted"),
        "timestamp_file": undist_cfg.get("timestamp_file", "timestamps.txt"),
        "prefix": undist_cfg.get("prefix", "undistorted_"),
        "undistorter": build_undistorter(calib),
    }

    tqdm.write(
        f"[UNDISTORT] Enabled export-time undistortion for "
        f"topic={topic} output={cfg['output_dir']}"
    )
    return cfg


def collect_scene_messages(
    scene_name: str, scene_cfg: dict[str, Any], bagfile_root: Path, topics: list[str]
) -> dict[str, list[tuple[float, object]]]:
    bagfiles = scene_cfg["bagfiles"]
    start = float(scene_cfg.get("start", 0))
    end = float(scene_cfg.get("end", -1))

    topics_to_msgs: dict[str, list[tuple[float, object]]] = defaultdict(list)

    for bagfile in bagfiles:
        bagfile_path = bagfile_root / bagfile
        tqdm.write(f"[READ] scene={scene_name} bagfile={bagfile_path} window=({start:.6f}, {end:.6f})")
        loaded = read_bagfile(bagfile_path, topics, start, end)
        for topic, msgs in loaded.items():
            topics_to_msgs[topic].extend(msgs)

    return topics_to_msgs


def process_csv_msgs(msgs: list[tuple[float, object]], outfile: Path) -> None:
    msg_type = msgs[0][1].__class__.__name__
    reader_fn = MSG_READERS.get(msg_type)
    if reader_fn is None:
        tqdm.write(f"[WARN] Unsupported message type {msg_type}: {outfile}")
        return

    df = msgs_to_dataframe(msgs, reader_fn)
    df.to_csv(outfile, float_format="%.8f", index=False)
    tqdm.write(f"[SUCCESS] Saved {len(df)} rows to {outfile}")


def process_raster_msgs(
    msgs: list[tuple[float, object]], outdir: Path, ts_file: Path, prefix: str, read_fn: Callable, save_fn: Callable
) -> None:
    timestamps = []

    for frame_idx, (ts, msg) in enumerate(tqdm(msgs, desc=f"process {outdir.name}", leave=False, dynamic_ncols=True)):
        if _has_invalid_header_timestamp(ts, msg):
            tqdm.write(f"[WARN] Invalid timestamp {ts} for message {msg}")
            continue

        outfile = str(outdir / f"{prefix}{frame_idx:06d}.png")

        try:
            image = read_fn(msg)
            if image is None:
                tqdm.write(f"[WARN] Empty image for message {msg}")
                continue
            saved = save_fn(image, outfile)
        except Exception as exc:
            tqdm.write(f"[WARN] Failed to process/save frame {outfile}: {exc}")
            continue

        if not saved:
            tqdm.write(f"[WARN] Failed to save frame {outfile}")
            continue

        timestamps.append(ts)

    np.savetxt(ts_file, np.array(timestamps).reshape(-1, 1), fmt="%.6f", delimiter=",")
    tqdm.write(f"[SUCCESS] Saved {len(timestamps)} images to {outdir}")


def process_rectified_stereo_msgs(
    rectification: dict[str, Any], topics_to_msgs: dict[str, list[tuple[float, object]]], output_dir: Path
) -> None:
    left_msgs = [
        (ts, msg)
        for ts, msg in topics_to_msgs.get(rectification["left_topic"], [])
        if not _has_invalid_header_timestamp(ts, msg)
    ]
    right_msgs = [
        (ts, msg)
        for ts, msg in topics_to_msgs.get(rectification["right_topic"], [])
        if not _has_invalid_header_timestamp(ts, msg)
    ]
    if not left_msgs or not right_msgs:
        tqdm.write(f"[RECTIFY] Missing stereo messages. left={len(left_msgs)} right={len(right_msgs)}")
        return

    left_ts = [ts for ts, _ in left_msgs]
    right_ts = [ts for ts, _ in right_msgs]
    left_idx, right_idx, synced_ts = sync_indices_closest(left_ts, right_ts, threshold=rectification["threshold"])

    if not left_idx:
        tqdm.write(f"[RECTIFY] No stereo pairs matched within threshold {rectification['threshold']:.6f}")
        return

    rectified_root = output_dir / rectification["output_dir"]
    left_outdir = rectified_root / rectification["left_subdir"]
    right_outdir = rectified_root / rectification["right_subdir"]
    left_outdir.mkdir(parents=True, exist_ok=True)
    right_outdir.mkdir(parents=True, exist_ok=True)

    rectifier = rectification["rectifier"]

    for frame_idx, (li, ri) in enumerate(
        tqdm(
            zip(left_idx, right_idx, strict=False),
            total=len(left_idx),
            desc="process stereo_rectified",
            leave=False,
            dynamic_ncols=True,
        )
    ):
        _, left_msg = left_msgs[li]
        _, right_msg = right_msgs[ri]

        left_img = read_image_msg(left_msg)
        right_img = read_image_msg(right_msg)
        left_rect = rectifier.rectify(left_img, left=True)
        right_rect = rectifier.rectify(right_img, left=False)

        save_image(left_rect, str(left_outdir / f"{rectification['left_prefix']}{frame_idx:06d}.png"))
        save_image(right_rect, str(right_outdir / f"{rectification['right_prefix']}{frame_idx:06d}.png"))

    np.savetxt(
        rectified_root / rectification["timestamp_file"],
        np.array(synced_ts).reshape(-1, 1),
        fmt="%.6f",
        delimiter=",",
    )
    tqdm.write(f"[SUCCESS] Saved {len(synced_ts)} rectified stereo pairs to {rectified_root}")


def process_undistorted_image_msgs(
    undistortion: dict[str, Any], topics_to_msgs: dict[str, list[tuple[float, object]]], output_dir: Path
) -> None:
    msgs = [
        (ts, msg)
        for ts, msg in topics_to_msgs.get(undistortion["topic"], [])
        if not _has_invalid_header_timestamp(ts, msg)
    ]
    if not msgs:
        tqdm.write(f"[UNDISTORT] No messages found for topic: {undistortion['topic']}")
        return

    outdir = output_dir / undistortion["output_dir"]
    outdir.mkdir(parents=True, exist_ok=True)

    timestamps: list[float] = []
    undistorter = undistortion["undistorter"]
    prefix = undistortion["prefix"]

    for frame_idx, (ts, msg) in enumerate(
        tqdm(msgs, desc=f"process {outdir.name}", leave=False, dynamic_ncols=True)
    ):
        outfile = outdir / f"{prefix}{frame_idx:06d}.png"

        try:
            image = read_image_msg(msg)
            image = undistorter.undistort(image)
            saved = save_image(image, str(outfile))
        except Exception as exc:
            tqdm.write(f"[WARN] Failed to process/save undistorted image {outfile}: {exc}")
            continue

        if not saved:
            tqdm.write(f"[WARN] Failed to save undistorted image {outfile}")
            continue

        timestamps.append(ts)

    np.savetxt(
        outdir / undistortion["timestamp_file"],
        np.array(timestamps).reshape(-1, 1),
        fmt="%.6f",
        delimiter=",",
    )
    tqdm.write(f"[SUCCESS] Saved {len(timestamps)} undistorted images to {outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data from ROS bagfiles.")
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    extract_bagfile(config)
