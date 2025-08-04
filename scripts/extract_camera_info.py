import argparse
import logging
import re
from pathlib import Path

import yaml

from rosbagkit.bagreader import get_topics_from_bagfile, read_bagfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_camera_info(bagfile: str):
    topics = get_topics_from_bagfile(bagfile)
    cam_info_topics = {topic for topic in topics if "camera_info" in topic}

    if not cam_info_topics:
        logger.warning(f"[TOPIC] No camera_info topics found in {bagfile}")
        return

    logger.info(f"[FOUND] {len(cam_info_topics)} camera_info topics found.")
    topics_to_msgs = read_bagfile(bagfile, cam_info_topics)

    for topic in cam_info_topics:
        msgs = topics_to_msgs.get(topic, [])
        if not msgs:
            logger.warning(f"[EMPTY] No messages for topic: {topic}")
            continue

        msg = msgs[0][1]

        camera_info = {
            "width": msg.width,
            "height": msg.height,
            "distortion_model": msg.distortion_model,
            "K": [float(x) for x in msg.k],
            "D": [float(x) for x in msg.d],
            "R": [float(x) for x in msg.r],
            "P": [float(x) for x in msg.p],
        }

        filename = sanitize_topic_name(topic) + ".yaml"
        outfile = Path(bagfile).parent / filename
        outfile.parent.mkdir(parents=True, exist_ok=True)

        with open(outfile, "w") as f:
            yaml.dump(camera_info, f, sort_keys=False)

        logger.info(f"[SAVED] {topic} -> {outfile}")


def sanitize_topic_name(topic: str) -> str:
    """Convert a ROS topic to a safe filename."""
    return re.sub(r"[^a-zA-Z0-9_]+", "_", topic.strip("/"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CameraInfo from ROS bagfile")
    parser.add_argument("--bagfile", required=True, help="Path to bagfile")
    args = parser.parse_args()

    extract_camera_info(args.bagfile)
