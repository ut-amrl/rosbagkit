import argparse
import logging
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from rosbagkit.bagreader import get_topic_frame_ids, read_bagfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_tf(bagfile: str, source_frame: str, target_frame: str, outfile: str):
    tf_topics = ["/tf", "/tf_static"]
    tf_msgs_by_topic = read_bagfile(bagfile, tf_topics)
    tf_msgs = tf_msgs_by_topic.get("/tf", []) + tf_msgs_by_topic.get("/tf_static", [])

    tf_map = build_tf_map(tf_msgs)
    T = find_chain(tf_map, source_frame, target_frame)

    if T is None:
        logger.warning(f"[NOT FOUND] No TF chain from {source_frame} to {target_frame}")
        available_frames = {frame for pair in tf_map.keys() for frame in pair}
        logger.info(f"Available frames: {available_frames}")
        topic_frame_ids = get_topic_frame_ids(bagfile)
        logger.info("[FRAME_ID] Topics and their frame_ids:")
        for topic, frame_id in topic_frame_ids.items():
            logger.info(f"  {topic}: {frame_id}")
        return

    tf_data = {
        "source_frame": source_frame,
        "target_frame": target_frame,
        "translation": T[:3, 3].tolist(),
        "rotation": T[:3, :3].flatten().tolist(),
    }

    logger.info(f"[FOUND] Transformation from {source_frame} to {target_frame}:\n{T}")

    with open(outfile, "w") as f:
        yaml.dump(tf_data, f, sort_keys=False, default_flow_style=None)
    logger.info(f"[SAVED] TF written to {outfile}")


def build_tf_map(tf_msgs):
    tf_map = {}
    for _, msg in tf_msgs:
        for tf in msg.transforms:
            parent = tf.header.frame_id.strip("/")
            child = tf.child_frame_id.strip("/")
            tf_matrix = transform_to_matrix(tf.transform)
            tf_map[(parent, child)] = tf_matrix
            tf_map[(child, parent)] = np.linalg.inv(tf_matrix)
    return tf_map


def find_chain(tf_map, source_frame, target_frame):
    def _search(current, target, visited):
        if current == target:
            return np.eye(4)
        visited.add(current)
        for (parent, child), tf_matrix in tf_map.items():
            if parent == current and child not in visited:
                result = _search(child, target, visited)
                if result is not None:
                    return tf_matrix @ result
        return None

    return _search(source_frame, target_frame, visited=set())


def transform_to_matrix(msg) -> np.ndarray:
    """Convert geometry_msgs/Transform into a 4x4 transformation matrix."""
    tx, ty, tz = (msg.translation.x, msg.translation.y, msg.translation.z)
    qx, qy, qz, qw = (msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w)
    rotation_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = [tx, ty, tz]

    return transformation_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract TF chain between frames from a bagfile")
    parser.add_argument("--bagfile", required=True, help="Path to ROS bagfile")
    parser.add_argument("--src", default="imu", help="Source frame (default: imu)")
    parser.add_argument("--tgt", default="lidar", help="Target frame (default: lidar)")
    args = parser.parse_args()

    source_frame = args.src.strip("/")
    target_frame = args.tgt.strip("/")

    filename = f"tf_{source_frame.split('/')[-1]}_to_{target_frame.split('/')[-1]}.yaml"
    outfile = Path(args.bagfile).parent / filename

    extract_tf(args.bagfile, source_frame, target_frame, outfile)
