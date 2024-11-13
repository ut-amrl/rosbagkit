from loguru import logger

import os
import argparse
import yaml

import numpy as np

from ros_utils import read_bagfile, transform_to_matrix


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bagfile", type=str, required=True)
    parser.add_argument(
        "--tf_specs",
        nargs="+",
        help="Transform specifications in the form source:target:outfile",
    )
    parser.add_argument("--timelimit", type=float, default=10.0)
    args = parser.parse_args()

    args.transforms = []
    for trans_spec in args.tf_specs:
        source, target, outfile = trans_spec.split(":")
        args.transforms.append(
            {"source_frame": source, "target_frame": target, "outfile": outfile}
        )
        os.makedirs(os.path.dirname(outfile), exist_ok=True)

    return args


def main(args):
    logger.info(f"\n\033[1mExtract TF from bagfile {args.bagfile}\033[0m")

    topics_to_msgs = read_bagfile(args.bagfile, topics=["/tf_static"])
    tf_static_msgs = topics_to_msgs["/tf_static"]

    for transform in args.transforms:
        source_frame = transform["source_frame"]
        target_frame = transform["target_frame"]
        transformation = build_tf_chain(tf_static_msgs, source_frame, target_frame)

        if transformation is not None:
            logger.info(
                f"Transformation ({source_frame} to {target_frame}):\n{transformation}"
            )
            save_matrix_to_yaml(transformation, transform["outfile"])
        else:
            logger.warning(
                f"No transformation found ({source_frame} to {target_frame})."
            )


def build_tf_chain(tf_msgs, source_frame, target_frame):
    """Find the chained transformation matrix from source_frame to target_frame."""
    tf_map = {}  # Maps (parent_frame, child_frame) -> transformation matrix

    # Populate the map with all static transforms
    for _, msg in tf_msgs:
        for tf in msg.transforms:
            parent_frame = tf.header.frame_id
            child_frame = tf.child_frame_id
            tf_matrix = transform_to_matrix(tf.transform)
            tf_map[(parent_frame, child_frame)] = tf_matrix
            tf_map[(child_frame, parent_frame)] = np.linalg.inv(tf_matrix)

    # Recursive search for transformation chain
    def find_chain(current_frame, target_frame, visited):
        if current_frame == target_frame:
            return np.eye(4)  # Identity matrix if source is target

        visited.add(current_frame)

        for (parent, child), tf_matrix in tf_map.items():
            if parent == current_frame and child not in visited:
                # Recursive call to continue the chain
                result = find_chain(child, target_frame, visited)
                if result is not None:
                    # H^parent_target = H^parent_child @ H^child_target
                    return tf_matrix @ result
        return None

    # Start recursive search
    visited = set()
    transformation = find_chain(source_frame, target_frame, visited)
    return transformation


def save_matrix_to_yaml(matrix, filepath):
    """
    Saves a 4x4 extrinsic matrix to a YAML file.

    Args:
        matrix (np.ndarray): 4x4 extrinsic matrix.
        filepath (str): Path to the YAML file.
    """
    matrix_dict = {
        "extrinsic_matrix": {
            "rows": matrix.shape[0],
            "cols": matrix.shape[1],
            "data": [float(f"{x:.8f}") for x in matrix.flatten()],
        }
    }

    with open(filepath, "w") as f:
        yaml.dump(matrix_dict, default_flow_style=None, sort_keys=False, stream=f)
    logger.info(f"Extrinsic matrix saved to {filepath}")


if __name__ == "__main__":
    args = get_args()
    main(args)
