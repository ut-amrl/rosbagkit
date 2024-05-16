import os
import argparse
import yaml

import numpy as np
from scipy.spatial.transform import Rotation as R

import rospy
import rosbag
import tf2_ros

from utils.ros_utils import log_tf


def save_extrinsic_matrix(matrix, outfile):
    matrix_dict = {
        "extrinsic_matrix": {
            "rows": matrix.shape[0],
            "cols": matrix.shape[1],
            "data": [float(f"{x:.8f}") for x in matrix.flatten()],
        }
    }

    with open(outfile, "w") as f:
        yaml.dump(matrix_dict, default_flow_style=None, sort_keys=False, stream=f)


def get_transform(tf_buffer, source_frame, target_frame, time):
    """Retrieve and return the transform from the tf buffer."""
    try:
        transform = tf_buffer.lookup_transform(target_frame, source_frame, time)
        trans = transform.transform.translation
        rot = transform.transform.rotation
        return trans, rot
    except tf2_ros.LookupException:
        rospy.logwarn(
            f"Transform not found for {time.to_sec()}: {source_frame} to {target_frame}"
        )
        return None, None
    except tf2_ros.ExtrapolationException:
        rospy.logwarn(f"Extrapolation needed but not possible for {time.to_sec()}")
        return None, None


def main(args):
    rospy.set_param("use_sim_time", True)
    rospy.init_node("extract_tf", anonymous=True)
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    timestamps = log_tf(tf_buffer, args.bagfile, args.timelimit)

    for transform in args.transforms:
        source_frame = transform["source_frame"]
        target_frame = transform["target_frame"]
        outfile = transform["outfile"]

        for t in timestamps:
            trans, rot = get_transform(tf_buffer, source_frame, target_frame, t)
            if trans is not None and rot is not None:
                print(f"Transform found for {source_frame} -> {target_frame}")
                break

        if trans is None or rot is None:
            print(f"Could not find transform from {source_frame} to {target_frame}")
            continue

        print(f"Transform received: {source_frame} -> {target_frame}")
        print(f"Trans (x, y, z): {trans.x}, {trans.y}, {trans.z}")
        print(f"Quat (qw, qx, qy, qz): {rot.w}, {rot.x}, {rot.y}, {rot.z}")

        transformation = np.eye(4)
        transformation[:3, 3] = np.array([trans.x, trans.y, trans.z])
        transformation[:3, :3] = R.from_quat([rot.x, rot.y, rot.z, rot.w]).as_matrix()

        save_extrinsic_matrix(transformation, outfile)

    print("* Finished extracting transforms")


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


if __name__ == "__main__":
    args = get_args()
    main(args)
