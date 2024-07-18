import os
import argparse
import yaml

import numpy as np
from scipy.spatial.transform import Rotation as R

import rospy
import tf2_ros

from src.utils.ros_utils import play_bagfile


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


def get_transform(tf_buffer, source_frame, target_frame):
    """Retrieve and return the transform from the tf buffer."""
    try:
        transform = tf_buffer.lookup_transform(
            target_frame, source_frame, rospy.Time(0), rospy.Duration(1.0)
        )
        trans = transform.transform.translation
        rot = transform.transform.rotation
        return trans, rot
    except tf2_ros.LookupException:
        rospy.logwarn(
            f"Transform not found for {rospy.Time(0)}: {source_frame} to {target_frame}"
        )
        return None, None
    except tf2_ros.ExtrapolationException:
        rospy.logwarn(f"Extrapolation needed but not possible for {rospy.Time(0)}")
        return None, None


def main(args):
    rospy.init_node("extract_tf", anonymous=True)
    bag_process = play_bagfile(args.bagfile, use_sim_time=True, silent=True)

    tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(100))
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    rospy.sleep(3)

    for transform in args.transforms:
        if rospy.is_shutdown():
            break

        source_frame = transform["source_frame"]
        target_frame = transform["target_frame"]
        outfile = transform["outfile"]
        print(f"* Extracting transform from {source_frame} to {target_frame}...")

        trans, rot = get_transform(tf_buffer, source_frame, target_frame)
        if trans is not None and rot is not None:
            print(f" - Transform received: {source_frame} -> {target_frame}")
            print(f" - Trans (x, y, z): {trans.x}, {trans.y}, {trans.z}")
            print(f" - Quat (qw, qx, qy, qz): {rot.w}, {rot.x}, {rot.y}, {rot.z}")

            transformation = np.eye(4)
            transformation[:3, 3] = np.array([trans.x, trans.y, trans.z])
            transformation[:3, :3] = R.from_quat(
                [rot.x, rot.y, rot.z, rot.w]
            ).as_matrix()

            save_extrinsic_matrix(transformation, outfile)
            print(f" - Saved extrinsic matrix to {outfile}", end="\n\n")

        if trans is None or rot is None:
            print(f"* Could not find transform from {source_frame} to {target_frame}")
            continue

    bag_process.terminate()


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
