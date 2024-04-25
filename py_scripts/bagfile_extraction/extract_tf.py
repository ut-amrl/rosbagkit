import os
import sys
import argparse
import pathlib
import yaml

import rospy
import tf2_ros
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from utils.ros_utils import start_roscore, play_bagfile


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


def main(args):
    print("* Extracting the extrinsic matrix from the bagfile ...")

    # Start the new roscore
    roscore_process = start_roscore(True)

    # Initialize the node
    rospy.init_node("tf_listener", anonymous=True)

    # Play the bagfile in the background
    bag_process = play_bagfile(args.bagfile, True, True)
    print(f"Processing bagfile {args.bagfile} ...")

    # Initialize the tf listener
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    rospy.sleep(3)  # Wait for the tf listener to start

    rate = rospy.Rate(1)
    trans, rot = None, None
    while not rospy.is_shutdown():
        try:
            transform = tf_buffer.lookup_transform(
                args.target_frame, args.source_frame, rospy.Time(0), rospy.Duration(1.0)
            )
            trans = transform.transform.translation
            rot = transform.transform.rotation
            if not np.isnan(
                [trans.x, trans.y, trans.z, rot.x, rot.y, rot.z, rot.w]
            ).any():
                print("Transform received:")
                print(f"Translation: {trans.x}, {trans.y}, {trans.z}")
                print(f"Rotation: {rot.x}, {rot.y}, {rot.z}, {rot.w}")
                break
            else:
                print("Received NaN in transform, skipping...")
        except Exception as e:
            # print(e)
            rate.sleep()
            continue

    if trans is None or rot is None:
        raise ValueError("Could not find the transform between the frames.")

    transformation = np.eye(4)
    transformation[:3, 3] = np.array([trans.x, trans.y, trans.z])
    transformation[:3, :3] = R.from_quat([rot.x, rot.y, rot.z, rot.w]).as_matrix()

    save_extrinsic_matrix(transformation, args.outfile)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bagfile", type=str, required=True)
    parser.add_argument("--source_frame", type=str, required=True)
    parser.add_argument("--target_frame", type=str, required=True)
    parser.add_argument("--outfile", type=str, required=True, help="yaml file to save")

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
