import os
import argparse
import yaml

import rosbag
from sensor_msgs.msg import CameraInfo


def save_camera_info(msg, outfile):
    data = {
        "frame_id": msg.header.frame_id,
        "width": msg.width,
        "height": msg.height,
        "camera_matrix": {"rows": 3, "cols": 3, "data": list(msg.K)},
        "distortion_model": msg.distortion_model,
        "distortion_coefficients": {"rows": 1, "cols": 5, "data": list(msg.D)},
        "rectification_matrix": {"rows": 3, "cols": 3, "data": list(msg.R)},
        "projection_matrix": {"rows": 3, "cols": 4, "data": list(msg.P)},
    }

    with open(outfile, "w") as f:
        yaml.dump(data, default_flow_style=None, sort_keys=False, stream=f)


def main(args):
    print(f"\nExtracting camera info from bagfile {args.bagfile} ...")

    topic_to_outfile = dict(zip(args.info_topics, args.outfiles))

    with rosbag.Bag(args.bagfile) as bag:
        for topic, msg, t in bag.read_messages(topics=args.info_topics):
            if topic in args.info_topics:
                save_camera_info(msg, topic_to_outfile[topic])

            # Break if all the outfiles have been written
            if all(os.path.exists(outfile) for outfile in args.outfiles):
                break

        for outfile, info_topic in zip(args.outfiles, args.info_topics):
            if not os.path.exists(outfile):
                raise ValueError(f"Could not find {info_topic} message in the bagfile")

    bag.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bagfile", type=str, required=True)
    parser.add_argument("--info_topics", type=str, nargs="+", required=True)
    parser.add_argument("--outfiles", type=str, nargs="+", required=True)
    args = parser.parse_args()

    assert len(args.info_topics) == len(args.outfiles)
    # delete the existing files
    for outfile in args.outfiles:
        if os.path.exists(outfile):
            os.remove(outfile)
        else:
            os.makedirs(os.path.dirname(outfile), exist_ok=True)

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
