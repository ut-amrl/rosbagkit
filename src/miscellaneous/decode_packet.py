"""
Convert ouster/lidar_packet to pointcloud2
Output: converted_<bagfile>.bag, <bagfile>/<frame>.bin, timestamps.txt
"""

import os
import argparse
import itertools
import yaml

import numpy as np

import rosbag
from ouster import client

from src.utils.msg_converter import np_to_pointcloud2


class BagDecoder(object):
    OS1_PACKETS_PER_FRAME = 64
    OS1_POINTCLOUD_SHAPE = [1024, 128, 3]

    def __init__(self, args):
        self._os1_info = client.SensorInfo(open(args.sensor_info, "r").read())
        self._qp_counter = 0
        self._qp_frame_id = -1
        self._qp_scan_queue = []
        self._qp_init_ts = 0
        self.packet_topic = args.packet_topic
        self.pointcloud_topic = args.pointcloud_topic
        self.bagfile = args.bagfile

        self.output_bag = rosbag.Bag(
            os.path.join(
                os.path.dirname(self.bagfile),
                "converted_" + os.path.basename(self.bagfile),
            ),
            "w",
        )

    def convert_bag(self):
        rosbag_info = yaml.safe_load(rosbag.Bag(self.bagfile)._get_yaml_info())
        frame = 0
        with rosbag.Bag(self.bagfile, "r") as bag:
            for topic, msg, ts in bag.read_messages():
                if topic == self.packet_topic:
                    pc, ts = self.qpacket(msg, ts)
                    if pc is not None:
                        pc_msg = np_to_pointcloud2(
                            pc,
                            "x y z intensity t reflectivity ring ambient range",
                            "os_sensor",
                            ts,
                        )
                        self.output_bag.write(self.pointcloud_topic, pc_msg, ts)
                else:
                    self.output_bag.write(topic, msg, ts)

    def qpacket(self, msg, ts):
        packet = client.LidarPacket(msg.buf, self._os1_info)

        pc_np, pc_timestamp = None, None
        if packet.frame_id != self._qp_frame_id:
            # Process packet queue when packets for a frame are complete
            if self._qp_counter == BagDecoder.OS1_PACKETS_PER_FRAME:
                print("Processing frame: ", self._qp_frame_id)
                pc_timestamp = self._qp_init_ts
                pc_np = self.process_ouster_packet(
                    self._os1_info,
                    self._qp_scan_queue,
                )
            # Reset queue for new frame
            self._qp_counter = 0
            self._qp_frame_id = packet.frame_id
            self._qp_scan_queue = []
            self._qp_init_ts = ts
        self._qp_counter += 1
        self._qp_scan_queue.append(packet)
        return pc_np, pc_timestamp

    @staticmethod
    def process_ouster_packet(os1_info, packet_arr):
        def nth(iterable, n, default=None):
            try:
                return next(itertools.islice(iterable, n, n + 1))
            except StopIteration:
                return default

        # Process Header
        packets = client.Packets(packet_arr, os1_info)
        scans = client.Scans(packets)
        rg = nth(scans, 0).field(client.ChanField.RANGE)
        rf = nth(scans, 0).field(client.ChanField.REFLECTIVITY)
        intensity = nth(scans, 0).field(client.ChanField.SIGNAL)
        ambient = nth(scans, 0).field(client.ChanField.NEAR_IR)
        ts = nth(scans, 0).timestamp

        # Set relative timestamp for each point
        init_ts = ts[0]
        ts_horizontal_rel = ts - init_ts
        ts_horizontal_rel[ts_horizontal_rel < 0] = 0
        ts_points = np.tile(ts_horizontal_rel, (BagDecoder.OS1_POINTCLOUD_SHAPE[1], 1))

        # Set ring to correspond to row idx
        ring_idx = np.arange(0, 128, 1).reshape(-1, 1)
        ring = np.tile(ring_idx, (1, BagDecoder.OS1_POINTCLOUD_SHAPE[0]))

        # Project Points to ouster LiDAR Frame
        xyzlut = client.XYZLut(os1_info)
        xyz_points = client.destagger(os1_info, xyzlut(rg))

        # Change from LiDAR to sensor coordinate system
        intensity = np.expand_dims(intensity, axis=-1)
        rf = np.expand_dims(rf, axis=-1)
        ts_points = np.expand_dims(ts_points, axis=-1)
        rg = np.expand_dims(rg, axis=-1)
        ambient = np.expand_dims(ambient, axis=-1)
        ring = np.expand_dims(ring, axis=-1)

        pc = np.dstack(
            (xyz_points, intensity, ts_points, rf, ring, ambient, rg)
        ).astype(np.float32)
        return pc

    @staticmethod
    def save_pointcloud_to_bin(pc, outfile):
        pc_flat = pc.reshape(-1, pc.shape[-1])
        with open(outfile, "wb") as f:
            f.write(pc_flat.flatten().tobytes())

    def __del__(self):
        self.output_bag.close()


def main(args):
    bag_decoder = BagDecoder(args)
    bag_decoder.convert_bag()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bagfile",
        type=str,
        default="/home/dongmyeong/Projects/AMRL/cs393r/bags/_2024-01-04-14-13-19.bag",
        help="path to bag file",
    )
    parser.add_argument(
        "--sensor_info",
        type=str,
        default="/home/dongmyeong/Projects/AMRL/coda-tools/config/OS1metadata.json",
        help="path to sensor info json",
    )
    parser.add_argument(
        "--packet_topic",
        type=str,
        default="/ouster/lidar_packets",
        help="topic name of lidar packets to decode",
    )
    parser.add_argument(
        "--pointcloud_topic",
        type=str,
        default="/ouster_points",
        help="topic name of pointcloud as output",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
