import warnings
from tqdm import tqdm
import pathlib

from rosbags.highlevel import AnyReader


def read_bagfile(
    bagfile: str,
    topics: list[str],
    start_time: float = 0.0,
    end_time: float = float("inf"),
) -> dict[str, list[tuple]]:
    """Read messages from a ROS bagfile."""
    # Initialize the dictionary to store messages
    topics_to_msgs = {topic: [] for topic in topics}  # list of tuples (timestamp, data)

    # Read the bagfile
    with AnyReader([pathlib.Path(bagfile)]) as reader:
        connections = [c for c in reader.connections if c.topic in topics]

        if not connections:
            warnings.warn("No valid topics found in the bagfile")
            return topics_to_msgs

        # Iterate over the messages
        for connection, timestamp, data in tqdm(
            reader.messages(connections=connections), desc="Reading messages"
        ):
            ts_sec = timestamp * 1e-9  # Convert nanoseconds to seconds
            if start_time <= ts_sec <= end_time:
                msg = reader.deserialize(data, connection.msgtype)
                topics_to_msgs[connection.topic].append((ts_sec, msg))

    # Sort the messages by timestamp
    for topic in topics:
        topics_to_msgs[topic].sort(key=lambda x: x[0])

    return topics_to_msgs
