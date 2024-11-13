from loguru import logger

import pathlib
from tqdm import tqdm

from rosbags.highlevel import AnyReader


def read_bagfile(
    bagfile: str,
    topics: list[str],
    start_time: float = 0.0,
    end_time: float = float("inf"),
) -> dict[str, list[tuple[float, object]]]:
    """Read messages from a ROS bagfile."""
    ## Initialize the dictionary to store messages
    topics_to_msgs = {topic: [] for topic in topics}  # list of tuples (timestamp, data)

    ## Read the bagfile
    bagfile = pathlib.Path(bagfile)
    bagfile_size = bagfile.stat().st_size / (1024**3)
    logger.info(f"Opening bagfile {bagfile} ({bagfile_size:.2f} GB) ...")

    with AnyReader([bagfile]) as reader:
        connections = [c for c in reader.connections if c.topic in topics]

        if set(topics) != set(c.topic for c in connections):
            logger.warning(
                f"Topics not found: {set(topics) - set(c.topic for c in connections)}"
            )
            logger.info(f"Available topics: {set(c.topic for c in reader.connections)}")
            return topics_to_msgs

        logger.success(f"Found all topics: {topics}")

        # Iterate over the messages
        for connection, timestamp, data in tqdm(
            reader.messages(connections=connections), desc="Reading messages"
        ):
            ts_sec = timestamp * 1e-9  # Convert nanoseconds to seconds
            if start_time <= ts_sec <= end_time:
                msg = reader.deserialize(data, connection.msgtype)
                topics_to_msgs[connection.topic].append((ts_sec, msg))

    ## Sort the messages by timestamp
    topics_to_msgs = {
        topic: sorted(msgs, key=lambda x: x[0])
        for topic, msgs in topics_to_msgs.items()
    }
    return topics_to_msgs
