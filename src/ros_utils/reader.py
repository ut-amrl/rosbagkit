from loguru import logger

import pathlib
from tqdm import tqdm

# https://gitlab.com/ternaris/rosbags
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
        available_topics = {c.topic for c in connections}
        missing_topics = set(topics) - available_topics

        if not missing_topics and available_topics:
            logger.success(f"Found all topics: {topics}")
        elif not connections:
            logger.error(f"Topics not found: {topics}")
            return topics_to_msgs
        else:
            logger.warning(f"Missing topics: {missing_topics}")
            logger.info(f"Available topics: {available_topics}")

        # Iterate over the messages
        for connection, timestamp, data in tqdm(
            reader.messages(connections=connections), desc="Reading messages"
        ):
            ts_sec = timestamp * 1e-9  # Convert nanoseconds to seconds
            if start_time <= ts_sec <= end_time:
                msg = reader.deserialize(data, connection.msgtype)
                topics_to_msgs[connection.topic].append((ts_sec, msg))

    ## Sort the messages by timestamp for each topic
    for msgs in topics_to_msgs.values():
        msgs.sort(key=lambda x: x[0])

    return topics_to_msgs
