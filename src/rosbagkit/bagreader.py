import logging
from collections import Counter
from pathlib import Path

# https://gitlab.com/ternaris/rosbags
from rosbags.highlevel import AnyReader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def read_bagfile(
    bagfile: str,
    topics: list[str],
    start_time: float = 0.0,
    end_time: float = float("inf"),
    early_return: bool = False,
) -> dict[str, list[tuple[float, object]]]:  # topic: [(timestamp, message), ...]
    if not topics:
        logger.warning("[TOPICS] No topics provided. Returning empty result.")
        return {}

    if end_time < 0:
        end_time = float("inf")

    topics_to_msgs = {topic: [] for topic in topics}

    # Read the bagfile
    logger.info(f"Opening bagfile -- {bagfile} ({format_bytes(get_path_size(bagfile))})")
    with AnyReader([Path(bagfile)]) as reader:
        connections = [c for c in reader.connections if c.topic in topics]
        available_topics = {c.topic for c in connections}
        missing_topics = set(topics) - available_topics

        if not missing_topics and available_topics:
            logger.info(f"[TOPICS] All found: {topics}")
        elif not connections:
            logger.error(f"[TOPICS] None found: {topics}")
            logger.debug(f"[TOPICS] Available: {[c.topic for c in reader.connections]}")
            return topics_to_msgs
        else:
            logger.warning(f"[TOPICS] Missing: {missing_topics}")
            logger.info(f"[TOPICS] Found: {available_topics}")
            logger.debug(f"[TOPICS] All available: {[c.topic for c in reader.connections]}")

        for connection, timestamp, rawdata in tqdm(reader.messages(connections=connections)):
            msg = reader.deserialize(rawdata, connection.msgtype)

            # Use the message header timestamp if available
            header = getattr(msg, "header", None)
            if header and hasattr(header, "stamp"):
                ts_sec = header.stamp.sec + header.stamp.nanosec * 1e-9
            else:
                ts_sec = timestamp * 1e-9

            if start_time <= ts_sec <= end_time:
                topics_to_msgs[connection.topic].append((ts_sec, msg))

            if early_return and all(len(msgs) > 0 for msgs in topics_to_msgs.values()):
                logger.info("\n[EARLY RETURN] All topics have at least one message.")
                break

    # Sort the messages by timestamp for each topic
    for msgs in topics_to_msgs.values():
        msgs.sort(key=lambda x: x[0])

    # Check for duplicate timestamps
    for topic, msgs in topics_to_msgs.items():
        timestamps = [ts for ts, _ in msgs]
        dupes = [ts for ts, count in Counter(timestamps).items() if count > 1]
        if dupes:
            logger.warning(
                f"[DUPLICATES] Topic {topic} has {len(dupes)} duplicate ts (e.g. {dupes[:5]})"
            )

    return topics_to_msgs


def get_topics_from_bagfile(bagfile: str) -> set[str]:
    try:
        with AnyReader([Path(bagfile)]) as reader:
            topics = {c.topic for c in reader.connections}
        logger.info(f"[TOPICS] Found {len(topics)} topics in {bagfile}")
        return topics
    except Exception as e:
        logger.exception(f"[ERROR] Failed to read topics from {bagfile}: {e}")
        return set()


def get_topic_frame_ids(bagfile: str, max_msg_per_topic: int = 1) -> dict[str, str]:
    frame_ids = {}

    try:
        with AnyReader([Path(bagfile)]) as reader:
            for connection in reader.connections:
                topic = connection.topic
                msgtype = connection.msgtype

                count = 0
                for _, _, rawdata in reader.messages(connections=[connection]):
                    msg = reader.deserialize(rawdata, msgtype)

                    header = getattr(msg, "header", None)
                    if header and hasattr(header, "frame_id"):
                        frame_ids[topic] = header.frame_id
                        break

                    count += 1
                    if count >= max_msg_per_topic:
                        break

        logger.info(f"[FRAME_ID] Found frame_id for {len(frame_ids)} topics.")
        return frame_ids

    except Exception as e:
        logger.exception(f"[ERROR] Failed to extract frame_id from bagfile: {e}")
        return {}


##### Helper Functions #####


def get_path_size(path: str) -> int:
    path = Path(path)
    if path.is_file():  # ROS1
        return path.stat().st_size
    elif path.is_dir():  # ROS2
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return 0


def format_bytes(size_in_bytes: int) -> str:
    if size_in_bytes >= 1024**3:
        return f"{size_in_bytes / 1024**3:.2f} GB"
    elif size_in_bytes >= 1024**2:
        return f"{size_in_bytes / 1024**2:.2f} MB"
    elif size_in_bytes >= 1024:
        return f"{size_in_bytes / 1024:.2f} KB"
    else:
        return f"{size_in_bytes} bytes"
