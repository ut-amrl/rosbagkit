import numpy as np
import cv2

import sensor_msgs.msg


def convert_image(msg: sensor_msgs.msg.Image) -> np.ndarray:
    np_arr = np.frombuffer(msg.data, np.uint8)

    if hasattr(msg, "format") and "compressed" in msg.format:
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    else:
        image = np_arr.reshape(msg.height, msg.width, -1)

    return image
