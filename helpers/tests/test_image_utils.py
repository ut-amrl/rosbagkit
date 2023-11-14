"""
Image Utils test module.

Author: Dongmyeong Lee [domlee[at]utexas.edu]
Date:   September 22, 2023
"""
import unittest
import numpy as np

from helpers.image_utils import *


class TestImageUtils(unittest.TestCase):
    def setUp(self):
        self.bbox1 = np.array([0, 0, 100, 100])
        self.bbox2 = np.array([50, 50, 150, 150])

    def test_00_compute_bbox_area(self):
        self.assertEqual(compute_bbox_area(self.bbox1), 10000)
        self.assertEqual(compute_bbox_area(self.bbox2), 10000)

    def test_10_compute_overlap(self):
        self.assertEqual(compute_overlap(self.bbox1, self.bbox2), 2500)

    def test_20_compute_iou(self):
        self.assertEqual(compute_iou(self.bbox1, self.bbox2), 1/7)

    def test_30_clip_line_with_image_size(self):
        image_size = (640, 480)
        p1 = (-10, 240)
        p2 = (320, 240)
        self.assertEqual(clip_line_with_image_size(p1, p2, image_size), 
                         ((0, 240), (320, 240)))

    def test_40_crop_2d_bbox(self):
        image_size = (640, 480)
        bbox = (-100, -100, 100, 100)
        cropped_bbox = (0, 0, 100, 100)
        cropped_ratio = 0.75
        self.assertEqual(crop_2d_bbox(bbox, image_size),
                         (cropped_bbox, cropped_ratio))

    def test_41_crop_2d_bbox(self):
        image_size = (640, 480)
        bbox = (-100, -100, -1, -1)
        cropped_bbox = (0, 0, 0, 0)
        cropped_ratio = 1.0
        self.assertEqual(crop_2d_bbox(bbox, image_size),
                         (cropped_bbox, cropped_ratio))


if __name__ == '__main__':
    unittest.main()
