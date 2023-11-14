"""
Geometry test module.

Author: Dongmyeong Lee [domlee[at]utexas.edu]
Date:   September 22, 2023
"""
import unittest
import numpy as np

from helpers.geometry import *

class TestGeometry(unittest.TestCase):
    def setUp(self):
        self.extrinsic = np.array([[0, -1,  0, 0],
                                   [0,  0, -1, 0],
                                   [1,  0,  0, 0],
                                   [0,  0,  0, 1]], dtype=np.float32)
        self.intrinsic = np.array([[500, 0,   320],
                                   [0,   500, 240],
                                   [0,   0,   1]], dtype=np.float32)
        self.image_size = np.array([640, 480], dtype=np.int32)

    def test_00_get_corners_3d_bbox(self):
        corners = np.array([
            [1, 2, 4], [1, 2, -4], [1, -2, 4], [1, -2, -4],
            [-1, 2, 4], [-1, 2, -4], [-1, -2, 4], [-1, -2, -4]])
        results, _ = get_corners_3d_bbox(0, 0, 0, 2, 4, 8, 0, 0, 0)
        np.testing.assert_array_equal(corners, results)
                                         
    def test_01_get_corners_3d_bbox(self):
        corners = np.array([
            [1, -4, 2], [1, 4, 2], [1, -4, -2], [1, 4, -2],
            [-1, -4, 2], [-1, 4, 2], [-1, -4, -2], [-1, 4, -2]])
        results, _ = get_corners_3d_bbox(0, 0, 0, 2, 4, 8, np.pi/2, 0, 0)
        np.testing.assert_almost_equal(corners, results, decimal=5)

    def test_10_project_points_3d_to_2d(self):
        points_3d = np.array([[1, 0, 0], [-1, 0, 0], [20, 1000, 0]])
        points_2d = np.array([[320, 240], [np.nan, np.nan], [np.nan, np.nan]])
        projected_points, depths = project_points_3d_to_2d(
            points_3d, self.extrinsic, self.intrinsic, self.image_size,
            keep_size=True)
        np.testing.assert_array_equal(projected_points, points_2d)
        np.testing.assert_array_equal(depths, np.array([1, -1, 20]))

    def test_11_project_points_3d_to_2d(self):
        points_3d = np.array([[1, 0, 0], [-1, 0, 0], [20, 1000, 0]])
        points_2d = np.array([[320, 240]])
        projected_points, depths = project_points_3d_to_2d(
            points_3d, self.extrinsic, self.intrinsic, self.image_size,
            keep_size=False)
        np.testing.assert_array_equal(projected_points, points_2d)
        np.testing.assert_array_equal(depths, np.array([1]))

    def test_20_project_bbox_3d_to_2d(self):
        bbox_3d = (11, 0, 0, 2, 2, 2, 0, 0, 0)
        bbox_2d = (270, 190, 370, 290)
        occlusion_ratio = 0.0
        results = project_bbox_3d_to_2d(
            bbox_3d, self.extrinsic, self.intrinsic, self.image_size)
        self.assertEqual(results, (bbox_2d, occlusion_ratio))

    def test_21_project_bbox_3d_to_2d(self):
        bbox_3d = (-11, 0, 0, 2, 2, 2, 0, 0, 0)
        bbox_2d = (None, None, None, None)
        occlusion_ratio = 1.0
        results = project_bbox_3d_to_2d(
            bbox_3d, self.extrinsic, self.intrinsic, self.image_size)
        self.assertEqual(results, (bbox_2d, occlusion_ratio))

    def test_22_project_bbox_3d_to_2d(self):
        bbox_3d = (1.5, -1, 0, 1, 1, 0.4, 0, 0, 0)
        bbox_2d = (445, 140, 640, 340)
        occlusion_ratio = float(125000 - 39000) / 125000
        results = project_bbox_3d_to_2d(
            bbox_3d, self.extrinsic, self.intrinsic, self.image_size)
        self.assertEqual(results, (bbox_2d, occlusion_ratio))

    def test_30_line_segment_intersection_2d(self):
        p1 = (0, 0)
        p2 = (1, 1)
        q1 = (1, 0)
        q2 = (0, 1)
        intersection = (0.5, 0.5)
        result = line_segment_intersection_2d(p1, p2, q1, q2)
        self.assertEqual(result, intersection)
        
    def test_31_line_segment_intersection_2d(self):
        p1 = (0, 0)
        p2 = (1, 1)
        q1 = (-1, 2)
        q2 = (0, 1)
        intersection = (None, None)
        result = line_segment_intersection_2d(p1, p2, q1, q2)
        self.assertEqual(result, intersection)

    def test_40_line_segment_intersection_plane(self):
        p1 = np.array([0, 0, -1])
        p2 = np.array([0, 0, 1])
        plane = np.array([0, 0, 1, 0])
        intersection = np.array([0, 0, 0])
        result = line_segment_intersection_plane(p1, p2, plane)
        np.testing.assert_array_equal(result, intersection)

    def test_41_line_segment_intersection_plane(self):
        p1 = np.array([0, 0, -1])
        p2 = np.array([0, 0, 1])
        plane = np.array([0, 0, 1, 3])
        intersection = None
        result = line_segment_intersection_plane(p1, p2, plane)
        np.testing.assert_array_equal(result, intersection)

if __name__ == '__main__':
    unittest.main()
