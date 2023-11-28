def project_points_3d_to_2d(
    points: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    image_size: Optional[np.ndarray] = None,
    dist_coeff: Optional[np.ndarray] = None,
    keep_size: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    if image_size is not None:
        mask_in_image = (



# When some corners are outside the image boundaries, compute the
# intersection of edges of the bounding box with the image boundaries
from helpers.image_utils import (
compute_bbox_area,
crop_2d_bbox,
clip_line_with_image_size,
)

corners_homo = np.column_stack((corners, np.ones(corners.shape[0])))
corners_camera = corners_homo @ extrinsic[:3, :].T

corners_outside = np.empty((0, 2))
for p1_idx, p2_idx in edges:
# Skip if both corners are inside or outside the image
if corners_validity[p1_idx] == corners_validity[p2_idx]:
continue
# Compute the intersection with xy plane in Camera Frame
# when one corner is behind the camera
p1_camera = corners_camera[p1_idx]
p2_camera = corners_camera[p2_idx]
intersection = line_segment_intersection_plane(
p1_camera, p2_camera, (0, 0, 1, 1)
)
# Replace behind corners with intersection
if intersection is not None:
if p1_camera[2] < 0:
p1_camera = intersection
if p2_camera[2] < 0:
p2_camera = intersection
# Project the corners onto the image plane and clip with the image
p1_image, _ = cv2.projectPoints(
p1_camera, np.zeros(3), np.zeros(3), intrinsic, dist_coeff
)
p2_image, _ = cv2.projectPoints(
p2_camera, np.zeros(3), np.zeros(3), intrinsic, dist_coeff
)
p1_image = p1_image.squeeze()
p2_image = p2_image.squeeze()

# Clip the line with the image boundaries
p1, p2 = clip_line_with_image_size(p1_image, p2_image, image_size)
if p1 is None or p2 is None:
continue
corners_image = np.vstack((corners_image, p1, p2))
if not np.isclose(p1_image, p1).all():
corners_outside = np.vstack((corners_outside, p1_image, p1))
if not np.isclose(p2_image, p2).all():
corners_outside = np.vstack((corners_outside, p2_image, p2))

# Remove NaN
corners_image = corners_image[~np.isnan(corners_image[:, 0])]
corners_outside = corners_outside[~np.isnan(corners_outside[:, 0])]

# Get valid 2d bounding box
x1 = int(corners_image[:, 0].min())
y1 = int(corners_image[:, 1].min())
x2 = int(corners_image[:, 0].max())
y2 = int(corners_image[:, 1].max())
valid_bbox_2d = (x1, y1, x2, y2)

if len(corners_outside) < 1:
return valid_bbox_2d, 0.0

# Get invalid 2d bounding box
x1_outside = int(corners_outside[:, 0].min())
y1_outside = int(corners_outside[:, 1].min())
x2_outside = int(corners_outside[:, 0].max())
y2_outside = int(corners_outside[:, 1].max())
invalid_bbox_2d = (x1_outside, y1_outside, x2_outside, y2_outside)

# Compute occlusion ratio
valid_bbox_2d_area = compute_bbox_area(valid_bbox_2d)
invalid_bbox_2d_area = compute_bbox_area(invalid_bbox_2d)
bbox_2d_area = valid_bbox_2d_area + invalid_bbox_2d_area
occlusion_ratio = invalid_bbox_2d_area / bbox_2d_area

return valid_bbox_2d, occlusion_ratio


def line_segment_intersection_2d(
p1: Tuple[float, float],
p2: Tuple[float, float],
q1: Tuple[float, float],
q2: Tuple[float, float],
) -> Tuple[float, float]:
"""
Calculate the intersection of two line segments in 2D.

Args:
p1, p2: end-points of the first line
q1, q2: end-points of the second line

Returns:
x, y: coordinates of the intersection point
"""
# ax + by + c = 0
a1 = p1[1] - p2[1]
b1 = p2[0] - p1[0]
c1 = -(b1 * p1[1] + a1 * p1[0])

a2 = q1[1] - q2[1]
b2 = q2[0] - q1[0]
c2 = -(b2 * q1[1] + a2 * q1[0])

# Calculate the intersection point
x, y = line_intersection_2d((a1, b1, c1), (a2, b2, c2))
# Check if the intersection point exists
if x is None or y is None:
return None, None
# Check if the intersection point is on the line segment
p_min_x = min(p1[0], p2[0])
p_max_x = max(p1[0], p2[0])
p_min_y = min(p1[1], p2[1])
p_max_y = max(p1[1], p2[1])
q_min_x = min(q1[0], q2[0])
q_max_x = max(q1[0], q2[0])
q_min_y = min(q1[1], q2[1])
q_max_y = max(q1[1], q2[1])
if (
(not np.isclose(x, p_min_x, atol=1e-6) and x < p_min_x)
or (not np.isclose(x, p_max_x, atol=1e-6) and x > p_max_x)
or (not np.isclose(y, p_min_y, atol=1e-6) and y < p_min_y)
or (not np.isclose(y, p_max_y, atol=1e-6) and y > p_max_y)
or (not np.isclose(x, q_min_x, atol=1e-6) and x < q_min_x)
or (not np.isclose(x, q_max_x, atol=1e-6) and x > q_max_x)
or (not np.isclose(y, q_min_y, atol=1e-6) and y < q_min_y)
or (not np.isclose(y, q_max_y, atol=1e-6) and y > q_max_y)
):
return None, None

return x, y


def line_intersection_2d(
line_coeff1: Tuple[float, float, float], line_coeff2: Tuple[float, float, float]
) -> Tuple[float, float]:
"""
Calculate the intersection of two lines in 2D.

Args:
coeff1: (a, b, c) coefficients of the first line  (ax + by + c = 0)
coeff2: (a, b, c) coefficients of the second line (ax + by + c = 0)

Returns:
x, y: coordinates of the intersection point
"""
# ax + by + c = 0
a1, b1, c1 = line_coeff1
a2, b2, c2 = line_coeff2

# Calculate the intersection point
denom = a1 * b2 - a2 * b1
if denom == 0:
return None, None
x = (b1 * c2 - b2 * c1) / denom
y = (a2 * c1 - a1 * c2) / denom

return x, y


def line_segment_intersection_plane(
p1: np.ndarray, p2: np.ndarray, plane_coeff: Tuple[float, float, float, float]
) -> Optional[np.ndarray]:
"""
Calculate the intersection of a line segment with a plane.
The line segment is defined by two end-points p1 and p2.

Args:
p1: end-point on the line (3,)
p2: another end-point on the line (3,)
plane_coeff: coefficient of the plane (ax + by + cz + d = 0)
Returns:
intersection point (3,)
"""
a, b, c, d = plane_coeff
p1p2 = p2 - p1
# Calculate the intersection point
denom = a * p1p2[0] + b * p1p2[1] + c * p1p2[2]
if denom == 0:
return None
t = -(a * p1[0] + b * p1[1] + c * p1[2] + d) / denom
intersection = p1 + t * p1p2

# Check if the intersection point is on the line segment
if t < 0 or t > 1:
return None

return intersection


def transform_plane(
plane_coeff: Tuple[float, float, float, float], extrinsic: np.ndarray
) -> Tuple[float, float, float, float]:
"""
Transform a plane with the given extrinsic matrix (LiDAR to camera).

Args:
plane_coeff: coefficients of the plane (ax + by + cZ + d = 0)
extrinsic: (4, 4) extrinsic matrix

Returns:
transformed_plane_coeff: coefficients of the transformed plane
"""
normal = np.array([a, b, c, 0]).reshape(4, 1)
transformed_normal = (extrinsic @ normal)[:3]

point = np.array([0, 0, -d / c, 1]).reshape(4, 1)
transformed_point = (extrinsic @ point)[:3]

transformed_a = transformed_normal[0].item()
transformed_b = transformed_normal[1].item()
transformed_c = transformed_normal[2].item()
transformed_d = (-transformed_normal.T @ transformed_point).item()

return transformed_a, transformed_b, transformed_c, transformed_d


def filter_points_inside_3d_bbox(points: np.ndarray, bbox_3d: Bbox3D) -> np.ndarray:
"""
Filter out points that are inside the 3D bounding box

Args:
points: (N, 3) array of 3D points
bbox_3d: (cX, cY, cZ, l, w, h, roll, pitch, yaw) of the bounding box

Returns:
points: (M, 3) array of 3D points that are inside the bounding box
"""
cX, cY, cZ, l, w, h, roll, pitch, yaw = bbox_3d

# Transformation matrix from the given roll, pitch, yaw
rot_mat = Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=False).as_matrix()

# Transform points to the local frame
transformed_points = (points - np.array([cX, cY, cZ])) @ rot_mat

# Filter out points that are outside the bounding box
mask_inside = (
(transformed_points[:, 0] >= -l / 2)
& (transformed_points[:, 0] <= l / 2)
& (transformed_points[:, 1] >= -w / 2)
& (transformed_points[:, 1] <= w / 2)
& (transformed_points[:, 2] >= -h / 2)
& (transformed_points[:, 2] <= h / 2)
)

return points[mask_inside]


