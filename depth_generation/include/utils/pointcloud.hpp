#ifndef POINTCLOUD_HPP
#define POINTCLOUD_HPP

#include <Eigen/Dense>
#include <fstream>
#include <pcl/filters/voxel_grid_occlusion_estimation.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr loadBinPointCloud(const std::string& filePath) {
  // clang-format off
  static_assert(std::is_same<PointT, pcl::PointXYZ>::value ||
                std::is_same<PointT, pcl::PointXYZI>::value,
                "Unsupported point type");
  // clang-format on

  std::ifstream file(filePath, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + filePath);
  }

  // Move to the end of the file to determine the size
  file.seekg(0, std::ios::end);
  size_t fileSize = file.tellg();
  file.seekg(0, std::ios::beg);

  // Calculate the number of floats in the file
  if (fileSize % sizeof(float) != 0) {
    throw std::runtime_error("File size is not a multiple of float size");
  }

  size_t numElementsPerPoint;
  if constexpr (std::is_same<PointT, pcl::PointXYZ>::value) {
    numElementsPerPoint = 3;  // pcl::PointXYZ has x, y, z
  } else if constexpr (std::is_same<PointT, pcl::PointXYZI>::value) {
    numElementsPerPoint = 4;  // pcl::PointXYZI has x, y, z, intensity
  }

  size_t numPoints = fileSize / sizeof(float) / numElementsPerPoint;

  // Read the data into a vector of floats
  std::vector<float> buffer(fileSize / sizeof(float));
  file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
  file.close();

  // Ensure that the number of floats in the buffer is divisible by numElementsPerPoint
  if (buffer.size() % numElementsPerPoint != 0) {
    throw std::runtime_error(
        "Invalid point cloud data: Number of floats not divisible by elements per "
        "point.");
  }

  // Create a point cloud and fill it with the data
  typename pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
  cloud->width = numPoints;
  cloud->height = 1;
  cloud->is_dense = false;
  cloud->points.resize(cloud->width);

  for (size_t i = 0; i < numPoints; ++i) {
    cloud->points[i].x = buffer[numElementsPerPoint * i];
    cloud->points[i].y = buffer[numElementsPerPoint * i + 1];
    cloud->points[i].z = buffer[numElementsPerPoint * i + 2];
    if constexpr (std::is_same_v<PointT, pcl::PointXYZI>) {
      cloud->points[i].intensity = buffer[numElementsPerPoint * i + 3];
    }
  }

  return cloud;
}

void filterOccludedPoints(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                          const Eigen::Vector3f& camOrigin,
                          const Eigen::Quaternionf& camOrientation,
                          pcl::PointCloud<pcl::PointXYZ>::Ptr& filteredCloud,
                          float leafSize = 0.1f,
                          bool visualize = false) {
  assert(leafSize > 0);
  filteredCloud->clear();
  if (!cloud || cloud->empty()) {
    std::cerr << "ERROR: Point cloud is empty" << std::endl;
    return;
  }

  cloud->sensor_origin_ =
      Eigen::Vector4f(camOrigin.x(), camOrigin.y(), camOrigin.z(), 1.0f);
  cloud->sensor_orientation_ = camOrientation;

  // Create a voxel grid filter
  pcl::VoxelGridOcclusionEstimation<pcl::PointXYZ> voxelGrid;
  voxelGrid.setInputCloud(cloud);
  voxelGrid.setLeafSize(leafSize, leafSize, leafSize);
  voxelGrid.initializeVoxelGrid();

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredCloud(
      new pcl::PointCloud<pcl::PointXYZRGB>);
  for (const auto& p : cloud->points) {
    Eigen::Vector3i gridCoordinates = voxelGrid.getGridCoordinates(p.x, p.y, p.z);
    int state;
    voxelGrid.occlusionEstimation(state, gridCoordinates);

    bool is_occluded = (state == 1);
    if (!is_occluded) filteredCloud->push_back(p);

    if (visualize) {
      pcl::PointXYZRGB coloredPoint;
      coloredPoint.x = p.x;
      coloredPoint.y = p.y;
      coloredPoint.z = p.z;
      coloredPoint.r = is_occluded ? 255 : 0;
      coloredPoint.g = is_occluded ? 0 : 255;
      coloredPoint.b = 0;
      coloredCloud->push_back(coloredPoint);
    }
  }

  // Visualize the filtered point cloud
  if (visualize) {
    pcl::visualization::PCLVisualizer viewer("Filtered Point Cloud");
    viewer.setBackgroundColor(0, 0, 0);
    viewer.addPointCloud<pcl::PointXYZRGB>(coloredCloud, "voxel occlusion");
    Eigen::Affine3f sensorPose = Eigen::Translation3f(camOrigin) * camOrientation;
    viewer.addCoordinateSystem(2.0, sensorPose, "sensor frame", 0);
    viewer.spin();
    viewer.close();
  }
}

pcl::PointCloud<pcl::PointXYZ>::Ptr randomPointCloudPCL(int numPoints) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

  cloud->resize(numPoints);

  for (int i = 0; i < numPoints; ++i) {
    cloud->points[i].x = static_cast<float>(rand()) / RAND_MAX;
    cloud->points[i].y = static_cast<float>(rand()) / RAND_MAX;
    cloud->points[i].z = static_cast<float>(rand()) / RAND_MAX;
  }

  return cloud;
}

#endif  // POINTCLOUD_HPP