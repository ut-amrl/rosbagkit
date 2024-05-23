#ifndef PROJECTION_HPP
#define PROJECTION_HPP

#include <iostream>
#include <vector>
//
#include <Eigen/Dense>
#include <glog/logging.h>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
//
#include <pcl/visualization/pcl_visualizer.h>

void projectToRectifiedImage(const cv::Mat& img,
                             const cv::Mat& R,
                             const cv::Mat& P,
                             const Eigen::Matrix4f& extrinsic,
                             const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                             std::vector<Eigen::Vector3f>& projectedPoints,
                             pcl::PointCloud<pcl::PointXYZ>::Ptr& validCloud,
                             bool visualize = false) {
  projectedPoints.clear();  // (x, y, depth)
  validCloud->clear();
  if (cloud->empty()) {
    return;
  }

  // Transform points to camera coordinate system
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcCam(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::transformPointCloud(*cloud, *pcCam, extrinsic);

  // TODO: clean this up
  Eigen::Matrix<float, 3, 4> Pmat;
  cv::cv2eigen(P, Pmat);
  Eigen::Matrix4f Rmat = Eigen::Matrix4f::Identity();
  Eigen::Matrix3f tmpRmat;
  cv::cv2eigen(R, tmpRmat);
  Rmat.block<3, 3>(0, 0) = tmpRmat;

  // Project points onto the image plane
  float minDepth = std::numeric_limits<float>::max();
  float maxDepth = 0.0;
  for (const auto& p : pcCam->points) {
    if (p.z < 0) {
      continue;
    }

    Eigen::Vector4f pointHomogeneous(p.x, p.y, p.z, 1.0);
    Eigen::Vector3f projectedPointHomo = Pmat * Rmat * pointHomogeneous;
    Eigen::Vector3f projectedPoint(projectedPointHomo.x() / projectedPointHomo.z(),
                                   projectedPointHomo.y() / projectedPointHomo.z(),
                                   p.z);
    if (projectedPoint.x() >= 0 && projectedPoint.x() < img.cols &&
        projectedPoint.y() >= 0 && projectedPoint.y() < img.rows) {
      projectedPoints.push_back(projectedPoint);
      validCloud->push_back(p);
      minDepth = std::min(minDepth, p.z);
      maxDepth = std::max(maxDepth, p.z);
    }
  }

  if (visualize) {
    cv::Mat imgCopy = img.clone();
    for (const auto& p : projectedPoints) {
      float normalizedDepth = (p[2] - minDepth) / (maxDepth - minDepth);
      int colorValue = static_cast<int>(normalizedDepth * 255);
      cv::Scalar color = cv::Scalar(0, 0, 255 - colorValue, colorValue);

      cv::circle(imgCopy, cv::Point(p[0], p[1]), 5, color, -1);
    }

    cv::imshow("Projected Points", imgCopy);
    cv::waitKey(0);
  }

  if (FLAGS_v > 5) {
    std::cout << "cloud: " << cloud->size() << ", projected: " << projectedPoints.size()
              << std::endl;
  }
}

#endif  // PROJECTION_HPP