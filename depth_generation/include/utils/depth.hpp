#ifndef DEPTH_HPP
#define DEPTH_HPP

#include <Eigen/Dense>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <vector>

void fillDepthBins(const std::vector<Eigen::Vector3f>& projectedPoints,
                   cv::Mat& depthBins,
                   const std::string& option = "max") {
  assert(depthBins.channels() == 3);

  int numUpdated = 0;
  for (size_t i = 0; i < projectedPoints.size(); ++i) {
    int x = std::round(projectedPoints[i].x());
    int y = std::round(projectedPoints[i].y());
    float depth = projectedPoints[i].z();

    if (x < 0 || x >= depthBins.cols || y < 0 || y >= depthBins.rows) {
      continue;
    }

    cv::Vec3f& depthPixel = depthBins.at<cv::Vec3f>(y, x);

    bool needUpdate = depthPixel[2] < 0 || (option == "max" && depth > depthPixel[2]) ||
                      (option == "min" && depth < depthPixel[2]);

    if (needUpdate) {
      depthPixel = cv::Vec3f(x, y, depth);
      numUpdated++;
    }
  }

  if (FLAGS_v > 0) {
    std::cout << "Updated " << numUpdated << " depth bins" << std::endl;
  }
}

void convertDepthBinsToImage(const cv::Mat& depthBins, cv::Mat& depthImage) {
  assert(depthBins.channels() == 3);
  depthImage = cv::Mat(depthBins.rows, depthBins.cols, CV_16UC1);

  // Iterate over each pixel in the depthBins image
  for (int y = 0; y < depthBins.rows; ++y) {
    for (int x = 0; x < depthBins.cols; ++x) {
      float depth = depthBins.at<cv::Vec3f>(y, x)[2];

      // Convert the depth from meters to millimeters and scale to uint16
      if (std::isfinite(depth) && depth > 0) {
        float depthInMillimeters = std::min(std::max(depth * 1000.0f, 0.0f), 65535.0f);
        depthImage.at<uint16_t>(y, x) = static_cast<uint16_t>(depthInMillimeters);
      } else {
        depthImage.at<uint16_t>(y, x) = 0;
      }
    }
  }
}

#endif  // DEPTH_HPP