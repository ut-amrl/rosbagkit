#ifndef DEPTH_HPP
#define DEPTH_HPP

#include <cmath>
#include <vector>
//
#include <Eigen/Dense>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

void fillDepthBins(const std::vector<Eigen::Vector3f>& projectedPoints,
                   cv::Mat& depthBins,
                   const std::string& option = "max") {
  assert(depthBins.type() == CV_32FC3);
  assert(option == "max" || option == "min");

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
      depthPixel = cv::Vec3f(projectedPoints[i].x(), projectedPoints[i].y(), depth);
      numUpdated++;
    }
  }

  if (FLAGS_v > 5) {
    std::cout << "num projected points: " << projectedPoints.size() << std::endl;
    std::cout << "Updated " << numUpdated << " depth bins" << std::endl;
  }
}

void saveDepthImage(const cv::Mat& depthImage, const std::string& filename) {
  CV_Assert(depthImage.type() == CV_32F);
  cv::Mat outDepthImage = cv::Mat::zeros(depthImage.rows, depthImage.cols, CV_16U);

  // Iterate over each pixel in the depthBins image
  for (int y = 0; y < depthImage.rows; ++y) {
    for (int x = 0; x < depthImage.cols; ++x) {
      float depth = depthImage.at<float>(y, x);

      // Convert the depth from meters to millimeters and scale to uint16
      if (std::isfinite(depth) && depth > 0) {
        float depthInMillimeters = std::min(std::max(depth * 1000.0f, 0.0f), 65535.0f);
        outDepthImage.at<uint16_t>(y, x) = static_cast<uint16_t>(depthInMillimeters);
      }
    }
  }

  cv::imwrite(filename, outDepthImage);

  if (FLAGS_v > 0) {
    std::cout << "Saved depth image to " << filename << std::endl;
  }
}

void densifyDepthBins(const cv::Mat& depthBins, cv::Mat& depthImage, int grid = 5) {
  CV_Assert(depthBins.type() == CV_32FC3);

  // Create a dense depth image
  int imgH = depthBins.rows;
  int imgW = depthBins.cols;
  int ng = 2 * grid + 1;

  cv::Mat mX = cv::Mat(imgH, imgW, CV_32F, std::numeric_limits<float>::infinity());
  cv::Mat mY = cv::Mat(imgH, imgW, CV_32F, std::numeric_limits<float>::infinity());
  cv::Mat mD = cv::Mat::zeros(imgH, imgW, CV_32F);

  for (int y = 0; y < imgH; ++y) {
    for (int x = 0; x < imgW; ++x) {
      cv::Vec3f point = depthBins.at<cv::Vec3f>(y, x);
      if (!std::isnan(point[2]) && !std::isinf(point[2]) && point[2] > 0) {
        mX.at<float>(y, x) = point[0] - x;
        mY.at<float>(y, x) = point[1] - y;
        mD.at<float>(y, x) = point[2];
      }
    }
  }

  int kernelH = imgH - ng + 1;
  int kernelW = imgW - ng + 1;

  // Initialize the kernel matrices
  std::vector<cv::Mat> KmX(ng * ng);
  std::vector<cv::Mat> KmY(ng * ng);
  std::vector<cv::Mat> KmD(ng * ng);

  for (int i = 0; i < ng; ++i) {
    for (int j = 0; j < ng; ++j) {
      int idx = i * ng + j;
      KmX[idx] = mX(cv::Range(i, kernelH + i), cv::Range(j, kernelW + j)) - grid + i;
      KmY[idx] = mY(cv::Range(i, kernelH + i), cv::Range(j, kernelW + j)) - grid + j;
      KmD[idx] = mD(cv::Range(i, kernelH + i), cv::Range(j, kernelW + j));
    }
  }

  cv::Mat S = cv::Mat::zeros(kernelH, kernelW, CV_32F);
  cv::Mat Y = cv::Mat::zeros(kernelH, kernelW, CV_32F);

  for (int i = 0; i < ng; ++i) {
    for (int j = 0; j < ng; ++j) {
      int idx = i * ng + j;
      cv::Mat squaredSum = KmX[idx].mul(KmX[idx]) + KmY[idx].mul(KmY[idx]);
      cv::Mat s;
      cv::sqrt(squaredSum, s);
      s = 1 / (s + 1e-12f);
      Y += s.mul(KmD[idx]);
      S += s;
    }
  }

  // Ensure no division by zero
  S.setTo(1, S == 0);

  depthImage = cv::Mat::zeros(imgH, imgW, CV_32F);
  depthImage(cv::Range(grid, imgH - grid), cv::Range(grid, imgW - grid)) = Y / S;
}

#endif  // DEPTH_HPP