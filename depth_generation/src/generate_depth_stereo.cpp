#include "generate_depth_stereo.h"

#include <algorithm>
#include <iostream>
//
#include <opencv2/opencv.hpp>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "utils/depth.hpp"
#include "utils/indicators.hpp"
#include "utils/pointcloud.hpp"
#include "utils/projection.hpp"

void generateDepthStereo(const std::vector<std::string>& pointcloudFiles,
                         const std::vector<manif::SE3d>& pcPoses,
                         const std::vector<double>& pcTimestamps,
                         const std::vector<std::string>& imageLeftFiles,
                         const std::vector<std::string>& imageRightFiles,
                         const std::vector<manif::SE3d>& imageLeftPoses,
                         const std::vector<manif::SE3d>& imageRightPoses,
                         const std::vector<double>& imageTimestamps,
                         const std::unordered_map<std::string, cv::Mat>& camLeftParams,
                         const std::unordered_map<std::string, cv::Mat>& camRightParams,
                         int windowSize,
                         const std::string& depthLeftDir,
                         const std::string& depthRightDir) {
  assert(pointcloudFiles.size() == pcPoses.size());
  assert(imageLeftFiles.size() == imageLeftPoses.size());
  assert(imageRightFiles.size() == imageRightPoses.size());
  assert(imageLeftPoses.size() == imageRightPoses.size());

  using namespace indicators;
  ProgressBar bar{option::BarWidth{50},
                  option::Start{"["},
                  option::Fill{"="},
                  option::Lead{">"},
                  option::Remainder{" "},
                  option::End{"]"},
                  option::PostfixText{"Generating depth images..."},
                  option::ForegroundColor{Color::green},
                  option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}};

  std::deque<pcl::PointCloud<pcl::PointXYZ>::Ptr> pcWorldWindow;
  int pcIdx = 0;

  for (size_t imgIdx = 5000; imgIdx < imageTimestamps.size(); ++imgIdx) {
    bar.set_progress(imgIdx / static_cast<float>(imageTimestamps.size()) * 100.0f);

    double imgTimestamp = imageTimestamps[imgIdx];

    // Accumulate the point clouds in the window just before the image timestamp
    auto it = std::upper_bound(pcTimestamps.begin(), pcTimestamps.end(), imgTimestamp);
    for (pcIdx; pcIdx < it - pcTimestamps.begin(); pcIdx++) {
      auto cloud = loadBinPointCloud<pcl::PointXYZ>(pointcloudFiles[pcIdx]);
      pcl::PointCloud<pcl::PointXYZ>::Ptr pcWorld(new pcl::PointCloud<pcl::PointXYZ>);

      // transform the point cloud to the world frame
      const manif::SE3d& pcPose = pcPoses[pcIdx];
      pcl::transformPointCloud(*cloud, *pcWorld, pcPose.transform());

      // keep the point cloud in the window
      pcWorldWindow.push_back(pcWorld);
      while (pcWorldWindow.size() > windowSize) {
        pcWorldWindow.pop_front();
      }
    }

    // Load the images and camera poses
    cv::Mat imgLeft = cv::imread(imageLeftFiles[imgIdx]);
    cv::Mat imgRight = cv::imread(imageRightFiles[imgIdx]);
    const manif::SE3d& camLeftPose = imageLeftPoses[imgIdx];
    const manif::SE3d& camRightPose = imageRightPoses[imgIdx];

    // Compute the stereo depth with the accumulated pointcloud
    cv::Mat depthLeft(imgLeft.size(), CV_16UC1);
    cv::Mat depthRight(imgRight.size(), CV_16UC1);
    computeStereoDepth(imgLeft,
                       imgRight,
                       camLeftPose,
                       camRightPose,
                       camLeftParams,
                       camRightParams,
                       pcWorldWindow,
                       depthLeft,
                       depthRight);

    // Save the depth images
    std::string depthLeftFile =
        depthLeftDir + "/2d_depth_left_" + std::to_string(imgIdx) + ".png";
    std::string depthRightFile =
        depthRightDir + "/2d_depth_right_" + std::to_string(imgIdx) + ".png";

    cv::imwrite(depthLeftFile, depthLeft);
    break;
  }
}

void computeStereoDepth(const cv::Mat& imgLeft,
                        const cv::Mat& imgRight,
                        const manif::SE3d& camLeftPose,
                        const manif::SE3d& camRightPose,
                        const std::unordered_map<std::string, cv::Mat>& camLeftParams,
                        const std::unordered_map<std::string, cv::Mat>& camRightParams,
                        std::deque<pcl::PointCloud<pcl::PointXYZ>::Ptr>& pcWorldWindow,
                        cv::Mat& depthLeft,
                        cv::Mat& depthRight) {
  assert(imgLeft.size() == imgRight.size());

  cv::Mat depthBinsLeft(imgLeft.size(), CV_32FC3, cv::Scalar(-1, -1, -1));
  cv::Mat depthBinsRight(imgRight.size(), CV_32FC3, cv::Scalar(-1, -1, -1));

  for (auto rit = pcWorldWindow.rbegin(); rit != pcWorldWindow.rend(); ++rit) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr filteredCloud(
        new pcl::PointCloud<pcl::PointXYZ>);

    Eigen::Vector3f camLeftOrigin = camLeftPose.translation().cast<float>();
    Eigen::Quaternionf camLeftOrientation = camLeftPose.quat().cast<float>();

    filterOccludedPoints(*rit, camLeftOrigin, camLeftOrientation, filteredCloud, true);

    std::vector<Eigen::Vector3f> projectedPoints;
    pcl::PointCloud<pcl::PointXYZ>::Ptr validCloud(new pcl::PointCloud<pcl::PointXYZ>);

    Eigen::Matrix4f Hwc = camLeftPose.transform().inverse().cast<float>();

    projectToRectifiedImage(imgLeft,
                            camLeftParams.at("R"),
                            camLeftParams.at("P"),
                            Hwc,
                            filteredCloud,
                            projectedPoints,
                            validCloud);

    pcl::visualization::PCLVisualizer viewer("Valid Cloud");
    viewer.addPointCloud<pcl::PointXYZ>(validCloud, "valid cloud");
    Eigen::Affine3f sensorPose =
        Eigen::Translation3f(camLeftOrigin) * camLeftOrientation;
    viewer.addCoordinateSystem(2.0, sensorPose, "sensor frame", 0);
    viewer.spin();

    // fillDepthBins(projectedPoints, depthBinsLeft);
  }
  std::cout << "Finish filling depth bins." << std::endl;

  // Convert Depth Bins to depth image
  // convertDepthBinsToImage(depthBinsLeft, depthLeft);
}