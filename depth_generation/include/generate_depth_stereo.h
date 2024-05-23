#ifndef GENERATE_DEPTH_STEREO_H
#define GENERATE_DEPTH_STEREO_H

#include <deque>
#include <string>
#include <unordered_map>
#include <vector>
//
#include <opencv2/core.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "manif/SE3.h"

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
                         const std::string& depthRightDir);

void computeStereoDepth(
    const cv::Mat& imgLeft,
    const cv::Mat& imgRight,
    const manif::SE3d& camLeftPose,
    const manif::SE3d& camRightPose,
    const std::unordered_map<std::string, cv::Mat>& camLeftParams,
    const std::unordered_map<std::string, cv::Mat>& camRightParams,
    const std::deque<pcl::PointCloud<pcl::PointXYZ>::Ptr>& pcWorldWindow,
    cv::Mat& depthLeft,
    cv::Mat& depthRight);

#endif  // GENERATE_DEPTH_STEREO_H