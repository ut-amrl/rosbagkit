#include <filesystem>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>
//
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "manif/manif.h"
//
#include "generate_depth_stereo.h"
#include "utils/camera.hpp"

DEFINE_string(dataset_dir,
              "/home/dongmyeong/Projects/datasets/SARA/wanda",
              "The directory of the wanda dataset.");
DEFINE_string(scene,
              "gq_appld_wandagq_32_field_foresttrail_06_2024-03-15-11-17-44",
              "The scene name.");
DEFINE_int32(window_size, 10, "The window size for accumulating pointclouds.");

void loadPointcloudData(const std::string &pointcloudDir,
                        const std::string &pcPoseFile,
                        std::vector<std::string> &pointcloudFiles,
                        std::vector<manif::SE3d> &pcPoses,
                        std::vector<double> &pcTimestamps);

void loadImageData(const std::string &imageLeftDir,
                   const std::string &imageRightDir,
                   const std::string &imageLeftPoseFile,
                   const std::string &imageRightPoseFile,
                   std::vector<std::string> &imageLeftFiles,
                   std::vector<std::string> &imageRightFiles,
                   std::vector<manif::SE3d> &imageLeftPoses,
                   std::vector<manif::SE3d> &imageRightPoses,
                   std::vector<double> &imageTimestamps);

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  // Pointcloud data
  std::string pointcloudDir = FLAGS_dataset_dir + "/3d_comp/" + FLAGS_scene;
  std::string pcPoseFile = FLAGS_dataset_dir + "/poses/" + FLAGS_scene + "/os1.txt";
  if (!std::filesystem::exists(pcPoseFile)) {
    std::cerr << "Pointcloud data or pose file does not exist." << std::endl;
    return -1;
  }

  // Stereo image data
  std::string imageLeftDir = FLAGS_dataset_dir + "/2d_rect/" + FLAGS_scene + "/left";
  std::string imageRightDir = FLAGS_dataset_dir + "/2d_rect/" + FLAGS_scene + "/right";
  std::string imageLeftPoseFile =
      FLAGS_dataset_dir + "/poses/" + FLAGS_scene + "/cam_left.txt";
  std::string imageRightPoseFile =
      FLAGS_dataset_dir + "/poses/" + FLAGS_scene + "/cam_right.txt";
  if (!std::filesystem::exists(imageLeftPoseFile) ||
      !std::filesystem::exists(imageRightPoseFile)) {
    std::cerr << "Stereo image pose files do not exist." << std::endl;
    return -1;
  }

  // Calibration data
  std::string calibLeftFile =
      FLAGS_dataset_dir + "/calibrations/" + FLAGS_scene + "/cam_left_intrinsics.yaml";
  std::string calibRightFile =
      FLAGS_dataset_dir + "/calibrations/" + FLAGS_scene + "/cam_right_intrinsics.yaml";
  if (!std::filesystem::exists(calibLeftFile) ||
      !std::filesystem::exists(calibRightFile)) {
    std::cerr << "Calibration files do not exist." << std::endl;
    return -1;
  }

  // Output directory and create it if it does not exist
  std::string depthLeftDir = FLAGS_dataset_dir + "/2d_depth/" + FLAGS_scene + "/left";
  std::string depthRightDir = FLAGS_dataset_dir + "/2d_depth/" + FLAGS_scene + "/right";
  std::filesystem::create_directories(depthLeftDir);
  std::filesystem::create_directories(depthRightDir);

  // Load the data
  std::vector<std::string> pointcloudFiles;
  std::vector<std::string> imageLeftFiles;
  std::vector<std::string> imageRightFiles;
  std::vector<manif::SE3d> pcPoses;
  std::vector<manif::SE3d> imageLeftPoses;
  std::vector<manif::SE3d> imageRightPoses;
  std::vector<double> pcTimestamps;
  std::vector<double> imageTimestamps;
  std::unordered_map<std::string, cv::Mat> camLeftParams;
  std::unordered_map<std::string, cv::Mat> camRightParams;

  loadPointcloudData(pointcloudDir, pcPoseFile, pointcloudFiles, pcPoses, pcTimestamps);
  loadImageData(imageLeftDir,
                imageRightDir,
                imageLeftPoseFile,
                imageRightPoseFile,
                imageLeftFiles,
                imageRightFiles,
                imageLeftPoses,
                imageRightPoses,
                imageTimestamps);
  loadCamParams(calibLeftFile, camLeftParams);
  loadCamParams(calibRightFile, camRightParams);

  generateDepthStereo(pointcloudFiles,
                      pcPoses,
                      pcTimestamps,
                      imageLeftFiles,
                      imageRightFiles,
                      imageLeftPoses,
                      imageRightPoses,
                      imageTimestamps,
                      camLeftParams,
                      camRightParams,
                      FLAGS_window_size,
                      depthLeftDir,
                      depthRightDir);

  return 0;
}

void loadPointcloudData(const std::string &pointcloudDir,
                        const std::string &pcPoseFile,
                        std::vector<std::string> &pointcloudFiles,
                        std::vector<manif::SE3d> &pcPoses,
                        std::vector<double> &pcTimestamps) {
  int pcIdx = 0;

  pointcloudFiles.clear();
  pcPoses.clear();
  pcTimestamps.clear();

  std::ifstream pcPoseStream(pcPoseFile);
  while (!pcPoseStream.eof()) {
    double timestamp;
    double tx, ty, tz, qw, qx, qy, qz;
    pcPoseStream >> timestamp >> tx >> ty >> tz >> qw >> qx >> qy >> qz;
    manif::SE3d Hlw(Eigen::Vector3d(tx, ty, tz),
                    Eigen::Quaterniond(qw, qx, qy, qz).normalized());
    pcPoses.push_back(Hlw);

    pcTimestamps.push_back(timestamp);

    std::string pcFile =
        pointcloudDir + "/3d_comp_os1_" + std::to_string(pcIdx) + ".bin";
    pointcloudFiles.push_back(pcFile);
    pcIdx++;
  }
  assert(pcPoses.size() == pointcloudFiles.size());

  if (FLAGS_v > 0) {
    std::cout << "Loaded " << pcIdx << " pointcloud files and poses." << std::endl;
  }
}

void loadImageData(const std::string &imageLeftDir,
                   const std::string &imageRightDir,
                   const std::string &imageLeftPoseFile,
                   const std::string &imageRightPoseFile,
                   std::vector<std::string> &imageLeftFiles,
                   std::vector<std::string> &imageRightFiles,
                   std::vector<manif::SE3d> &imageLeftPoses,
                   std::vector<manif::SE3d> &imageRightPoses,
                   std::vector<double> &imageTimestamps) {
  int imgIdx = 0;

  imageLeftFiles.clear();
  imageRightFiles.clear();
  imageLeftPoses.clear();
  imageRightPoses.clear();
  imageTimestamps.clear();

  std::ifstream imageLeftPoseStream(imageLeftPoseFile);
  std::ifstream imageRightPoseStream(imageRightPoseFile);
  while (!imageLeftPoseStream.eof() && !imageRightPoseStream.eof()) {
    double timestampLeft, timestampRight;
    double tx, ty, tz, qw, qx, qy, qz;
    // Read the left camera pose
    imageLeftPoseStream >> timestampLeft >> tx >> ty >> tz >> qw >> qx >> qy >> qz;
    manif::SE3d HcwLeft(Eigen::Vector3d(tx, ty, tz),
                        Eigen::Quaterniond(qw, qx, qy, qz).normalized());
    imageLeftPoses.push_back(HcwLeft);

    // Read the right camera pose
    imageRightPoseStream >> timestampRight >> tx >> ty >> tz >> qw >> qx >> qy >> qz;
    manif::SE3d HcwRight(Eigen::Vector3d(tx, ty, tz),
                         Eigen::Quaterniond(qw, qx, qy, qz).normalized());
    imageRightPoses.push_back(HcwRight);

    // Check if the timestamps are the same
    assert(std::abs(timestampLeft - timestampRight) < 1e-1);
    imageTimestamps.push_back((timestampLeft + timestampRight) / 2.0);

    // Read the image files
    std::string imageLeftFile =
        imageLeftDir + "/2d_rect_left_" + std::to_string(imgIdx) + ".jpg";
    std::string imageRightFile =
        imageRightDir + "/2d_rect_right_" + std::to_string(imgIdx) + ".jpg";
    imageLeftFiles.push_back(imageLeftFile);
    imageRightFiles.push_back(imageRightFile);

    imgIdx++;
  }
  assert(imageLeftPoses.size() == imageRightPoses.size());
  assert(imageLeftPoses.size() == imageLeftFiles.size());
  assert(imageRightPoses.size() == imageRightFiles.size());

  if (FLAGS_v > 0) {
    std::cout << "Loaded " << imgIdx << " stereo image files and poses." << std::endl;
  }
}