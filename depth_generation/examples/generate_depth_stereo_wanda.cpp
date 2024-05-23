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
              //"gq_appld_wandagq_32_field_foresttrail_06_2024-03-15-11-17-44",
              "The scene name.");
DEFINE_int32(window_size, 20, "The window size for accumulating pointclouds.");

void loadPointcloudData(const std::string &pointcloudDir,
                        const std::string &pcPoseFile,
                        std::vector<std::string> &pointcloudFiles,
                        std::vector<manif::SE3d> &pcPoses,
                        std::vector<double> &pcTimestamps);

void loadImageData(const std::string &imageLeftDir,
                   const std::string &imageRightDir,
                   const std::string &leftPoseFile,
                   const std::string &rightPoseFile,
                   std::vector<std::string> &imageLeftFiles,
                   std::vector<std::string> &imageRightFiles,
                   std::vector<manif::SE3d> &leftPoses,
                   std::vector<manif::SE3d> &rightPoses,
                   std::vector<double> &imageTimestamps);

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  std::cout << "Dataset directory: " << FLAGS_dataset_dir << std::endl;
  std::cout << "Scene: " << FLAGS_scene << std::endl;

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
  std::string leftPoseFile =
      FLAGS_dataset_dir + "/poses/" + FLAGS_scene + "/cam_left.txt";
  std::string rightPoseFile =
      FLAGS_dataset_dir + "/poses/" + FLAGS_scene + "/cam_right.txt";
  if (!std::filesystem::exists(leftPoseFile) ||
      !std::filesystem::exists(rightPoseFile)) {
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
  std::vector<manif::SE3d> leftPoses;
  std::vector<manif::SE3d> rightPoses;
  std::vector<double> pcTimestamps;
  std::vector<double> imageTimestamps;
  std::unordered_map<std::string, cv::Mat> camLeftParams;
  std::unordered_map<std::string, cv::Mat> camRightParams;

  loadPointcloudData(pointcloudDir, pcPoseFile, pointcloudFiles, pcPoses, pcTimestamps);
  loadImageData(imageLeftDir,
                imageRightDir,
                leftPoseFile,
                rightPoseFile,
                imageLeftFiles,
                imageRightFiles,
                leftPoses,
                rightPoses,
                imageTimestamps);
  loadCamParams(calibLeftFile, camLeftParams);
  loadCamParams(calibRightFile, camRightParams);

  generateDepthStereo(pointcloudFiles,
                      pcPoses,
                      pcTimestamps,
                      imageLeftFiles,
                      imageRightFiles,
                      leftPoses,
                      rightPoses,
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
  while (true) {
    double stamp;
    double tx, ty, tz, qw, qx, qy, qz;

    if (!(pcPoseStream >> stamp >> tx >> ty >> tz >> qw >> qx >> qy >> qz)) {
      break;
    }

    manif::SE3d Hwl(Eigen::Vector3d(tx, ty, tz),
                    Eigen::Quaterniond(qw, qx, qy, qz).normalized());
    pcPoses.push_back(Hwl);

    pcTimestamps.push_back(stamp);

    std::string pcFile =
        pointcloudDir + "/3d_comp_os1_" + std::to_string(pcIdx) + ".bin";
    pointcloudFiles.push_back(pcFile);
    pcIdx++;
  }

  if (FLAGS_v > 0) {
    std::cout << "Loaded " << pcIdx << " pointcloud files and poses." << std::endl;
  }
}

void loadImageData(const std::string &imageLeftDir,
                   const std::string &imageRightDir,
                   const std::string &leftPoseFile,
                   const std::string &rightPoseFile,
                   std::vector<std::string> &imageLeftFiles,
                   std::vector<std::string> &imageRightFiles,
                   std::vector<manif::SE3d> &leftPoses,
                   std::vector<manif::SE3d> &rightPoses,
                   std::vector<double> &imageTimestamps) {
  int imgIdx = 0;

  imageLeftFiles.clear();
  imageRightFiles.clear();
  leftPoses.clear();
  rightPoses.clear();
  imageTimestamps.clear();

  std::ifstream leftPoseStream(leftPoseFile);
  std::ifstream rightPoseStream(rightPoseFile);
  while (true) {
    double stampLeft, stampRight;
    double tx, ty, tz, qw, qx, qy, qz;

    // Read the left camera pose
    if (!(leftPoseStream >> stampLeft >> tx >> ty >> tz >> qw >> qx >> qy >> qz) ||
        !(rightPoseStream >> stampRight >> tx >> ty >> tz >> qw >> qx >> qy >> qz)) {
      break;
    }

    leftPoses.emplace_back(Eigen::Vector3d(tx, ty, tz),
                           Eigen::Quaterniond(qw, qx, qy, qz).normalized());

    rightPoses.emplace_back(Eigen::Vector3d(tx, ty, tz),
                            Eigen::Quaterniond(qw, qx, qy, qz).normalized());

    // Check if the timestamps are the same
    assert(std::abs(stampLeft - stampRight) < 1e-1);
    imageTimestamps.push_back((stampLeft + stampRight) / 2.0);

    // Read the image files
    std::string imageLeftFile =
        imageLeftDir + "/2d_rect_left_" + std::to_string(imgIdx) + ".jpg";
    std::string imageRightFile =
        imageRightDir + "/2d_rect_right_" + std::to_string(imgIdx) + ".jpg";
    imageLeftFiles.push_back(imageLeftFile);
    imageRightFiles.push_back(imageRightFile);

    imgIdx++;
  }

  if (FLAGS_v > 0) {
    std::cout << "Loaded " << imgIdx << " stereo image files and poses." << std::endl;
  }
}
