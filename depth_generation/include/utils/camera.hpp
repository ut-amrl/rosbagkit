#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>

bool loadMatrix(const YAML::Node& node, cv::Mat& mat, int rows, int cols) {
  if (!node["data"]) {
    std::cerr << "Error: Missing 'data' key." << std::endl;
    return false;
  }
  auto data = node["data"].as<std::vector<double>>();
  if (data.size() != rows * cols) {
    std::cerr << "Error: Expected " << rows * cols << " elements, but found "
              << data.size() << "." << std::endl;
    return false;
  }
  mat = cv::Mat(rows, cols, CV_64F, data.data()).clone();
  return true;
}

void loadCamParams(const std::string& configFile,
                   std::unordered_map<std::string, cv::Mat>& camParams) {
  YAML::Node config;
  try {
    config = YAML::LoadFile(configFile);
  } catch (const YAML::Exception& e) {
    std::cerr << "Error loading configuration file: " << e.what() << std::endl;
    return;
  }

  // Load image size
  try {
    int width = config["width"].as<int>();
    int height = config["height"].as<int>();
    camParams["image_size"] = (cv::Mat_<int>(1, 2) << width, height);
  } catch (const YAML::TypedBadConversion<int>& e) {
    std::cerr << "Error converting image size: " << e.what() << std::endl;
    return;
  }

  // Load camera matrix
  if (!loadMatrix(config["camera_matrix"], camParams["K"], 3, 3)) return;

  // Load distortion coefficients
  if (!loadMatrix(config["distortion_coefficients"], camParams["D"], 1, 5)) return;

  // Load rectification matrix
  if (!loadMatrix(config["rectification_matrix"], camParams["R"], 3, 3)) return;

  // Load projection matrix
  if (!loadMatrix(config["projection_matrix"], camParams["P"], 3, 4)) return;

  if (FLAGS_v > 0) {
    std::cout << "Loaded camera parameters from " << configFile << std::endl;
    std::cout << "Image size: " << camParams["image_size"] << std::endl;
    std::cout << "K:\n" << camParams["K"] << std::endl;
    std::cout << "D:\n" << camParams["D"] << std::endl;
    std::cout << "R:\n" << camParams["R"] << std::endl;
    std::cout << "P:\n" << camParams["P"] << std::endl;
  }
}

#endif  // CAMERA_HPP