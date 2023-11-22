// Author: Dongmyeong Lee [domlee@utexas.edu]
// Date:   Nov 20, 2023
// Description: Reads a file containing timestamps and poses

#ifndef UT_AMRL_SLAM_UTILS_READ_POSE_H_
#define UT_AMRL_SLAM_UTILS_READ_POSE_H_

#include <fstream>
#include <string>

#include "glog/logging.h"

namespace ut_amrl::slam {

// Read a single pose from the input and inserts it into the map.
template <typename Pose, typename Allocator>
bool readPose(std::ifstream* infile,
              std::map<double, Pose, std::less<double>, Allocator>* poses) {
  double timestamp;
  Pose pose;
  pose.format = "wxyz";
  *infile >> timestamp >> pose;

  // Ensure we don't have duplicate timestamps
  if (poses->find(timestamp) != poses->end()) {
    LOG(ERROR) << "Duplicate timestamp: " << timestamp;
    return false;
  }
  (*poses)[timestamp] = pose;

  return true;
}

// Reads a pose file. The file format is:
// Timestamp x y z qw qx qy qz
template <typename Pose, typename MapAllocator>
bool readPoseFile(const std::string& filename,
                  std::map<double, Pose, std::less<double>, MapAllocator>* poses) {
  CHECK_NOTNULL(poses);
  poses->clear();

  std::ifstream infile(filename.c_str());
  if (!infile.is_open()) {
    LOG(ERROR) << "Failed to open file: " << filename;
    return false;
  }

  while (infile.good()) {
    if (!readPose(&infile, poses)) {
      LOG(ERROR) << "Failed to read pose from file: " << filename;
      return false;
    }

    // Skip any whitespace
    infile >> std::ws;
  }

  return true;
}

}  // namespace ut_amrl::slam

#endif  // UT_AMRL_SLAM_UTILS_READ_POSE_H_
