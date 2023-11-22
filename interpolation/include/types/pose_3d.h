// Author: Dongmyeong Lee [domlee@utexas.edu]
// Date:   Nov 20, 2023
// Description: 3D Pose type definition for pose graph optimization.

#ifndef UT_AMRL_SLAM_TYPES_POSE_3D_H_
#define UT_AMRL_SLAM_TYPES_POSE_3D_H_

#include <functional>
#include <iomanip>
#include <istream>
#include <map>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"

namespace ut_amrl::slam {

struct Pose3d {
  Eigen::Vector3d p;
  Eigen::Quaterniond q;
  std::string format = "wxyz";

  explicit Pose3d(const std::string& format = "wxyz") : format(format) {}

  Pose3d(const Eigen::Vector3d& p,
         const Eigen::Quaterniond& q,
         const std::string& format = "wxyz")
      : p(p), q(q), format(format) {}

  // The name of the data type in g2o file format
  static std::string name() { return "VERTEX_SE3:QUAT"; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

inline std::istream& operator>>(std::istream& input, Pose3d& pose) {
  if (pose.format == "wxyz") {
    input >> pose.p.x() >> pose.p.y() >> pose.p.z() >> pose.q.w() >> pose.q.x() >>
        pose.q.y() >> pose.q.z();
  } else if (pose.format == "xyzw") {
    input >> pose.p.x() >> pose.p.y() >> pose.p.z() >> pose.q.x() >> pose.q.y() >>
        pose.q.z() >> pose.q.w();
  } else {
    throw std::runtime_error("Unknown quaternion format.");
  }

  // Normalize the quaternion to account for precision loss due to
  // serialization.
  pose.q.normalize();
  return input;
}

inline std::ostream& operator<<(std::ostream& output, const Pose3d& pose) {
  output << std::fixed << std::setprecision(8);
  if (pose.format == "wxyz") {
    output << pose.p.x() << " " << pose.p.y() << " " << pose.p.z() << " " << pose.q.w()
           << " " << pose.q.x() << " " << pose.q.y() << " " << pose.q.z();
  } else if (pose.format == "xyzw") {
    output << pose.p.x() << " " << pose.p.y() << " " << pose.p.z() << " " << pose.q.x()
           << " " << pose.q.y() << " " << pose.q.z() << " " << pose.q.w();
  } else {
    throw std::runtime_error("Unknown quaternion format.");
  }
  return output;
}

template <typename T>
using MapOfPoses3d = std::
    map<T, Pose3d, std::less<T>, Eigen::aligned_allocator<std::pair<const T, Pose3d>>>;

}  // namespace ut_amrl::slam

#endif  // UT_AMRL_SLAM_TYPES_POSE_3D_H_
