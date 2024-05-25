// Author: Dongmyeong Lee [domlee@utexas.edu]
// Date:   Nov 20, 2023
// Description: Relative pose 3d type for g2o file format.

#ifndef UT_AMRL_SLAM_TYPES_RELATIVE_POSE_3D_H_
#define UT_AMRL_SLAM_TYPES_RELATIVE_POSE_3D_H_
#include <functional>
#include <istream>
#include <map>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "pose_3d.h"

namespace ut_amrl::slam {

template <typename T>
struct RelativePose3d {
  T id_begin;
  T id_end;

  // The transformation that represents the pose of the end frame E w.r.t. the begin
  // frame B. In other words, it transforms a vector in the E frame to the B frame.
  Pose3d t_be;

  // The square root of the information matrix.
  // Information matrix is the inverse of the covariance matrix.
  Eigen::Matrix<double, 6, 6> sqrt_information;

  // The name of the data type in the g2o file format.
  static std::string name() { return "EDGE_SE3:QUAT"; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template <typename T>
inline std::istream& operator>>(std::istream& input, RelativePose3d<T>& relative_pose) {
  Pose3d& t_be = relative_pose.t_be;
  input >> relative_pose.id_begin >> relative_pose.id_end >> t_be;

  // Read the information matrix.
  Eigen::Matrix<double, 6, 6> information;
  for (int i = 0; i < 6 && input.good(); ++i) {
    for (int j = i; j < 6 && input.good(); ++j) {
      input >> information(i, j);
      if (i != j) {
        information(j, i) = information(i, j);
      }
    }
  }
  // Compute the square root of the information matrix using its Cholesky
  relative_pose.sqrt_information = information.llt().matrixL();

  return input;
}

template <typename T>
using VectorOfRelativePoses3d =
    std::vector<RelativePose3d<T>, Eigen::aligned_allocator<RelativePose3d<T>>>;

}  // namespace ut_amrl::slam
#endif  // UT_AMRL_SLAM_TYPES_RELATIVE_POSE_3D_H_
