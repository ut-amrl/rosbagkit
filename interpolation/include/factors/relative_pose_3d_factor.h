// Author: Dongmyeong Lee [domlee@utexas.edu]
// Date:   Nov 20, 2023
// Description: Relative pose 3d factor for pose graph optimization.

#ifndef UT_AMRL_SLAM_FACTOR_RELATIVE_POSE_3D_FACTOR_H_
#define UT_AMRL_SLAM_FACTOR_RELATIVE_POSE_3D_FACTOR_H_

#include <cmath>
#include <utility>

#include "Eigen/Core"
#include "ceres/autodiff_cost_function.h"
#include "types/pose_3d.h"
#include "types/relative_pose_3d.h"

namespace ut_amrl::slam {

class RelativePose3dFactor {
 public:
  RelativePose3dFactor(Pose3d t_ab_measured,
                       Eigen::Matrix<double, 6, 6> sqrt_information)
      : t_ab_measured_(std::move(t_ab_measured)),
        sqrt_information_(std::move(sqrt_information)) {}

  template <typename T>
  bool operator()(const T* const p_a_ptr,
                  const T* const q_a_ptr,
                  const T* const p_b_ptr,
                  const T* const q_b_ptr,
                  T* residuals_ptr) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_a(p_a_ptr);
    Eigen::Map<const Eigen::Quaternion<T>> q_a(q_a_ptr);

    Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_b(p_b_ptr);
    Eigen::Map<const Eigen::Quaternion<T>> q_b(q_b_ptr);

    // Compute the relative transformation between the two frames.
    // q_ab = q_a^{-1} * q_b
    Eigen::Quaternion<T> q_a_inverse = q_a.conjugate();
    Eigen::Quaternion<T> q_ab_estimated = q_a_inverse * q_b;

    // Represent the displacement between the two frames in the A frame.
    // p_ab = q_a^{-1} * (p_b - p_a)
    Eigen::Matrix<T, 3, 1> p_ab_estimated = q_a_inverse * (p_b - p_a);

    // Compute the error between the two orientation estimates.
    // delta_q = q_ab * \hat{q_ab}^{-1}
    Eigen::Quaternion<T> delta_q =
        q_ab_estimated * t_ab_measured_.q.template cast<T>().conjugate();

    // Compute the residuals.
    // [ position    ] = [ delta_p ]
    // [ orientation ]   [ 2 * delta_q ]
    Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr);
    residuals.template block<3, 1>(0, 0) =
        p_ab_estimated - t_ab_measured_.p.template cast<T>();
    residuals.template block<3, 1>(3, 0) = T(2.0) * delta_q.vec();

    // Scale the residuals by the measurement uncertainty.
    residuals.applyOnTheLeft(sqrt_information_.template cast<T>());

    return true;
  }

  static ceres::CostFunction* Create(
      const Pose3d& t_ab_measured,
      const Eigen::Matrix<double, 6, 6>& sqrt_information) {
    return new ceres::AutoDiffCostFunction<RelativePose3dFactor, 6, 3, 4, 3, 4>(
        new RelativePose3dFactor(t_ab_measured, sqrt_information));
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  // The measurement for the position of B relative to A in the A frame.
  const Pose3d t_ab_measured_;
  // The square root of the measurement information matrix. (LL^T = W^-1)
  const Eigen::Matrix<double, 6, 6> sqrt_information_;
};

}  // namespace ut_amrl::slam

#endif  // UT_AMRL_SLAM_FACTOR_RELATIVE_POSE_3D_FACTOR_H_
