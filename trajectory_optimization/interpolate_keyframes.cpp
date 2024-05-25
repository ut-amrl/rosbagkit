// Author: Dongmyeong Lee [domlee@utexas.edu]
// Date:   Nov 20, 2023
// Description: Interpolate the keyframe poses with odometry data to get poses at
// required time stamps using Pose Graph Optimization.

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "ceres/ceres.h"
#include "factors/relative_pose_3d_factor.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "manif/SE3.h"
#include "types/pose_3d.h"
#include "types/relative_pose_3d.h"
#include "utils/read_pose.h"

DEFINE_string(input_kf_file,
              "/home/dongmyeong/Projects/AMRL/CODa/poses/global_keyframe/0.txt",
              "The input keyframe poses.");
DEFINE_string(input_odom_file,
              "/home/dongmyeong/Projects/AMRL/CODa/poses/dense_keyframe/1.txt",
              "The input odometry poses.");
DEFINE_string(output_file,
              "/home/dongmyeong/Projects/AMRL/CODa/poses/os1.txt",
              "The output poses.");

namespace ut_amrl::slam {
namespace {

template <typename T>
RelativePose3d<T> getRelativePose(const T id_begin,
                                  const Pose3d& pose_begin,
                                  const T id_end,
                                  const Pose3d& pose_end,
                                  double weight = 1.0) {
  // Get the relative pose between the odometry poses.
  RelativePose3d<T> relative_pose;
  relative_pose.id_begin = id_begin;
  relative_pose.id_end = id_end;

  Eigen::Quaterniond q_begin_inverse = pose_begin.q.conjugate();
  relative_pose.t_be.q = q_begin_inverse * pose_end.q;
  relative_pose.t_be.p = q_begin_inverse * (pose_end.p - pose_begin.p);

  // Set the information matrix for the relative pose.
  relative_pose.sqrt_information = Eigen::Matrix<T, 6, 6>::Identity() * weight;
  relative_pose.sqrt_information(3, 3) *= 30.0;
  relative_pose.sqrt_information(4, 4) *= 30.0;
  relative_pose.sqrt_information(5, 5) *= 30.0;

  return relative_pose;
}

template <typename T>
void buildInterpolationOptimizationProblem(const MapOfPoses3d<T>& kf_poses,
                                           const MapOfPoses3d<T>& odom_poses,
                                           MapOfPoses3d<T>* poses,
                                           ceres::Problem* problem) {
  CHECK_NOTNULL(poses);
  CHECK_NOTNULL(problem);
  poses->clear();
  if (kf_poses.empty() || odom_poses.empty()) {
    LOG(ERROR) << "The keyframe or odometry poses are empty.";
    return;
  }

  // #1 Initialize the poses with the odometry poses.
  for (const auto& [timestamp, odom_pose] : odom_poses) {
    auto kf_up_it = kf_poses.lower_bound(timestamp);
    if (timestamp == kf_up_it->first) {
      // The timestamp matches the keyframe timestamp exactly.
      poses->emplace(timestamp, kf_up_it->second);
    } else {
      poses->emplace(timestamp, odom_pose);
    }
  }

  // #2
  // Insert the keyframe that is not in the odometry poses. This is to make sure
  // that the keyframe poses are included in the optimization problem.
  // Add factor between the keyframe and nearest odometry pose.
  VectorOfRelativePoses3d<T> relative_poses;
  for (const auto& [kf_timestamp, kf_pose] : kf_poses) {
    if (poses->find(kf_timestamp) == poses->end()) {
      const auto& odom_up_it = odom_poses.lower_bound(kf_timestamp);
      // If keyframe is out of range of odometry, skip.
      // odom_lower -> keyframe -> odom_upper
      if (odom_up_it == odom_poses.begin() || odom_up_it == odom_poses.end()) {
        break;
      }

      // Insert the keyframe pose to the poses.
      poses->emplace(kf_timestamp, kf_pose);

      const auto& odom_lo_it = std::prev(odom_up_it);
      // get the relative pose between the keyframe and odometry pose.
      double t =
          (kf_timestamp - odom_lo_it->first) / (odom_up_it->first - odom_lo_it->first);

      // Linear interpolation of the odometry pose.
      manif::SE3d odom_lower_SE3(odom_lo_it->second.p, odom_lo_it->second.q);
      manif::SE3d odom_upper_SE3(odom_up_it->second.p, odom_up_it->second.q);

      // xi = log(X_lo_it^{-1} * X_up_it)
      manif::SE3Tangentd xi = odom_upper_SE3 - odom_lower_SE3;

      // X_interpolated = X_lower * exp(t * xi)
      manif::SE3d odom_interp_SE3 = odom_lower_SE3 + t * xi;
      Pose3d odom_interp(odom_interp_SE3.translation(), odom_interp_SE3.quat());

      relative_poses.push_back(getRelativePose(
          odom_lo_it->first, odom_lo_it->second, kf_timestamp, odom_interp));
      relative_poses.push_back(getRelativePose(
          kf_timestamp, odom_interp, odom_up_it->first, odom_up_it->second));
    }
  }

  // #3
  // Get the relative poses between the odometry poses.
  // The relative poses are used as constraints for the pose graph optimization.
  for (auto odom_it = odom_poses.begin(); odom_it != odom_poses.end(); ++odom_it) {
    auto odom_next_it = std::next(odom_it);
    if (odom_next_it == odom_poses.end()) {
      break;
    }

    RelativePose3d<T> relative_pose = getRelativePose(
        odom_it->first, odom_it->second, odom_next_it->first, odom_next_it->second);
    relative_poses.push_back(relative_pose);
  }

  // #4
  // Build the pose graph optimization problem.
  ceres::LossFunction* loss_function = nullptr;
  ceres::Manifold* quaternion_manifold = new ceres::EigenQuaternionManifold;

  for (const auto& relative_pose : relative_poses) {
    auto pose_begin_it = poses->find(relative_pose.id_begin);
    CHECK(pose_begin_it != poses->end())
        << "The pose with timestamp " << relative_pose.id_begin << " is not found.";
    auto pose_end_it = poses->find(relative_pose.id_end);
    CHECK(pose_end_it != poses->end())
        << "The pose with timestamp " << relative_pose.id_end << " is not found.";

    // Ceres will take ownership of the pointer.
    ceres::CostFunction* cost_function = RelativePose3dFactor::Create(
        relative_pose.t_be, relative_pose.sqrt_information);

    problem->AddResidualBlock(cost_function,
                              loss_function,
                              pose_begin_it->second.p.data(),
                              pose_begin_it->second.q.coeffs().data(),
                              pose_end_it->second.p.data(),
                              pose_end_it->second.q.coeffs().data());

    // Set the local parameterization for the quaternion.
    problem->SetManifold(pose_begin_it->second.q.coeffs().data(), quaternion_manifold);
    problem->SetManifold(pose_end_it->second.q.coeffs().data(), quaternion_manifold);
  }

  // #5
  // Fix the keyframe poses.
  for (const auto& [timestamp, kf_pose] : kf_poses) {
    auto pose_it = poses->find(timestamp);
    if (pose_it == poses->end()) {
      continue;
    }
    problem->SetParameterBlockConstant(pose_it->second.p.data());
    problem->SetParameterBlockConstant(pose_it->second.q.coeffs().data());
  }
}

// Return true if the optimization problem is solved.
bool SolveInterpOptimizationProblem(ceres::Problem* problem) {
  std::cout << "Solving the interpolation optimization problem..." << std::endl;
  CHECK_NOTNULL(problem);

  ceres::Solver::Options options;
  options.max_num_iterations = 200;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, problem, &summary);

  std::cout << summary.FullReport() << std::endl;

  return summary.IsSolutionUsable();
}

// Output the poses to the file with format: timestamp tx ty tz qw qx qy qz
bool outInterpolatedPoses(const std::string& filename,
                          const MapOfPoses3d<double>& poses) {
  std::ofstream outfile(filename);
  if (!outfile.is_open()) {
    LOG(ERROR) << "Failed to open the output file: " << filename;
    return false;
  }

  for (const auto& [timestamp, pose] : poses) {
    outfile << std::fixed << std::setprecision(6) << timestamp << " " << pose << '\n';
  }

  return true;
}

}  // namespace
}  // namespace ut_amrl::slam

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  CHECK(!FLAGS_input_kf_file.empty()) << "The input keyframe file is empty.";
  CHECK(!FLAGS_input_odom_file.empty()) << "The input odometry file is empty.";
  CHECK(!FLAGS_output_file.empty()) << "The output directory is empty.";

  // Read the keyframe and odometry poses.
  ut_amrl::slam::MapOfPoses3d<double> kf_poses;
  CHECK(ut_amrl::slam::readPoseFile(FLAGS_input_kf_file, &kf_poses));
  LOG(INFO) << "Number of keyframe poses: " << kf_poses.size();

  ut_amrl::slam::MapOfPoses3d<double> odom_poses;
  CHECK(ut_amrl::slam::readPoseFile(FLAGS_input_odom_file, &odom_poses));
  LOG(INFO) << "Number of odometry poses: " << odom_poses.size();

  // Interpolate the keyframe poses with odometry poses using pose graph optimization
  ut_amrl::slam::MapOfPoses3d<double> interpolated_poses;
  ceres::Problem problem;
  ut_amrl::slam::buildInterpolationOptimizationProblem(
      kf_poses, odom_poses, &interpolated_poses, &problem);

  CHECK(ut_amrl::slam::SolveInterpOptimizationProblem(&problem))
      << "Failed to solve the interpolation optimization problem.";

  // Write the interpolated poses.
  CHECK(ut_amrl::slam::outInterpolatedPoses(FLAGS_output_file, interpolated_poses))
      << "Failed to write the interpolated poses.";

  return 0;
}
