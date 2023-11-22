cmake_minimum_required(VERSION 3.10)
project(PoseGraph3D)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find Eigen
find_package(Eigen3 REQUIRED)

# Find Ceres
find_package(Ceres REQUIRED)

# Include directories
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CERES_INCLUDE_DIRS})

# Add executable
add_executable(pose_graph_3d pose_graph_3d.cpp)

# Link libraries
target_link_libraries(pose_graph_3d ${CERES_LIBRARIES})

