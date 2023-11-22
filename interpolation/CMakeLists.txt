cmake_minimum_required(VERSION 3.10)
project(interpolate_keyframes)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find Eigen, Ceres
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(manif REQUIRED)

# Include directories
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CERES_INCLUDE_DIRS})
include_directories(${manif_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add executable
add_executable(interpolate_keyframes interpolate_keyframes.cpp)

# Link libraries
target_link_libraries(interpolate_keyframes ${CERES_LIBRARIES} gflags glog)
