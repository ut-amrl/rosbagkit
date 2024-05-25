cmake_minimum_required(VERSION 3.10)
project(interpolate_keyframes)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find Eigen, Ceres
find_package(Eigen3 3.3 REQUIRED)
find_package(Ceres REQUIRED)

# print all variables
message(STATUS "EIGEN3_INCLUDE_DIR: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "CERES_INCLUDE_DIRS: ${CERES_INCLUDE_DIRS}")
message(STATUS "CERES_LIBRARIES: ${CERES_LIBRARIES}")

# Include directories
# include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CERES_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(LIBS PRIVATE
    ${CERES_LIBRARIES}
    # ${EIGEN3_LIBRARIES}
    gflags
    glog
)


# Add executable
add_executable(interpolate_keyframes interpolate_keyframes.cpp)

# Link libraries
target_link_libraries(interpolate_keyframes ${LIBS})
