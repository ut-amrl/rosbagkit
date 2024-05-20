cmake_minimum_required(VERSION 3.10)
project(depth_generation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(PCL CONFIG REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# FetchContent for yaml-cpp
include(FetchContent)
FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG yaml-cpp-0.7.0 # Specifying the version 0.8.0
)
FetchContent_GetProperties(yaml-cpp)
if(NOT yaml-cpp_POPULATED)
  message(STATUS "Fetching yaml-cpp...")
  FetchContent_Populate(yaml-cpp)
  add_subdirectory(${yaml-cpp_SOURCE_DIR} ${yaml-cpp_BINARY_DIR})
endif()


include_directories(${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR}
${CMAKE_CURRENT_SOURCE_DIR}/include)

# Group libraries into a single variable
set(LIBS PRIVATE
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
    ${EIGEN3_LIBRARIES}
    yaml-cpp::yaml-cpp
    gflags
    glog
)

# Set the output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Add the executables
add_executable(generate_depth_wanda examples/generate_depth_stereo_wanda.cpp
               src/generate_depth_stereo.cpp)
target_link_libraries(generate_depth_wanda ${LIBS})
