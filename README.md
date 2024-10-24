# Dataset Toolkit

## Build

1. Clone the repositories

   ```bash
   git clone git@github.com:ut-amrl/dataset-tools.git
   git submodule update --init --recursive
   ```

2. build third_party (FAST_LIO, odometry_saver) ROS packages

   ```bash
   cd third_party/catkin_ws (or docker_ws)
   catkin_make
   source devel/setup.bash
   ```

3. install the toolkit

   ```bash
   pip install -e .
   ```

### Docker Container

1. change `DATA_DIR` in `.run_docker.sh` file

2. launch `.run_docker.sh`

3. install package

   ```bash
   pip install -e .
   ```

## Process Bagfile for Introspective-Vision

> **NOTE**: Ensure `ROBOT` and `scenes` variables are correctly set in each script.

### 1. Extract data from bagfile

- **Input**: bagfile
- **Output**: Images, timestamps, extrinsics, intrinsics
- **Example**:
   1. Extract Images: `./scripts/bagfile_extraction/extract_images_warthog.sh`
   2. Extract Parameters `./scripts/bagfile_extraction/extract_params_warthog.sh`

### 2. Generate LiDAR Odometry (FAST-LIO)

- > NOTE: need to set LiDAR-IMU extrinsics for FAST-LIO and odometry_saver from the results parameter extraction
  > e.g., `third_party/FAST_LIO/config/ouster128_trevor.yaml`, `third_party/odometry_saver/launch/static_tf_trevor.launch`
- **Input**: bagfile (pointcloud, IMU)
- **Output**: LiDAR trajectory
- **Example**: `./scripts/LiDAR-Odometry/run_fast_lio_warthog.sh`

### 3. Compensate Pointcloud

- **Input**: LiDAR trajectory, Pointcloud (bagfile)
- **Output**: Compensated pointcloud
- **Example**: `./scripts/pointcloud_compensation/compensate_pointcloud_warthog.sh`

### 4. Generate Depth Image

- **Input**: LiDAR trajectory, Compensated Pointcloud, Images, timestamps, extrinsic, intrinsics, (optional) Disparity
- **Output**: Depth images
- **Example**: `./scripts/depth_generation/generate_depth_warthog.sh`

### 5. Get camera pose based on image timestamps

- **Input**: LiDAR trajectory, LiDAR-Camera extrinsic, timestamps
- **Output**: Camera trajectory
- **Example**: `./scripts/sync_poses/sync_cam_pose_warthog.sh`

### 6. Generate Reliable Map Points and indices per frame

- **Input**: LiDAR trajectory, Compensated Pointcloud
- **Output**: pointcloud of entire map
- **Example**: `./scripts/map_points_generation/generate_map_points_iris_trevor.sh`

## Generate LiDAR Odometry and Compensated Pointcloud

1. Get extrinsic between LiDAR and IMU
   - Option1: Calibration file
   - Option2: Extract from TF in bagfile
     - Example: `src/bagfile_extraction/extract_tf.py`
   - Option3: Manual Calibrations (e.g., [LiDAR_IMU_Init](https://github.com/hku-mars/LiDAR_IMU_Init))

2. Get High-Frequency LiDAR Odometry (/w [FAST-LIO](https://github.com/ut-amrl/FAST_LIO) and [odometry_saver](https://github.com/ut-amrl/odometry_saver))
   - Example: `./scripts/LiDAR-Odometry/run_fast_lio_coda.sh`

3. Compensate PointCloud with the high-frequency Odometry
   - Example: `./scripts/pointcloud_compensation/compensate_pointcloud_coda.sh`

## Interactive SLAM for global graph optimization

1. Convert Odometry and Compensated Pointcloud

2. Automatic/Manual loop closure with Interative SLAM

