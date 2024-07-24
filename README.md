# Dataset Toolkit

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

## Process Bagfile for Introspective-Vision

1. Extract data from bagfile
   - Input: bagfile
   - Output: Images, timestamps, extrinsics, intrinsics
   - Example:
     - `./scripts/bagfile_extraction/extract_images_wilbur.sh`
     - `./scripts/bagfile_extraction/extract_params_wilbur.sh`

2. Generate LiDAR Odometry
   - Input: bagfile (pointcloud, IMU)
   - Output: LiDAR trajectory
   - Example:
     - `./scripts/LiDAR-Odometry/run_fast_lio_wilbur.sh`

3. Compensate Pointcloud
   - Input: LiDAR trajectory, Pointcloud (bagfile)
   - Output: Compensated pointcloud
   - Example:
     - `./scripts/pointcloud_compensation/compensate_pointcloud_wilbur.sh`

4. Generate Depth Image
   - Input: LiDAR trajectory, Compensated Pointcloud, Images, timestamps, extrinsic, intrinsics, (optional) Disparity
   - Output: Depth image
   - Example:
     - `./scripts/depth_generation/generate_depth_wilbur.sh`

5. Get camera pose based on image timestamps
   - Input: LiDAR trajectory, LiDAR-Camera extrinsic, timestamps
   - Example: `./scripts/sync_cam_poses/sync_cam_pose_wilbur.sh`

6. Generate Static Map
   - Input: LiDAR trajectory, Compensated Pointcloud
   - Example: `scripts/static_map_generation/generate_static_map_warthog.sh`