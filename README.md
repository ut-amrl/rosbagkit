# Dataset Toolkit

## Generate LiDAR Odometry and Compensated Pointcloud

1. Get extrinsic between LiDAR and IMU
   - extract from TF: `py_scripts/bagfile_extraction/extract_tf.py`
   - Manual Calibrations
     - [LiDAR_IMU_Init](https://github.com/hku-mars/LiDAR_IMU_Init)

2. Get High-Frequency LiDAR Odometry (/w [Point-LIO](https://github.com/ut-amrl/Point-LIO) and [odometry_saver](git@github.com:ut-amrl/odometry_saver.git))
   - Example: `./bash_scripts/LiDAR-Inertial-Odometry/run_point_lio_coda.sh`
