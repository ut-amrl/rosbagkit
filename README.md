### Install
```bash
conda create -n rosbagkit python=3.12
pip install -e .
```

### Process (for BEV-Patch-PF)

#### 1. Extract data (RGB, Depth, GPS ...) from rosbag files
- Set configuration (e.g., `config/extract/jackal_ahg_courtyard.yaml`)
- `python scripts/extract_bagfile.py --config=<config_path>`

#### 2. Get ground truth UTM pose
- RTK-GPS or LiDAR-Inertial Odometry
- Align the odometry in UTM coordinates (check bev-patch-pf repo)

#### 3. Generate depth image (for stereo images)
- Foundation Stereo (https://github.com/NVlabs/FoundationStereo/) 