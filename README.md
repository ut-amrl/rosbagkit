### Install
```bash
conda create -n rosbagkit python=3.12
pip install -e .
```

### Process

#### 1. extract data from rosbag files
- Set configuration (e.g., `config/extract/jackal_ahg_courtyard.yaml`)
- `python scripts/extract_bagfile.py --config=<config_path>`