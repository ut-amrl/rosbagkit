import argparse
import pathlib
from natsort import natsorted


def main(args):
    odom_files = natsorted(args.odom_dir.glob("*.odom"))
    pcd_files = natsorted(args.odom_dir.glob("*.pcd"))

    for odom_file, pcd_file in zip(odom_files, pcd_files):
        odom_sec = int(odom_file.stem.split("_")[-2])
        odom_nsec = int(odom_file.stem.split("_")[-1])
        odom_usec = str(odom_nsec // 1000).zfill(6)

        pcd_sec = int(pcd_file.stem.split("_")[-2])
        pcd_nsec = int(pcd_file.stem.split("_")[-1])
        pcd_usec = str(pcd_nsec // 1000).zfill(6)

        odom_new_filename = f"{odom_file.parent}/{odom_sec}_{odom_usec}.odom"
        pcd_new_filename = f"{pcd_file.parent}/{pcd_sec}_{pcd_usec}.pcd"

        odom_file.rename(odom_new_filename)
        pcd_file.rename(pcd_new_filename)


def get_args():
    parser = argparse.ArgumentParser(description="Sync filename")
    parser.add_argument("--odom_dir", type=str, help="Path to the odometry directory")
    args = parser.parse_args()

    args.odom_dir = pathlib.Path(args.odom_dir)
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
