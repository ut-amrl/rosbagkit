import argparse
import open3d


def get_parser():
    parser = argparse.ArgumentParser(description="Downsample point cloud")
    parser.add_argument(
        "--input",
        type=str,
        default="/home/dongmyeong/Projects/AMRL/CODa/correction/map_0_4.pcd",
        help="input point cloud",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/dongmyeong/Projects/AMRL/CODa/correction/downsampled_map_0_4.pcd",
        help="output point cloud",
    )
    parser.add_argument("--voxel_size", type=float, default=0.05, help="voxel size")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    pcd = open3d.io.read_point_cloud(args.input)
    pcd_downsampled = pcd.voxel_down_sample(voxel_size=args.voxel_size)
    open3d.io.write_point_cloud(args.output, pcd_downsampled)
    print(f"Downsampled point cloud saved to {args.output}")
