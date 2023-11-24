import argparse
import open3d


def get_parser():
    parser = argparse.ArgumentParser(description="Downsample point cloud")
    parser.add_argument(
        "--input",
        type=str,
        default="/home/dongmyeong/Projects/AMRL/CODa/correction/map_0_6.pcd",
        help="input point cloud",
    )
    parser.add_argument("--voxel_size", type=float, default=0.05, help="voxel size")
    parser.add_argument(
        "--nb_neighbors",
        type=int,
        default=20,
        help="number of neighbors for statistical outlier removal",
    )
    parser.add_argument(
        "--std_ratio",
        type=float,
        default=2.0,
        help="standard deviation ratio for statistical outlier removal",
    )
    parser.add_argument(
        "--radius", type=float, default=0.05, help="radius for radius outlier removal"
    )
    parser.add_argument(
        "--min_nb_points",
        type=int,
        default=4,
        help="minimum number of points for radius outlier removal",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    ## Read the point cloud
    pcd = open3d.io.read_point_cloud(args.input)

    ## Downsample the point cloud
    pcd_downsampled = pcd.voxel_down_sample(voxel_size=args.voxel_size)
    print(f"Point cloud downsampled with a voxel size of {args.voxel_size}")

    ## Statistical outlier removal
    pcd_downsampled, _ = pcd_downsampled.remove_statistical_outlier(
        nb_neighbors=args.nb_neighbors, std_ratio=args.std_ratio
    )
    print("Done statistical outlier removal")

    ## Radius outlier removal
    pcd_downsampled, _ = pcd_downsampled.remove_radius_outlier(
        nb_points=args.min_nb_points, radius=args.radius
    )

    # Write the processed point cloud
    output = args.input.split(".")[0] + "_downsampled.pcd"
    open3d.io.write_point_cloud(output, pcd_downsampled)
    print(f"Downsampled point cloud saved to {output}")
