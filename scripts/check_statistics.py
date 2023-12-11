"""
Author: Dongmyeong Lee (domlee[at]utexas.edu)
Date:   Dec 1, 2023
Description: This file is for checking the statistics of the dataset.
"""
import os
import pathlib
import argparse
from tqdm import tqdm
import pandas as pd


def get_parser():
    parser = argparse.ArgumentParser(description="Check the statistics of the dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="/home/dongmyeong/Projects/AMRL/CODa",
        help="Path to the dataset.",
    )
    return parser


def main(args):
    dataset = pathlib.Path(args.dataset)

    bbox_2d_data = []  # {"class": str, "instance_id": int, "trajectory": int}
    bbox_2d_dir = dataset / "2d_bbox"
    count = 0
    for cam in ["cam0", "cam1"]:
        cam_dir = bbox_2d_dir / cam
        for seq_dir in tqdm(cam_dir.iterdir()):
            for file in seq_dir.glob("*.txt"):
                count += 1
                with open(file, "r") as f:
                    for line in f:
                        line = line.strip().split()
                        bbox_2d_data.append(
                            {
                                "class": line[0],
                                "instance_id": int(line[1]),
                                "trajectory": int(seq_dir.name),
                            }
                        )
    print(f"Total number of 2D bbox: {count}")

    df = pd.DataFrame(bbox_2d_data)
    stats = (
        df.groupby(["class", "instance_id", "trajectory"])
        .size()
        .reset_index(name="counts")
    )

    # Count unique instances for each class
    instance_counts = df.groupby("class")["instance_id"].nunique()
    print(instance_counts)

    # Calculate the average count per instance_id
    sum_per_class_instance = stats.groupby(["class", "instance_id"])["counts"].sum()
    average_count_per_class = sum_per_class_instance.groupby("class").mean()
    print(f"Average count per instance_id: {average_count_per_class}")

    # Define trajectories for each environmental condition
    sunny_trajectories = [3, 6, 7, 11, 17, 18, 19, 20, 22]
    cloudy_trajectories = [0, 1, 8, 9, 10, 12, 14, 21]
    rainy_trajectories = [13, 15, 16]
    dark_trajectories = [2, 4, 5]
    # Filter stats for each environmental condition
    sunny_stats = stats[stats["trajectory"].isin(sunny_trajectories)]
    cloudy_stats = stats[stats["trajectory"].isin(cloudy_trajectories)]
    rainy_stats = stats[stats["trajectory"].isin(rainy_trajectories)]
    dark_stats = stats[stats["trajectory"].isin(dark_trajectories)]
    # Calculate the average count for each environmental condition
    average_count_sunny = (
        sunny_stats.groupby(["class", "instance_id"])["counts"]
        .sum()
        .groupby("class")
        .mean()
    )
    average_count_cloudy = (
        cloudy_stats.groupby(["class", "instance_id"])["counts"]
        .sum()
        .groupby("class")
        .mean()
    )
    average_count_rainy = (
        rainy_stats.groupby(["class", "instance_id"])["counts"]
        .sum()
        .groupby("class")
        .mean()
    )
    average_count_dark = (
        dark_stats.groupby(["class", "instance_id"])["counts"]
        .sum()
        .groupby("class")
        .mean()
    )

    # Print the average counts
    print(f"Average count in sunny conditions: {average_count_sunny}")
    print(f"Average count in cloudy conditions: {average_count_cloudy}")
    print(f"Average count in rainy conditions: {average_count_rainy}")
    print(f"Average count in dark conditions: {average_count_dark}")


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
