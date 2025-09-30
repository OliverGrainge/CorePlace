import argparse
import os
import pickle
from glob import glob

import numpy as np
from omegaconf import OmegaConf
from sklearn.neighbors import NearestNeighbors

from utils import load_config


def read_images_paths(dataset_folder, get_abs_path=False):
    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"Folder {dataset_folder} does not exist")

    file_with_paths = dataset_folder + "_images_paths.txt"
    if os.path.exists(file_with_paths):
        with open(file_with_paths, "r") as file:
            images_paths = file.read().splitlines()
        images_paths = [dataset_folder + "/" + path for path in images_paths]
        # Sanity check that paths within the file exist
        if not os.path.exists(images_paths[0]):
            raise FileNotFoundError(
                f"Image with path {images_paths[0]} "
                f"does not exist within {dataset_folder}. It is likely "
                f"that the content of {file_with_paths} is wrong."
            )
    else:
        images_paths = sorted(glob(f"{dataset_folder}/**/*.jpg", recursive=True))
        if len(images_paths) == 0:
            raise FileNotFoundError(
                f"Directory {dataset_folder} does not contain any JPEG images"
            )

    if not get_abs_path:  # Remove dataset_folder from the path
        images_paths = [p[len(dataset_folder) + 1 :] for p in images_paths]
    return images_paths


def create_groundtruth_pickle(dataset_folder, positive_dist_threshold, output_path):
    """
    Create a pickle file containing query paths, database paths, and ground truth.

    Parameters
    ----------
    dataset_folder : str
        Path to the dataset root directory
    query_folder : str
        Relative path to queries folder from dataset_folder
    database_folder : str
        Relative path to database folder from dataset_folder
    positive_dist_threshold : float
        Distance threshold in meters for positive matches
    output_path : str
        Path where the pickle file will be saved
    """

    # Construct full paths

    database_path = os.path.join(dataset_folder, "images", "test", "database")
    queries_path = os.path.join(dataset_folder, "images", "test", "queries")

    # Read image paths
    database_paths = read_images_paths(database_path, get_abs_path=True)
    queries_paths = read_images_paths(queries_path, get_abs_path=True)

    print(f"  Found {len(database_paths)} database images")
    print(f"  Found {len(queries_paths)} query images")

    # Extract UTM coordinates from paths
    # Format: path/to/file/@utm_easting@utm_northing@...@.jpg
    database_utms = np.array(
        [(path.split("@")[1], path.split("@")[2]) for path in database_paths]
    ).astype(float)

    queries_utms = np.array(
        [(path.split("@")[1], path.split("@")[2]) for path in queries_paths]
    ).astype(float)

    # Compute ground truth using NearestNeighbors
    print(f"  Computing ground truth with threshold: {positive_dist_threshold}m")
    knn = NearestNeighbors(n_jobs=-1)
    knn.fit(database_utms)
    positives_per_query = knn.radius_neighbors(
        queries_utms, radius=positive_dist_threshold, return_distance=False
    )

    # Convert paths to relative paths from dataset_folder
    database_paths_relative = [
        os.path.relpath(path, dataset_folder) for path in database_paths
    ]
    queries_paths_relative = [
        os.path.relpath(path, dataset_folder) for path in queries_paths
    ]

    # Create dictionary
    groundtruth_dict = {
        "dataset_folder": dataset_folder,
        "query": queries_paths_relative,
        "database": database_paths_relative,
        "groundtruth": positives_per_query,
    }

    # Save to pickle
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(groundtruth_dict, f)

    print(f"  Saved ground truth to: {output_path}")

    # Print some statistics
    num_positives = [len(pos) for pos in positives_per_query]
    print(
        f"  Queries with at least 1 positive: {sum(1 for n in num_positives if n > 0)}"
    )
    print(f"  Average positives per query: {np.mean(num_positives):.2f}")
    print(f"  Max positives for a query: {max(num_positives)}")
    print(f"  Min positives for a query: {min(num_positives)}")


def main():
    config = load_config("config.yaml")
    for ds_name, ds_path in config.val_datasets.items():
        print(f"\n\n Processing {ds_name}...\n")
        create_groundtruth_pickle(
            dataset_folder=ds_path,
            positive_dist_threshold=25,
            output_path="dataloaders/val/valconfigs/" + ds_name + ".pkl",
        )


if __name__ == "__main__":
    main()
