import argparse
import os
import pickle
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

CITIES = [
    "Bangkok", "Barcelona", "Boston", "Brussels", "BuenosAires", "Chicago",
    "Lisbon", "London", "LosAngeles", "Madrid", "Melbourne", "MexicoCity",
    "Miami", "Minneapolis", "OSL", "Osaka", "PRG", "PRS", "Phoenix",
    "Rome", "TRT", "WashingtonDC",
]


def get_img_name(row):
    """Generate image filename from row data."""
    city = row["city_id"]
    pl_id = int(row["place_id"]) % 10**5
    pl_id = str(pl_id).zfill(7)
    panoid = row["panoid"]
    year = str(row["year"]).zfill(4)
    month = str(row["month"]).zfill(2)
    northdeg = str(row["northdeg"]).zfill(3)
    lat, lon = str(row["lat"]), str(row["lon"])
    return f"{city}_{pl_id}_{year}_{month}_{northdeg}_{lat}_{lon}_{panoid}.jpg"


def load_single_city_dataframe(gsv_cities_path: str, city: str):
    """
    Load CSV file for a single city.
    
    Args:
        gsv_cities_path: Base path to the GSV cities data directory
        city: City name to load data for
    
    Returns:
        Dataframe with city data
    """
    csv_path = os.path.join(gsv_cities_path, "Dataframes", f"{city}.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    city_dataconfig = pd.read_csv(csv_path)
    
    # Generate filename for each image
    city_dataconfig["filename"] = city_dataconfig.apply(get_img_name, axis=1)
    
    # Build relative image paths from dataset root
    city_dataconfig["image_path"] = city_dataconfig.apply(
        lambda row: os.path.join("Images", str(row["city_id"]), row["filename"]),
        axis=1,
    )
    
    return city_dataconfig


def split_query_database(dataframe, random_state=42):
    """
    Split dataframe into query and database sets.
    
    For each place_id, randomly selects exactly ONE image as query,
    and all remaining images go to the database.
    
    Args:
        dataframe: Input dataframe with place_id column
        random_state: Random seed for reproducibility
    
    Returns:
        query_df, database_df: Two dataframes for queries and database
    """
    np.random.seed(random_state)
    
    query_indices = []
    database_indices = []
    
    # Group by place_id and select one query per place
    for place_id, group in dataframe.groupby("place_id"):
        indices = group.index.tolist()
        
        # If only one image for this place, skip it (no positives for retrieval)
        if len(indices) == 1:
            continue
        
        # Randomly select one image as query
        query_idx = np.random.choice(indices)
        query_indices.append(query_idx)
        
        # All other images go to database
        database_indices.extend([idx for idx in indices if idx != query_idx])
    
    query_df = dataframe.loc[query_indices].reset_index(drop=True)
    database_df = dataframe.loc[database_indices].reset_index(drop=True)
    
    return query_df, database_df


def compute_groundtruth_from_place_ids(query_place_ids, database_place_ids):
    """
    Compute ground truth matches based on place_id.
    
    Args:
        query_place_ids: Array of place_ids for queries
        database_place_ids: Array of place_ids for database
    
    Returns:
        List of arrays, where each array contains indices of positive matches
    """
    groundtruth = []
    
    for query_place_id in tqdm(query_place_ids, desc="Computing groundtruth"):
        # Find all database images with the same place_id
        positive_indices = np.where(database_place_ids == query_place_id)[0]
        groundtruth.append(positive_indices)
    
    return groundtruth


def create_groundtruth_pickle(gsv_cities_path, city, output_dir):
    """
    Create a pickle file containing query paths, database paths, and ground truth
    for a single city in GSV Cities dataset.
    
    For each place (class), one random image is selected as a query and all
    remaining images become the database.
    
    Args:
        gsv_cities_path: Path to GSV Cities dataset root
        city: City name to process
        output_dir: Directory where the pickle file will be saved
    """
    print(f"\nProcessing city: {city}")
    
    # Load dataframe for this city
    dataframe = load_single_city_dataframe(gsv_cities_path, city)
    print(f"  Loaded {len(dataframe)} total images")
    print(f"  Found {dataframe['place_id'].nunique()} unique places")
    
    # Split into query and database (1 query per place)
    query_df, database_df = split_query_database(dataframe)
    
    print(f"  Split into {len(query_df)} queries (1 per place) and {len(database_df)} database images")
    
    # Extract paths and place_ids
    query_paths = query_df["image_path"].tolist()
    database_paths = database_df["image_path"].tolist()
    
    query_place_ids = query_df["place_id"].values
    database_place_ids = database_df["place_id"].values
    
    # Compute ground truth
    print("  Computing ground truth based on place_id matching...")
    groundtruth = compute_groundtruth_from_place_ids(query_place_ids, database_place_ids)
    
    # Create dictionary
    groundtruth_dict = {
        "dataset_folder": gsv_cities_path,
        "query": query_paths,
        "database": database_paths,
        "groundtruth": groundtruth,
    }
    
    # Create output path with city name
    output_path = os.path.join(output_dir, f"GSVCities-{city}.pkl")
    
    # Save to pickle
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(groundtruth_dict, f)
    
    print(f"  Saved ground truth to: {output_path}")
    
    # Print statistics
    num_positives = [len(pos) for pos in groundtruth]
    print(f"  Queries with at least 1 positive: {sum(1 for n in num_positives if n > 0)}")
    print(f"  Average positives per query: {np.mean(num_positives):.2f}")
    print(f"  Max positives for a query: {max(num_positives)}")
    print(f"  Min positives for a query: {min(num_positives)}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Create ground truth pickle files for GSV Cities dataset (one per city)"
    )
    parser.add_argument(
        "--gsv_cities_path",
        type=str,
        required=True,
        help="Path to GSV Cities dataset root directory"
    )
    parser.add_argument(
        "--cities",
        type=str,
        nargs="+",
        default=CITIES,
        help="List of cities to process (default: all cities)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataloaders/val/valconfigs",
        help="Output directory for pickle files (default: dataloaders/val/valconfigs)"
    )
    
    args = parser.parse_args()
    
    print(f"Processing {len(args.cities)} cities...")
    print(f"Output directory: {args.output_dir}")
    print(f"Query selection: 1 random image per place")
    
    created_files = []
    failed_cities = []
    
    for city in args.cities:
        try:
            output_path = create_groundtruth_pickle(
                gsv_cities_path=args.gsv_cities_path,
                city=city,
                output_dir=args.output_dir
            )
            created_files.append(output_path)
        except Exception as e:
            print(f"  Error processing {city}: {e}")
            failed_cities.append(city)
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Successfully created {len(created_files)} pickle files")
    if failed_cities:
        print(f"  Failed cities: {', '.join(failed_cities)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()