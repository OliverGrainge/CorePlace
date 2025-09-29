import os
from typing import List

import pandas as pd
from tqdm import tqdm

from pipeline.base import CorePlaceStep

CITIES = [
    "Bangkok",
    "Barcelona",
    "Boston",
    "Brussels",
    "BuenosAires",
    "Chicago",
    "Lisbon",
    "London",
    "LosAngeles",
    "Madrid",
    "Melbourne",
    "MexicoCity",
    "Miami",
    "Minneapolis",
    "OSL",
    "Osaka",
    "PRG",
    "PRS",
    "Phoenix",
    "Rome",
    "TRT",
    "WashingtonDC",
]


def get_img_name(row):
    city = row["city_id"]
    pl_id = int(row["place_id"]) % 10**5
    pl_id = str(pl_id).zfill(7)
    panoid = row["panoid"]
    year = str(row["year"]).zfill(4)
    month = str(row["month"]).zfill(2)
    northdeg = str(row["northdeg"]).zfill(3)
    lat, lon = str(row["lat"]), str(row["lon"])
    return f"{city}_{pl_id}_{year}_{month}_{northdeg}_{lat}_{lon}_{panoid}.jpg"


def get_dataframes(gsv_cities_path: str, cities: List[str]):
    """
    Load and combine CSV files for multiple cities into a single dataframe.

    Args:
        gsv_cities_path: Base path to the GSV cities data directory
        cities: List of city names to load data for

    Returns:
        Combined dataframe with all city data
    """
    # Construct CSV file paths
    csv_paths = [
        os.path.join(gsv_cities_path, "Dataframes", f"{city}.csv") for city in cities
    ]

    # Validate that CSV files exist
    if not csv_paths:
        raise ValueError(
            f"No CSV files found in {os.path.join(gsv_cities_path, 'Dataframes')}"
        )

    # Read and process all CSV files with detailed progress tracking
    dataconfig_list = []

    for i, csv_path in tqdm(
        enumerate(csv_paths), desc="Reading GSVCities", total=len(csv_paths)
    ):
        # Read CSV file
        city_dataconfig = pd.read_csv(csv_path).sample(frac=1)

        # Add city prefix to place_id to ensure uniqueness across cities
        if i > 0:
            city_dataconfig["place_id"] = city_dataconfig["place_id"] + (i * 10**5)

        # Generate filename for each image (moved inside loop)
        city_dataconfig["filename"] = city_dataconfig.apply(get_img_name, axis=1)

        # Build absolute image paths: <image_dir>/Images/<city>/<filename> (moved inside loop)
        city_dataconfig["image_path"] = city_dataconfig.apply(
            lambda row: os.path.join(
                gsv_cities_path, "Images", str(row["city_id"]), row["filename"]
            ),
            axis=1,
        )

        # Set class_id for downstream processing (moved inside loop)
        city_dataconfig["class_id"] = city_dataconfig["place_id"]
        city_dataconfig["supclass_id"] = 1

        dataconfig_list.append(city_dataconfig)

    # Combine all dataframes
    dataconfig = pd.concat(dataconfig_list, ignore_index=True)

    # Final shuffle of combined data and reset index
    dataconfig = dataconfig.sample(frac=1).reset_index(drop=True)

    return dataconfig


class ReadGsvCities(CorePlaceStep):
    def __init__(self, gsv_cities_path: str, cities: List[str] = CITIES):
        self.gsv_cities_path = gsv_cities_path
        self.cities = cities

    def run(self, pipe_state: dict) -> dict:
        dataconfig = get_dataframes(self.gsv_cities_path, self.cities)
        pipe_state["dataconfig"] = dataconfig
        return pipe_state

    def __repr__(self) -> str:
        return f"ReadGsvCities(gsv_cities_path={self.gsv_cities_path}, cities={self.cities})"


if __name__ == "__main__":
    gsv_cities_path = "/Users/olivergrainge/datasets/gsv-cities"
    step = ReadGsvCities(gsv_cities_path, CITIES)
    pipe_state = step.run({})  # Pass empty dict as initial pipe_state
    print(pipe_state["dataconfig"].head())
