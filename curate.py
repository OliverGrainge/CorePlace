import sys

import pandas as pd

from pipeline import CorePlacePipeline
from utils import load_config


def save_dataconfig(dataconfig: pd.DataFrame, name: str):
    pd.to_pickle(dataconfig, f"registry/datasets/{name}.pkl")
    print_summary(dataconfig, name)


def print_summary(dataconfig: pd.DataFrame, name: str):
    num_classes = dataconfig["class_id"].nunique()
    num_instances = len(dataconfig)
    avg_instances = dataconfig["class_id"].value_counts().mean()
    min_instances = dataconfig["class_id"].value_counts().min()

    print("\n" + "=" * 60)
    print("  DATASET SUMMARY".center(60))
    print("=" * 60)
    print(f"\n  Dataset: {name}")
    print(f"\n  Classes:              {num_classes:,}")
    print(f"  Total Instances:      {num_instances:,}")
    print(f"  Avg per Class:        {avg_instances:.1f}")
    print(f"  Min per Class:        {min_instances:,}")
    print(f"\n  Saved to: registry/datasets/{name}.pkl")
    print("\n" + "=" * 60 + "\n")


def main():
    if len(sys.argv) < 2:
        print("\n❌ Error: Missing configuration file")
        print("Usage: python script.py <config.yaml>\n")
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_config(config_path)
    name = config_path.split("/")[-1].split(".")[0]

    print("\n" + "=" * 60)
    print("  DATA PIPELINE".center(60))
    print("=" * 60)
    print(f"\n  Config: {name}")
    print(f"\n  Starting pipeline...\n")

    pipeline = CorePlacePipeline.from_config(config)
    pipe_state = pipeline.run()
    dataconfig = pipe_state["dataconfig"]
    save_dataconfig(dataconfig, name)


if __name__ == "__main__":
    main()
