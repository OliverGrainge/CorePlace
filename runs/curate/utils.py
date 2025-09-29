from typing import Union
import pandas as pd


def save_dataconfig(dataconfig: pd.DataFrame, name: str):
    pd.to_pickle(dataconfig, f"registry/datasets/{name}.pkl")
    print("\n\n ==== Dataset Summary: ====\n")
    summary(dataconfig)
    print(f"Saved config to registry/datasets/{name}.pkl")


def summary(dataconfig: pd.DataFrame):
    print(f"Number of classes: {dataconfig['class_id'].nunique()}")
    print(f"Number of instances: {len(dataconfig)}")
    print(
        f"Average number of instances per class: {dataconfig['class_id'].value_counts().mean()}"
    )
    print(
        f"Min number of instances per class: {dataconfig['class_id'].value_counts().min()}"
    )
