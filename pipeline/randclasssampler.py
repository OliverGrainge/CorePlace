import numpy as np
import pandas as pd
from typing import Union

from .base import CorePlaceStep


class RandomClassSampler(CorePlaceStep):
    def __init__(
        self,
        num_classes: Union[int, None] = None,
        num_instances_per_class: Union[int, None] = None,
        num_instances: Union[int, None] = None,
        min_instances_per_class: Union[int, None] = None,
    ):
        self.num_classes = num_classes
        self.num_instances_per_class = num_instances_per_class
        self.num_instances = num_instances
        self.min_instances_per_class = min_instances_per_class

    def _filter_num_classes(self, dataconfig: pd.DataFrame, num_classes: int):
        unique_classes = dataconfig["class_id"].unique()
        if len(unique_classes) < num_classes:
            raise ValueError(
                f"Not enough classes to sample. Only {len(unique_classes)} classes found."
            )
        sampled_classes = np.random.choice(
            unique_classes, size=num_classes, replace=False
        )
        filtered = dataconfig[dataconfig["class_id"].isin(sampled_classes)]
        return filtered

    def _filter_num_instances_per_class(self, dataconfig: pd.DataFrame, num_instances_per_class: int):
        filtered = dataconfig.groupby("class_id", group_keys=False).sample(
            n=num_instances_per_class, replace=False
        )
        return filtered
        

    def _filter_num_instances(self, dataconfig: pd.DataFrame, num_instances: int):
        if len(dataconfig) <= num_instances:
            return dataconfig
        filtered = dataconfig.sample(n=num_instances, replace=False)
        return filtered

    def _filter_min_instances_per_class(self, dataconfig: pd.DataFrame, min_instances_per_class: int):
        filtered = dataconfig.groupby("class_id", group_keys=False).filter(
            lambda x: len(x) >= min_instances_per_class
        )
        return filtered

    def run(self, pipe_state: dict) -> dict:
        dataconfig = pipe_state["dataconfig"]
        
        if self.num_classes is not None:
            dataconfig = self._filter_num_classes(dataconfig, self.num_classes)
        
        if self.num_instances_per_class is not None:
            dataconfig = self._filter_num_instances_per_class(
                dataconfig, self.num_instances_per_class
            )
        
        if self.num_instances is not None:
            assert self.num_instances_per_class is None, (
                "num_instances and num_instances_per_class cannot be set at the same time"
            )
            dataconfig = self._filter_num_instances(dataconfig, self.num_instances)
        
        if self.min_instances_per_class is not None:
            dataconfig = self._filter_min_instances_per_class(
                dataconfig, self.min_instances_per_class
            )

        pipe_state["dataconfig"] = dataconfig
        return pipe_state

    def __repr__(self) -> str:
        return (
            f"RandomClassSample(num_classes={self.num_classes}, "
            f"num_instances_per_class={self.num_instances_per_class}, "
            f"num_instances={self.num_instances}, "
            f"min_instances_per_class={self.min_instances_per_class})"
        )