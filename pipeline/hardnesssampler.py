from pipeline.base import CorePlaceStep
from typing import Union, Tuple
import pandas as pd 


class HardnessSampler(CorePlaceStep):
    def __init__(
        self,
        hardest_first: bool = True, 
        num_classes: Union[int, None] = None,
        num_instances_per_class: Union[int, None] = None,
        num_instances: Union[int, None] = None,
        min_instances_per_class: Union[int, None] = None,
        percentile: Union[Tuple[int, int], None] = None,
    ):
        self.num_classes = num_classes
        self.num_instances_per_class = num_instances_per_class
        self.num_instances = num_instances
        self.min_instances_per_class = min_instances_per_class
        self.percentile = percentile
        self.hardest_first = hardest_first

    def _filter_percentile(self, dataconfig: pd.DataFrame, percentile: Tuple[int, int]):
        lower_percentile, upper_percentile = percentile
        
        lower_threshold = dataconfig["hardness"].quantile(lower_percentile / 100)
        upper_threshold = dataconfig["hardness"].quantile(upper_percentile / 100)
        
        filtered = dataconfig[
            (dataconfig["hardness"] >= lower_threshold) & 
            (dataconfig["hardness"] <= upper_threshold)
        ]
        
        return filtered

    def _filter_num_classes(self, dataconfig: pd.DataFrame, num_classes: int, hardest_first: bool = True): 
        class_hardness = dataconfig.groupby("class_id")["hardness"].sum()
        class_hardness = class_hardness.sort_values(ascending=not hardest_first)
        top_classes = class_hardness.head(num_classes).index
        filtered = dataconfig[dataconfig["class_id"].isin(top_classes)]
        return filtered

    def _filter_num_instances_per_class(self, dataconfig: pd.DataFrame, num_instances_per_class: int, hardest_first: bool = True): 
        dataconfig = dataconfig.sort_values(by="hardness", ascending=not hardest_first)
        filtered = dataconfig.groupby("class_id", group_keys=False).head(num_instances_per_class)
        return filtered

    def _filter_num_instances(self, dataconfig: pd.DataFrame, num_instances: int, hardest_first: bool = True): 
        dataconfig = dataconfig.sort_values(by="hardness", ascending=not hardest_first)
        filtered = dataconfig.head(num_instances)
        return filtered

    def _filter_min_instances_per_class(self, dataconfig: pd.DataFrame, min_instances_per_class: int, hardest_first: bool = True): 
        filtered = dataconfig.groupby("class_id", group_keys=False).filter(lambda x: len(x) >= min_instances_per_class)
        return filtered

    def run(self, pipe_state: dict) -> dict:
        dataconfig = pipe_state["dataconfig"]
        if self.percentile is not None: 
            dataconfig = self._filter_percentile(dataconfig, self.percentile)
        if self.num_classes is not None: 
            dataconfig = self._filter_num_classes(dataconfig, self.num_classes, self.hardest_first)
        if self.num_instances_per_class is not None: 
            dataconfig = self._filter_num_instances_per_class(dataconfig, self.num_instances_per_class, self.hardest_first)
        if self.num_instances is not None: 
            assert self.num_instances_per_class is None, "num_instances and num_instances_per_class cannot be set at the same time"
            dataconfig = self._filter_num_instances(dataconfig, self.num_instances, self.hardest_first)
        if self.min_instances_per_class is not None: 
            dataconfig = self._filter_min_instances_per_class(dataconfig, self.min_instances_per_class, self.hardest_first)

        pipe_state["dataconfig"] = dataconfig
        return pipe_state

    def __repr__(self) -> str:
        return f"HardnessSampler(num_classes={self.num_classes}, num_instances_per_class={self.num_instances_per_class}, num_instances={self.num_instances}, percentile={self.percentile})"
