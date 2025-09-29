from typing import List, Optional

from pipeline.base import CorePlaceStep
from pipeline.embeddings import Embeddings
from pipeline.hardness import ComputeHardness
from pipeline.randclasssampler import RandomClassSampler
from pipeline.read_gsvcities import ReadGsvCities
from pipeline.hardnesssampler import HardnessSampler

def load_step(step_name: str, **kwargs) -> CorePlaceStep:
    step_name = step_name.lower()
    print(step_name)
    if step_name == "readgsvcities":
        return ReadGsvCities(**kwargs)
    elif step_name == "embeddings":
        return Embeddings(**kwargs)
    elif step_name == "randomclasssampler":
        return RandomClassSampler(**kwargs)
    elif step_name == "computehardness":
        return ComputeHardness(**kwargs)
    elif step_name == "hardnesssampler":
        return HardnessSampler(**kwargs)
    else:
        raise ValueError(f"Step {step_name} not found")

class CorePlacePipeline:
    def __init__(self, steps: Optional[List[CorePlaceStep]] = None):
        self.steps = steps or []
        self.pipe_state = {}

    def run(self) -> dict:
        for step in self.steps:
            self.pipe_state = step.run(self.pipe_state)
        return self.pipe_state

    @classmethod
    def from_config(cls, config: dict) -> "CorePlacePipeline":
        steps = []
        for key, value in config.items():
            steps.append(load_step(key, **value))
        return cls(steps=steps)

    def __repr__(self) -> str:
        lines = ["CorePlacePipeline("]
        for i, step in enumerate(self.steps):
            lines.append(f"  {i+1}. {repr(step)}")
        lines.append(")")
        return "\n".join(lines)