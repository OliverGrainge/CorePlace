from typing import List, Optional

from pipeline.base import CorePlaceStep
from pipeline.embeddings import Embeddings
from pipeline.hardness import MultiSimilarityHardness
from pipeline.sampler import Sampler
from pipeline.read_gsvcities import ReadGsvCities
from pipeline.entropy import Entropy
from pipeline.labelmixer import LabelMixer
from pipeline.confusionmixer import ConfusionMixer
from pipeline.blurdetection import BlurDetection
from pipeline.edgedensity import EdgeDensity
from pipeline.colourvariance import ColorVariance


def load_step(step_name: str, **kwargs) -> CorePlaceStep:
    step_name = step_name.lower()
    if step_name == "readgsvcities":
        return ReadGsvCities(**kwargs)
    elif step_name == "embeddings":
        return Embeddings(**kwargs)
    elif step_name == "multisimilarityhardness":
        return MultiSimilarityHardness(**kwargs)
    elif step_name == "sampler":
        return Sampler(**kwargs)
    elif step_name == "entropy":
        return Entropy(**kwargs)
    elif step_name == "labelmixer":
        return LabelMixer(**kwargs)
    elif step_name == "confusionmixer":
        return ConfusionMixer(**kwargs)
    elif step_name == "blurdetection":
        return BlurDetection(**kwargs)
    elif step_name == "edgedensity":
        return EdgeDensity(**kwargs)
    elif step_name == "colourvariance":
        return ColorVariance(**kwargs)
    else:
        raise ValueError(f"Step {step_name} not found")


class CorePlacePipeline:
    def __init__(self, steps: Optional[List[CorePlaceStep]] = None):
        self.steps = steps or []
        self.pipe_state = {}

    def run(self) -> dict:
        self.pipe_state["plots"] = []
        for step in self.steps:
            self.pipe_state = step.run(self.pipe_state)
        return self.pipe_state

    @classmethod
    def from_config(cls, config: dict) -> "CorePlacePipeline":
        steps = []
        
        # Handle both old dict format and new list format
        if 'pipeline' in config:
            # New list-based format
            for step_config in config['pipeline']:
                step_name = step_config['step']
                step_params = step_config.get('params', {})
                steps.append(load_step(step_name, **step_params))
        else:
            # Old dict-based format (backward compatible)
            for step_name, step_params in config.items():
                steps.append(load_step(step_name, **(step_params or {})))
        
        return cls(steps=steps)

    def __repr__(self) -> str:
        lines = ["CorePlacePipeline("]
        for i, step in enumerate(self.steps):
            lines.append(f"  {i+1}. {repr(step)}")
        lines.append(")")
        return "\n".join(lines)
