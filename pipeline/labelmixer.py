from pipeline.base import CorePlaceStep
import numpy as np
from tqdm import tqdm 


class LabelMixer(CorePlaceStep):
    """Randomly corrupts labels for a specified percentage of samples in the dataconfig."""
    
    def __init__(self, rate: float = 0.1):
        """
        Args:
            corruption_rate: Percentage of samples (0.0 to 1.0) that will have their labels randomly shuffled.
        """
        self.rate = rate

    def run(self, pipe_state: dict) -> dict:
        dataconfig = pipe_state["dataconfig"]
        
        # Calculate number of samples to corrupt
        n_samples = len(dataconfig)
        n_samples_to_corrupt = int(n_samples * self.rate)
        
        # Randomly select indices of samples to corrupt
        indices_to_corrupt = np.random.choice(
            n_samples, 
            size=n_samples_to_corrupt, 
            replace=False
        )
        
        # Get all unique class IDs for random reassignment
        all_class_ids = dataconfig["class_id"].unique()
        
        # Corrupt labels for selected samples
        for idx in tqdm(indices_to_corrupt, desc="Corrupting labels"):
            dataconfig.loc[dataconfig.index[idx], "class_id"] = np.random.choice(all_class_ids)
        pipe_state["dataconfig"] = dataconfig
        return pipe_state
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(rate={self.rate})"