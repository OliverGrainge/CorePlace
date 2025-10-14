from collections import defaultdict
from typing import List

import faiss
import numpy as np
from pytorch_lightning import LightningModule
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import (CosineSimilarity,
                                               DotProductSimilarity)
from torch.nn import Linear
from torch.optim import AdamW
import torch 

from models.archs import get_arch


def recallatk(
    query_embeddings: np.ndarray,
    database_embeddings: np.ndarray,
    groundtruth: np.ndarray,
    recallatks: List[int],
):

    index = faiss.IndexFlatIP(query_embeddings.shape[1])
    index.add(database_embeddings)
    _, I = index.search(query_embeddings, max(recallatks))
    results = {}
    num_queries = query_embeddings.shape[0]
    for k in recallatks:
        I_k = I[:, :k]  # Top-k predictions
        success = 0
        for i in range(num_queries):
            if len(groundtruth[i]) > 0:
                if np.isin(I_k[i], groundtruth[i]).any():
                    success += 1
            else:
                num_queries -= 1
        results[k] = (success / num_queries) * 100
    return results



def compute_mAPatk(
    query_embeddings: np.ndarray,
    database_embeddings: np.ndarray,
    groundtruth: np.ndarray,
    mAPatks: List[int] = [50, 100, 200],
):
    """
    Compute mAP at multiple k values
    
    Args:
        query_embeddings: [num_queries, embedding_dim]
        database_embeddings: [num_db, embedding_dim]
        groundtruth: list/array where groundtruth[i] contains indices of correct matches for query i
        ks: list of k values to compute mAP at
    
    Returns:
        dict mapping k -> mAP@k score
    """
    index = faiss.IndexFlatIP(query_embeddings.shape[1])
    index.add(database_embeddings)
    
    # Retrieve up to max k
    max_k = max(mAPatks)
    _, I = index.search(query_embeddings, max_k)
    
    results = {}
    
    for k in mAPatks:
        average_precisions = []
        
        for i in range(query_embeddings.shape[0]):
            # Skip queries with no ground truth
            if len(groundtruth[i]) == 0:
                continue
            
            # Only look at top-k for this specific k value
            retrieved = I[i, :k]
            
            # Check which retrieved items are correct
            correct_matches = np.isin(retrieved, groundtruth[i])
            
            # Total number of relevant items for this query
            num_relevant = len(groundtruth[i])
            
            # Compute Average Precision for this query
            if not correct_matches.any():
                ap = 0.0
            else:
                # Positions where we have correct matches (0-indexed)
                correct_positions = np.where(correct_matches)[0]
                
                # Precision at each correct position
                precisions = []
                for idx, pos in enumerate(correct_positions, start=1):
                    # Precision@position = (number correct so far) / (position in ranking)
                    precision_at_k = idx / (pos + 1)
                    precisions.append(precision_at_k)
                
                # IMPORTANT: Divide by TOTAL relevant, not just found
                ap = np.sum(precisions) / num_relevant
            
            average_precisions.append(ap)
        
        # Mean of all Average Precisions for this k
        results[k] = np.mean(average_precisions) if len(average_precisions) > 0 else 0.0
    
    return results



def compute_MRR(
    query_embeddings: np.ndarray,
    database_embeddings: np.ndarray,
    groundtruth: np.ndarray,
    k: int = 100,
):
    """
    Compute Mean Reciprocal Rank
    MRR = average of 1/(rank of first correct match)
    
    Args:
        query_embeddings: [num_queries, embedding_dim]
        database_embeddings: [num_db, embedding_dim]
        groundtruth: array/list where groundtruth[i] contains indices of relevant items for query i
        k: number of top retrievals to consider
    
    Returns:
        MRR score (float)
    """
    index = faiss.IndexFlatIP(query_embeddings.shape[1])
    index.add(database_embeddings)
    _, I = index.search(query_embeddings, k)
    
    reciprocal_ranks = []
    num_queries = query_embeddings.shape[0]
    
    for i in range(num_queries):
        # Skip queries with no ground truth
        if len(groundtruth[i]) == 0:
            continue
        
        # Get retrieved indices for this query
        retrieved = I[i]
        
        # Check which retrieved items are correct
        correct_matches = np.isin(retrieved, groundtruth[i])
        
        if correct_matches.any():
            # Find rank of first correct match (1-indexed)
            first_correct_rank = np.where(correct_matches)[0][0] + 1
            reciprocal_ranks.append(1.0 / first_correct_rank)
        else:
            # No correct match found in top-k
            reciprocal_ranks.append(0.0)
    
    # Mean of all reciprocal ranks
    mrr = np.mean(reciprocal_ranks) if len(reciprocal_ranks) > 0 else 0.0
    
    return mrr


class CorePlaceModel(LightningModule):
    def __init__(
        self,
        arch_name: str,
        pretrained: bool = True,
        desc_dim: int = 2048,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        recallatks=[1, 5, 10, 25, 50],
        mAPatks=[100],
    ):
        super().__init__()
        self.arch_name = arch_name
        self.pretrained = pretrained
        self.desc_dim = desc_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.recallatks = recallatks
        self.mAPatks = mAPatks

        self.loss_fn = losses.MultiSimilarityLoss(
            alpha=1.0, beta=50, base=0.0, distance=DotProductSimilarity()
        )
        self.miner_fn = miners.MultiSimilarityMiner(
            epsilon=0.1, distance=CosineSimilarity()
        )

    @classmethod
    def from_config(cls, config: dict):
        arch_name = config.get("arch_name", "resnet50gem")
        pretrained = config.get("pretrained", True)
        desc_dim = config.get("desc_dim", 2048)
        learning_rate = config.get("learning_rate", 0.001)
        weight_decay = config.get("weight_decay", 0.0001)
        recallatks = config.get("recallatks", [1, 5, 10])

        return cls(
            arch_name=arch_name,
            pretrained=pretrained,
            desc_dim=desc_dim,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            recallatks=recallatks,
        )

    def setup(self, stage: str):
        self.arch = get_arch(self.arch_name, self.pretrained, self.desc_dim)

    def forward(self, x):
        return self.arch(x)

    def configure_optimizers(self):
        return AdamW(
            self.arch.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def training_step(self, batch):
        images, labels = batch
        descs = self(images)
        pairs = self.miner_fn(descs, labels)
        loss = self.loss_fn(descs, labels, pairs)
        self.log("train_loss", loss)
        return loss

    def on_fit_start(self):
        datamodule = self.trainer.datamodule
        class_counts = datamodule.dataconfig["class_id"].value_counts()

        # Log as hyperparameters (appears in W&B config)
        hparams = {
            "dataset/num_classes": class_counts.size,
            "dataset/min_per_class": int(class_counts.min()),
            "dataset/max_per_class": int(class_counts.max()),
            "dataset/avg_per_class": float(class_counts.mean()),
            "dataset/median_per_class": float(class_counts.median()),
            "dataset/std_per_class": float(class_counts.std()),
            "dataset/num_images": int(class_counts.sum()),
        }

        # Option 1: Use log_hyperparams (good for config-like values)
        self.logger.log_hyperparams(hparams)

    def on_validation_epoch_start(self):
        val_loaders = self.trainer.val_dataloaders
        # Use a dict to store only the minimal necessary embeddings, and clear any previous state
        self.embeddings = {}
        for idx, loader in enumerate(val_loaders):
            dataset = loader.dataset
            assert len(dataset.query) == len(
                dataset.groundtruth()
            ), "Number of queries and groundtruth must be the same"
            # Preallocate as before, but will delete after use to save memory
            self.embeddings[idx] = np.empty(
                (len(dataset), self.desc_dim), dtype=np.float16
            )

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        images, idxs = batch
        descs = self(images)
        if dataloader_idx is None:
            dataloader_idx = 0
        # Store only the embeddings for this batch in the preallocated array
        self.embeddings[dataloader_idx][idxs.cpu().numpy()] = (
            descs.cpu().detach().numpy().astype(np.float16)
        )
        # Explicitly delete variables to free memory
        del images, idxs, descs

    def on_validation_epoch_end(self):
        val_loaders = self.trainer.val_dataloaders
        for idx, loader in enumerate(val_loaders):
            dataset = loader.dataset
            # Extract only the slices needed for evaluation
            query_embeddings = self.embeddings[idx][: dataset.n_query]
            database_embeddings = self.embeddings[idx][dataset.n_query :]
            groundtruth = dataset.groundtruth()
            recalls = recallatk(
                query_embeddings, database_embeddings, groundtruth, self.recallatks
            )
            mAPs = compute_mAPatk(
                query_embeddings, database_embeddings, groundtruth, self.mAPatks
            )

            mrr = compute_MRR(
                query_embeddings, database_embeddings, groundtruth, self.mAPatks[-1]
            )

            for k in self.recallatks:
                self.log(f"{str(dataset)}/recallat{k}", recalls[k])
            
            for k in self.mAPatks:
                self.log(f"{str(dataset)}/mAPat{k}", mAPs[k])

            self.log(f"{str(dataset)}/MRR", mrr)

            # Explicitly delete large arrays to free memory
            del self.embeddings[idx]
            del query_embeddings, database_embeddings, groundtruth, recalls, mAPs, mrr

        # Optionally, clear the embeddings dict entirely
        self.embeddings.clear()


