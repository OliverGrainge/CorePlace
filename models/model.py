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
        results[k] = success / num_queries
    return results


class CorePlaceModel(LightningModule):
    def __init__(
        self,
        arch_name: str,
        pretrained: bool = True,
        desc_dim: int = 2048,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        recallatks=[1, 5, 10],
    ):
        super().__init__()
        self.arch_name = arch_name
        self.pretrained = pretrained
        self.desc_dim = desc_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.recallatks = recallatks

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
        self.embeddings = {}
        for idx, loader in enumerate(val_loaders):
            dataset = loader.dataset
            assert len(dataset.query) == len(
                dataset.groundtruth()
            ), "Number of queries and groundtruth must be the same"
            self.embeddings[idx] = np.empty(
                (len(dataset), self.desc_dim), dtype=np.float16
            )

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        images, idxs = batch
        descs = self(images)
        if dataloader_idx is None:
            dataloader_idx = 0
        self.embeddings[dataloader_idx][idxs.cpu().numpy()] = (
            descs.cpu().detach().numpy().astype(np.float16)
        )

    def on_validation_epoch_end(self):
        val_loaders = self.trainer.val_dataloaders
        for idx, loader in enumerate(val_loaders):
            dataset = loader.dataset
            query_embeddings = self.embeddings[idx][: dataset.n_query]
            database_embeddings = self.embeddings[idx][dataset.n_query :]
            groundtruth = dataset.groundtruth()
            recalls = recallatk(
                query_embeddings, database_embeddings, groundtruth, self.recallatks
            )
            for k in self.recallatks:
                self.log(f"{str(dataset)}/recallat{k}", recalls[k])
