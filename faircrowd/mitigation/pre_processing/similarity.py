import numpy as np
import pandas as pd
from numpy import typing as npt
from typing import Union
from ...core.algorithms import PreProcessingAlgorithm
from ...core.fair_crowd_dataset import FairCrowdDataset


class SimilarityPreProcessing(PreProcessingAlgorithm):
    """
    Mitigates the unfairness using the similarity-based pre-processing algorithm
    """

    distance_matrix: npt.NDArray
    k: Union[int, float, None]
    dist: Union[int, float, None]

    def __init__(
        self,
        distance_matrix: npt.NDArray,
        k: Union[int, float, None] = None,
        dist: Union[int, float, None] = None,
    ) -> None:
        if k is None and dist is None or k is not None and dist is not None:
            raise ValueError("Exactly one between k and dist must be defined")
        self.k = k
        self.dist = dist
        self.distance_matrix = distance_matrix

    def run(self, df: FairCrowdDataset) -> FairCrowdDataset:
        answers = df.answers.values
        new_answers = np.empty(answers.shape)
        new_answers.fill(np.nan)
        distance_matrix = np.ma.array(self.distance_matrix, mask=False)
        for worker in range(answers.shape[1]):
            worker_subset = np.argwhere(np.isfinite(answers[:, worker])).flatten()
            for sample in worker_subset:
                distances = distance_matrix[sample, worker_subset]
                sample_idx = np.argwhere(worker_subset == sample)[0][0]
                distances.mask[sample_idx] = True
                if self.k is None:
                    neighbors = np.append(
                        np.argwhere(distances <= self.dist)[:, 0], sample_idx
                    )
                else:
                    dist_idxs = distances.argsort()

                    if isinstance(self.k, float):
                        n = int(self.k * len(dist_idxs) / 2)
                    else:
                        n = self.k
                    sensit_idxs = dist_idxs[df["s"].values[dist_idxs] == 1][:n]
                    others_idxs = dist_idxs[df["s"].values[dist_idxs] == 0][:n]
                    neighbors = np.concatenate([sensit_idxs, others_idxs, [sample_idx]])

                distances.mask[sample_idx] = False
                neighbors = worker_subset[neighbors]
                new_answers[sample, worker] = np.mean(answers[neighbors, worker])

        return FairCrowdDataset(
            pd.DataFrame(
                new_answers, columns=df.answers.columns, index=df.answers.index
            ),
            df.s,
            df.x,
            df.y,
        )
