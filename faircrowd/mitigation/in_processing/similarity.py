import numpy as np
from numpy import typing as npt
from typing import Union
from ...core.algorithms import TDAlgorithm, TDOutput
from ...core.fair_crowd_dataset import FairCrowdDataset


class SimilarityInProcessing(TDAlgorithm):
    """
    Mitigates the unfairness using the similarity-based in-processing algorithm
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

    def run(self, df: FairCrowdDataset) -> TDOutput:
        answers = df.answers.values
        probabilities = np.zeros(df.answers.shape[0])
        distance_matrix = np.ma.array(self.distance_matrix, mask=False)
        for sample in range(answers.shape[0]):
            distances = distance_matrix[sample]
            distances.mask[sample] = True
            if self.k is None:
                neighbors = np.append(np.argwhere(distances <= self.dist)[:, 0], sample)
            else:
                dist_idxs = distances.argsort()

                if isinstance(self.k, float):
                    n = int(self.k * len(dist_idxs) / 2)
                else:
                    n = self.k
                sensit_idxs = dist_idxs[df["s"].values[dist_idxs] == 1][:n]
                others_idxs = dist_idxs[df["s"].values[dist_idxs] == 0][:n]
                neighbors = np.concatenate([sensit_idxs, others_idxs, [sample]])

            distances.mask[sample] = False
            probabilities[sample] = np.nanmean(answers[neighbors])

        return TDOutput(probabilities=probabilities)
