import numpy as np
import pandas as pd
from numpy import typing as npt
from typing import Union
from ...core.algorithms import PreProcessingAlgorithm
from ...core.fair_crowd_dataset import FairCrowdDataset


class PreFairTD(PreProcessingAlgorithm):
    """
    Description & reference
    Different from original implementation: Majority Voting is used for truth inference to check if we respect theta-disparity
    Also removes the biggest biases in absolute value
    PreTD requires that the same workers annotate the same samples
    """

    theta: float

    def __init__(self, theta: float = 0.01) -> None:
        self.theta = theta

    # def run(gtruth, bias, answer):
    def run(self, df: FairCrowdDataset) -> FairCrowdDataset:
        # Preference = demographic parity of worker without abs
        preferences = []
        for worker in df.answers.columns:
            preferences.append(
                np.nanmean(df.answers[worker][df["s"] == 0])
                - np.nanmean(df.answers[worker][df["s"] == 1])
            )
        preferences = np.array(preferences)

        # Bias = abs(preference - average of other workers' preference in the same group)
        biases = []
        for i, worker in enumerate(df.answers.columns):
            samples = np.isfinite(df.answers[worker].values)
            other_workers = df.answers.loc[samples].dropna(axis=1).columns
            other_workers = [
                j
                for j in range(df.answers.shape[1])
                if df.answers.columns[i] in other_workers and i != j
            ]
            biases.append(np.abs(preferences[i] - np.mean(preferences[other_workers])))

        biases = sorted(
            zip(range(df.answers.shape[1]), biases),
            key=lambda x: x[1],
            reverse=False,
        )

        new_answers = df.answers.values.copy()
        for worker, _ in biases:
            # Remove answers of worker (keep at least one per sample)
            samples = np.argwhere(np.isfinite(new_answers[:, worker]))[:, 0]
            # At least one of the samples has only the answer of this worker
            if np.min(np.isfinite(new_answers[samples]).sum(axis=1)) == 1:
                continue
            new_answers[:, worker] = np.nan
            # Check if disparity is now respected
            predictions = np.nanmean(new_answers, axis=1)
            predictions = np.where(np.isfinite(predictions), predictions, 0)
            groups = np.unique(df["s"])
            positive_rates = [
                (predictions[df["s"] == group] > 0.5).mean() for group in groups
            ]
            disparity = (
                np.nanmax(positive_rates) - np.nanmin(positive_rates)
                if len(groups) > 1
                else np.nan
            )
            if disparity < self.theta:
                break

        return FairCrowdDataset(
            pd.DataFrame(
                new_answers, columns=df.answers.columns, index=df.answers.index
            ),
            df.s,
            df.x,
            df.y,
        )
