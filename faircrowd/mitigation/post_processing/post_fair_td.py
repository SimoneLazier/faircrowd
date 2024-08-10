import math
from typing import Union
import pandas as pd
from ...core.algorithms import PostProcessingAlgorithm, TDOutput
from ...core.fair_crowd_dataset import FairCrowdDataset


class PostFairTD(PostProcessingAlgorithm):
    """
    Description & reference
    """

    theta: float
    random_state: Union[int, None]

    def __init__(
        self,
        theta: float = 0.01,
        random_state: Union[int, None] = None,
    ) -> None:
        self.theta = theta
        self.random_state = random_state

    def run(self, df: FairCrowdDataset, td_output: TDOutput) -> TDOutput:
        # Adapted code from the original paper
        groups = df["s"].unique()
        lengths = []
        negatives = []
        for s in groups:
            lengths.append(len(df["s"][df["s"] == s]))
            negatives.append(len(df["s"][(df["s"] == s) & (td_output.labels == 0)]))
        negative_rate = [
            negative / length for negative, length in zip(negatives, lengths)
        ]
        # Protected group has the max positive rate (min negative rate)
        protected_group = groups[negative_rate.index(min(negative_rate))]

        labels_to_switch = math.ceil(
            (
                lengths[1 - protected_group] * negatives[protected_group]
                - lengths[protected_group] * negatives[1 - protected_group]
                - self.theta * lengths[protected_group] * lengths[1 - protected_group]
            )
            / (lengths[protected_group] + lengths[1 - protected_group])
        )
        labels_to_switch = max(0, labels_to_switch)

        # Select n=labels_to_switch labels randomly for both groups and switch them
        labels_to_switch_indices = (
            df.s[(df["s"] == protected_group) & (td_output.labels == 1)]
            .sample(n=labels_to_switch, random_state=self.random_state)
            .index.tolist()
        ) + (
            df.s[(df["s"] == 1 - protected_group) & (td_output.labels == 0)]
            .sample(n=labels_to_switch, random_state=self.random_state)
            .index.tolist()
        )
        new_labels = pd.DataFrame(
            td_output.labels, index=df.answers.index, columns=["labels"]
        )
        new_labels.loc[labels_to_switch_indices] = (
            1 - new_labels.loc[labels_to_switch_indices]
        )

        return TDOutput(probabilities=new_labels["labels"].values)
