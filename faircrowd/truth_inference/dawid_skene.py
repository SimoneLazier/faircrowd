import numpy as np
import numpy.typing as npt
from ..core.algorithms import TDAlgorithm, TDOutput
from ..core.fair_crowd_dataset import FairCrowdDataset


class DawidSkene(TDAlgorithm):
    """
    Implementation of the Dawid-Skene algorithm for truth inference.
    """

    iterations: int
    initial_quality: float
    predicted_probabilities: npt.NDArray[np.float_]
    positive_probability: npt.NDArray[np.float_]
    confusion_matrices: npt.NDArray

    def __init__(self, iterations: int = 20, initial_quality: float = 0.7) -> None:
        self.iterations = iterations
        self.initial_quality = initial_quality

    def e_step(self, answers: npt.NDArray) -> None:
        """
        Update the predicted probabilities for each sample.
        """
        finite_elements = np.isfinite(answers)

        for sample in range(answers.shape[0]):
            # Get the subset of workers that labelled that sample
            workers = finite_elements[sample]
            responses = answers[sample, workers].astype(np.int32)

            weight_neg = (1 - self.positive_probability) * self.confusion_matrices[
                workers, 0, responses
            ].prod()
            weight_pos = (
                self.positive_probability
                * self.confusion_matrices[workers, 1, responses].prod()
            )
            total_weight = weight_neg + weight_pos

            self.predicted_probabilities[sample] = (
                np.array([0.5, 0.5])
                if total_weight == 0
                else np.array([weight_neg, weight_pos]) / total_weight
            )

    def m_step(self, answers: npt.NDArray) -> None:
        """
        Update the confusion matrices and the label probabilities.
        """
        # Update label probabilities
        self.positive_probability = self.predicted_probabilities.mean(axis=0)[1]

        # Update confusion matrices
        finite_elements = np.isfinite(answers)
        self.confusion_matrices = np.tile(
            [[0.0, 0.0], [0.0, 0.0]], (answers.shape[1], 1, 1)
        )
        for worker in range(answers.shape[1]):
            samples = finite_elements[:, worker]
            weights = self.predicted_probabilities[samples].sum(axis=0)
            responses = answers[samples, worker].astype(np.int32)

            for tlabel in [0, 1]:
                if weights[tlabel] == 0:
                    if tlabel == 0:
                        self.confusion_matrices[worker, tlabel] = np.array(
                            [self.initial_quality, 1 - self.initial_quality]
                        )
                    else:
                        self.confusion_matrices[worker, tlabel] = np.array(
                            [1 - self.initial_quality, self.initial_quality]
                        )
                else:
                    np.add.at(
                        self.confusion_matrices[worker, tlabel],
                        responses,
                        self.predicted_probabilities[samples, tlabel] / weights[tlabel],
                    )

    def initialize_parameters(self, df: FairCrowdDataset) -> None:
        """
        Initialize the parameters of the algorithm.
        """
        self.predicted_probabilities = np.tile(
            [np.nan, np.nan], (df.answers.shape[0], 1)
        )
        self.positive_probability = 0.5
        self.confusion_matrices = np.tile(
            [
                [self.initial_quality, 1 - self.initial_quality],
                [1 - self.initial_quality, self.initial_quality],
            ],
            (df.answers.shape[1], 1, 1),
        )

    def run(self, df: FairCrowdDataset) -> TDOutput:
        """
        Run the Dawid-Skene algorithm.
        """
        self.initialize_parameters(df)

        iterations = self.iterations
        while iterations > 0:
            self.e_step(df.answers.values)
            self.m_step(df.answers.values)
            iterations -= 1

        # Normalize confusion matrices
        confusion_matrices = self.confusion_matrices / 2

        return TDOutput(
            probabilities=self.predicted_probabilities[:, 1],
            confusion_matrices=confusion_matrices,
        )
