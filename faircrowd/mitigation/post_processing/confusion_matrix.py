import numpy as np
import numpy.typing as npt
from typing import Dict, Any, Union, Sequence
from ...core.metrics import FairnessMetric
from ...core.algorithms import PostProcessingAlgorithm, TDOutput
from ...core.fair_crowd_dataset import FairCrowdDataset

div = lambda num, den: 0 if den == 0 else num / den


class DemographicParity(FairnessMetric):
    confusion_matrix_updater = np.array(
        [
            [1, -1],
            [-1, 1],
        ]
    )

    def compute(
        self,
        predictions: npt.NDArray,
        sensitive: npt.NDArray,
        truth: npt.NDArray,
    ) -> float:
        groups = np.unique(sensitive)
        if len(groups) < 2:
            return 0
        return (predictions[sensitive == groups[0]] > 0.5).mean() - (
            predictions[sensitive == groups[1]] > 0.5
        ).mean()


class EqualOpportunities(FairnessMetric):
    confusion_matrix_updater = np.array(
        [
            [1, 1],
            [-1, -1],
        ]
    )

    def compute(
        self,
        predictions: npt.NDArray,
        sensitive: npt.NDArray,
        truth: npt.NDArray,
    ) -> float:
        groups = np.unique(sensitive)
        if len(groups) < 2:
            return 0
        true_positive = (predictions > 0.5) & (truth == 1)

        return div(
            len(predictions[(sensitive == groups[0]) & true_positive]),
            len(truth[(sensitive == groups[0]) & (truth == 1)]),
        ) - div(
            len(predictions[(sensitive == groups[1]) & true_positive]),
            len(truth[(sensitive == groups[1]) & (truth == 1)]),
        )


class PredictiveParity(FairnessMetric):
    confusion_matrix_updater = np.array(
        [
            [0, 0],
            [1, -1],
        ]
    )

    def compute(
        self,
        predictions: npt.NDArray,
        sensitive: npt.NDArray,
        truth: npt.NDArray,
    ) -> float:
        groups = np.unique(sensitive)
        if len(groups) < 2:
            return 0
        true_positive = (predictions > 0.5) & (truth == 1)

        return div(
            len(predictions[(sensitive == groups[0]) & true_positive]),
            len(predictions[(sensitive == groups[0]) & (predictions > 0.5)]),
        ) - div(
            len(predictions[(sensitive == groups[1]) & true_positive]),
            len(predictions[(sensitive == groups[1]) & (predictions > 0.5)]),
        )


metrics: Sequence[FairnessMetric] = [
    DemographicParity(),
    EqualOpportunities(),
    PredictiveParity(),
]


class ConfusionMatrix(PostProcessingAlgorithm):
    """
    Description
    """

    fairness_metric: FairnessMetric
    theta: float
    golden_tasks_indexes: npt.NDArray
    golden_tasks: npt.NDArray
    max_num_iter: Union[int, None]
    tolerance: float

    groups: npt.NDArray

    def __init__(
        self,
        fairness_metric: FairnessMetric,
        golden_tasks: npt.NDArray,
        theta: Union[float, None] = 0.05,
        max_num_iter: Union[int, None] = None,
        tolerance: float = 0.005,
    ):
        # TODO: find a way to avoid this duplication
        # Find the modified fairness metric class
        self.fairness_metric = next(
            (
                x
                for x in metrics
                if x.__class__.__name__ == fairness_metric.__class__.__name__
            ),
            None,
        )
        # Demographic Parity can be computed without golden tasks
        if isinstance(self.fairness_metric, DemographicParity):
            self.golden_tasks_indexes = np.ones(golden_tasks.shape, dtype=np.bool)
        else:
            self.golden_tasks_indexes = np.isfinite(golden_tasks)
        if self.fairness_metric is None:
            raise ValueError(
                f'Fairness metric "{fairness_metric.__class__.__name__}" is not supported'
            )
        self.theta = theta
        self.max_num_iter = max_num_iter
        self.golden_tasks = golden_tasks
        self.tolerance = tolerance

    def safe_add(self, x, y):
        """
        x + y with a lower bound of 0 and an upper bound of 1
        """
        return np.maximum(np.minimum(x + y, 1), 0)

    def update_confusion_matrices(
        self,
        confusion_matrices: Dict[Any, npt.NDArray],
        unfairnesses: float,
    ) -> Dict[Any, npt.NDArray]:
        confusion_matrices = {
            key: value.copy() for key, value in confusion_matrices.items()
        }
        for group in self.groups:
            sign = 1 if group == self.groups[0] else -1
            confusion_matrices[group] = np.array(
                [
                    self.safe_add(
                        confusion_matrices[group][worker],
                        self.fairness_metric.confusion_matrix_updater
                        * unfairnesses[worker]
                        * sign,
                    )
                    for worker in range(len(confusion_matrices[group]))
                ]
            )

        return confusion_matrices

    def update_probabilities(
        self,
        answers: npt.NDArray,
        sensitive: npt.NDArray,
        probabilities: npt.NDArray,
        confusion_matrices: Dict[Any, npt.NDArray],
    ) -> npt.NDArray:
        # Update label probabilities
        finite_elements = np.isfinite(answers)
        positive_probability = probabilities.mean()
        new_probabilities = np.zeros_like(probabilities)

        for sample in range(answers.shape[0]):
            # Get the subset of workers that labelled that sample
            workers = finite_elements[sample]
            responses = answers[sample, workers].astype(np.int32)

            weight_neg = (1 - positive_probability) * confusion_matrices[
                sensitive[sample]
            ][workers, 0, responses].prod()
            weight_pos = (
                positive_probability
                * confusion_matrices[sensitive[sample]][workers, 1, responses].prod()
            )
            total_weight = weight_neg + weight_pos

            new_probabilities[sample] = (
                0.5 if total_weight == 0 else weight_pos / total_weight
            )

        return new_probabilities

    def run(self, df: FairCrowdDataset, td_output: TDOutput) -> TDOutput:
        answers = df.answers.values
        sensitive = df["s"].values
        self.groups = np.unique(sensitive)
        finite_elements = np.isfinite(answers)
        confusion_matrices = td_output.confusion_matrices
        probabilities = td_output.probabilities
        unfairness = np.abs(
            self.fairness_metric.compute(
                probabilities[self.golden_tasks_indexes] > 0.5,
                sensitive[self.golden_tasks_indexes],
                self.golden_tasks[self.golden_tasks_indexes],
            )
        )
        unfairnesses = np.array(
            [
                self.fairness_metric.compute(
                    answers[
                        finite_elements[:, worker] & self.golden_tasks_indexes, worker
                    ],
                    sensitive[finite_elements[:, worker] & self.golden_tasks_indexes],
                    self.golden_tasks[
                        finite_elements[:, worker] & self.golden_tasks_indexes
                    ],
                )
                for worker in range(answers.shape[1])
            ]
        )

        weights = np.zeros(answers.shape[1])
        num_iter = 0
        while (self.max_num_iter is None or num_iter < self.max_num_iter) and (
            self.theta is None or unfairness > self.theta
        ):
            num_iter += 1
            for worker in range(answers.shape[1]):
                weights[worker] += unfairnesses[worker]

                new_confusion_matrices = self.update_confusion_matrices(
                    confusion_matrices, weights
                )
                new_probabilities = self.update_probabilities(
                    answers, sensitive, probabilities, new_confusion_matrices
                )
                new_unfairness = np.abs(
                    self.fairness_metric.compute(
                        new_probabilities[self.golden_tasks_indexes] > 0.5,
                        sensitive[self.golden_tasks_indexes],
                        self.golden_tasks[self.golden_tasks_indexes],
                    )
                )
                if new_unfairness - unfairness <= self.tolerance:
                    unfairness = min(new_unfairness, unfairness)
                    probabilities = new_probabilities
                    if self.theta is not None and unfairness < self.theta:
                        break
                else:
                    weights[worker] -= unfairnesses[worker]

        if self.theta is not None and unfairness > self.theta:
            print("Warning: the algorithm did not converge")

        return TDOutput(
            probabilities=probabilities,
            confusion_matrices=new_confusion_matrices
            if num_iter > 0
            else confusion_matrices,
        )
