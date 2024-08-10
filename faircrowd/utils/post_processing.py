from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import Sequence
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from ..mitigation import ConfusionMatrix
from ..core import FairnessMetric, FairCrowdDataset, TDAlgorithm
from .string_utils import name, title_case


def get_golden_tasks(
    df: FairCrowdDataset,
    proportion: float,
) -> pd.Series:
    golden_tasks_per_worker = {
        worker: int(proportion * len(df.answers[worker].dropna()))
        for worker in df.answers.columns
    }
    y = df["y"].copy()
    for worker in df.answers.columns:
        y.loc[
            df.answers[worker].dropna().index[golden_tasks_per_worker[worker] :]
        ] = np.nan
    return y


def post_processing_stats(
    df: FairCrowdDataset,
    golden_tasks: Sequence[float],
    td_algorithm: TDAlgorithm,
    metrics: Sequence[FairnessMetric],
    theta: float = 0.05,
    max_num_iter: float = 10,
) -> None:
    """
    Print the post-processing stats for different proportions of golden tasks.
    """
    predictions = np.nanmean(df.answers.values, axis=1)
    index = list(map(name, metrics))
    difference = pd.DataFrame(index=index, columns=golden_tasks)
    pct_diff = difference.copy()
    initial_unfairnesses = pd.DataFrame(
        [m.compute(predictions, df["s"].values, df["y"].values) for m in metrics],
        index=list(map(name, metrics)),
    ).transpose()
    initial_accuracy = accuracy_score(predictions > 0.5, df["y"].values)

    td_output = td_algorithm.run(df)

    for x in tqdm(golden_tasks):
        y = get_golden_tasks(df, x)
        for metric in metrics:
            post_processing = ConfusionMatrix(
                golden_tasks=y,
                fairness_metric=metric,
                theta=theta,
                max_num_iter=max_num_iter,
            )
            new_predictions = post_processing.run(df, td_output).labels
            difference.loc[name(metric), x] = (
                metric.compute(new_predictions, df["s"].values, df["y"].values)
                - initial_unfairnesses[name(metric)].values[0]
            )
            pct_diff.loc[name(metric), x] = (
                (
                    metric.compute(new_predictions, df["s"].values, df["y"].values)
                    - initial_unfairnesses[name(metric)].values[0]
                )
                / initial_unfairnesses[name(metric)].values[0]
                * 100
            )

            difference.loc["Accuracy " + name(metric), x] = (
                accuracy_score(new_predictions > 0.5, df["y"].values) - initial_accuracy
            )
            pct_diff.loc["Accuracy " + name(metric), x] = (
                (
                    accuracy_score(new_predictions > 0.5, df["y"].values)
                    - initial_accuracy
                )
                / initial_accuracy
                * 100
            )

    difference.index = list(map(title_case, difference.index))
    difference.index.name = "Abs. Change"
    print(difference.to_string())
    print()

    pct_diff.index = list(map(title_case, difference.index))
    pct_diff.index.name = "% Change"
    print(pct_diff.to_string())
    print()


def post_processing_plot(
    df: FairCrowdDataset,
    golden_tasks: Sequence[float],
    td_algorithm: TDAlgorithm,
    metrics: Sequence[FairnessMetric],
    theta: float = 0.05,
    max_num_iter: float = 10,
) -> None:
    """
    Plot the pre-processing resulting unfairness and accuracy for different values of k or distance.
    """
    predictions = np.nanmean(df.answers.values, axis=1)
    initial_unfairnesses = [
        m.compute(predictions, df["s"].values, df["y"].values) for m in metrics
    ]
    initial_accuracy = accuracy_score(predictions > 0.5, df["y"].values)

    unfairnesses = [[] for _ in metrics]
    accuracies = [[] for _ in metrics]

    td_output = td_algorithm.run(df)

    for x in tqdm(golden_tasks):
        y = get_golden_tasks(df, x)
        for i, metric in enumerate(metrics):
            post_processing = ConfusionMatrix(
                golden_tasks=y,
                fairness_metric=metric,
                theta=theta,
                max_num_iter=max_num_iter,
            )
            new_predictions = post_processing.run(df, td_output).labels
            unfairnesses[i].append(
                np.abs(metric.compute(new_predictions, df["s"].values, df["y"].values))
            )
            accuracies[i].append(accuracy_score(new_predictions > 0.5, df["y"].values))

    for i, unfairness in enumerate(unfairnesses):
        style = next(plt.gca()._get_lines.prop_cycler)
        plt.plot(
            golden_tasks,
            unfairness,
            color=style["color"],
            label=title_case(metrics[i]),
            linestyle=style["linestyle"],
            marker=style["marker"],
        )
        plt.hlines(
            initial_unfairnesses[i],
            np.min(golden_tasks),
            np.max(golden_tasks),
            colors=style["color"],
            label="Initial " + title_case(metrics[i]),
            linestyle=style["linestyle"],
        )
    plt.xlabel("Proportion of Golden Tasks")
    plt.ylabel("Unfairness")
    # plt.legend()
    plt.show()

    for i, accuracy in enumerate(accuracies):
        style = next(plt.gca()._get_lines.prop_cycler)
        plt.plot(
            golden_tasks,
            accuracy,
            label=title_case(metrics[i]),
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
        )
        if i == 0:
            plt.hlines(
                initial_accuracy,
                np.min(golden_tasks),
                np.max(golden_tasks),
                colors=style["color"],
                label="Initial Accuracy",
                linestyles=style["linestyle"],
            )
    plt.xlabel("Proportion of Golden Tasks")
    plt.ylabel("Accuracy")
    # plt.legend()
    plt.show()
