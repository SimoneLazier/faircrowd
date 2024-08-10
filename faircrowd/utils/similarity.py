from tqdm import tqdm
import numpy as np
import pandas as pd
from numpy import typing as npt
from typing import Callable, Sequence, Any, Union
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from ..mitigation import SimilarityInProcessing, SimilarityPreProcessing
from ..truth_inference import MajorityVoting
from ..metrics import SimilarityFairness
from ..core import FairnessMetric, FairCrowdDataset, PostProcessingAlgorithm
from .string_utils import name, title_case
from sklearn.base import BaseEstimator


def compute_distance_matrix(
    X: npt.NDArray, d: Callable[[Any, Any], float]
) -> npt.NDArray:
    """
    Compute the distance matrix between samples in X using the distance function d.
    """
    res = np.zeros((len(X), len(X)))
    idxs = np.triu_indices(len(X), 1)
    with tqdm(total=len(idxs[0])) as pbar:
        for x, y in zip(idxs[0], idxs[1]):
            res[x, y] = d(X[x], X[y])
            pbar.update(1)
    return res + res.T


def global_estimate_stats(
    df: FairCrowdDataset,
    distance_matrix: npt.NDArray,
    ks: npt.NDArray = None,
    dists: npt.NDArray = None,
    metrics: Sequence[FairnessMetric] = [],
) -> None:
    """
    Print the global estimate of unfairness for different values of k or distance.
    """
    predictions = np.nanmean(df.answers.values, axis=1)
    difference = pd.DataFrame(
        index=list(map(name, metrics)),
        columns=ks if ks is not None else dists,
    )
    pct_diff = difference.copy()
    unfairnesses = pd.DataFrame(
        [m.compute(predictions, df["s"].values, df["y"].values) for m in metrics],
        index=list(map(name, metrics)),
    ).transpose()

    xs = ks if ks is not None else dists
    estimates = []
    for x in xs:
        if ks is not None:
            metric = SimilarityFairness(distance_matrix, k=x)
        else:
            metric = SimilarityFairness(distance_matrix, dist=x)
        estimates.append(metric.compute(df.answers.values, df["s"].values))

    for metric in difference.index:
        true_value = unfairnesses[metric].values[0]
        difference.loc[metric] = np.array(estimates) - true_value
        pct_diff.loc[metric] = (np.array(estimates) - true_value) / true_value * 100

    unfairnesses.index = ["True Value"]
    unfairnesses.columns = map(title_case, unfairnesses.columns)
    print(unfairnesses.to_string())
    print()

    difference.index = list(map(title_case, difference.index))
    difference.index.name = "Abs. Change"
    print(difference.to_string())
    print()

    pct_diff.index = list(map(title_case, difference.index))
    pct_diff.index.name = "% Change"
    print(pct_diff.to_string())
    print()


def global_estimate_plot(
    df: FairCrowdDataset,
    distance_matrix: npt.NDArray,
    ks: npt.NDArray = None,
    dists: npt.NDArray = None,
    metrics: Sequence[FairnessMetric] = [],
) -> None:
    """
    Plot the global estimate of unfairness for different values of k or distance.
    """
    predictions = np.nanmean(df.answers.values, axis=1)
    unfairnesses = [
        metric.compute(predictions, df["s"].values, df["y"].values)
        for metric in metrics
    ]

    xs = ks if ks is not None else dists
    estimates = []
    for x in tqdm(xs):
        if ks is not None:
            metric = SimilarityFairness(distance_matrix, k=x)
        else:
            metric = SimilarityFairness(distance_matrix, dist=x)
        estimates.append(metric.compute(df.answers.values, df["s"].values))

    plt.plot(xs, estimates, label="Estimate")
    for i, unfairness in enumerate(unfairnesses):
        style = next(plt.gca()._get_lines.prop_cycler)
        plt.hlines(
            unfairness,
            np.min(xs),
            np.max(xs),
            colors=style["color"],
            label=title_case(metrics[i]),
            linestyle=style["linestyle"],
        )
    plt.xlabel("K" if ks is not None else "Dissimilarity Threshold")
    plt.ylabel("Unfairness")
    # plt.legend()
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, frameon=False, ncol=50)
    plt.show()


def workers_estimate_plot(
    df: FairCrowdDataset,
    distance_matrix: npt.NDArray,
    ks: npt.NDArray = None,
    dists: npt.NDArray = None,
    metrics: Sequence[FairnessMetric] = [],
) -> None:
    """
    Plot the workers' estimate of unfairness for different values of k or distance.
    """
    xs = ks if ks is not None else dists
    estimates = []
    unfairnesses = []
    # Compute workers' unfairnesses
    for metric in metrics:
        workers_unfairness = []
        for worker in df.answers.columns:
            idxs = df[worker].dropna().index
            workers_unfairness.append(
                metric.compute(
                    df.loc[idxs, worker].values,
                    df["s"].loc[idxs].values,
                    df["y"].loc[idxs].values,
                )
            )
        unfairnesses.append(workers_unfairness)

    for x in tqdm(xs):
        scores = []
        for worker in df.answers.columns:
            subset = np.argwhere(np.isfinite(df[worker].values)).flatten()
            idxs = df.iloc[subset].index
            distance_subset = distance_matrix[np.ix_(subset, subset)]
            if ks is not None:
                metric = SimilarityFairness(distance_subset, k=x)
            else:
                metric = SimilarityFairness(distance_subset, dist=x)
            scores.append(
                metric.compute(df.loc[idxs, [worker]].values, df["s"].loc[idxs].values)
            )
        estimates.append(scores)

    corrs = []
    for unfairness in unfairnesses:
        not_nan = np.argwhere(~np.isnan(unfairness)).flatten()
        corrs.append(
            [
                np.corrcoef(np.array(estimate)[not_nan], np.array(unfairness)[not_nan])[
                    0, 1
                ]
                for estimate in estimates
            ]
        )

    for i, corr in enumerate(corrs):
        plt.plot(xs, corr, label=title_case(metrics[i]))
    plt.xlabel("K" if ks is not None else "Dissimilarity Threshold")
    plt.ylabel("Correlation")
    # plt.legend()
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, frameon=False, ncol=50)
    plt.show()


def in_processing_stats(
    df: FairCrowdDataset,
    distance_matrix: npt.NDArray,
    ks: npt.NDArray = None,
    dists: npt.NDArray = None,
    metrics: Sequence[FairnessMetric] = [],
) -> None:
    """
    Print the in-processing stats for different values of k or distance.
    """
    predictions = np.nanmean(df.answers.values, axis=1)
    index = list(map(name, metrics))
    index.append("Accuracy")
    difference = pd.DataFrame(index=index, columns=ks if ks is not None else dists)
    pct_diff = difference.copy()
    initial_unfairnesses = pd.DataFrame(
        [m.compute(predictions, df["s"].values, df["y"].values) for m in metrics],
        index=list(map(name, metrics)),
    ).transpose()
    initial_accuracy = accuracy_score(predictions > 0.5, df["y"].values)

    xs = ks if ks is not None else dists
    for x in xs:
        if ks is not None:
            in_processing = SimilarityInProcessing(distance_matrix, k=x)
        else:
            in_processing = SimilarityInProcessing(distance_matrix, dist=x)
        new_predictions = in_processing.run(df).labels
        for metric in metrics:
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

        difference.loc["Accuracy", x] = (
            accuracy_score(new_predictions > 0.5, df["y"].values) - initial_accuracy
        )
        pct_diff.loc["Accuracy", x] = (
            (accuracy_score(new_predictions > 0.5, df["y"].values) - initial_accuracy)
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


def in_processing_plot(
    df: FairCrowdDataset,
    distance_matrix: npt.NDArray,
    ks: npt.NDArray = None,
    dists: npt.NDArray = None,
    metrics: Sequence[FairnessMetric] = [],
) -> None:
    """
    Plot the in-processing resulting unfairness and accuracy for different values of k or distance.
    """
    predictions = np.nanmean(df.answers.values, axis=1)
    initial_unfairnesses = [
        m.compute(predictions, df["s"].values, df["y"].values) for m in metrics
    ]
    initial_accuracy = accuracy_score(predictions > 0.5, df["y"].values)

    xs = ks if ks is not None else dists
    unfairnesses = [[] for _ in metrics]
    accuracies = []

    for x in tqdm(xs):
        if ks is not None:
            in_processing = SimilarityInProcessing(distance_matrix, k=x)
        else:
            in_processing = SimilarityInProcessing(distance_matrix, dist=x)
        new_predictions = in_processing.run(df).labels
        for i, metric in enumerate(metrics):
            unfairnesses[i].append(
                np.abs(metric.compute(new_predictions, df["s"].values, df["y"].values))
            )
        accuracies.append(accuracy_score(new_predictions > 0.5, df["y"].values))

    for i, unfairness in enumerate(unfairnesses):
        style = next(plt.gca()._get_lines.prop_cycler)
        plt.plot(
            xs,
            unfairness,
            color=style["color"],
            label=title_case(metrics[i]),
            linestyle=style["linestyle"],
            marker=style["marker"],
        )
        plt.hlines(
            initial_unfairnesses[i],
            np.min(xs),
            np.max(xs),
            colors=style["color"],
            label="Initial " + title_case(metrics[i]),
            linestyle=style["linestyle"],
        )
    plt.xlabel("K" if ks is not None else "Dissimilarity Threshold")
    plt.ylabel("Unfairness")
    # plt.legend()
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, frameon=False, ncol=50)
    plt.show()

    style = next(plt.gca()._get_lines.prop_cycler)
    plt.plot(
        xs,
        accuracies,
        label="In-Processing",
        color=style["color"],
        linestyle=style["linestyle"],
        marker=style["marker"],
    )
    plt.hlines(
        initial_accuracy,
        np.min(xs),
        np.max(xs),
        colors=style["color"],
        label="Initial Accuracy",
        linestyles=style["linestyle"],
    )
    plt.xlabel("K" if ks is not None else "Dissimilarity Threshold")
    plt.ylabel("Accuracy")
    # plt.legend()
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, frameon=False, ncol=50)
    plt.show()


def pre_processing_stats(
    df: FairCrowdDataset,
    distance_matrix: npt.NDArray,
    ks: npt.NDArray = None,
    dists: npt.NDArray = None,
    metrics: Sequence[FairnessMetric] = [],
) -> None:
    """
    Print the pre-processing stats for different values of k or distance.
    """
    predictions = np.nanmean(df.answers.values, axis=1)
    index = list(map(name, metrics))
    index.append("Accuracy")
    difference = pd.DataFrame(index=index, columns=ks if ks is not None else dists)
    pct_diff = difference.copy()
    initial_unfairnesses = pd.DataFrame(
        [m.compute(predictions, df["s"].values, df["y"].values) for m in metrics],
        index=list(map(name, metrics)),
    ).transpose()
    initial_accuracy = accuracy_score(predictions > 0.5, df["y"].values)

    xs = ks if ks is not None else dists
    for x in xs:
        if ks is not None:
            pre_processing = SimilarityPreProcessing(distance_matrix, k=x)
        else:
            pre_processing = SimilarityPreProcessing(distance_matrix, dist=x)
        new_answers = pre_processing.run(df).answers
        new_predictions = np.nanmean(new_answers, axis=1)
        for metric in metrics:
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

        difference.loc["Accuracy", x] = (
            accuracy_score(new_predictions > 0.5, df["y"].values) - initial_accuracy
        )
        pct_diff.loc["Accuracy", x] = (
            (accuracy_score(new_predictions > 0.5, df["y"].values) - initial_accuracy)
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


def pre_processing_plot(
    df: FairCrowdDataset,
    distance_matrix: npt.NDArray,
    ks: npt.NDArray = None,
    dists: npt.NDArray = None,
    metrics: Sequence[FairnessMetric] = [],
    estimation_k: int = None,
    estimation_dist: float = None,
) -> None:
    """
    Plot the pre-processing resulting unfairness and accuracy for different values of k or distance.
    """
    predictions = np.nanmean(df.answers.values, axis=1)
    initial_unfairnesses = [
        m.compute(predictions, df["s"].values, df["y"].values) for m in metrics
    ]
    initial_accuracy = accuracy_score(predictions > 0.5, df["y"].values)

    xs = ks if ks is not None else dists
    unfairnesses = [[] for _ in metrics]
    accuracies = []
    estimated_unfairness = []

    for x in tqdm(xs):
        if ks is not None:
            pre_processing = SimilarityPreProcessing(distance_matrix, k=x)
        else:
            pre_processing = SimilarityPreProcessing(distance_matrix, dist=x)
        new_answers = pre_processing.run(df).answers
        new_predictions = np.nanmean(new_answers, axis=1)
        for i, metric in enumerate(metrics):
            unfairnesses[i].append(
                np.abs(metric.compute(new_predictions, df["s"].values, df["y"].values))
            )
        if estimation_k is not None:
            estimated_unfairness.append(
                SimilarityFairness(distance_matrix, k=estimation_k).compute(
                    new_answers, df["s"].values
                )
            )
        else:
            estimated_unfairness.append(
                SimilarityFairness(distance_matrix, dist=estimation_dist).compute(
                    new_answers, df["s"].values
                )
            )
        accuracies.append(accuracy_score(new_predictions > 0.5, df["y"].values))

    for i, unfairness in enumerate(unfairnesses):
        style = next(plt.gca()._get_lines.prop_cycler)
        plt.plot(
            xs,
            unfairness,
            color=style["color"],
            label=title_case(metrics[i].__class__.__name__),
            linestyle=style["linestyle"],
            marker=style["marker"],
        )
        plt.hlines(
            initial_unfairnesses[i],
            np.min(xs),
            np.max(xs),
            colors=style["color"],
            label="Initial " + title_case(metrics[i].__class__.__name__),
            linestyle=style["linestyle"],
        )

    style = next(plt.gca()._get_lines.prop_cycler)
    plt.plot(
        xs,
        estimated_unfairness,
        color=style["color"],
        label="Estimated Unfairness",
        linestyle=style["linestyle"],
        marker=style["marker"],
    )
    plt.xlabel("K" if ks is not None else "Dissimilarity Threshold")
    plt.ylabel("Unfairness")
    # plt.legend()
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, frameon=False, ncol=50)
    plt.show()

    style = next(plt.gca()._get_lines.prop_cycler)
    plt.plot(
        xs,
        accuracies,
        label="Pre-Processing",
        color=style["color"],
        linestyle=style["linestyle"],
        marker=style["marker"],
    )
    plt.hlines(
        initial_accuracy,
        np.min(xs),
        np.max(xs),
        colors=style["color"],
        label="Initial Accuracy",
        linestyle=style["linestyle"],
    )
    plt.xlabel("K" if ks is not None else "Dissimilarity Threshold")
    plt.ylabel("Accuracy")
    # plt.legend()
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, frameon=False, ncol=50)
    plt.show()


def downstream_tasks(
    df: FairCrowdDataset,
    pipeline: Sequence[Any],
    ml_model: Union[BaseEstimator, None] = None,
    metrics: Sequence[FairnessMetric] = [],
) -> None:
    """
    Evaluate the effects of a fair crowdsourcing pipeline on downstream tasks, compared to majority voting.

    N.B. If ml_model is None, only the majority voting and pipeline results are evaluated.
    """
    columns = list(map(name, metrics))
    columns.append("Accuracy")
    results = pd.DataFrame(index=["Majority Voting"], columns=columns)

    if ml_model is None:
        predictions = MajorityVoting().run(df).labels
    else:
        model = ml_model.fit(df.x, MajorityVoting().run(df).labels)
        predictions = model.predict(df.x) > 0.5

    results["Accuracy"] = accuracy_score(df["y"], predictions)
    for metric in metrics:
        results[name(metric)] = metric.compute(
            predictions, df["s"].values, df["y"].values
        )

    results.columns = map(title_case, results.columns)
    print(results.to_string())
    print()

    pipeline_results = df
    for step in pipeline:
        if issubclass(step.__class__, PostProcessingAlgorithm):
            pipeline_results = step.run(df, pipeline_results)
        else:
            pipeline_results = step.run(pipeline_results)

    # At this point pipeline_results must be an instance of TDOutput
    if ml_model is None:
        predictions = pipeline_results.labels
    else:
        model = ml_model.fit(df.x, pipeline_results.labels)
        predictions = model.predict(df.x) > 0.5

    results = pd.DataFrame(index=["Fair Pipeline"], columns=columns)
    results["Accuracy"] = accuracy_score(df["y"], predictions)
    for metric in metrics:
        results[name(metric)] = metric.compute(
            predictions, df["s"].values, df["y"].values
        )

    results.columns = map(title_case, results.columns)
    print(results.to_string())
    print()
