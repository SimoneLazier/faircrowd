from random import uniform
from copy import deepcopy
from collections import OrderedDict
import numpy as np
import pandas as pd
from ...core.algorithms import TDAlgorithm, TDOutput
from ...core.fair_crowd_dataset import FairCrowdDataset


class FairTD(TDAlgorithm):
    """
    Description & reference
    """

    theta: float

    def __init__(
        self,
        theta: float = 0.01,
    ):
        self.theta = theta

    def run(self, df: FairCrowdDataset) -> TDOutput:
        raw_answers = df.answers

        # Convert to the format used by FairTD
        raw_answers_none = (
            raw_answers.dropna(how="all", axis=0)
            .dropna(how="all", axis=1)
            .astype(object)
            .where(pd.notnull(df.answers), None)
        )
        sensitive = raw_answers_none[df["s"] == 1]
        others = raw_answers_none[df["s"] == 0]
        answer = []
        for worker in raw_answers.columns:
            if (
                len(sensitive[worker].dropna(how="all")) > 0
                and len(others[worker].dropna(how="all")) > 0
            ):
                answer.append([sensitive[worker].values, others[worker].values])
        del raw_answers_none
        del sensitive
        del others

        # Original FairTD code
        n = len(answer)
        m = len(answer[0])
        r = [len(answer[0][j]) for j in range(0, m)]

        truth = [[uniform(0.1, 0.9) for k in range(0, r[j])] for j in range(0, m)]
        bias = [[uniform(-0.1, 0.1) for j in range(0, m)] for i in range(0, n)]
        sigma = [0.0 for i in range(0, n)]
        quality = [1.0 / n for i in range(0, n)]

        last_truth = np.array([])

        disparity = np.array([0.0 for i in range(0, m)])

        maxdiff = -1

        repeat_times = -1

        while True:
            repeat_times += 1

            if self.theta != None:
                disparity_self = [[0.0, 0.0] for i in range(0, m)]
                for j in range(0, m):
                    for k in range(0, len(truth[j])):
                        if truth[j][k] == None:
                            continue
                        disparity_self[j][1] += 1
                        if truth[j][k] > 0.5:
                            disparity_self[j][0] += 1
                for j in range(0, m):
                    disparity_self[j] = disparity_self[j][0] / disparity_self[j][1]

                disparity_other = [[0.0, 0.0] for i in range(0, m)]
                for j_ in range(0, m):
                    for j in range(0, m):
                        if j == j_:
                            continue
                        for k in range(0, len(truth[j])):
                            if truth[j][k] == None:
                                continue
                            disparity_other[j_][1] += 1
                            if truth[j][k] > 0.5:
                                disparity_other[j_][0] += 1
                for j in range(0, m):
                    disparity_other[j] = disparity_other[j][0] / disparity_other[j][1]

                disparity = []
                for j in range(0, m):
                    disparity.append(disparity_self[j] - disparity_other[j])
                disparity = np.array(disparity)

            if len(last_truth) != 0:
                maxdiff = -1
                for j in range(0, m):
                    for k in range(0, r[j]):
                        if last_truth[j][k] != None and truth[j][k] != None:
                            maxdiff = max(maxdiff, abs(last_truth[j][k] - truth[j][k]))
                if self.theta == None or (
                    self.theta != None and max(abs(disparity)) < self.theta
                ):
                    break
                else:
                    pass
            if self.theta != None:
                rejection_list = OrderedDict()
                sum_score = 0.0
                for i in range(0, n):
                    for j in range(0, m):
                        z = disparity[j] * bias[i][j]
                        if z < 0:
                            rejection_list[(i, j)] = -z
                            sum_score += -z
                p = uniform(0, 1)
                acc = 0.0
                selected = None
                for item in rejection_list:
                    rejection_list[item] /= sum_score
                    acc += rejection_list[item]
                    if acc >= p:
                        selected = item
                        break

            last_truth = deepcopy(truth)
            for j in range(0, m):
                for k in range(0, r[j]):
                    truth[j][k] = 0.0
                    acc_q = 0.0
                    for i in range(0, n):
                        if answer[i][j][k] != None:
                            if self.theta != None:
                                # if bias[i][j]*disparity[j]>0 or uniform(0,1)>abs(disparity[j]):# or max(abs(disparity))<theta:
                                if (i, j) != selected or max(
                                    abs(disparity)
                                ) < self.theta:
                                    truth[j][k] += (
                                        answer[i][j][k] - bias[i][j]
                                    ) * quality[i]
                                else:
                                    truth[j][k] += answer[i][j][k] * quality[i]
                            else:
                                truth[j][k] += (answer[i][j][k] - bias[i][j]) * quality[
                                    i
                                ]
                            acc_q += quality[i]
                    if acc_q == 0:
                        truth[j][k] = None
                    else:
                        truth[j][k] /= acc_q

            for i in range(0, n):
                for j in range(0, m):
                    bias[i][j] = 0.0
                    acc_n = 0
                    for k in range(0, r[j]):
                        if answer[i][j][k] != None:
                            bias[i][j] += answer[i][j][k] - truth[j][k]
                            acc_n += 1
                    if acc_n == 0:
                        bias[i][j] = None
                    else:
                        bias[i][j] /= acc_n

            for i in range(0, n):
                sigma[i] = 0.0
                acc_n = 0
                for j in range(0, m):
                    for k in range(0, r[j]):
                        if answer[i][j][k] != None:
                            sigma[i] += (
                                answer[i][j][k] - truth[j][k] - bias[i][j]
                            ) ** 2
                            acc_n += 1
                sigma[i] = np.sqrt(sigma[i] / acc_n)

            sum_q = 0.0
            for i in range(0, len(quality)):
                quality[i] = 1.0 / sigma[i] ** 2
                sum_q += quality[i]
            for i in range(0, len(quality)):
                quality[i] = quality[i] / sum_q + 1e-10

        # Convert back to the format used by TDAlgorithm
        predictions_sensitive = pd.DataFrame(
            truth[0], index=df.answers[df["s"] == 1].index, columns=["probabilities"]
        )
        predictions_others = pd.DataFrame(
            truth[1], index=df.answers[df["s"] == 0].index, columns=["probabilities"]
        )
        predictions = pd.concat([predictions_sensitive, predictions_others]).reindex(
            df.answers.index
        )
        return TDOutput(probabilities=predictions["probabilities"])
