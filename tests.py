"""
Run tests to ensure that metrics are computed as expected

metrics are inspired from the website:
https://towardsdatascience.com/mean-average-precision-at-k-map-k-clearly-explained-538d8e032d2
"""

import constants
import config
from llama_index_qa import (
    get_rank,
    compute_average_precision_at_i,
    compute_precision_at_i,
)
from sklearn.metrics import ndcg_score
import numpy as np

# evaluations represent a list of relevance scores for top 10 retrieved documents (sorted by cosine similarity)
evaluations_1 = [1, 0, 0, 0, 1, 1, 1, 0, 0, 0]
evaluations_2 = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
evaluations_3 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
evaluations_4 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
evaluations_5 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
evaluations_6 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
all_evaluations = [
    evaluations_1,
    evaluations_2,
    evaluations_3,
    evaluations_4,
    evaluations_5,
    evaluations_6,
]

precision_solutions_1 = [
    sum(evaluations_1[:i]) / i for i in range(1, len(evaluations_1) + 1)
]
precision_solutions_2 = [
    sum(evaluations_2[:i]) / i for i in range(1, len(evaluations_2) + 1)
]
precision_solutions_3 = [
    sum(evaluations_3[:i]) / i for i in range(1, len(evaluations_3) + 1)
]
precision_solutions_4 = [
    sum(evaluations_4[:i]) / i for i in range(1, len(evaluations_4) + 1)
]
precision_solutions_5 = [
    sum(evaluations_5[:i]) / i for i in range(1, len(evaluations_5) + 1)
]
precision_solutions_6 = [
    sum(evaluations_6[:i]) / i for i in range(1, len(evaluations_6) + 1)
]
all_precision_solutions = [
    precision_solutions_1,
    precision_solutions_2,
    precision_solutions_3,
    precision_solutions_4,
    precision_solutions_5,
    precision_solutions_6,
]

all_rank_solutions = []
for evaluations in all_evaluations:
    rank_solutions = []
    for i in range(10):
        try:
            rank_solutions.append(evaluations[: i + 1].index(1) + 1)
        except ValueError:
            rank_solutions.append(np.inf)
    all_rank_solutions.append(rank_solutions)

# solutions for average precision at i
all_average_precision_solutions = []
for evaluations, precision_solutions in zip(all_evaluations, all_precision_solutions):
    average_precision_solutions = []
    for i in range(10):
        m = sum(evaluations[: i + 1])
        if m == 0:
            average_precision_solutions.append(0)
        else:
            dot_product = np.array(precision_solutions[: i + 1]) @ np.array(
                evaluations[: i + 1]
            )

            average_precision_solutions.append(dot_product / m)

    all_average_precision_solutions.append(average_precision_solutions)


def test_precision_at_k(precision_fn, all_evaluations, all_precision_solutions, k=10):
    """
    precision_fn: computes precision at k
        args: evaluations, k
        return: float, precision at i

    all_evaluations: list of list of ints
    solutions: list of floats, contains precision at i for i in [1 ... k]
    """
    test_index = 1
    for evaluations, solutions in zip(all_evaluations, all_precision_solutions):
        precision_at_i = [precision_fn(evaluations, i) for i in range(1, k + 1)]
        assert precision_at_i == solutions, f"Precision at k test {test_index} failed"
        test_index += 1


def test_average_precision_at_k(
    average_precision_fn, all_average_precision_solutions, k=10
):
    """
    average_precision_fn: computes average precision at k
        args:   evaluations
                cpis, list of p@i for i in [1 ... k]
                i, int representing AP@i
        return: float, average precision at i

    evaluations: list of ints
    solutions: list of floats, contains average precision at i for i in [1 ... k]
    """
    test_index = 1
    for evaluations, solutions in zip(all_evaluations, all_average_precision_solutions):
        cpis = [compute_precision_at_i(evaluations, i) for i in range(1, k + 1)]
        average_precision_at_i = [
            average_precision_fn(evaluations, cpis, i) for i in range(1, k + 1)
        ]
        assert (
            average_precision_at_i == solutions
        ), f"Average precision at k test {test_index} failed"
        test_index += 1


def test_get_rank_at_k(rank_fn, all_rank_solutions, k=10):
    test_index = 1
    for evaluations, solutions in zip(all_evaluations, all_rank_solutions):
        ranks = [rank_fn(evaluations[: i + 1]) for i in range(k)]


if __name__ == "__main__":
    print("Running tests...")

    test_precision_at_k(
        compute_precision_at_i, all_evaluations, all_precision_solutions, 10
    )
    print("Precision at k tests passed!")

    test_average_precision_at_k(
        compute_average_precision_at_i, all_average_precision_solutions, 10
    )
    print("Average precision at k tests passed!")

    test_get_rank_at_k(get_rank, all_rank_solutions, 10)
    print("Get rank at k tests passed!")

    print("All tests passed!")
