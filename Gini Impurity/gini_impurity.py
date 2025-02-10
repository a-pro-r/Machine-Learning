import numpy as np
from collections import Counter

def gini_impurity(y):
    """
    Calculate Gini Impurity for a list of class labels.

    :param y: List of class labels
    :return: Gini Impurity rounded to three decimal places
    """
    counts = Counter(y)

    sum_sq = 0
    for item in counts:
        sum_sq += (counts[item] / len(y)) ** 2
    return round(1 - sum_sq,3)


def test_gini_impurity() -> None:
    classes_1 = [0, 0, 0, 0, 1, 1, 1, 1]
    assert gini_impurity(classes_1) == 0.5

    classes_2 = [0, 0, 0, 0, 0, 1]
    assert gini_impurity(classes_2) == 0.278

    classes_3 = [0, 1, 2, 2, 2, 1, 2]
    assert gini_impurity(classes_3) == 0.571


if __name__ == "__main__":
    test_gini_impurity()
    print("All Gini Impurity tests passed.")