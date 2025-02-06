import numpy as np


def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
    mean = np.mean(data, axis = 0)
    std_dev = np.std(data, axis = 0)
    std_data = [[] for _ in range(len(data))]
    min_max = [[] for _ in range(len(data))]

    min_data = np.min(data, axis = 0)
    max_data = np.max(data, axis = 0)
    col_idx = 0
    for item in zip(*data):
        row_idx = 0
        sd = std_dev[col_idx]
        m = mean[col_idx]
        for i in item:
            res_stand = (i - m) / sd
            std_data[row_idx].append(np.round(res_stand, 4))
            res_min_max = (i - min_data[col_idx]) / (max_data[col_idx] - min_data[col_idx])
            min_max[row_idx].append(np.round(res_min_max, 4))
            row_idx += 1
        col_idx += 1

    return std_data, min_max


def test_feature_scaling() -> None:
    # Test case 1
    data = np.array([[1, 2], [3, 4], [5, 6]])
    standardized, normalized = feature_scaling(data)
    assert standardized == [[-1.2247, -1.2247], [0.0, 0.0], [1.2247, 1.2247]], "Test case 1 failed"
    assert normalized == [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], "Test case 1 failed"


if __name__ == "__main__":
    test_feature_scaling()
    print("All feature_scaling tests passed.")