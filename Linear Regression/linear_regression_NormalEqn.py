import numpy as np

"""
    Implements linear regression using the normal equation.
    Finds optimal coefficients for the line.

    Problem and test case from : https://www.deep-ml.com/problems/14

    Args:
        X: Feature matrix of shape (m, n) where m = examples, n = features
        y: Target values array of shape (m,)
        
    Returns:
        Optimized coefficients as 1D array, rounded to 4 decimal places
    """


def linear_regression_normal_equation(X, y):
    X = np.array(X)
    y = np.array(y)

    y.reshape(-1,1)

    X_trans = X.T
    mul = X_trans @ X
    theta = np.linalg.inv(mul) @ X_trans @ y
    return np.round(theta, 4).flatten().tolist()



def test_linear_regression_normal_equation() -> None:
    # Test case 1
    X = [[1, 1], [1, 2], [1, 3]]
    y = [1, 2, 3]
    assert linear_regression_normal_equation(X, y) == [-0.0, 1.0], "Test case 1 failed"

    # Test case 2
    X = [[1, 3, 4], [1, 2, 5], [1, 3, 2]]
    y = [1, 2, 1]
    assert linear_regression_normal_equation(X, y) == [4.0, -1.0, -0.0], "Test case 2 failed"


if __name__ == "__main__":
    test_linear_regression_normal_equation()
    print("All linear_regression_normal_equation tests passed.")