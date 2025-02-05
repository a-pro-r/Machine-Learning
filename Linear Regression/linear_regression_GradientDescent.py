import numpy as np

"""
    Implements linear regression using gradient descent optimization.
    Finds optimal coefficients through iterative updates.

    Args:
        X: Feature matrix of shape (m, n) where m = examples, n = features
        y: Target values array of shape (m,)
        alpha: Learning rate for gradient descent
        iterations: Number of gradient descent iterations

    Returns:
        Optimized coefficients as 1D array, rounded to 4 decimal places
    """


def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    m,n = X.shape
    theta = np.zeros((n,1))

    for i in range(iterations):

        # Calculate predictions
        predictions = X @ theta

        # Calculate errors
        errors = predictions - y.reshape(-1, 1)

        # Calculate Gradient
        gradient = X.T @ errors / m

        # Update theta

        theta -= alpha * gradient

    return np.round(theta.flatten(), 4) # round to 4 decimals


def test_linear_regression_gradient_descent() -> None:
    # Test case 1
    X = np.array([[1, 1], [1, 2], [1, 3]])
    y = np.array([1, 2, 3])
    alpha = 0.01
    iterations = 1000
    assert np.allclose(linear_regression_gradient_descent(X, y, alpha, iterations), np.array([0.1107, 0.9513]),
                       atol=1e-4), "Test case 1 failed"


if __name__ == "__main__":
    test_linear_regression_gradient_descent()
    print("All linear_regression_gradient_descent tests passed.")