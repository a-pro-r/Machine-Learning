"""
How to PCA?
1. Standardize the data (Standard Scaling) (scale the features so mean is 0 and Std dev is 1)
2. Find Covariance Matrix ( explains the variance within the variables <diagnonal values> and how pairs move <off diag>)
3. Find EigenValue and EigenVector ( Gives the factor of variance and the direction)
4. Return top k EigenVectors sorted by EigenValues
USAGE:
Feature reduction -> can be used to train and test

Transform = Original_Data × Eigenvector_Matrix
(n×p)     =    (n×d)    ×     (d×p)

"""
from audioop import reverse

import numpy as np
from numpy.linalg.linalg import eigvals


def pca(data, k):
    # Apply Standard Scalar
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    scaled = (data - mean) / std_dev

    # Find Covariance
    """
    rowvar -> is variable each row or each column?
        In this case, rows are observation and columns represents variables 
    """
    cov_mat = np.cov(scaled, rowvar= False)

    # find eigenvalues and eigenvectors

    eigen_val, eigen_vec = np.linalg.eig(cov_mat)

    # sort by eigenvalues descending
    idx = eigen_val.argsort()[::-1]
    sorted_eigen_values = eigen_val[idx]
    sorted_eigen_vec = eigen_vec[:,idx]

    principal_comp = sorted_eigen_vec[:,:k]

    print("Transformed Data")
    transformed_data = np.dot(data, principal_comp)
    print(transformed_data)

    return np.round(principal_comp, 4).tolist()


def test_pca():
    # Test case 1
    data = np.array([[4, 2, 1], [5, 6, 7], [9, 12, 1], [4, 6, 7]])
    k = 2
    expected_output = [[0.6855, 0.0776], [0.6202, 0.4586], [-0.3814, 0.8853]]
    assert pca(data, k) == expected_output, "Test case 1 failed"

    # Test case 2
    data = np.array([[1, 2], [3, 4], [5, 6]])
    k = 1
    expected_output = [[0.7071], [0.7071]]
    assert pca(data, k) == expected_output, "Test case 2 failed"


if __name__ == "__main__":
    test_pca()
    print("All pca tests passed.")
