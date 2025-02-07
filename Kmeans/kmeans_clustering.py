from collections import defaultdict

import numpy as np
from isort.literal import assignment


def k_means_clustering(points, k, initial_centroids, max_iterations):
    """
    1. Calculate euclidian distance between initial centroids and points
    2. assign points to the nearest centroids
    3. repeat
    4. stop if prev centroid is same as new centroid
    """
    points = np.array(points)
    initial_centroids = np.array(initial_centroids)
    n_points, n_dimen = points.shape
    centroids = initial_centroids.copy()

    assignments = np.zeros(n_points, dtype= int)

    for _ in range(max_iterations):

        # step 1: calculate min distance and assignments
        for i in range(n_points):
            distances = np.sqrt(np.sum((centroids - points[i]) ** 2, axis= 1 ))
            assignments[i] = np.argmin(distances)

        # step 2: update centroids

        new_centroids = np.zeros_like(centroids, dtype=float)

        for j in range(k):
            pts = points[assignments == j]
            if len(pts) > 0:
                new_centroids[j] = np.mean(pts, axis=0)
            else:
                new_centroids[j] = centroids[j]

        # break if no update in centroids
        if np.all(new_centroids == centroids):
            break
        centroids = np.round(new_centroids, 4)

    return centroids


def test_k_means_clustering() -> None:
    # Test case 1
    points = [(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)]
    k = 2
    initial_centroids = [(1, 1), (10, 1)]
    max_iterations = 10
    assert np.all(k_means_clustering(points, k, initial_centroids, max_iterations) == [(1.0, 2.0),
                                                                                (10.0, 2.0)]), "Test case 1 failed"

    # Test case 2
    points = [(0, 0, 0), (2, 2, 2), (1, 1, 1), (9, 10, 9), (10, 11, 10), (12, 11, 12)]
    k = 2
    initial_centroids = [(1, 1, 1), (10, 10, 10)]
    max_iterations = 10
    assert np.all(k_means_clustering(points, k, initial_centroids, max_iterations) == [(1.0, 1.0, 1.0), (
    10.3333, 10.6667, 10.3333)]), "Test case 2 failed"

    # Test case 3: Single cluster
    points = [(1, 1), (2, 2), (3, 3), (4, 4)]
    k = 1
    initial_centroids = [(0, 0)]
    max_iterations = 10
    assert np.all(k_means_clustering(points, k, initial_centroids, max_iterations) == [(2.5, 2.5)]), "Test case 3 failed"

    # Test case 4: Four clusters in 2D space
    points = [(0, 0), (1, 0), (0, 1), (1, 1), (5, 5), (6, 5), (5, 6), (6, 6),
              (0, 5), (1, 5), (0, 6), (1, 6), (5, 0), (6, 0), (5, 1), (6, 1)]
    k = 4
    initial_centroids = [(0, 0), (0, 5), (5, 0), (5, 5)]
    max_iterations = 10
    result = k_means_clustering(points, k, initial_centroids, max_iterations)
    expected = [(0.5, 0.5), (0.5, 5.5), (5.5, 0.5), (5.5, 5.5)]
    assert all(np.allclose(r, e) for r, e in zip(result, expected)), "Test case 4 failed"

    # Test case 5: Clusters with different densities
    points = [(0, 0), (0.5, 0), (0, 0.5), (0.5, 0.5),  # Dense cluster
              (4, 4), (6, 6)]  # Sparse cluster
    k = 2
    initial_centroids = [(0, 0), (5, 5)]
    max_iterations = 10
    result = k_means_clustering(points, k, initial_centroids, max_iterations)
    expected = [(0.25, 0.25), (5.0, 5.0)]
    assert all(np.allclose(r, e) for r, e in zip(result, expected)), "Test case 5 failed"


if __name__ == "__main__":
    test_k_means_clustering()
    print("All k_means_clustering tests passed.")
