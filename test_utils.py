## Please fill in all the parts labeled as ### YOUR CODE HERE

import numpy as np
import pytest
from utils import *


def test_dot_product():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])

    result = dot_product(vector1, vector2)

    assert result == 32, f"Expected 32, but got {result}"


def test_cosine_similarity():
    vector1 = np.array([1, 0, 0])
    vector2 = np.array([0, 1, 0])

    result = cosine_similarity(vector1, vector2)

    expected_result = 0.0  # Cosine similarity for orthogonal vectors is 0

    assert np.isclose(
        result, expected_result
    ), f"Expected {expected_result}, but got {result}"


def test_nearest_neighbor():
    vectors = np.array([[1, 2], [3, 4], [5, 6]])
    query_vector = np.array([4, 4])

    result = nearest_neighbor(query_vector, vectors)

    expected_index = 1  # Assuming the nearest vector to [4, 4] is [3, 4], which is at index 1

    assert (
        result == expected_index
    ), f"Expected index {expected_index}, but got {result}"
