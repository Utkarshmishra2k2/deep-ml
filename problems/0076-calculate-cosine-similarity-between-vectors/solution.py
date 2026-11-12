import numpy as np

def cosine_similarity(v1, v2):
    """
    Calculate the cosine_similarity of two vectors.
    Args:
        vec1 (numpy.ndarray): 1D array representing the first vector.
        vec2 (numpy.ndarray): 1D array representing the second vector.
    Returns:
        The cosine_similarity of the two vectors.
    """
    # Implement your code here

    if v1.shape != v2.shape:
        raise ValueError("Vectors must have the same shape")
    if v1.size == 0 | v2.size == 0:
        raise ValueError("Vectors cannot be empty")

    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product == 0:
        raise ValueError("Vectors must have nonzero magnitude")

    return float(np.dot(v1, v2) / norm_product)
