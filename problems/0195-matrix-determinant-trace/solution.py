import numpy as np

def matrix_determinant_and_trace(matrix: list[list[float]]) -> tuple[float, float]:
    """
    Compute the determinant and trace of a square matrix.
    """
    arr = np.asarray(matrix, dtype=float)

    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Matrix must be square")

    return float(np.linalg.det(arr)), float(np.trace(arr))
