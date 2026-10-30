import numpy as np

def lu_decomposition(A: list) -> tuple:
    """
    Perform LU decomposition on a square matrix using Doolittle's method.

    Args:
        A: Square matrix as a list of lists

    Returns:
        tuple: (L, U) where L is lower triangular with 1s on diagonal,
               U is upper triangular, and A = L @ U
    """
    A = np.asarray(A, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")

    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros((n, n))

    for i in range(n):
        # Compute the i-th row of U
        for j in range(i, n):
            U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])

        if np.isclose(U[i, i], 0):
            raise ValueError("Zero pivot encountered; pivoting is required")

        # Compute the i-th column of L
        for j in range(i + 1, n):
            L[j, i] = (
                A[j, i] - np.dot(L[j, :i], U[:i, i])
            ) / U[i, i]

    return L, U