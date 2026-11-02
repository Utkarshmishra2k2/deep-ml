import numpy as np
import math
def cholesky_decomposition(A):
    """
    Perform Cholesky decomposition on a symmetric positive-definite matrix.
    
    Args:
        A: A symmetric positive-definite matrix (2D list or numpy array)
    
    Returns:
        L: Lower triangular matrix such that A = L @ L.T as a 2D list,
           or -1 if decomposition is not possible
    """
    # Your code here
    # try:
    #     return np.linalg.cholesky(np.array(A)).tolist()
    # except:
    #     return -1

    n = len(A)
    if any(len(row) != n for row in A):
        return -1


    if any(A[i][j] != A[j][i] for i in range(n) for j in range(n)):
        return -1

    L = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            value = A[i][j]

            for k in range(j):
                value -= L[i][k] * L[j][k]

            if i == j:
                if value <= 0:
                    return -1
                L[i][j] = math.sqrt(value)
            else:
                L[i][j] = value / L[j][j]

    return L