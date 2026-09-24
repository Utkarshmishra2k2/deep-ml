import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.

    Args:
        A: A 2x2 NumPy array

    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
    """
    A = np.asarray(A, dtype=float)

    if A.shape != (2, 2):
        raise ValueError("A must be a 2x2 matrix")

    # Calculate the symmetric matrix A transpose times A
    B = A.T @ A

    a = B[0, 0]
    b = B[0, 1]
    d = B[1, 1]

    # Calculate the Jacobi rotation angle
    theta = 0.5 * np.arctan2(2 * b, a - d)

    c = np.cos(theta)
    s = np.sin(theta)

    # Right singular vectors
    V = np.array([
        [c, -s],
        [s,  c]
    ])

    # Calculate A @ V
    AV = A @ V

    # Singular values are the lengths of the columns of A @ V
    S = np.sqrt(np.sum(AV ** 2, axis=0))

    # Sort singular values from largest to smallest
    order = np.argsort(S)[::-1]
    S = S[order]
    V = V[:, order]
    AV = A @ V

    # Calculate the left singular vectors
    U = np.zeros((2, 2))
    tolerance = 1e-12

    if S[0] > tolerance:
        U[:, 0] = AV[:, 0] / S[0]
    else:
        U[:, 0] = np.array([1.0, 0.0])

    if S[1] > tolerance:
        U[:, 1] = AV[:, 1] / S[1]
    else:
        # Make the second column perpendicular to the first
        U[:, 1] = np.array([-U[1, 0], U[0, 0]])

    Vt = V.T

    return U, S, Vt