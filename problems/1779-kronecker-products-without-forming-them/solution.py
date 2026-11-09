import numpy as np

def kron_matvec(A, B, C):
    A = np.asarray(A)
    B = np.asarray(B)
    C = np.asarray(C)
    return (B @ C @ A.T).reshape(-1, order="F")

def kron_solve(A, B, y):
    A = np.asarray(A)
    B = np.asarray(B)
    y = np.asarray(y)

    n = A.shape[0]
    q = B.shape[0]
    Y = y.reshape((q, n), order="F")

    Z = np.linalg.solve(B, Y)
    X = np.linalg.solve(A, Z.T).T
    return X.reshape(-1, order="F")

def bow_logits(S, W, V):
    return np.einsum("ntk,kd,dc->nc", S, W, V)