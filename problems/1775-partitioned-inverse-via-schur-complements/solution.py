import numpy as np

def partitioned_inverse(M, k):
    M = np.asarray(M)
    E = M[:k, :k]
    F = M[:k, k:]
    G = M[k:, :k]
    H = M[k:, k:]

    # Schur complement: S = E - F H^-1 G
    H_inv_G = np.linalg.solve(H, G)
    H_inv = np.linalg.solve(H, np.eye(H.shape[0], dtype=M.dtype))
    S = E - F @ H_inv_G
    S_inv = np.linalg.solve(S, np.eye(S.shape[0], dtype=M.dtype))

    top_left = S_inv
    top_right = -S_inv @ F @ H_inv
    bottom_left = -H_inv_G @ S_inv
    bottom_right = H_inv + H_inv_G @ S_inv @ F @ H_inv

    return np.block([
        [top_left, top_right],
        [bottom_left, bottom_right],
    ])

def block_slogdet(M, k):
    M = np.asarray(M)
    E = M[:k, :k]
    F = M[:k, k:]
    G = M[k:, :k]
    H = M[k:, k:]

    H_inv_G = np.linalg.solve(H, G)
    S = E - F @ H_inv_G

    sign_s, logabs_s = np.linalg.slogdet(S)
    sign_h, logabs_h = np.linalg.slogdet(H)

    return sign_s * sign_h, logabs_s + logabs_h

def bordered_inverse(A_inv, b, c):
    A_inv = np.asarray(A_inv)
    b = np.asarray(b).reshape(-1)

    x = A_inv @ b
    s = c - b @ x

    if abs(s) < 1e-12:
        raise ValueError("Schur complement is too close to zero")

    top_left = A_inv + np.outer(x, x) / s
    top_right = -x / s
    bottom_right = 1.0 / s

    return np.block([
        [top_left, top_right[:, None]],
        [top_right[None, :], np.array([[bottom_right]])],
    ])
