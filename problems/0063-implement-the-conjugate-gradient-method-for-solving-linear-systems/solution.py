import numpy as np

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x = np.zeros_like(b) if x0 is None else np.asarray(x0, dtype=float).copy()

    r = b - A @ x
    p = r.copy()
    rs_old = r @ r

    for _ in range(n):
        if np.sqrt(rs_old) <= tol:
            break

        Ap = A @ p
        denom = p @ Ap
        if denom <= 0:
            break

        alpha = rs_old / denom
        x += alpha * p
        r -= alpha * Ap

        rs_new = r @ r
        if np.sqrt(rs_new) <= tol:
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x.tolist()
