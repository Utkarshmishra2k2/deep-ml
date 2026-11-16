import numpy as np
from itertools import product

def polynomial_basis(s, n):
    # return the (n+1)**k polynomial features of state s
    s = np.asarray(s, dtype=float).ravel()
    exponents = product(range(n + 1), repeat=len(s))

    features = [
        np.prod(s ** np.array(powers))
        for powers in exponents
    ]

    return np.array(features, dtype=float)

def gradient_mc_poly(samples, n, alpha, n_passes):
    # return the learned weight vector
    if not samples:
        return np.array([], dtype=float)

    k = np.asarray(samples[0][0]).size
    weights = np.zeros((n + 1) ** k, dtype=float)

    for _ in range(n_passes):
        for state, observed_return in samples:
            features = polynomial_basis(state, n)
            prediction = weights @ features

            weights += alpha * (observed_return - prediction) * features

    return weights
