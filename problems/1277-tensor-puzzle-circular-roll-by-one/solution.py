import numpy as np

def roll(a: np.ndarray) -> np.ndarray:
    """Circular left shift by one."""
    # Your code here
    n = len(a)
    if n == 0:
        return a.copy()

    indices = (np.arange(n) + 1) % n
    return a[indices]