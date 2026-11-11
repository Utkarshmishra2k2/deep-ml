import numpy as np

def pin(x, cell, digit):
    """Copy of x where cell (r, c) keeps only the candidate `digit` (1-based)."""
    r, c = cell
    y = x.copy()
    y[r, c, :] = False
    y[r, c, digit - 1] = True
    return y

def choose_cell(x, rng, strategy):
    counts = x.sum(axis=2)
    open_mask = (counts >= 2).ravel()
    open_idx = np.flatnonzero(open_mask)
    if open_idx.size == 0:
        return None
    n = x.shape[1]
    if strategy == "mrv":
        flat = counts.ravel()[open_idx]
        k = open_idx[int(np.argmin(flat))]      # argmin -> first (row-major) on ties
    elif strategy == "random":
        k = rng.choice(open_idx)
    else:
        raise ValueError("unknown strategy: %r" % (strategy,))
    k = int(k)
    return (k // n, k % n)

def solve(x, deduce, rng, strategy="mrv", max_nodes=100000):
    """Depth-first search. Returns (solved state or None, number of deduce calls)."""
    nodes = 0

    def search(state):
        nonlocal nodes
        if nodes >= max_nodes:
            return None
        nodes += 1
        state = deduce(state)
        counts = state.sum(axis=2)
        if (counts == 0).any():
            return None
        if (counts == 1).all():
            return state
        cell = choose_cell(state, rng, strategy)
        digits = np.flatnonzero(state[cell[0], cell[1]]) + 1
        if strategy == "random":
            digits = rng.permutation(digits)
        for d in digits:
            res = search(pin(state, cell, int(d)))
            if res is not None:
                return res
            if nodes >= max_nodes:
                return None
        return None

    sol = search(x)
    return sol, nodes

def count_solutions(x, deduce, limit):
    """Number of distinct solved states reachable, stopping early at `limit`."""
    total = 0

    def search(state):
        nonlocal total
        if total >= limit:
            return
        state = deduce(state)
        counts = state.sum(axis=2)
        if (counts == 0).any():
            return
        if (counts == 1).all():
            total += 1
            return
        cell = choose_cell(state, None, "mrv")
        for d in np.flatnonzero(state[cell[0], cell[1]]) + 1:
            search(pin(state, cell, int(d)))
            if total >= limit:
                return

    search(x)
    return total