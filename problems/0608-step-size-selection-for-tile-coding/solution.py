# Cell 1
import numpy as np

def tile_coding_step(base_alpha: float, n_tilings: int, weights: list, active_tiles: list, target: float) -> dict:
    """
    Perform a single value function update using tile coding with adjusted step size.
    
    Args:
        base_alpha: The base learning rate before adjustment
        n_tilings: Number of tilings used in the tile coding scheme
        weights: List of current weight values for all tiles
        active_tiles: List of indices of tiles active for the current state
        target: The target value (e.g., a return or TD target)
    
    Returns:
        Dictionary with 'adjusted_alpha', 'prediction_before',
        'prediction_after', and 'updated_weights'
    """
    # Step size scaled by the number of active features (one per tiling)
    adjusted_alpha = base_alpha / n_tilings

    w = np.array(weights, dtype=float)  # copy, so the input list isn't mutated
    idx = np.asarray(active_tiles, dtype=int)

    # Current estimate: sum of weights at active tiles
    prediction_before = w[idx].sum()
    error = target - prediction_before

    # Update only the active tiles
    # (np.add.at handles repeated indices correctly, unlike w[idx] += ...)
    np.add.at(w, idx, adjusted_alpha * error)

    # Estimate after the update
    prediction_after = w[idx].sum()

    return {
        'adjusted_alpha': round(float(adjusted_alpha), 4),
        'prediction_before': round(float(prediction_before), 4),
        'prediction_after': round(float(prediction_after), 4),
        'updated_weights': [round(float(x), 4) for x in w],
    }