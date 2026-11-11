# --- cell 1 ---
import numpy as np

def activation(x):
    '''
    Apply an activation function element-wise.
    
    Args:
        x: numpy array of any shape (raw neuron outputs)
    
    Returns:
        numpy array of same shape (activated outputs)
    
    Requirements:
        - Must be non-linear (not just returning x)
        - Must work on arrays of any shape
        - Must be deterministic
    '''
    x = np.asarray(x, dtype=np.float64)
    # e = np.exp(-np.abs(x))
    # sig = np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
    # result = x * sig  # Replace with your activation
    
    return np.where(x > 0, 2.0 * x, 0.02 * x)