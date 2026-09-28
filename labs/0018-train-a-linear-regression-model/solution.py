import numpy as np

def train(X, y, W, b):
    """
    Train linear regression weights on standardized data.
    
    Args:
        X: numpy array of shape (n_samples, n_features) -- standardized features
        y: numpy array of shape (n_samples,) -- standardized targets
        W: numpy array of shape (n_features,) -- initial random weights
        b: float -- initial bias (0.0)
    
    Returns:
        W: numpy array of shape (n_features,) -- trained weights
        b: float -- trained bias
    """
    # TODO: implement your training strategy here
    # You can use ANY approach: gradient descent, normal equation,
    # momentum, adaptive learning rates, mini-batching, etc.
    y = np.asarray(y).reshape(-1)
    X_withBias = np.column_stack((
        np.asarray(X,dtype = float),
        np.ones(X.shape[0])
    ))
    
    parameters,_,_,_ = np.linalg.lstsq(
        X_withBias,
        y,
        rcond = None
    )

    W = parameters[:-1]
    b = float((parameters[-1]))

    return W,b
