import numpy as np

def feature_density_approximation(
    train_states: np.ndarray,
    train_values: np.ndarray,
    test_states: np.ndarray,
    test_values: np.ndarray,
    n_centers: int,
    sigma: float
) -> dict:
    """
    Approximate a value function using RBF features and evaluate quality.
    
    Args:
        train_states: 1D array of training state positions in [0, 1]
        train_values: 1D array of true values at training states
        test_states: 1D array of test state positions in [0, 1]
        test_values: 1D array of true values at test states
        n_centers: number of RBF centers placed uniformly in [0, 1]
        sigma: bandwidth parameter for the RBF features
    
    Returns:
        dict with keys 'centers', 'weights', 'predictions', 'rmse'
    """
    if n_centers < 1:
        raise ValueError("n_centers must be at least 1.")
    if sigma <= 0:
        raise ValueError("sigma must be greater than 0.")

    train_states = np.asarray(train_states,dtype=float).reshape(-1)
    train_values = np.asarray(train_values,dtype=float).reshape(-1)
    test_states = np.asarray(test_states,dtype=float).reshape(-1)
    test_values = np.asarray(test_values,dtype=float).reshape(-1)

    if train_states.size != train_values.size:
        raise ValueError("train_states and train_values must have the same length.")
    if test_states.size != test_values.size:
        raise ValueError("test_states and test_values must have the same length.")

    if train_states.size == 0:
        raise ValueError("Training data must not be empty.")
    if test_states.size == 0:
        raise ValueError("Testing data must not be empty.")

    if n_centers == 1:
        centers = np.array([0.5])
    else:
        centers = np.linspace(0.0, 1.0, n_centers)

    def rbf_features(states: np.ndarray) -> np.ndarray:
        differences = states[:, np.newaxis] - centers[np.newaxis, :]
        return np.exp(
            -(differences ** 2) / (2.0 * sigma ** 2)
        )

    # Construct the training feature matrix and fit the weights.
    train_features = rbf_features(train_states)
    weights, _, _, _ = np.linalg.lstsq(
        train_features,
        train_values,
        rcond=None
    )

    # Evaluate the fitted approximation on the test states.
    test_features = rbf_features(test_states)
    predictions = test_features @ weights

    rmse = np.sqrt(np.mean((predictions - test_values) ** 2))

    return {
        "centers": np.round(centers, 4).tolist(),
        "weights": np.round(weights, 4).tolist(),
        "predictions": np.round(predictions, 4).tolist(),
        "rmse": round(float(rmse), 4)
    }