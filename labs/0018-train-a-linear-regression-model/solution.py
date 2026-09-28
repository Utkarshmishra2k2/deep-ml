import numpy as np

def train(X, y, W, b):
    """
    Train a regularized linear regression model using
    5-fold cross-validation to select the ridge penalty.

    Args:
        X: array of shape (n_samples, n_features)
        y: array of shape (n_samples,)
        W: initial weights, not used
        b: initial bias, not used

    Returns:
        W: trained weights
        b: trained bias
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    n_samples, n_features = X.shape

    # Ridge strengths to test.
    # Alpha = 0 is ordinary least squares.
    alphas = np.concatenate((
        np.array([0.0]),
        np.logspace(-4, 4, 25)
    ))

    n_folds = 5

    # Deterministic shuffled folds for reproducible results
    rng = np.random.default_rng(42)
    indices = rng.permutation(n_samples)
    folds = np.array_split(indices, n_folds)

    best_alpha = 0.0
    best_score = np.inf

    for alpha in alphas:
        fold_errors = []

        for fold_index in range(n_folds):
            val_idx = folds[fold_index]
            train_idx = np.concatenate([
                folds[i]
                for i in range(n_folds)
                if i != fold_index
            ])

            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]

            # Learn the intercept separately so it is not regularized
            x_mean = X_train.mean(axis=0)
            y_mean = y_train.mean()

            X_centered = X_train - x_mean
            y_centered = y_train - y_mean

            # Solve ridge regression:
            # (X.T @ X + alpha * I) @ W = X.T @ y
            matrix = X_centered.T @ X_centered
            matrix.flat[::n_features + 1] += alpha

            rhs = X_centered.T @ y_centered

            try:
                fold_W = np.linalg.solve(matrix, rhs)
            except np.linalg.LinAlgError:
                fold_W = np.linalg.lstsq(matrix, rhs, rcond=None)[0]

            fold_b = y_mean - x_mean @ fold_W
            predictions = X_val @ fold_W + fold_b

            mse = np.mean((y_val - predictions) ** 2)
            fold_errors.append(mse)

        average_error = np.mean(fold_errors)

        if average_error < best_score:
            best_score = average_error
            best_alpha = alpha

    # Retrain on all available training data using the best alpha
    x_mean = X.mean(axis=0)
    y_mean = y.mean()

    X_centered = X - x_mean
    y_centered = y - y_mean

    matrix = X_centered.T @ X_centered
    matrix.flat[::n_features + 1] += best_alpha

    rhs = X_centered.T @ y_centered

    try:
        W = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        W = np.linalg.lstsq(matrix, rhs, rcond=None)[0]

    b = float(y_mean - x_mean @ W)

    return W, b