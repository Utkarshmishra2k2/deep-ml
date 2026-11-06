import numpy as np


def f_score(y_true, y_pred, beta):
    """
    Calculate the F-Score for a binary classification task.

    Returns 0.0 when the denominator is zero.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    beta_squared = beta ** 2
    denominator = (1 + beta_squared) * tp + beta_squared * fn + fp

    if denominator == 0:
        return 0.0

    return round((1 + beta_squared) * tp / denominator, 3)
