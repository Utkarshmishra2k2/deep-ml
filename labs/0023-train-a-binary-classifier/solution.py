# --- cell 1 ---
import numpy as np
from sklearn.linear_model import LogisticRegression
# You can import any sklearn module you need

def train(X_train, y_train, X_val, y_val):
    """
    Train a binary classifier.
    
    Args:
        X_train: numpy array of shape (n_samples, 30) -- standardized features
        y_train: numpy array of shape (n_samples,) -- binary labels (0 or 1)
        X_val:   numpy array of shape (n_val, 30) -- standardized
        y_val:   numpy array of shape (n_val,) -- validation labels
    
    Returns:
        predict: callable that takes X (n, 30) and returns y_pred (n,) of 0s and 1s
    """
    # TODO: train a model and return a predict function
    
    model = LogisticRegression(
    C=1.0,
    penalty = "l2",
    solver = "liblinear",
    max_iter = 2000,
    random_state = 42
    )
    
    model.fit(X_train,y_train)
    def predict(X):
        return model.predict(X)
        
    return predict