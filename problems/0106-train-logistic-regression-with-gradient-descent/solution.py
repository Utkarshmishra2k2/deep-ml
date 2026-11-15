# Cell 1
import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
    """
    Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
    """
    # Your code here
    X_bias = np.column_stack((np.ones(X.shape[0]),X))
    coefficients = np.zeros(X_bias.shape[1])
    losses = []
    
    for _ in range(iterations):
        z = X_bias @ coefficients # @ operator performs matrix multiplication
        predictions = 1 / (1 + np.exp(-z))
        
        loss = -np.sum(y*np.log(predictions) + (1-y)* np.log(1-predictions))
        losses.append(loss)
        gradient = X_bias.T @ (predictions - y)
        coefficients -= learning_rate * gradient
    return(
        [round(value,4) for value in coefficients],
        [round(value, 4) for value in losses]
        )