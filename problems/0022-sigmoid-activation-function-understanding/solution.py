import math

def sigmoid(z: float) -> float:
    result = 1.0 / (1.0 + math.exp(-z))
    return result