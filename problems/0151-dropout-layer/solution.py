import numpy as np


class DropoutLayer:
    def __init__(self, p: float):
        """Initialize the dropout layer."""
        if p < 0 or p >= 1:
            raise ValueError("p must satisfy 0 <= p < 1")

        self.p = p
        self.mask = None

    def forward(
        self,
        x: np.ndarray,
        training: bool = True
    ) -> np.ndarray:
        """Forward pass of the dropout layer."""

        # During inference, return x without changing the stored mask.
        if not training:
            return x

        # Generate a fresh binary mask for every training forward pass.
        self.mask = np.random.binomial(
            1,
            1.0 - self.p,
            size=x.shape
        )

        # Apply inverted dropout scaling.
        return x * self.mask / (1.0 - self.p)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass using the latest training mask."""
        if self.mask is None:
            raise RuntimeError(
                "A training forward pass must occur before backward."
            )

        return grad * self.mask / (1.0 - self.p)