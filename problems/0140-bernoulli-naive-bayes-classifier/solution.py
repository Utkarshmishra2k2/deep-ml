import numpy as np

class NaiveBayes():
    def __init__(self, smoothing=1.0):
        # Initialize smoothing
        self.smoothing = smoothing

    def forward(self, X, y):
        # Fit model to binary features X and labels y
        self.prior = []
        self.prob = []
        for c in [0,1]:
            X_c = X[y == c]
            count = len(X_c)
            if count == 0:
                self.prior.append(-np.inf)
                self.prob.append(np.full(X.shape[1],0.5))
            else:
                self.prior.append(np.log(count/len(y)))
                ones = X_c.sum(axis = 0)
                p = (ones + self.smoothing) / (count + 2 * self.smoothing )
                self.prob.append(p)

    def predict(self, X):
        # Predict class labels for test set X
        scores = []
        for c in [0,1]:
            p = self.prob[c]
            score = (
                self.prior[c]
                +
                (X * np.log(p)).sum(axis = 1)
                +
                ((1 - X) * np.log(1 - p)).sum(axis=1)
            )
            scores.append(score)

        scores = np.array(scores).T

        return np.argmax(scores, axis=1)