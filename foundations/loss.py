import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)

        n = np.size(y_pred)
        arr = y_true * np.log(y_pred + 1e-7) + (1 - y_true) * np.log(1 - y_pred + 1e-7)
        total = np.sum(arr)
        loss = -1/n * total
        return np.round(loss, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        n_samples = y_true.shape[0] 
        epsilon = 1e-7
        
        # Calculate the sum of losses across all classes, then average over samples
        # We sum over axis 1 (classes) then mean over axis 0 (samples)
        loss = -np.sum(y_true * np.log(y_pred + epsilon)) / n_samples
        
        return float(np.round(loss, 4))

