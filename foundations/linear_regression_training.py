import numpy as np
from numpy.typing import NDArray

class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # This calculates the partial derivative for a single weight w_j
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.matmul(X, weights)

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        
        learning_rate = 0.01
        cur_weights = initial_weights.copy()
        N = len(X)
        num_weights = len(initial_weights)

        for _ in range(num_iterations):
            # 1. Compute predictions
            pred = self.get_model_prediction(X, cur_weights)
            
            # Create a temporary array to store gradients for this iteration
            # so we don't update weights while still calculating gradients
            gradients = np.zeros(num_weights)
            
            # 2. Compute gradient for each weight index j
            for j in range(num_weights):
                gradients[j] = self.get_derivative(pred, Y, N, X, j)
            
            # 3. Update all weights
            cur_weights -= learning_rate * gradients
            
        return np.round(cur_weights, 5)