import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)

        nmax = np.max(z)
        total = np.sum(np.exp(z -  nmax))

        dist = np.exp(z - nmax)/ total
        return np.round(dist, 4)
