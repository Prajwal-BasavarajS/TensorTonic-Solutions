import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: scalar, list, or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    
    # Convert input to NumPy array
    x = np.asarray(x, dtype=float)

    # Vectorized GELU computation
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))