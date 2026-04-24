import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.

    Args:
        x: Input (scalar, list, or NumPy array)
        alpha: Negative slope

    Returns:
        NumPy array with Leaky ReLU applied
    """
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, x, alpha * x)