import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x = np.array(x)
    p = np.array(p)

    # Check shape match
    if x.shape != p.shape:
        raise ValueError("x and p must have the same shape")

    # Check probabilities sum to 1 (within tolerance)
    if not np.allclose(np.sum(p), 1.0, atol=1e-6):
        raise ValueError("Probabilities must sum to 1")

    # Compute expected value
    return float(np.sum(x * p))