import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    
    Parameters
    ----------
    a : np.ndarray
        First input vector.
    b : np.ndarray
        Second input vector.
    
    Returns
    -------
    float
        Cosine similarity in the range [-1, 1].
        Returns 0.0 if either vector has zero magnitude.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))