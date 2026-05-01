import numpy as np

def zscore_standardize(X, axis=0, eps=1e-12):
    """
    Standardize X: (X - mean) / std.
    If 2D and axis=0, standardizes each column independently.
    
    Parameters:
        X : array-like
            Input data (1D or 2D).
        axis : int, default=0
            Axis along which to compute mean and std.
        eps : float, default=1e-12
            Small constant to prevent division by zero.
    
    Returns:
        np.ndarray
            Standardized array.
    """
    X = np.asarray(X, dtype=float)
    
    mean = np.mean(X, axis=axis, keepdims=True)
    std = np.std(X, axis=axis, keepdims=True)
    
    return (X - mean) / (std + eps)