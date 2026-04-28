import numpy as np

def minmax_scale(X, axis=0, eps=1e-12):
    """
    Scale X to the [0, 1] range.

    Parameters
    ----------
    X : np.ndarray
        Input array (1D or 2D).
    axis : int, default=0
        Axis along which to compute min/max for 2D arrays.
    eps : float, default=1e-12
        Small constant to avoid division by zero.

    Returns
    -------
    np.ndarray
        Min-max scaled array.
    """
    X = np.asarray(X, dtype=np.float64)

    x_min = np.min(X, axis=axis, keepdims=True)
    x_max = np.max(X, axis=axis, keepdims=True)

    return (X - x_min) / (x_max - x_min + eps)