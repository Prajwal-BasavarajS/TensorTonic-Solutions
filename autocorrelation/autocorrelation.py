import numpy as np

def autocorrelation(series, max_lag):
    """
    Compute autocorrelation for lags 0..max_lag.
    Returns a list of floats of length max_lag + 1.
    """
    x = np.asarray(series, dtype=np.float64)
    n = x.shape[0]

    if n < 2:
        raise ValueError("series must have at least 2 elements")
    if not (0 <= max_lag < n):
        raise ValueError("max_lag must satisfy 0 <= max_lag < len(series)")

    # Mean-center the series
    x_mean = x.mean()
    x_centered = x - x_mean

    # Total variance (autocovariance at lag 0)
    gamma0 = np.dot(x_centered, x_centered)

    # Handle constant series
    if gamma0 == 0:
        result = [0.0] * (max_lag + 1)
        result[0] = 1.0
        return result

    # Compute autocorrelation for each lag
    result = []
    for k in range(max_lag + 1):
        # Dot product of overlapping segments
        cov = np.dot(x_centered[:n - k], x_centered[k:])
        result.append(cov / gamma0)

    return result