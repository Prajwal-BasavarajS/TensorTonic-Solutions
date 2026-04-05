import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    x = np.asarray(x)

    if x.ndim == 3:  # (C, H, W)
        # Average over H and W
        return x.mean(axis=(1, 2), dtype=np.float64)

    elif x.ndim == 4:  # (N, C, H, W)
        # Average over H and W
        return x.mean(axis=(2, 3), dtype=np.float64)

    else:
        raise ValueError("Input must have shape (C,H,W) or (N,C,H,W)")