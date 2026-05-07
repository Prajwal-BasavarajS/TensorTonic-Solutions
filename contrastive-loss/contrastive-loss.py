import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    a, b: arrays of shape (N, D) or (D,)  (will broadcast to (N,D))
    y:    array of shape (N,) with values in {0,1}; 1=similar, 0=dissimilar
    margin: float > 0
    reduction: "mean" (default) or "sum"
    Return: float
    """
    
    # Convert inputs to arrays
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    y = np.asarray(y)

    # Validate margin
    if margin <= 0:
        raise ValueError("margin must be > 0")

    # Validate reduction
    if reduction not in ("mean", "sum"):
        raise ValueError("reduction must be 'mean' or 'sum'")

    # Validate labels
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must contain only 0 or 1")

    # Ensure at least 2D for broadcasting
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)

    # Compute Euclidean distances
    d = np.linalg.norm(a - b, axis=1)

    # Contrastive loss
    pos_loss = y * (d ** 2)
    neg_loss = (1 - y) * np.maximum(0.0, margin - d) ** 2

    loss = pos_loss + neg_loss

    if reduction == "mean":
        return float(np.mean(loss))
    else:
        return float(np.sum(loss))