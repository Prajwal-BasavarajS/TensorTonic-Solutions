import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    """
    y_true: 1D array of {-1,+1}
    y_score: 1D array of real scores, same shape as y_true
    reduction: "mean" or "sum"
    Return: float
    """
    # Convert to NumPy arrays (handles list inputs)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Validate shapes
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape")

    # Validate labels
    if not np.all(np.isin(y_true, [-1, 1])):
        raise ValueError("y_true must contain only -1 and +1")

    # Compute hinge loss vectorized
    loss = np.maximum(0, margin - y_true * y_score)

    # Reduction
    if reduction == "mean":
        return float(np.mean(loss))
    elif reduction == "sum":
        return float(np.sum(loss))
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")