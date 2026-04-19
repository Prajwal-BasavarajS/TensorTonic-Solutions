import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Computes average cross-entropy loss.

    Parameters:
    y_true : array-like of shape (N,)
        True class indices
    y_pred : array-like of shape (N, K)
        Predicted probabilities

    Returns:
    float
        Average cross-entropy loss
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Select the predicted probabilities for the correct classes
    correct_class_probs = y_pred[np.arange(len(y_true)), y_true]

    # Compute negative log likelihood and return mean
    loss = -np.mean(np.log(correct_class_probs))

    return loss