def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    n = len(y_true)
    
    # Count true positives (correct predictions)
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    
    # In single-label multi-class:
    # FP = FN = total errors
    fp = n - tp
    fn = n - tp
    
    # Micro-F1 formula
    return float((2 * tp) / (2 * tp + fp + fn))