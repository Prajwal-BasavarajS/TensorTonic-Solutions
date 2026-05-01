import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    """
    Compute accuracy, precision, recall, and F1 for single-label classification.

    Parameters:
        y_true : array-like
            Ground truth labels.
        y_pred : array-like
            Predicted labels.
        average : str
            One of: 'micro', 'macro', 'weighted', 'binary'.
        pos_label : int
            Positive class for binary averaging.

    Returns:
        dict
            {
                "accuracy": float,
                "precision": float,
                "recall": float,
                "f1": float
            }
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    labels = np.union1d(y_true, y_pred)
    n = len(y_true)

    accuracy = np.mean(y_true == y_pred)

    def safe_div(a, b):
        return a / b if b != 0 else 0.0

    # Binary averaging
    if average == "binary":
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)

    # Micro averaging
    elif average == "micro":
        tp = np.sum(y_true == y_pred)
        precision = tp / n
        recall = tp / n
        f1 = tp / n

    # Macro / Weighted averaging
    elif average in ("macro", "weighted"):
        precisions = []
        recalls = []
        f1s = []
        supports = []

        for label in labels:
            tp = np.sum((y_true == label) & (y_pred == label))
            fp = np.sum((y_true != label) & (y_pred == label))
            fn = np.sum((y_true == label) & (y_pred != label))
            support = np.sum(y_true == label)

            p = safe_div(tp, tp + fp)
            r = safe_div(tp, tp + fn)
            f = safe_div(2 * p * r, p + r)

            precisions.append(p)
            recalls.append(r)
            f1s.append(f)
            supports.append(support)

        precisions = np.array(precisions)
        recalls = np.array(recalls)
        f1s = np.array(f1s)
        supports = np.array(supports)

        if average == "macro":
            precision = np.mean(precisions)
            recall = np.mean(recalls)
            f1 = np.mean(f1s)
        else:  # weighted
            weights = supports / supports.sum()
            precision = np.sum(precisions * weights)
            recall = np.sum(recalls * weights)
            f1 = np.sum(f1s * weights)

    else:
        raise ValueError("average must be one of: 'micro', 'macro', 'weighted', 'binary'")

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }