import numpy as np

def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Score for given points and cluster labels.
    
    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input data.
    labels : np.ndarray of shape (n_samples,)
        Cluster labels.
    
    Returns
    -------
    float
        Mean silhouette score.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)

    n_samples = X.shape[0]
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        raise ValueError("Silhouette score requires at least 2 clusters.")

    # Pairwise Euclidean distance matrix
    diff = X[:, None, :] - X[None, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))

    silhouette_vals = np.zeros(n_samples)

    for i in range(n_samples):
        current_label = labels[i]

        # Points in the same cluster (excluding itself)
        same_cluster = (labels == current_label)
        same_cluster[i] = False

        # a(i): mean intra-cluster distance
        if np.any(same_cluster):
            a = np.mean(distances[i, same_cluster])
        else:
            silhouette_vals[i] = 0.0
            continue

        # b(i): minimum mean distance to points in another cluster
        b = np.inf
        for label in unique_labels:
            if label == current_label:
                continue

            other_cluster = (labels == label)
            mean_dist = np.mean(distances[i, other_cluster])
            b = min(b, mean_dist)

        silhouette_vals[i] = (b - a) / max(a, b)

    return np.mean(silhouette_vals)