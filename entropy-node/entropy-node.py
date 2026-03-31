import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Convert to numpy array
    y = np.asarray(y)
    
    # Handle empty input
    if y.size == 0:
        return 0.0
    
    # Get class counts
    _, counts = np.unique(y, return_counts=True)
    
    # Convert to probabilities
    probs = counts / counts.sum()
    
    # Stable log: ignore zero probabilities
    # (though np.unique ensures counts > 0, this keeps it safe)
    probs = probs[probs > 0]
    
    # Compute entropy
    entropy = -np.sum(probs * np.log2(probs))
    
    return float(entropy)