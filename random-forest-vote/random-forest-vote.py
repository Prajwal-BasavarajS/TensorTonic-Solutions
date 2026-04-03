import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    preds = np.array(predictions)  # shape (T, N)
    T, N = preds.shape
    
    result = []
    
    for i in range(N):
        # Get predictions for sample i across all trees
        votes = preds[:, i]
        
        # Count occurrences
        values, counts = np.unique(votes, return_counts=True)
        
        # Find max vote count
        max_count = np.max(counts)
        
        # Get all classes with max count
        candidates = values[counts == max_count]
        
        # Choose smallest class label (tie-break)
        result.append(int(np.min(candidates)))
    
    return result