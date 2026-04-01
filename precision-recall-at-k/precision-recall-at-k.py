def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Step 1: Take top-k recommendations
    top_k = recommended[:k]
    
    # Step 2: Convert relevant list to a set for fast lookup
    relevant_set = set(relevant)
    
    # Step 3: Count hits (intersection of top_k and relevant)
    hits = sum(1 for item in top_k if item in relevant_set)
    
    # Step 4: Compute metrics
    precision = hits / k
    recall = hits / len(relevant_set)
    
    return [precision, recall]