import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L)
    """
    N = len(seqs)
    
    # Handle empty input
    if N == 0:
        return np.array([])
    
    # Determine max length
    if max_len is None:
        L = max(len(seq) for seq in seqs) if seqs else 0
    else:
        L = max_len
    
    # Initialize output with pad_value
    out = np.full((N, L), pad_value, dtype=float)
    
    # Fill sequences
    for i, seq in enumerate(seqs):
        seq = list(seq)  # ensure indexable
        length = min(len(seq), L)  # truncate if needed
        out[i, :length] = seq[:length]
    
    return out