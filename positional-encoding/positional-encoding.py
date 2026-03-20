import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # positions (seq_len, 1)
    positions = np.arange(seq_len)[:, np.newaxis]
    
    # dimensions (1, d_model)
    dims = np.arange(d_model)[np.newaxis, :]
    
    # compute angle rates
    angle_rates = 1 / (base ** (2 * (dims // 2) / d_model))
    
    # compute angles
    angles = positions * angle_rates  # (seq_len, d_model)
    
    # initialize encoding
    pe = np.zeros((seq_len, d_model))
    
    # apply sin to even indices
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    
    # apply cos to odd indices
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    
    return pe