import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Convert input to NumPy array
    x = np.array(x)

    if rng is not None:
        rand_vals = rng.random(x.shape)
    else:
        rand_vals = np.random.random(x.shape)

    scale = 1.0 / (1.0 - p)

    # Use correct comparison from hint
    mask = (rand_vals < (1.0 - p)).astype(x.dtype) * scale

    out = x * mask

    return out, mask