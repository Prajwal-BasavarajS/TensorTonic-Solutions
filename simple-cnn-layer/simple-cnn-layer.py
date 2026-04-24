import numpy as np

def conv2d(x, W, b):
    """
    Simple 2D convolution layer forward pass.
    Valid padding, stride=1.

    Args:
        x: Input tensor of shape (N, C_in, H, W)
        W: Weights of shape (C_out, C_in, KH, KW)
        b: Bias of shape (C_out,)

    Returns:
        Output tensor of shape (N, C_out, H_out, W_out)
    """
    N, C_in, H, W_in = x.shape
    C_out, C_in_w, KH, KW = W.shape

    if C_in != C_in_w:
        raise ValueError("Input channels must match weight channels")

    H_out = H - KH + 1
    W_out = W_in - KW + 1

    # Initialize output
    y = np.zeros((N, C_out, H_out, W_out), dtype=float)

    # Convolution
    for n in range(N):
        for c_out in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    patch = x[n, :, i:i+KH, j:j+KW]
                    y[n, c_out, i, j] = np.sum(patch * W[c_out]) + b[c_out]

    return y