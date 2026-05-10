import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    Returns a NumPy array for valid inputs, otherwise None.
    """
    try:
        matrix = np.asarray(matrix, dtype=float)

        # Must be 2D
        if matrix.ndim != 2:
            return None

        # Validate axis
        if axis not in (None, 0, 1):
            return None

        # Compute norms
        if norm_type == 'l2':
            norms = np.sqrt(np.sum(matrix ** 2, axis=axis, keepdims=True))
        elif norm_type == 'l1':
            norms = np.sum(np.abs(matrix), axis=axis, keepdims=True)
        elif norm_type == 'max':
            norms = np.max(np.abs(matrix), axis=axis, keepdims=True)
        else:
            return None

        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)

        return matrix / norms

    except:
        return None