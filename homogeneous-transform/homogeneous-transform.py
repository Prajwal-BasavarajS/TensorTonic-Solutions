import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    
    Parameters:
        T: (4,4) numpy array
        points: (3,) or (N,3) numpy array
        
    Returns:
        (3,) or (N,3) numpy array
    """
    points = np.asarray(points)
    
    # Check if single point
    single_point = points.ndim == 1
    if single_point:
        points = points.reshape(1, 3)
    
    # Convert to homogeneous coordinates (append 1)
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack([points, ones])  # (N,4)
    
    # Apply transformation
    transformed_h = (T @ points_h.T).T  # (N,4)
    
    # Convert back to 3D (drop last coordinate)
    transformed = transformed_h[:, :3]
    
    # Return original shape
    return transformed[0] if single_point else transformed