import numpy as np

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    Returns a list of [x_min, y_min, x_max, y_max].
    """

    if isinstance(feature_size, int):
        feature_h = feature_w = feature_size
    else:
        feature_h, feature_w = feature_size

    if isinstance(image_size, int):
        image_h = image_w = image_size
    else:
        image_h, image_w = image_size

    stride_y = image_h / feature_h
    stride_x = image_w / feature_w

    anchors = []

    for i in range(feature_h):
        for j in range(feature_w):
            cx = (j + 0.5) * stride_x
            cy = (i + 0.5) * stride_y

            for scale in scales:
                for ar in aspect_ratios:
                    w = scale * np.sqrt(ar)
                    h = scale / np.sqrt(ar)

                    x_min = cx - w / 2
                    y_min = cy - h / 2
                    x_max = cx + w / 2
                    y_max = cy + h / 2

                    anchors.append([x_min, y_min, x_max, y_max])

    return anchors