"""
preprocess.py  
"""

import cv2
import numpy as np


def preprocess(image_path: str, min_short_edge: int = 1500, max_long_edge: int = 8000):
    """
     poster 

    Args:
        image_path: 
        min_short_edge: 
        max_long_edge: 

    Returns:
        img:  RGB  (numpy array)
        original_size:  (width, height)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f": {image_path}")

    h, w = img.shape[:2]
    original_size = (w, h)

    # 
    short_edge = min(h, w)
    if short_edge < min_short_edge:
        scale = min_short_edge / short_edge
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge > max_long_edge:
        scale = max_long_edge / long_edge
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img, original_size
