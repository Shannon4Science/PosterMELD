"""
visualize.py  

 poster  block 
"""

import cv2
import numpy as np
from typing import List, Dict


def _imread_unicode(path: str):
    """/Unicode cv2.imread """
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f": {path}")
    return img


def _imwrite_unicode(path: str, img):
    """/Unicode """
    ext = path[path.rfind("."):]
    success, buf = cv2.imencode(ext, img)
    if success:
        buf.tofile(path)
    else:
        raise IOError(f": {path}")


#  (BGR)
COLOR_PALETTE = [
    (46, 204, 113),   # 
    (52, 152, 219),   # 
    (231, 76, 60),    # 
    (241, 196, 15),   # 
    (155, 89, 182),   # 
    (26, 188, 156),   # 
    (230, 126, 34),   # 
    (149, 165, 166),  # 
    (192, 57, 43),    # 
    (41, 128, 185),   # 
]


def _parse_polygon(polygon, bbox, img_w, img_h, coord_range):
    """
     numpy 

    :
    - : {"exterior": [[x,y],...], "holes": [[[x,y],...], ...]}
    - / polygon:  bbox 

    Returns:
        exterior_pts: np.array of shape (N, 2)
        hole_pts_list: list of np.array, each of shape (M, 2)
    """
    def to_pixels(points):
        return np.array([
            [int(p[0] / coord_range * img_w), int(p[1] / coord_range * img_h)]
            for p in points
        ], dtype=np.int32)

    if isinstance(polygon, dict) and "exterior" in polygon:
        exterior_pts = to_pixels(polygon["exterior"])
        hole_pts_list = [to_pixels(h) for h in polygon.get("holes", [])]
        return exterior_pts, hole_pts_list

    # fallback: bbox  
    x1 = int(bbox[0] / coord_range * img_w)
    y1 = int(bbox[1] / coord_range * img_h)
    x2 = int(bbox[2] / coord_range * img_w)
    y2 = int(bbox[3] / coord_range * img_h)
    rect = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
    return rect, []


def visualize_blocks(image_path: str, blocks: List[Dict], output_path: str,
                     coord_range: float = 1000.0):
    """
     poster  block 
    L 

    Args:
        image_path: poster 
        blocks: block  bbox/polygonlabeltype
        output_path: 
        coord_range: bbox MinerU  [0, 1000]
    """
    img = _imread_unicode(image_path)
    if img is None:
        raise FileNotFoundError(f": {image_path}")

    h, w = img.shape[:2]
    original = img.copy()

    #  block 
    color_layer = np.zeros_like(img)
    mask = np.zeros((h, w), dtype=np.uint8)

    label_colors = {}
    block_draw_info = []  # 

    for i, block in enumerate(blocks):
        label = block.get("label", f"block_{i}")

        if label not in label_colors:
            label_colors[label] = COLOR_PALETTE[len(label_colors) % len(COLOR_PALETTE)]
        color = label_colors[label]

        exterior_pts, hole_pts_list = _parse_polygon(polygon=block.get("polygon"),
                                                      bbox=block["bbox"],
                                                      img_w=w, img_h=h,
                                                      coord_range=coord_range)

        # 
        cv2.fillPoly(color_layer, [exterior_pts], color)
        cv2.fillPoly(mask, [exterior_pts], 255)

        # 
        for hole_pts in hole_pts_list:
            cv2.fillPoly(color_layer, [hole_pts], (0, 0, 0))
            cv2.fillPoly(mask, [hole_pts], 0)

        block_draw_info.append((exterior_pts, hole_pts_list, color, label, block))

    #  mask 
    alpha = 0.15
    mask_3ch = cv2.merge([mask, mask, mask])
    img = np.where(mask_3ch > 0,
                   cv2.addWeighted(color_layer, alpha, original, 1 - alpha, 0),
                   original).astype(np.uint8)

    # 
    for exterior_pts, hole_pts_list, color, label, block in block_draw_info:
        cv2.polylines(img, [exterior_pts], isClosed=True, color=color, thickness=3)
        for hole_pts in hole_pts_list:
            cv2.polylines(img, [hole_pts], isClosed=True, color=color, thickness=3)

        top_y = exterior_pts[:, 1].min()
        top_x = int(exterior_pts[exterior_pts[:, 1].argmin(), 0])

        label_text = f"{label} ({block['type']})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, min(w, h) / 2000)
        thickness = max(1, int(font_scale * 2))
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

        label_y = max(int(top_y) - 5, text_h + 5)
        cv2.rectangle(img, (top_x, label_y - text_h - 5),
                      (top_x + text_w + 5, label_y + 5), color, -1)
        cv2.putText(img, label_text, (top_x + 2, label_y),
                    font, font_scale, (255, 255, 255), thickness)

    _imwrite_unicode(output_path, img)
    print(f"[Visualize]   {output_path}")


def visualize_raw_blocks(image_path: str, blocks: List[Dict], output_path: str,
                         coord_range: float = 1000.0):
    """
     MinerU 
    
    """
    img = _imread_unicode(image_path)
    if img is None:
        raise FileNotFoundError(f": {image_path}")

    h, w = img.shape[:2]

    type_colors = {
        "text": (0, 200, 0),       # 
        "image": (200, 0, 0),      # 
        "table": (0, 0, 200),      # 
        "equation": (200, 200, 0), # 
    }

    for block in blocks:
        bbox = block["bbox"]
        x1 = int(bbox[0] / coord_range * w)
        y1 = int(bbox[1] / coord_range * h)
        x2 = int(bbox[2] / coord_range * w)
        y2 = int(bbox[3] / coord_range * h)

        color = type_colors.get(block["type"], (128, 128, 128))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

        # 
        font_scale = max(0.3, min(w, h) / 3000)
        cv2.putText(img, str(block["id"]), (x1 + 2, y1 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)

    _imwrite_unicode(output_path, img)
    print(f"[Visualize]   {output_path}")
