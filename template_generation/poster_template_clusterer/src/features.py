"""
features.py  Poster 

 poster ( blocks JSON)  
 bbox  polygon  type  label 

 config.yaml 
  - global geometry: aspect_ratio, num_blocks, mean_block_area      (3)
  - occupancy grid: GG                                        (G*G,  1212=144)
  - shape stats:                                (2)
  - structure:         (4)

3 + 144 + 2 + 4 = 153
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


COORD_MAX = 1000.0  # bbox 


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _bbox_to_unit(bbox: List[float]) -> Tuple[float, float, float, float]:
    """bbox [x1,y1,x2,y2] in [0,1000]  unit square [0,1]."""
    x1, y1, x2, y2 = bbox
    return (x1 / COORD_MAX, y1 / COORD_MAX, x2 / COORD_MAX, y2 / COORD_MAX)


def _occupancy_grid(bboxes: List[Tuple[float, float, float, float]],
                    grid_size: int) -> np.ndarray:
    """
     bbox  [0,1]
     bbox ""max coverage
    """
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    cell = 1.0 / grid_size
    for x1, y1, x2, y2 in bboxes:
        gx1 = max(0, int(np.floor(x1 / cell)))
        gx2 = min(grid_size, int(np.ceil(x2 / cell)))
        gy1 = max(0, int(np.floor(y1 / cell)))
        gy2 = min(grid_size, int(np.ceil(y2 / cell)))
        for gy in range(gy1, gy2):
            cy1 = gy * cell
            cy2 = cy1 + cell
            inter_y = max(0.0, min(cy2, y2) - max(cy1, y1))
            if inter_y <= 0:
                continue
            for gx in range(gx1, gx2):
                cx1 = gx * cell
                cx2 = cx1 + cell
                inter_x = max(0.0, min(cx2, x2) - max(cx1, x1))
                if inter_x <= 0:
                    continue
                cov = (inter_x * inter_y) / (cell * cell)
                if cov > grid[gy, gx]:
                    grid[gy, gx] = cov
    return grid


def _estimate_columns(bboxes: List[Tuple[float, float, float, float]]) -> int:
    """
     bbox  x  100  bin bin 
    ""
    """
    if not bboxes:
        return 0
    nbin = 100
    cover = np.zeros(nbin, dtype=np.int32)
    for x1, _, x2, _ in bboxes:
        b1 = max(0, int(np.floor(x1 * nbin)))
        b2 = min(nbin, int(np.ceil(x2 * nbin)))
        cover[b1:b2] += 1
    threshold = max(1, cover.max() // 2)
    in_seg = False
    segs = 0
    gap_run = 0
    min_gap = 3  #  3% 
    for v in cover:
        if v >= threshold:
            if not in_seg:
                if segs == 0 or gap_run >= min_gap:
                    segs += 1
                in_seg = True
            gap_run = 0
        else:
            if in_seg:
                in_seg = False
            gap_run += 1
    return segs


def _estimate_rows(bboxes: List[Tuple[float, float, float, float]]) -> int:
    """ _estimate_columns y """
    if not bboxes:
        return 0
    nbin = 100
    cover = np.zeros(nbin, dtype=np.int32)
    for _, y1, _, y2 in bboxes:
        b1 = max(0, int(np.floor(y1 * nbin)))
        b2 = min(nbin, int(np.ceil(y2 * nbin)))
        cover[b1:b2] += 1
    threshold = max(1, cover.max() // 2)
    in_seg = False
    segs = 0
    gap_run = 0
    min_gap = 3
    for v in cover:
        if v >= threshold:
            if not in_seg:
                if segs == 0 or gap_run >= min_gap:
                    segs += 1
                in_seg = True
            gap_run = 0
        else:
            if in_seg:
                in_seg = False
            gap_run += 1
    return segs


def extract_features(poster_json: dict, grid_size: int):
    """
    Args:
        poster_json: blocks JSON dict.
        grid_size: occupancy  12
    Returns:
        feature dict with arrays + a flat vector.
    """
    blocks = poster_json.get("blocks", [])
    img = poster_json.get("image_size", {})
    width = img.get("width", 1)
    height = img.get("height", 1)
    aspect_ratio = width / max(height, 1)

    bboxes_unit = [_bbox_to_unit(b["bbox"]) for b in blocks]

    num_blocks = len(blocks)
    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in bboxes_unit]
    mean_area = float(np.mean(areas)) if areas else 0.0

    # 
    occ = _occupancy_grid(bboxes_unit, grid_size)

    #  type 
    if areas:
        ratios = []
        for x1, y1, x2, y2 in bboxes_unit:
            w = max(x2 - x1, 1e-6)
            h = max(y2 - y1, 1e-6)
            ratios.append(w / h)
        ratio_mean = float(np.mean(ratios))
        ratio_std = float(np.std(ratios))
    else:
        ratio_mean = 0.0
        ratio_std = 0.0

    # 
    n_cols = _estimate_columns(bboxes_unit)
    n_rows = _estimate_rows(bboxes_unit)
    if areas:
        widest = max((x2 - x1) for x1, y1, x2, y2 in bboxes_unit)
        tallest = max((y2 - y1) for x1, y1, x2, y2 in bboxes_unit)
    else:
        widest = 0.0
        tallest = 0.0

    global_feats = np.array([aspect_ratio, num_blocks, mean_area], dtype=np.float32)
    shape_feats = np.array([ratio_mean, ratio_std], dtype=np.float32)
    struct_feats = np.array([n_cols, n_rows, widest, tallest], dtype=np.float32)

    flat = np.concatenate([
        global_feats,
        occ.flatten(),
        shape_feats,
        struct_feats,
    ]).astype(np.float32)

    return {
        "global": global_feats,
        "occupancy": occ,
        "shape": shape_feats,
        "structure": struct_feats,
        "vector": flat,
    }


def build_feature_matrix(json_dir: str, grid_size: int):
    """
     json_dir  *.json (N, D) 
     (names, X, sample_feats_first)   feature dict 
    """
    paths = sorted(Path(json_dir).glob("*.json"))
    names: List[str] = []
    vectors: List[np.ndarray] = []
    feature_blocks: Dict[str, dict] = {}
    sizes: List[Tuple[int, int]] = []
    block_counts: List[int] = []

    for p in paths:
        try:
            data = _load_json(str(p))
        except Exception as e:
            print(f"[features]  {p.name}: {e}")
            continue
        feats = extract_features(data, grid_size)
        names.append(p.stem)
        vectors.append(feats["vector"])
        img = data.get("image_size", {})
        sizes.append((img.get("width", 0), img.get("height", 0)))
        block_counts.append(data.get("num_blocks", len(data.get("blocks", []))))
        feature_blocks[p.stem] = {
            "global": feats["global"].tolist(),
            "shape": feats["shape"].tolist(),
            "structure": feats["structure"].tolist(),
            "occupancy_shape": list(feats["occupancy"].shape),
        }

    if not vectors:
        raise RuntimeError(f"No JSON found in {json_dir}")

    X = np.stack(vectors, axis=0)
    return names, X, sizes, block_counts, feature_blocks


def feature_segments(grid_size: int) -> Dict[str, Tuple[int, int]]:
    """ [start, end) """
    out: Dict[str, Tuple[int, int]] = {}
    cur = 0
    out["global"] = (cur, cur + 3); cur += 3
    occ_d = grid_size * grid_size
    out["occupancy"] = (cur, cur + occ_d); cur += occ_d
    out["shape"] = (cur, cur + 2); cur += 2
    out["structure"] = (cur, cur + 4); cur += 4
    return out
