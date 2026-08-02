"""
template_proto.py  ""bbox 

 slot 4 bbox 
 slot 

slot 
  1.  block  polygon  shapely Polygon
  2.  polygon IoU = 1 - IoUaverage linkage
  3.  =  candidate slot < min_freq 
  4. Medoid  IoU  polygon  bbox
  5.  (, )  candidate
      candidate bbox slot 
      < min_keep_ratio * 
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.validation import make_valid


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _aspect_ratio(d: dict) -> float:
    img = d.get("image_size", {})
    return img.get("width", 1) / max(img.get("height", 1), 1)


def _make_polygon(poly_field: Optional[dict],
                  bbox_fallback: Optional[List[float]]) -> Optional[Polygon]:
    poly = None
    if poly_field and poly_field.get("exterior"):
        ext = poly_field["exterior"]
        holes = poly_field.get("holes") or []
        try:
            poly = Polygon(ext, holes=holes)
            if not poly.is_valid:
                poly = make_valid(poly)
                if not isinstance(poly, Polygon):
                    polys = [g for g in getattr(poly, 'geoms', [])
                             if isinstance(g, Polygon)]
                    poly = max(polys, key=lambda g: g.area) if polys else None
        except Exception:
            poly = None
    if (poly is None or poly.is_empty) and bbox_fallback:
        x1, y1, x2, y2 = bbox_fallback
        poly = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    return poly if (poly and not poly.is_empty) else None


def _shapely_iou(a: Polygon, b: Polygon) -> float:
    if a.is_empty or b.is_empty:
        return 0.0
    inter = a.intersection(b).area
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _bbox_area(b: List[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _bbox_overlap(a: List[float], b: List[float]) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _trim_bbox(candidate: List[float], blocker: List[float]) -> List[float]:
    """
     candidate bbox  blocker 
     4 
    """
    cx1, cy1, cx2, cy2 = candidate
    bx1, by1, bx2, by2 = blocker

    if _bbox_overlap(candidate, blocker) <= 0:
        return candidate

    options = []
    #  candidate  top  blocker  bottom 
    if by2 < cy2:
        options.append([cx1, by2, cx2, cy2])
    #  candidate  bottom  blocker  top 
    if by1 > cy1:
        options.append([cx1, cy1, cx2, by1])
    #  candidate  left  blocker  right 
    if bx2 < cx2:
        options.append([bx2, cy1, cx2, cy2])
    #  candidate  right  blocker  left 
    if bx1 > cx1:
        options.append([cx1, cy1, bx1, cy2])

    if not options:
        return [0.0, 0.0, 0.0, 0.0]

    return max(options, key=_bbox_area)


def _collect_blocks(json_paths: List[str]) -> Tuple[List[dict], int, float]:
    rows: List[dict] = []
    ratios: List[float] = []
    for p in json_paths:
        d = _load_json(p)
        ratios.append(_aspect_ratio(d))
        name = Path(p).stem
        for b in d.get("blocks", []):
            poly = _make_polygon(b.get("polygon"), b.get("bbox"))
            if poly is None:
                continue
            rows.append({"poster": name, "polygon": poly})
    ar = float(np.median(ratios)) if ratios else 0.78
    return rows, len(json_paths), ar


def extract_template(json_paths: List[str],
                     iou_threshold: float = 0.30,
                     min_freq: float = 0.30,
                     min_keep_ratio: float = 0.40,
                     morph_gap: float = 10.0) -> dict:
    rows, n_posters, aspect_ratio = _collect_blocks(json_paths)
    if not rows:
        return {"num_posters": 0, "aspect_ratio": aspect_ratio,
                "num_slots": 0, "slots": [], "occupancy_heatmap": [[]]}

    polys = [r["polygon"] for r in rows]

    # 
    if len(polys) >= 2:
        n = len(polys)
        cond = []
        iou_mat = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                iou = _shapely_iou(polys[i], polys[j])
                iou_mat[i, j] = iou
                iou_mat[j, i] = iou
                cond.append(1.0 - iou)
        cond = np.array(cond, dtype=np.float64)
        Z = linkage(cond, method="average")
        labels = fcluster(Z, t=1.0 - iou_threshold, criterion="distance")
    else:
        iou_mat = np.zeros((1, 1))
        labels = np.array([1])

    #  slot medoid  bbox 
    candidates: List[dict] = []
    for slot_id in sorted(set(labels.tolist())):
        idxs = np.where(labels == slot_id)[0].tolist()
        members = [rows[i] for i in idxs]
        unique_posters = {m["poster"] for m in members}
        frequency = len(unique_posters) / n_posters
        if frequency < min_freq:
            continue
        # Medoid
        if len(idxs) == 1:
            medoid_idx = 0
        else:
            sub = iou_mat[np.ix_(idxs, idxs)]
            medoid_idx = int(np.argmax(sub.sum(axis=1)))
        rep_poly = rows[idxs[medoid_idx]]["polygon"]
        #  medoid polygon  bbox 
        minx, miny, maxx, maxy = rep_poly.bounds
        candidates.append({
            "frequency": float(frequency),
            "bbox": [round(minx, 1), round(miny, 1), round(maxx, 1), round(maxy, 1)],
        })

    # 
    candidates.sort(key=lambda c: (-c["frequency"], -_bbox_area(c["bbox"])))
    kept: List[dict] = []

    for c in candidates:
        bbox = list(c["bbox"])
        orig_area = _bbox_area(bbox)
        if orig_area <= 0:
            continue

        #  slot 
        for k in kept:
            if _bbox_overlap(bbox, k["bbox"]) > 0:
                bbox = _trim_bbox(bbox, k["bbox"])
                if _bbox_area(bbox) <= 0:
                    break

        area = _bbox_area(bbox)
        if area <= 0:
            continue
        if area / orig_area < min_keep_ratio:
            continue

        kept.append({
            "frequency": c["frequency"],
            "bbox": [round(v, 1) for v in bbox],
        })

    # y x 
    kept.sort(key=lambda s: (s["bbox"][1], s["bbox"][0]))

    out_slots = []
    for new_id, s in enumerate(kept):
        x1, y1, x2, y2 = s["bbox"]
        out_slots.append({
            "slot_id": new_id,
            "frequency": round(s["frequency"], 3),
            "bbox": s["bbox"],
            "polygon": {
                "exterior": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "holes": [],
            },
        })

    return {
        "num_posters": n_posters,
        "aspect_ratio": round(aspect_ratio, 4),
        "num_slots": len(out_slots),
        "slots": out_slots,
    }
