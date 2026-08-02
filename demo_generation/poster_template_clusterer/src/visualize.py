"""
visualize.py  

- 2D scatterPCA 
-  poster 
- slot  +  + type 
-  occupancy 

Windows  ASCII  PIL Unicode + np.fromfile/tofile
 cv2.imread/imwrite
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path as MplPath
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA


def pca_2d(X: np.ndarray) -> np.ndarray:
    """PCA  2 sklearn"""
    return PCA(n_components=2, random_state=0).fit_transform(X)


def scatter_plot(X2: np.ndarray, labels: np.ndarray, out_path: str,
                 title: str = "Clusters (PCA-2D)") -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    uniq = sorted(set(labels.tolist()))
    for c in uniq:
        sel = labels == c
        ax.scatter(X2[sel, 0], X2[sel, 1],
                   c=[cmap(c % 10)], label=f"cluster {c} (n={int(sel.sum())})",
                   s=40, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _load_image_unicode(path: str) -> Optional[Image.Image]:
    try:
        with open(path, "rb") as f:
            data = f.read()
        from io import BytesIO
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception as e:
        print(f"[viz] read failed: {path}: {e}")
        return None


def _find_poster_path(name: str, posters_dir: str) -> Optional[str]:
    """name  JSON  stemposter """
    d = Path(posters_dir)
    if not d.exists():
        return None
    candidates = []
    target = name.lower().replace(" ", "")
    for p in d.iterdir():
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        if p.stem.lower().replace(" ", "") == target:
            return str(p)
        candidates.append(p)
    # fallback: substring match (handles minor punctuation differences)
    for p in candidates:
        if p.stem.lower().replace(" ", "")[:30] == target[:30]:
            return str(p)
    return None


def cluster_grids(reps: Dict[int, dict], posters_dir: str,
                  out_dir: str, thumb_max_side: int = 400) -> Dict[int, str]:
    """
     poster 
     {cluster_id: }
    """
    os.makedirs(out_dir, exist_ok=True)
    paths_out: Dict[int, str] = {}
    for cid, info in reps.items():
        names = [n for n, _ in info["reps"]]
        thumbs = []
        for name in names:
            ppath = _find_poster_path(name, posters_dir)
            if not ppath:
                continue
            img = _load_image_unicode(ppath)
            if img is None:
                continue
            img.thumbnail((thumb_max_side, thumb_max_side))
            thumbs.append((name, img))
        if not thumbs:
            print(f"[viz] cluster {cid}: ")
            continue

        max_h = max(im.height for _, im in thumbs)
        total_w = sum(im.width for _, im in thumbs) + 10 * (len(thumbs) - 1)
        canvas = Image.new("RGB", (total_w, max_h + 30), color=(255, 255, 255))
        x = 0
        for _, im in thumbs:
            canvas.paste(im, (x, 0))
            x += im.width + 10

        #  + 
        from PIL import ImageDraw
        draw = ImageDraw.Draw(canvas)
        label = f"cluster {cid}   size={info['size']}   reps={', '.join(names)}"
        draw.text((5, max_h + 5), label[:140], fill=(0, 0, 0))

        out_path = os.path.join(out_dir, f"cluster_{cid}.png")
        canvas.save(out_path)
        paths_out[cid] = out_path
        print(f"[viz] cluster {cid}: saved -> {out_path}")
    return paths_out


# ============================================================
# Step 4: polygon-based
# ============================================================


def _polygon_to_mplpath(poly_dict: dict) -> Optional[MplPath]:
    """{exterior, holes}  matplotlib Path"""
    ext = poly_dict.get("exterior") or []
    if len(ext) < 3:
        return None
    verts: List[Tuple[float, float]] = []
    codes: List[int] = []

    def add_ring(ring):
        if len(ring) < 3:
            return
        verts.append((ring[0][0], ring[0][1]))
        codes.append(MplPath.MOVETO)
        for x, y in ring[1:]:
            verts.append((x, y))
            codes.append(MplPath.LINETO)
        verts.append((ring[0][0], ring[0][1]))
        codes.append(MplPath.CLOSEPOLY)

    add_ring(ext)
    for hole in poly_dict.get("holes") or []:
        add_ring(hole)
    return MplPath(verts, codes)


def _polygon_centroid(poly_dict: dict) -> Tuple[float, float]:
    ext = poly_dict.get("exterior") or []
    if not ext:
        return (500.0, 500.0)
    xs = [p[0] for p in ext]
    ys = [p[1] for p in ext]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def render_template(template: dict, out_path: str,
                    canvas_long_side: int = 1000) -> None:
    """
    
      -  aspect_ratio 
      -  slot  5 
      -  = tab10  slot_id
      -  = 0.35 + 0.50 * frequency
    """
    aspect = template.get("aspect_ratio", 0.78)
    if aspect >= 1:
        W = canvas_long_side
        H = int(canvas_long_side / aspect)
    else:
        H = canvas_long_side
        W = int(canvas_long_side * aspect)

    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=120)
    ax.set_xlim(0, 1000)
    ax.set_ylim(1000, 0)
    ax.set_aspect('auto')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Template (n={template['num_posters']}, "
                 f"slots={template['num_slots']})", fontsize=10)

    cmap = plt.get_cmap("tab10")
    gap = 4  # JSON  bbox 

    for s in template["slots"]:
        x1, y1, x2, y2 = s["bbox"]
        # 
        rx1 = x1 + gap; ry1 = y1 + gap
        rx2 = x2 - gap; ry2 = y2 - gap
        if rx2 <= rx1 or ry2 <= ry1:
            rx1, ry1, rx2, ry2 = x1, y1, x2, y2

        rgba_face = list(cmap(s["slot_id"] % 10))
        rgba_face[3] = 0.35 + 0.50 * s["frequency"]
        rgba_edge = list(cmap(s["slot_id"] % 10))
        rgba_edge[3] = 1.0

        ax.add_patch(Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1,
                               linewidth=1.5,
                               edgecolor=tuple(rgba_edge),
                               facecolor=tuple(rgba_face)))

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        label = f"#{s['slot_id']}\n{s['frequency']:.0%}"
        ax.text(cx, cy, label,
                ha="center", va="center",
                fontsize=10, color="black", weight="bold",
                bbox=dict(facecolor="white", alpha=0.8,
                          edgecolor="none", pad=2))

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def render_heatmap(heatmap: np.ndarray, aspect_ratio: float,
                   out_path: str, canvas_long_side: int = 1000) -> None:
    """ occupancy """
    H_g, W_g = heatmap.shape
    if aspect_ratio >= 1:
        W = canvas_long_side
        H = int(canvas_long_side / aspect_ratio)
    else:
        H = canvas_long_side
        W = int(canvas_long_side * aspect_ratio)

    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=120)
    im = ax.imshow(heatmap, cmap="hot", interpolation="nearest",
                   extent=[0, 1000, 1000, 0], aspect=aspect_ratio,
                   vmin=0, vmax=max(heatmap.max(), 1e-3))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Avg occupancy ({W_g}{H_g} grid)")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
