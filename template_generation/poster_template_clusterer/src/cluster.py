"""
cluster.py   +  + 
"""

from typing import Dict, List, Tuple

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score


def standardize_and_weight(X: np.ndarray,
                           segments: Dict[str, Tuple[int, int]],
                           occupancy_weight: float = 1.5) -> Tuple[np.ndarray, dict]:
    """
     z-score  occupancy 
     (X_processed, stats)stats  mean/std 
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std_safe = np.where(std < 1e-8, 1.0, std)
    Xn = (X - mean) / std_safe

    weights = np.ones(X.shape[1], dtype=np.float32)
    if "occupancy" in segments:
        s, e = segments["occupancy"]
        weights[s:e] *= occupancy_weight
    Xw = Xn * weights

    return Xw, {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weights": weights.tolist(),
    }


def cluster_ward(X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ward  (labels in [0,k-1], linkage_matrix)
    """
    Z = linkage(X, method="ward")
    labels = fcluster(Z, t=k, criterion="maxclust") - 1
    return labels.astype(np.int32), Z


def representatives(X: np.ndarray, labels: np.ndarray,
                    names: List[str], top_n: int = 6) -> Dict[int, dict]:
    """
     top_n 
     {cluster_id: {"size", "members": [...],
                       "member_dists": [...], "reps": [(name, dist)...]}}.

    members 
     M 
    """
    out: Dict[int, dict] = {}
    for c in sorted(set(labels.tolist())):
        idxs = np.where(labels == c)[0]
        if len(idxs) == 0:
            continue
        sub = X[idxs]
        centroid = sub.mean(axis=0)
        dists = np.linalg.norm(sub - centroid, axis=1)
        order = np.argsort(dists)
        sorted_idxs = idxs[order]
        sorted_dists = dists[order]
        rep = [(names[sorted_idxs[i]], float(sorted_dists[i]))
               for i in range(min(top_n, len(sorted_idxs)))]
        out[int(c)] = {
            "size": int(len(idxs)),
            "members": [names[i] for i in sorted_idxs.tolist()],
            "member_dists": [float(d) for d in sorted_dists.tolist()],
            "reps": rep,
        }
    return out


def silhouette_like(X: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score (sklearn )"""
    if len(set(labels.tolist())) < 2 or X.shape[0] < 3:
        return 0.0
    return float(silhouette_score(X, labels, metric="euclidean"))
