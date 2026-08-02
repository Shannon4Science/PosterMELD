"""End-to-end entry point for layout clustering and template extraction.

Examples:
    python run.py
    python run.py --k 5
    python run.py --k-sweep 3 4 5 6
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from src.features import build_feature_matrix, feature_segments
from src.cluster import (
    standardize_and_weight,
    cluster_ward,
    representatives,
    silhouette_like,
)
from src.template_proto import extract_template
from src.visualize import (
    pca_2d, scatter_plot, cluster_grids,
    render_template,
)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(rel_or_abs: str, base: Path) -> str:
    p = Path(rel_or_abs)
    if not p.is_absolute():
        p = (base / p).resolve()
    return str(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    ap.add_argument("--k", type=int, default=None, help="Override default_k")
    ap.add_argument("--k-sweep", type=int, nargs="*", default=None,
                    help="Sweep multiple K values and write reports for each.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    json_dir = resolve(cfg["input"]["blocks_json_dir"], SCRIPT_DIR)
    posters_dir = resolve(cfg["input"]["posters_dir"], SCRIPT_DIR)
    out_dir = resolve(cfg["output"]["dir"], SCRIPT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    grid_size = int(cfg["features"]["occupancy_grid_size"])
    occ_w = float(cfg["features"]["occupancy_weight"])

    print(f"[run] Reading block JSON files from: {json_dir}")
    names, X, sizes, block_counts, fblocks = build_feature_matrix(
        json_dir, grid_size)
    print(f"[run] N={len(names)} samples, D={X.shape[1]} features")

    segs = feature_segments(grid_size)
    Xw, stats = standardize_and_weight(X, segs, occupancy_weight=occ_w)

    # Reuse the same PCA projection for all K values.
    X2 = pca_2d(Xw)

    if args.k_sweep:
        ks = list(args.k_sweep)
    else:
        ks = [args.k or int(cfg["cluster"]["default_k"])]

    sweep_summary = []
    for k in ks:
        print(f"\n[run] ==== K = {k} ====")
        labels, Z = cluster_ward(Xw, k)
        sil = silhouette_like(Xw, labels)
        reps = representatives(Xw, labels, names,
                               top_n=int(cfg["viz"]["num_representatives"]))
        print(f"[run] silhouette-like = {sil:.3f}")
        for cid, info in reps.items():
            print(f"  cluster {cid}: size={info['size']}, "
                  f"reps={[n for n,_ in info['reps'][:3]]}")

        k_dir = os.path.join(out_dir, f"k{k}")
        os.makedirs(k_dir, exist_ok=True)

        # 1. labels json
        labels_json = {
            "k": k,
            "silhouette_like": sil,
            "assignments": {names[i]: int(labels[i]) for i in range(len(names))},
            "cluster_sizes": {int(c): int((labels == c).sum())
                              for c in sorted(set(labels.tolist()))},
        }
        with open(os.path.join(k_dir, "clusters.json"), "w", encoding="utf-8") as f:
            json.dump(labels_json, f, ensure_ascii=False, indent=2)

        # 2. cluster summary
        summary = {
            "k": k,
            "silhouette_like": sil,
            "clusters": {
                str(cid): {
                    "size": info["size"],
                    "representatives": info["reps"],
                    "all_members": info["members"],
                } for cid, info in reps.items()
            },
        }
        with open(os.path.join(k_dir, "cluster_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # 3. scatter
        scatter_plot(X2, labels,
                     os.path.join(k_dir, "scatter_pca2.png"),
                     title=f"Poster clusters (K={k}, ward)")

        # 4. Representative poster grids.
        cluster_grids(reps, posters_dir,
                      os.path.join(k_dir, "cluster_grids"),
                      thumb_max_side=int(cfg["viz"]["thumb_max_side"]))

        # 5. Template prototype extraction.
        tpl_cfg = cfg.get("template", {})
        iou_thr = float(tpl_cfg.get("slot_iou_threshold", 0.30))
        min_freq = float(tpl_cfg.get("slot_min_freq", 0.30))
        min_keep = float(tpl_cfg.get("min_keep_ratio", 0.40))
        morph_gap = float(tpl_cfg.get("morph_gap", 10.0))
        canvas_side = int(tpl_cfg.get("canvas_size", 1000))
        top_m = int(tpl_cfg.get("top_m_for_template", 20))

        templates_dir = os.path.join(k_dir, "templates")
        os.makedirs(templates_dir, exist_ok=True)

        md_lines = [f"# Templates (K={k})\n",
                    f"silhouette = {sil:.3f}\n",
                    f"top_m_for_template = {top_m}\n"]
        for cid, info in reps.items():
            # Members are sorted by distance to the cluster centroid.
            members_sorted = info["members"]
            used_members = members_sorted[:top_m]
            json_paths = [os.path.join(json_dir, f"{m}.json") for m in used_members]
            json_paths = [p for p in json_paths if os.path.exists(p)]
            tpl = extract_template(json_paths,
                                   iou_threshold=iou_thr,
                                   min_freq=min_freq,
                                   min_keep_ratio=min_keep,
                                   morph_gap=morph_gap)

            with open(os.path.join(templates_dir, f"cluster_{cid}_template.json"),
                      "w", encoding="utf-8") as f:
                json.dump(tpl, f, ensure_ascii=False, indent=2)

            layout_png = os.path.join(templates_dir, f"cluster_{cid}_layout.png")
            render_template(tpl, layout_png, canvas_long_side=canvas_side)

            print(f"[run] template cluster {cid}: "
                  f"used {len(json_paths)}/{info['size']} posters (top-{top_m}), "
                  f"slots={tpl['num_slots']}, aspect={tpl['aspect_ratio']:.2f}")

            md_lines.append(f"## Cluster {cid}  (cluster_size={info['size']}, "
                            f"used={len(json_paths)}, slots={tpl['num_slots']})\n")
            md_lines.append(f"![layout](templates/cluster_{cid}_layout.png)\n")
            md_lines.append(f"![heatmap](templates/cluster_{cid}_heatmap.png)\n")
            md_lines.append("\n| slot | freq | bbox (x1,y1,x2,y2) |")
            md_lines.append("|---|---|---|")
            for s in tpl["slots"]:
                md_lines.append(
                    f"| {s['slot_id']} | {s['frequency']:.0%} | "
                    f"({s['bbox'][0]:.0f}, {s['bbox'][1]:.0f}, "
                    f"{s['bbox'][2]:.0f}, {s['bbox'][3]:.0f}) |")
            md_lines.append("")

        with open(os.path.join(k_dir, "templates.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        sweep_summary.append({"k": k, "silhouette_like": sil,
                              "sizes": labels_json["cluster_sizes"]})

    # Global metadata.
    with open(os.path.join(out_dir, "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "num_samples": len(names),
            "feature_dim": int(X.shape[1]),
            "feature_segments": {k: list(v) for k, v in segs.items()},
            "standardization": {
                "mean_dim": len(stats["mean"]),
                "weights_dim": len(stats["weights"]),
            },
            "sweep": sweep_summary,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[run] Done. Output directory: {out_dir}")


if __name__ == "__main__":
    main()
