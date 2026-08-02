from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common.io import atomic_write_json, atomic_write_text, load_manifest, now, read_json
from keypoint_bertscore.state import BERTSCORE_NUM_LAYERS, score_path


METRICS = ("bert_precision", "bert_recall", "bert_f1")


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "median": None, "min": None, "max": None}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize keypoint BERTScore results.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--result-dir", default="outputs/keypoint_bertscore")
    parser.add_argument("--report-dir", default="reports/keypoint_bertscore")
    args = parser.parse_args()

    items = load_manifest(args.manifest)
    result_dir = Path(args.result_dir).resolve()
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for item in items:
        path = score_path(result_dir, item)
        result = read_json(path, {})
        success = isinstance(result, dict) and result.get("status") == "success"
        rows.append(
            {
                "id": item.id,
                "method": item.method,
                "subset": item.subset,
                "paper_name": item.paper_name,
                "status": result.get("status", "not_run") if isinstance(result, dict) else "not_run",
                "bert_precision": float(result.get("bert_precision", 0.0)) if success else 0.0,
                "bert_recall": float(result.get("bert_recall", 0.0)) if success else 0.0,
                "bert_f1": float(result.get("bert_f1", 0.0)) if success else 0.0,
                "result_path": str(path),
            }
        )

    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    method_order: list[str] = []
    for row in rows:
        if row["method"] not in by_method:
            method_order.append(row["method"])
        by_method[row["method"]].append(row)

    methods = []
    for method in method_order:
        selected = by_method[method]
        successful_rows = [row for row in selected if row["status"] == "success"]
        methods.append(
            {
                "method": method,
                "expected": len(selected),
                "successful": len(successful_rows),
                "coverage": len(successful_rows) / len(selected) if selected else 0.0,
                "status_counts": dict(Counter(row["status"] for row in selected)),
                "available_only": {metric: stats([row[metric] for row in successful_rows]) for metric in METRICS},
                "full_benchmark_missing_as_zero": {metric: stats([row[metric] for row in selected]) for metric in METRICS},
            }
        )

    summary = {
        "schema_version": 1,
        "created_at": now(),
        "evaluation": {
            "ocr_model": "opendatalab/MinerU2.5-Pro-2605-1.2B",
            "bertscore_model": "roberta-large",
            "num_layers": BERTSCORE_NUM_LAYERS,
            "idf": False,
            "rescale_with_baseline": False,
            "reference": "paper_poster_keypoints concatenated in reading_order",
        },
        "total": len(rows),
        "method_summaries": methods,
    }
    atomic_write_json(report_dir / "summary.json", summary)

    with (report_dir / "per_sample.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fields = ["method", "expected", "successful", "coverage", "precision", "recall", "f1", "available_f1"]
    with (report_dir / "method_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in methods:
            full = item["full_benchmark_missing_as_zero"]
            available = item["available_only"]
            writer.writerow(
                {
                    "method": item["method"],
                    "expected": item["expected"],
                    "successful": item["successful"],
                    "coverage": item["coverage"],
                    "precision": full["bert_precision"]["mean"],
                    "recall": full["bert_recall"]["mean"],
                    "f1": full["bert_f1"]["mean"],
                    "available_f1": available["bert_f1"]["mean"],
                }
            )

    lines = [
        "# Keypoint BERTScore",
        "",
        "Poster text is extracted with MinerU2.5-Pro. References are ordered paper keypoints. Missing or failed posters receive zero in the full-benchmark aggregate.",
        "",
        "| Method | Successful / Expected | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in methods:
        full = item["full_benchmark_missing_as_zero"]
        lines.append(
            f"| {item['method']} | {item['successful']} / {item['expected']} | "
            f"{full['bert_precision']['mean']:.6f} | {full['bert_recall']['mean']:.6f} | {full['bert_f1']['mean']:.6f} |"
        )
    atomic_write_text(report_dir / "keypoint_bertscore_results.md", "\n".join(lines) + "\n")
    print(report_dir / "keypoint_bertscore_results.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
