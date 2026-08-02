from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common.io import atomic_write_json, atomic_write_text, load_manifest, mean, now, read_json, result_path


def fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize PRR and conditional CHE results.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--result-dir", default="outputs/prr_che")
    parser.add_argument("--report-dir", default="reports/prr_che")
    args = parser.parse_args()

    items = load_manifest(args.manifest)
    result_dir = Path(args.result_dir).resolve()
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for item in items:
        path = result_path(result_dir, item, "prr_che.json")
        record = read_json(path, {})
        prr = record.get("prr") if isinstance(record, dict) else {}
        prr_parsed = prr.get("parsed") if isinstance(prr, dict) else {}
        che = record.get("che") if isinstance(record, dict) else {}
        che_parsed = che.get("parsed") if isinstance(che, dict) else {}
        che_parsed = che_parsed if isinstance(che_parsed, dict) else {}
        rows.append(
            {
                "id": item.id,
                "method": item.method,
                "subset": item.subset,
                "paper_name": item.paper_name,
                "pipeline_status": record.get("pipeline_status", "not_run") if isinstance(record, dict) else "not_run",
                "print_ready": prr_parsed.get("print_ready") if isinstance(prr_parsed, dict) else None,
                "craftsmanship": (che_parsed.get("craftsmanship") or {}).get("score"),
                "harmony": (che_parsed.get("harmony") or {}).get("score"),
                "expressiveness": (che_parsed.get("expressiveness") or {}).get("score"),
                "che_score": che_parsed.get("che_score"),
                "result_path": str(path),
            }
        )

    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    method_order: list[str] = []
    for row in rows:
        if row["method"] not in by_method:
            method_order.append(row["method"])
        by_method[row["method"]].append(row)

    methods: dict[str, Any] = {}
    for method in method_order:
        selected = by_method[method]
        ready = [row for row in selected if row["print_ready"] is True]
        prr_valid = [row for row in selected if isinstance(row["print_ready"], bool)]
        che_valid = [row for row in ready if isinstance(row["che_score"], (int, float))]
        methods[method] = {
            "total": len(selected),
            "prr_valid": len(prr_valid),
            "print_ready": len(ready),
            "print_ready_rate": len(ready) / len(selected) if selected else None,
            "che_expected": len(ready),
            "che_valid": len(che_valid),
            "craftsmanship_mean": mean(float(row["craftsmanship"]) for row in che_valid),
            "harmony_mean": mean(float(row["harmony"]) for row in che_valid),
            "expressiveness_mean": mean(float(row["expressiveness"]) for row in che_valid),
            "che_mean": mean(float(row["che_score"]) for row in che_valid),
            "status_counts": dict(Counter(row["pipeline_status"] for row in selected)),
        }

    summary = {
        "schema_version": 1,
        "created_at": now(),
        "definition": {
            "prr_denominator": "all manifest items; missing posters are false",
            "che_condition": "print_ready=true",
            "che_formula": "(craftsmanship + harmony + expressiveness) / 3",
        },
        "total": len(rows),
        "methods": methods,
    }
    atomic_write_json(report_dir / "summary.json", summary)

    fields = list(rows[0])
    with (report_dir / "per_sample.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    method_fields = ["method", *next(iter(methods.values())).keys()]
    with (report_dir / "method_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=method_fields)
        writer.writeheader()
        for method in method_order:
            writer.writerow({"method": method, **methods[method]})

    lines = [
        "# Print-Ready Rate and Conditional CHE",
        "",
        "Missing posters count as not print-ready. CHE is averaged only over posters with PRR=true.",
        "",
        "| Method | Print-Ready Rate | Craftsmanship | Harmony | Expressiveness | CHE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in method_order:
        value = methods[method]
        lines.append(
            f"| {method} | {fmt(value['print_ready_rate'])} | {fmt(value['craftsmanship_mean'])} | "
            f"{fmt(value['harmony_mean'])} | {fmt(value['expressiveness_mean'])} | {fmt(value['che_mean'])} |"
        )
    atomic_write_text(report_dir / "prr_che_results.md", "\n".join(lines) + "\n")
    print(report_dir / "prr_che_results.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
