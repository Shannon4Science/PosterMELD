from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common.io import atomic_write_json, atomic_write_text, load_manifest, mean, now, read_json, result_path
from universal_score.evaluate import load_checklist


def fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def main() -> int:
    module_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Summarize Universal Score results.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--result-dir", default="outputs/universal")
    parser.add_argument("--report-dir", default="reports/universal")
    parser.add_argument("--checklist", type=Path, default=module_root / "checklist.yaml")
    args = parser.parse_args()

    items = load_manifest(args.manifest)
    result_dir = Path(args.result_dir).resolve()
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    checklist = load_checklist(args.checklist.resolve())
    rows: list[dict[str, Any]] = []
    for item in items:
        path = result_path(result_dir, item, "universal.json")
        record = read_json(path, {})
        scores = record.get("scores") if isinstance(record, dict) else []
        by_index = {
            int(score["criterion_index"]): float(score["score"])
            for score in scores or []
            if isinstance(score, dict) and "criterion_index" in score and "score" in score
        }
        row = {
            "id": item.id,
            "method": item.method,
            "subset": item.subset,
            "paper_name": item.paper_name,
            "status": record.get("status", "not_run") if isinstance(record, dict) else "not_run",
            "universal_score": record.get("universal_score") if isinstance(record, dict) else None,
            "xgboost_score": record.get("xgboost_score") if isinstance(record, dict) else None,
            **{f"criterion_{index}": by_index.get(index) for index in range(1, 11)},
            "result_path": str(path),
        }
        rows.append(row)

    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    method_order: list[str] = []
    for row in rows:
        if row["method"] not in by_method:
            method_order.append(row["method"])
        by_method[row["method"]].append(row)

    methods: dict[str, Any] = {}
    for method in method_order:
        selected = by_method[method]
        methods[method] = {
            "total": len(selected),
            "valid": sum(isinstance(row["universal_score"], (int, float)) for row in selected),
            "missing_as_zero": sum(row["status"] == "missing_poster" for row in selected),
            "universal_score": mean(float(row["universal_score"]) for row in selected if isinstance(row["universal_score"], (int, float))),
            "xgboost_score": mean(float(row["xgboost_score"]) for row in selected if isinstance(row["xgboost_score"], (int, float))),
            "criteria": {
                str(index): mean(float(row[f"criterion_{index}"]) for row in selected if isinstance(row[f"criterion_{index}"], (int, float)))
                for index in range(1, 11)
            },
            "status_counts": dict(Counter(row["status"] for row in selected)),
        }

    summary = {
        "schema_version": 1,
        "created_at": now(),
        "definition": {
            "universal_score": "arithmetic mean of the ten 0-5 criterion scores",
            "missing_policy": "missing poster receives zero on all ten criteria",
            "xgboost_feature_order": checklist,
        },
        "total": len(rows),
        "methods": methods,
    }
    atomic_write_json(report_dir / "summary.json", summary)

    with (report_dir / "per_sample.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    method_fields = ["method", "total", "valid", "missing_as_zero", "universal_score", "xgboost_score", *[f"criterion_{i}" for i in range(1, 11)]]
    with (report_dir / "method_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=method_fields)
        writer.writeheader()
        for method in method_order:
            value = methods[method]
            writer.writerow(
                {
                    "method": method,
                    "total": value["total"],
                    "valid": value["valid"],
                    "missing_as_zero": value["missing_as_zero"],
                    "universal_score": value["universal_score"],
                    "xgboost_score": value["xgboost_score"],
                    **{f"criterion_{i}": value["criteria"][str(i)] for i in range(1, 11)},
                }
            )

    lines = [
        "# Universal Score",
        "",
        "All VLM scores use a 0--5 scale. Missing posters receive zero on all ten dimensions.",
        "",
        "| Method | Universal Score | XGBoost Score | Valid / Total |",
        "|---|---:|---:|---:|",
    ]
    for method in method_order:
        value = methods[method]
        lines.append(f"| {method} | {fmt(value['universal_score'])} | {fmt(value['xgboost_score'])} | {value['valid']} / {value['total']} |")
    lines += ["", "## Ten Dimensions", ""]
    lines.append("| Dimension | " + " | ".join(method_order) + " |")
    lines.append("|---|" + "---:|" * len(method_order))
    for index, description in enumerate(checklist, start=1):
        name = description.split(" - ", 1)[0]
        values = " | ".join(fmt(methods[method]["criteria"][str(index)]) for method in method_order)
        lines.append(f"| {name} | {values} |")
    atomic_write_text(report_dir / "universal_results.md", "\n".join(lines) + "\n")
    print(report_dir / "universal_results.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
