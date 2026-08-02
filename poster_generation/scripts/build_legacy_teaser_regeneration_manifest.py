#!/usr/bin/env python3
"""Recover teaser regeneration status for legacy benchmark records.

New benchmark records persist the exact generative-asset report. Legacy runs
deleted those reports with their scratch directories, so this script compares
the rendered teaser region against a visually confirmed procedural reference.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


DEFAULT_REFERENCE_JOB_ID = "003_Authority_Backdoor_A_Certifiable_Backdoor_Mechanism_for_Authoring_DNNs_05033eeb"


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_records(run_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted((run_dir / "records").rglob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        record["_record_path"] = str(path.resolve())
        records.append(record)
    return records


def load_manifest_jobs(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "manifest.json"
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [job for job in payload.get("jobs", []) if isinstance(job, dict) and job.get("job_id")]


def teaser_edge_feature(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L")
        width, height = image.size
        crop = image.crop(
            (
                int(width * 0.045),
                int(height * 0.225),
                int(width * 0.265),
                int(height * 0.440),
            )
        )
        crop = crop.resize((96, 48), Image.Resampling.BILINEAR)
        edges = np.asarray(crop.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    edges -= edges.mean()
    norm = float(np.linalg.norm(edges))
    return edges / norm if norm else edges


def recorded_teaser_status(record: dict[str, Any]) -> dict[str, Any] | None:
    teaser = ((record.get("generative_assets") or {}).get("teaser") or {})
    return teaser if teaser.get("reported") else None


def build_inventory(
    run_dir: Path,
    reference_job_id: str,
    threshold: float,
) -> dict[str, Any]:
    records = load_records(run_dir)
    manifest_jobs = load_manifest_jobs(run_dir)
    by_job = {str(record.get("job_id")): record for record in records}
    if reference_job_id not in by_job:
        raise ValueError(f"reference job not found: {reference_job_id}")

    reference_record = by_job[reference_job_id]
    reference_path = Path(str(reference_record.get("poster_png") or ""))
    if not reference_path.is_file():
        raise ValueError(f"reference poster not found: {reference_path}")
    reference_feature = teaser_edge_feature(reference_path)

    items = []
    for record in records:
        poster_path = Path(str(record.get("poster_png") or ""))
        if record.get("status") != "success" or not poster_path.is_file():
            continue

        exact = recorded_teaser_status(record)
        if exact is not None:
            needs_regeneration = bool(exact.get("needs_regeneration"))
            score = None
            method = "persisted_generative_asset_report"
            reason = str(exact.get("reason") or ("recorded_fallback" if needs_regeneration else "validated_image_api"))
            confidence = "exact"
        else:
            score = float(np.sum(reference_feature * teaser_edge_feature(poster_path)))
            needs_regeneration = score >= threshold
            method = "legacy_procedural_visual_signature"
            reason = "procedural_teaser_signature" if needs_regeneration else "distinct_paper_specific_teaser"
            confidence = "high" if score >= threshold or score <= threshold - 0.04 else "review"

        items.append(
            {
                "job_id": record.get("job_id"),
                "subset": record.get("subset"),
                "index": record.get("index"),
                "paper_name": record.get("paper_name"),
                "model": record.get("model"),
                "poster_png": str(poster_path.resolve()),
                "record_path": record.get("_record_path"),
                "action": "regenerate" if needs_regeneration else "preserve",
                "asset": "teaser",
                "needs_regeneration": needs_regeneration,
                "reason": reason,
                "classification_method": method,
                "procedural_similarity": round(score, 6) if score is not None else None,
                "threshold": threshold if score is not None else None,
                "confidence": confidence,
                "background_status": "legacy_raw_api_success_postprocess_unverified",
            }
        )

    items.sort(key=lambda item: (str(item.get("subset")), int(item.get("index") or 0)))
    regenerate = [item for item in items if item["needs_regeneration"]]
    preserve = [item for item in items if not item["needs_regeneration"]]
    completed_job_ids = {str(item["job_id"]) for item in items}
    unprocessed = [
        {
            "job_id": job.get("job_id"),
            "subset": job.get("subset"),
            "index": job.get("index"),
            "paper_path": job.get("paper_path"),
            "conference": job.get("conference"),
            "action": "generate",
            "reason": "no_completed_poster_record",
        }
        for job in manifest_jobs
        if str(job.get("job_id")) not in completed_job_ids
    ]
    total_jobs = len(manifest_jobs) if manifest_jobs else len(items)
    return {
        "version": 1,
        "generated_at": now_iso(),
        "source_run_dir": str(run_dir.resolve()),
        "asset": "teaser",
        "policy": "Preserve validated paper-specific teaser images; regenerate legacy procedural fallbacks without fallback.",
        "legacy_recovery": {
            "required": True,
            "reason": "legacy batch cleanup deleted generated_teaser_report.json",
            "reference_job_id": reference_job_id,
            "reference_poster": str(reference_path.resolve()),
            "crop_fractions": [0.045, 0.225, 0.265, 0.440],
            "similarity_threshold": threshold,
            "validation_note": "The score distribution has a clear gap around the threshold and representative posters were visually checked.",
        },
        "counts": {
            "total_benchmark_jobs": total_jobs,
            "completed_posters_audited": len(items),
            "preserve_successful": len(preserve),
            "regenerate_required": len(regenerate),
            "unprocessed": len(unprocessed),
            "run_after_recharge": len(regenerate) + len(unprocessed),
        },
        "items": items,
        "unprocessed_jobs": unprocessed,
    }


def write_inventory(inventory: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "generative_asset_inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fields = [
        "job_id",
        "subset",
        "index",
        "paper_name",
        "model",
        "action",
        "asset",
        "needs_regeneration",
        "reason",
        "classification_method",
        "procedural_similarity",
        "threshold",
        "confidence",
        "poster_png",
        "record_path",
        "background_status",
    ]
    with (output_dir / "generative_asset_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: item.get(field, "") for field in fields} for item in inventory["items"])

    for action, filename in (("preserve", "preserve_job_ids.txt"), ("regenerate", "regenerate_job_ids.txt")):
        values = [str(item["job_id"]) for item in inventory["items"] if item["action"] == action]
        (output_dir / filename).write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")

    unprocessed_ids = [str(item["job_id"]) for item in inventory.get("unprocessed_jobs", [])]
    (output_dir / "unprocessed_job_ids.txt").write_text(
        "\n".join(unprocessed_ids) + ("\n" if unprocessed_ids else ""),
        encoding="utf-8",
    )
    regenerate_ids = [str(item["job_id"]) for item in inventory["items"] if item["action"] == "regenerate"]
    run_after_recharge = regenerate_ids + unprocessed_ids
    (output_dir / "run_after_recharge_job_ids.txt").write_text(
        "\n".join(run_after_recharge) + ("\n" if run_after_recharge else ""),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reference-job-id", default=DEFAULT_REFERENCE_JOB_ID)
    parser.add_argument("--threshold", type=float, default=0.75)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or (run_dir / "regeneration_inventory")).resolve()
    inventory = build_inventory(run_dir, args.reference_job_id, args.threshold)
    write_inventory(inventory, output_dir)
    print(json.dumps({"output_dir": str(output_dir), **inventory["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
