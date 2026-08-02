#!/usr/bin/env python3
"""Regenerate rejected benchmark teaser/background assets without rerunning LLM stages."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageStat

from src.template_extraction.block_template_registry import load_block_template_layout
from src.tools.image_api import ImageQuotaError, ImageTools
from src.tools.layout_api import LayoutTemplates
from src.utils.image_text_detector import detect_readable_text


POSTER_WIDTH_INCHES = 54.0
POSTER_HEIGHT_INCHES = 27.0
REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")


@dataclass(frozen=True)
class RepairJob:
    record_path: Path
    job_id: str
    subset: str
    poster_path: Path
    asset_report_path: Path
    teaser_needed: bool
    background_needed: bool


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def discover_jobs(run_dir: Path, selected_job_ids: set[str]) -> list[RepairJob]:
    jobs: list[RepairJob] = []
    for record_path in sorted((run_dir / "records").rglob("*.json")):
        record = load_json(record_path)
        job_id = str(record.get("job_id") or "")
        if selected_job_ids and job_id not in selected_job_ids:
            continue
        if record.get("status") != "success":
            continue
        assets = record.get("generative_assets") or {}
        teaser_needed = bool((assets.get("teaser") or {}).get("needs_regeneration"))
        background_needed = bool((assets.get("background") or {}).get("needs_regeneration"))
        if not teaser_needed and not background_needed:
            continue
        poster_path = Path(str(record.get("poster_png") or ""))
        report_path = Path(str(record.get("generative_asset_report") or ""))
        if not poster_path.is_file() or not report_path.is_file():
            continue
        jobs.append(
            RepairJob(
                record_path=record_path,
                job_id=job_id,
                subset=str(record.get("subset") or record_path.parent.name),
                poster_path=poster_path,
                asset_report_path=report_path,
                teaser_needed=teaser_needed,
                background_needed=background_needed,
            )
        )
    return jobs


def cover_resize(image: Image.Image, width: int, height: int) -> Image.Image:
    ratio = max(width / image.width, height / image.height)
    resized = image.resize(
        (max(1, int(round(image.width * ratio))), max(1, int(round(image.height * ratio)))),
        Image.Resampling.LANCZOS,
    )
    left = max(0, (resized.width - width) // 2)
    top = max(0, (resized.height - height) // 2)
    return resized.crop((left, top, left + width, top + height))


def is_placeholder(image: Image.Image) -> bool:
    stats = ImageStat.Stat(image.resize((32, 32)).convert("RGB"))
    mean = sum(stats.mean) / 3
    variance = sum(stats.var) / 3
    return variance < 18 and 175 <= mean <= 230


def strong_readable_text_report(path: Path, min_confidence: float) -> dict[str, Any]:
    report = detect_readable_text(path, min_confidence=min_confidence, timeout_seconds=20)
    tokens = report.get("tokens") or []
    strong = []
    for token in tokens:
        text = re.sub(r"[^A-Za-z0-9]", "", str(token.get("text") or ""))
        confidence = float(token.get("confidence") or 0.0)
        if len(text) >= 4 and confidence >= 82:
            strong.append(token)
        elif text.isdigit() and len(text) >= 3 and confidence >= 75:
            strong.append(token)
    rejected = len(tokens) >= 2 or bool(strong)
    return {
        **report,
        "rejected": rejected,
        "strong_tokens": strong,
        "reason": "readable_text_detected" if rejected else "",
    }


def generation_prompt(prompt: str, kind: str, attempt: int) -> str:
    if kind == "teaser":
        suffix = (
            " Asset-only repair: create one continuous scientific illustration with no panels, captions, labels, "
            "screens, signs, documents, code, UI, axes, legends, letters, digits, or typographic marks. "
            "Use only unlabeled objects, materials, spatial structures, light, color, and geometry."
        )
    else:
        suffix = (
            " Asset-only repair: output a continuous pale decorative texture only. Do not draw panels, headings, "
            "documents, screens, labels, glyphs, digits, logos, charts, or any poster-like composition."
        )
    if attempt > 1:
        suffix += f" This is regeneration attempt {attempt}; use a visibly different composition."
    return f"{prompt}{suffix}"


def generate_asset(
    report: dict[str, Any],
    kind: str,
    output_dir: Path,
    max_attempts: int,
    ocr_min_confidence: float,
) -> tuple[Path | None, dict[str, Any]]:
    prompt = str(report.get("prompt") or "")
    width = max(64, int(report.get("width_px") or (1292 if kind == "teaser" else 2035)))
    height = max(64, int(report.get("height_px") or (650 if kind == "teaser" else 1018)))
    attempts: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / f"{kind}.png"

    if final_path.is_file():
        with Image.open(final_path) as existing:
            placeholder = is_placeholder(existing)
        ocr = strong_readable_text_report(final_path, ocr_min_confidence)
        if not placeholder and not ocr.get("rejected"):
            return final_path, {
                "accepted": True,
                "attempts": [{"attempt": 0, "accepted": True, "reason": "reused_existing_asset"}],
                "prompt": prompt,
                "reused_existing": True,
            }

    for attempt in range(1, max_attempts + 1):
        raw_path = output_dir / f"raw_{kind}_attempt_{attempt}.png"
        raw_path.unlink(missing_ok=True)
        ImageTools().generate_image(
            generation_prompt(prompt, kind, attempt),
            width=width,
            height=height,
            output_path=str(raw_path),
        )
        if not raw_path.is_file():
            attempts.append({"attempt": attempt, "accepted": False, "reason": "missing_image"})
            continue
        with Image.open(raw_path) as image:
            placeholder = is_placeholder(image)
        ocr = strong_readable_text_report(raw_path, ocr_min_confidence)
        accepted = not placeholder and not ocr.get("rejected")
        attempts.append(
            {
                "attempt": attempt,
                "accepted": accepted,
                "reason": "placeholder" if placeholder else "readable_text_artifacts" if ocr.get("rejected") else "",
                "ocr_report": ocr,
                "raw_path": str(raw_path),
            }
        )
        if accepted:
            with Image.open(raw_path) as image:
                image = cover_resize(image.convert("RGB"), width, height)
                image.save(final_path)
            return final_path, {"accepted": True, "attempts": attempts, "prompt": prompt}

    final_path.unlink(missing_ok=True)
    return None, {"accepted": False, "attempts": attempts, "prompt": prompt}


def process_background(background_path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(background_path) as image:
        image = cover_resize(image.convert("RGB"), *size)
    image = ImageEnhance.Color(image).enhance(0.28)
    image = image.filter(ImageFilter.GaussianBlur(1.2))
    image = Image.blend(image, Image.new("RGB", size, "white"), 0.76)
    return image


def replace_light_background(poster: Image.Image, background: Image.Image) -> Image.Image:
    poster = poster.convert("RGB")
    array = np.asarray(poster, dtype=np.int16)
    minimum = array.min(axis=2)
    maximum = array.max(axis=2)
    neutral = np.clip((22 - (maximum - minimum)) / 16.0, 0.0, 1.0)
    light = np.clip((minimum - 222) / 30.0, 0.0, 1.0)
    replace = neutral * light

    gray = poster.convert("L")
    nearby_dark = gray.point(lambda value: 255 if value < 185 else 0).filter(ImageFilter.MaxFilter(7))
    preserve = np.asarray(nearby_dark, dtype=np.float32) / 255.0
    replace *= 1.0 - preserve
    mask = Image.fromarray(np.uint8(np.clip(replace * 255, 0, 255)), mode="L").filter(ImageFilter.GaussianBlur(1.0))
    return Image.composite(background, poster, mask)


def teaser_box(report: dict[str, Any], poster_size: tuple[int, int]) -> tuple[int, int, int, int]:
    geometry = report.get("geometry") or {}
    template_id = str(geometry.get("template_id") or "cluster_43_landscape")
    slot_id = str(geometry.get("slot_id") or "slot_1")
    poster_width, poster_height = poster_size
    page_height_inches = POSTER_WIDTH_INCHES * poster_height / max(poster_width, 1)
    layout = load_block_template_layout(template_id, POSTER_WIDTH_INCHES, page_height_inches)
    slots = (layout or {}).get("content_slots") or []
    if not slots:
        adaptive_layout = LayoutTemplates(POSTER_WIDTH_INCHES, page_height_inches).get_template(
            template_id,
            header_height=6.0,
        )
        slots = adaptive_layout.get("lanes") or adaptive_layout.get("columns") or []
    slot = next(
        item
        for item in slots
        if str(item.get("slot_id") or item.get("id")) == slot_id
    )
    target_width = min(float(geometry.get("target_width_inches") or slot["w"] - 0.6), float(slot["w"]) - 0.4)
    target_height = min(float(geometry.get("target_height_inches") or slot["h"] * 0.65), float(slot["h"]) - 2.0)
    x = float(slot["x"]) + (float(slot["w"]) - target_width) / 2
    y = float(slot["y"]) + 1.05
    return (
        int(round(x / POSTER_WIDTH_INCHES * poster_width)),
        int(round(y / page_height_inches * poster_height)),
        int(round((x + target_width) / POSTER_WIDTH_INCHES * poster_width)),
        int(round((y + target_height) / page_height_inches * poster_height)),
    )


def apply_teaser(poster: Image.Image, teaser_path: Path, report: dict[str, Any]) -> Image.Image:
    box = teaser_box(report, poster.size)
    width = max(1, box[2] - box[0])
    height = max(1, box[3] - box[1])
    with Image.open(teaser_path) as teaser:
        teaser = cover_resize(teaser.convert("RGB"), width, height)
    result = poster.convert("RGB")
    panel = Image.new("RGB", (width + 8, height + 8), "white")
    result.paste(panel, (max(0, box[0] - 4), max(0, box[1] - 4)))
    result.paste(teaser, (box[0], box[1]))
    return result


def remove_repaired_asset_failures(record: dict[str, Any], repaired_assets: set[str]) -> None:
    failures = []
    for failure in record.get("quality_gate_failures") or []:
        if not isinstance(failure, dict):
            continue
        if failure.get("category") == "generated_asset" and failure.get("asset") in repaired_assets:
            continue
        failures.append(failure)
    record["quality_gate_failures"] = failures
    record["quality_gate_accepted"] = not failures


def repair_job(
    job: RepairJob,
    run_dir: Path,
    max_attempts: int,
    ocr_min_confidence: float,
) -> dict[str, Any]:
    record = load_json(job.record_path)
    asset_report = load_json(job.asset_report_path)
    raw_reports = asset_report.get("raw_reports") or {}
    repair_dir = run_dir / "repaired_assets" / job.subset / job.job_id
    report_path = run_dir / "asset_repair_reports" / job.subset / f"{job.job_id}.json"
    backup_path = run_dir / "original_posters_before_asset_repair" / job.subset / f"{job.job_id}.png"
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if not backup_path.exists():
        shutil.copy2(job.poster_path, backup_path)

    result: dict[str, Any] = {
        "job_id": job.job_id,
        "subset": job.subset,
        "started_at": now_iso(),
        "original_poster": str(backup_path),
        "poster_png": str(job.poster_path),
        "assets": {},
    }
    repaired_assets: set[str] = set()
    generated_paths: dict[str, Path] = {}

    for kind, needed in (("background", job.background_needed), ("teaser", job.teaser_needed)):
        if not needed:
            continue
        generated, generation_report = generate_asset(
            raw_reports.get(kind) or {},
            kind,
            repair_dir,
            max_attempts,
            ocr_min_confidence,
        )
        result["assets"][kind] = generation_report
        if generated:
            generated_paths[kind] = generated
            repaired_assets.add(kind)

    with Image.open(backup_path) as source:
        poster = source.convert("RGB")
    if "background" in generated_paths:
        background = process_background(generated_paths["background"], poster.size)
        poster = replace_light_background(poster, background)
    if "teaser" in generated_paths:
        poster = apply_teaser(poster, generated_paths["teaser"], raw_reports.get("teaser") or {})
    if repaired_assets:
        temporary = job.poster_path.with_name(f".{job.poster_path.name}.asset-repair.tmp.png")
        poster.save(temporary)
        temporary.replace(job.poster_path)

    assets = record.get("generative_assets")
    if not isinstance(assets, dict):
        assets = {}
        record["generative_assets"] = assets
    for kind in repaired_assets:
        previous = assets.get(kind) or {}
        assets[kind] = {
            **previous,
            "reported": True,
            "enabled": True,
            "applied": True,
            "asset_source": "image_api_asset_only_repair",
            "degraded": False,
            "needs_regeneration": False,
            "used_procedural_fallback": False,
            "reason": "",
            "generation_attempt_count": len(result["assets"][kind].get("attempts") or []),
            "repair_report": str(report_path),
        }
    remove_repaired_asset_failures(record, repaired_assets)
    asset_only_repairs = record.get("asset_only_repairs")
    if not isinstance(asset_only_repairs, list):
        asset_only_repairs = []
        record["asset_only_repairs"] = asset_only_repairs
    asset_only_repairs.append(
        {
            "repaired_at": now_iso(),
            "assets": sorted(repaired_assets),
            "report": str(report_path),
            "poster_backup": str(backup_path),
        }
    )
    atomic_json(job.record_path, record)
    result["finished_at"] = now_iso()
    result["repaired_assets"] = sorted(repaired_assets)
    result["remaining_assets"] = sorted(
        kind for kind, needed in (("teaser", job.teaser_needed), ("background", job.background_needed)) if needed and kind not in repaired_assets
    )
    atomic_json(report_path, result)
    return result


def update_run_summary(run_dir: Path) -> None:
    records = [load_json(path) for path in (run_dir / "records").rglob("*.json")]
    summary_path = run_dir / "summary.json"
    summary = load_json(summary_path)
    summary.update(
        {
            "updated_at": now_iso(),
            "success": sum(record.get("status") == "success" for record in records),
            "failed": sum(record.get("status") == "failed" for record in records),
            "quality_rejected": sum(
                record.get("status") == "success" and record.get("quality_gate_accepted") is False for record in records
            ),
            "teaser_needs_regeneration": sum(
                bool(((record.get("generative_assets") or {}).get("teaser") or {}).get("needs_regeneration"))
                for record in records
            ),
            "background_needs_regeneration": sum(
                bool(((record.get("generative_assets") or {}).get("background") or {}).get("needs_regeneration"))
                for record in records
            ),
        }
    )
    atomic_json(summary_path, summary)


def write_repair_summary(run_dir: Path, jobs: list[RepairJob], results: list[dict[str, Any]]) -> None:
    payload = {
        "updated_at": now_iso(),
        "jobs_selected": len(jobs),
        "jobs_completed": len(results),
        "jobs_fully_repaired": sum(not item.get("remaining_assets") for item in results),
        "jobs_still_needing_repair": sum(bool(item.get("remaining_assets")) for item in results),
        "assets_repaired": {
            "teaser": sum("teaser" in (item.get("repaired_assets") or []) for item in results),
            "background": sum("background" in (item.get("repaired_assets") or []) for item in results),
        },
        "results": results,
    }
    atomic_json(run_dir / "asset_repair_summary.json", payload)
    csv_path = run_dir / "asset_repair_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["subset", "job_id", "repaired_assets", "remaining_assets", "poster_png"],
        )
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "subset": item.get("subset"),
                    "job_id": item.get("job_id"),
                    "repaired_assets": ",".join(item.get("repaired_assets") or []),
                    "remaining_assets": ",".join(item.get("remaining_assets") or []),
                    "poster_png": item.get("poster_png"),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--ocr-min-confidence", type=float, default=70.0)
    parser.add_argument("--max-papers", type=int, default=0)
    parser.add_argument("--job-id", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    jobs = discover_jobs(run_dir, set(args.job_id))
    if args.max_papers > 0:
        jobs = jobs[: args.max_papers]
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "jobs": len(jobs),
                "teaser": sum(job.teaser_needed for job in jobs),
                "background": sum(job.background_needed for job in jobs),
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        return 0

    results: list[dict[str, Any]] = []
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency), thread_name_prefix="asset-repair") as executor:
            futures = {
                executor.submit(repair_job, job, run_dir, max(1, args.max_attempts), args.ocr_min_confidence): job
                for job in jobs
            }
            for future in as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(
                        f"[{now_iso()}] REPAIRED {job.job_id} assets={result['repaired_assets']} remaining={result['remaining_assets']}",
                        flush=True,
                    )
                except ImageQuotaError as exc:
                    atomic_json(
                        run_dir / "ASSET_REPAIR_QUOTA_STOPPED.json",
                        {"stopped_at": now_iso(), "job_id": job.job_id, "reason": str(exc)},
                    )
                    for pending in futures:
                        pending.cancel()
                    raise
                except Exception as exc:
                    results.append(
                        {
                            "job_id": job.job_id,
                            "subset": job.subset,
                            "poster_png": str(job.poster_path),
                            "repaired_assets": [],
                            "remaining_assets": [
                                kind
                                for kind, needed in (("teaser", job.teaser_needed), ("background", job.background_needed))
                                if needed
                            ],
                            "error": str(exc),
                        }
                    )
                    print(f"[{now_iso()}] ERROR {job.job_id}: {exc}", flush=True)
    finally:
        write_repair_summary(run_dir, jobs, results)
        update_run_summary(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
