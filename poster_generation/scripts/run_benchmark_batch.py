#!/usr/bin/env python3
"""Run PosterMELD over benchmark subsets with bounded local concurrency.

The runner isolates every paper in a scratch directory, copies only the final
PNG and audit logs, then deletes all intermediate pipeline artifacts. Existing
successful records are skipped, so the same command is safe to resume.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_SUBSETS = ("aaai2026", "cvpr2025", "neurips2025", "p2peval", "pairs")
DEFAULT_SUBSETS = ("aaai2026", "cvpr2025")
CONFERENCES = {
    "aaai2026": "AAAI 2026",
    "cvpr2025": "CVPR 2025",
    "neurips2025": "NeurIPS 2025",
    # p2peval and pairs aggregate papers from multiple venues. Leaving the
    # conference blank avoids attaching a confidently wrong venue logo.
    "p2peval": "",
    "pairs": "",
}
LOCAL_AFFILIATION_LOGO_NAMES = (
    "affiliation_logo.png",
    "affiliation-logo.png",
    "aff_logo.png",
    "aff.png",
    "affiliation_logo.jpg",
    "affiliation_logo.jpeg",
    "affiliation_logo.webp",
)


@dataclass(frozen=True)
class Job:
    subset: str
    index: int
    paper_path: Path
    job_id: str
    conference: str


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def slugify(value: str, limit: int = 88) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_") or "paper"
    return slug[:limit].rstrip("_")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def valid_png(path: Path) -> tuple[bool, dict[str, int]]:
    if not path.exists() or path.stat().st_size <= 0:
        return False, {}
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
        return width > 0 and height > 0, {"width": width, "height": height}
    except Exception:
        return False, {}


def classify_attempt(return_code: int, png_available: bool) -> tuple[str, bool]:
    """Separate poster generation success from the stricter quality gate result."""
    return ("success" if png_available else "failed", return_code == 0)


def summarize_generative_asset_report(report: dict[str, Any], asset: str) -> dict[str, Any]:
    """Keep audit-critical generation state without copying large prompts into records."""
    if not report:
        return {
            "asset": asset,
            "reported": False,
            "applied": False,
            "asset_source": "missing_report",
            "needs_regeneration": True,
            "used_procedural_fallback": False,
            "reason": "missing_report",
        }

    applied = bool(report.get("applied", True))
    used_fallback = bool(report.get("used_procedural_fallback", False))
    source = str(report.get("asset_source") or ("procedural" if used_fallback else "image_api"))
    reason = str(report.get("fallback_reason") or report.get("reason") or report.get("image_api_error") or "")
    needs_regeneration = bool(
        report.get("needs_regeneration", False)
        or not applied
        or used_fallback
        or source in {"none", "missing", "missing_report"}
    )
    return {
        "asset": asset,
        "reported": True,
        "enabled": bool(report.get("enabled", True)),
        "applied": applied,
        "asset_source": source,
        "degraded": bool(report.get("degraded", False)),
        "needs_regeneration": needs_regeneration,
        "used_procedural_fallback": used_fallback,
        "reason": reason,
        "generation_mode": str(report.get("generation_mode") or ""),
        "generation_attempt_count": int(report.get("generation_attempt_count", 0) or 0),
        "image_api_error": str(report.get("image_api_error") or ""),
    }


def discover_jobs(benchmark_root: Path, subsets: list[str]) -> list[Job]:
    jobs: list[Job] = []
    for subset in subsets:
        subset_root = benchmark_root / subset
        paper_paths = sorted(subset_root.glob("*/paper.pdf"), key=lambda path: path.parent.name.lower())
        for index, paper_path in enumerate(paper_paths, start=1):
            relative = str(paper_path.relative_to(benchmark_root))
            digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:8]
            job_id = f"{index:03d}_{slugify(paper_path.parent.name)}_{digest}"
            jobs.append(
                Job(
                    subset=subset,
                    index=index,
                    paper_path=paper_path.resolve(),
                    job_id=job_id,
                    conference=CONFERENCES.get(subset, ""),
                )
            )
    return jobs


def find_local_affiliation_logo(paper_path: Path) -> Path | None:
    """Return a pre-vetted local institution logo for a benchmark paper."""
    paper_dir = paper_path.resolve().parent
    for filename in LOCAL_AFFILIATION_LOGO_NAMES:
        candidate = paper_dir / filename
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate.resolve()

    logo_dir = paper_dir / "affiliation_logos"
    if logo_dir.is_dir():
        candidates = sorted(
            path.resolve()
            for path in logo_dir.iterdir()
            if path.is_file()
            and path.stat().st_size > 0
            and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        if len(candidates) == 1:
            return candidates[0]
    return None


def has_no_affiliation_logo_marker(paper_path: Path) -> bool:
    paper_dir = paper_path.resolve().parent
    return any((paper_dir / name).is_file() for name in ("no_affiliation_logo", ".no_affiliation_logo"))


class BatchRunner:
    def __init__(self, args: argparse.Namespace, jobs: list[Job]):
        self.args = args
        self.jobs = jobs
        self.run_dir = args.run_dir.resolve()
        self.records_dir = self.run_dir / "records"
        self.logs_dir = self.run_dir / "logs"
        self.tokens_dir = self.run_dir / "token_logs"
        self.asset_reports_dir = self.run_dir / "asset_reports"
        self.posters_dir = self.run_dir / "posters"
        self.work_dir = self.run_dir / ".work"
        self.summary_lock = threading.Lock()
        self.launch_lock = threading.Lock()
        self.last_launch = 0.0
        self.running: set[str] = set()

        for directory in (
            self.records_dir,
            self.logs_dir,
            self.tokens_dir,
            self.asset_reports_dir,
            self.posters_dir,
            self.work_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def record_path(self, job: Job) -> Path:
        return self.records_dir / job.subset / f"{job.job_id}.json"

    def poster_path(self, job: Job) -> Path:
        return self.posters_dir / job.subset / f"{job.job_id}.png"

    def read_record(self, job: Job) -> dict[str, Any]:
        path = self.record_path(job)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def is_complete(self, job: Job) -> bool:
        record = self.read_record(job)
        png_ok, _ = valid_png(self.poster_path(job))
        # Generated teaser/background failures are repaired image-only after the
        # main run. A valid poster must never trigger an expensive full rerun.
        return record.get("status") == "success" and png_ok

    def write_manifest(self) -> None:
        manifest = {
            "created_at": now_iso(),
            "repository": str(REPO_ROOT),
            "benchmark_root": str(self.args.benchmark_root.resolve()),
            "subsets": self.args.subsets,
            "paper_count": len(self.jobs),
            "concurrency": self.args.concurrency,
            "max_attempts_per_paper": self.args.max_attempts,
            "job_timeout_minutes": self.args.job_timeout_minutes,
            "model": self.args.model,
            "layout_template": self.args.layout_template,
            "poster_style": "navy_serif",
            "visual_density": self.args.visual_density,
            "keypoint_limit": self.args.keypoint_limit,
            "quality_review": {
                "vlm_layout_review": True,
                "visual_legibility_review": True,
                "block_vlm_review": True,
                "adaptive_column_width": True,
            },
            "generated_assets": {
                "background": self.args.enable_generated_background,
                "teaser": self.args.enable_generated_teaser,
                "accounting_note": "Image APIs are billed per generated image and are not included in text token totals.",
            },
            "retained_artifacts": [
                "final poster PNG",
                "per-attempt stdout log",
                "per-attempt token log",
                "per-attempt generative asset report",
                "per-paper record",
            ],
            "success_policy": (
                "A valid final PNG is retained as generation success. The pipeline return code and final quality "
                "gate outcome are recorded separately; quality rejection does not trigger a costly rerun."
            ),
            "token_accounting": "Successful API responses reported by text/vision model endpoints; attempts are accumulated per paper.",
            "jobs": [
                {
                    "job_id": job.job_id,
                    "subset": job.subset,
                    "index": job.index,
                    "paper_path": str(job.paper_path),
                    "conference": job.conference,
                    "local_affiliation_logo": str(find_local_affiliation_logo(job.paper_path) or ""),
                    "affiliation_logo_policy": (
                        "local"
                        if find_local_affiliation_logo(job.paper_path)
                        else "disabled"
                        if has_no_affiliation_logo_marker(job.paper_path)
                        else "automatic"
                    ),
                }
                for job in self.jobs
            ],
        }
        manifest_path = self.run_dir / "manifest.json"
        if not manifest_path.exists():
            atomic_json(manifest_path, manifest)

    def wait_for_resources(self) -> None:
        while True:
            free_gb = shutil.disk_usage(self.run_dir).free / (1024**3)
            if free_gb < self.args.min_free_disk_gb:
                print(
                    f"[{now_iso()}] PAUSED: free disk {free_gb:.1f}GB is below "
                    f"{self.args.min_free_disk_gb:.1f}GB",
                    flush=True,
                )
                time.sleep(60)
                continue
            break

        with self.launch_lock:
            delay = self.args.launch_stagger_seconds - (time.monotonic() - self.last_launch)
            if delay > 0:
                time.sleep(delay)
            self.last_launch = time.monotonic()

    def command_for(self, job: Job) -> list[str]:
        local_affiliation_logo = find_local_affiliation_logo(job.paper_path)
        disable_automatic_affiliation_logo = bool(local_affiliation_logo) or has_no_affiliation_logo_marker(job.paper_path)
        command = [
            str(REPO_ROOT / ".venv" / "bin" / "python"),
            "-m",
            "src.workflow.pipeline",
            str(job.paper_path),
            "--text_model",
            self.args.model,
            "--vision_model",
            self.args.model,
            "--vlm-model",
            self.args.model,
            "--layout-template",
            self.args.layout_template,
        ]
        if job.conference:
            command.extend(["--conference", job.conference])
        command.extend(
            [
                "--disable-affiliation-logos" if disable_automatic_affiliation_logo else "--enable-affiliation-logos",
                "--affiliation-logo-mode",
                "single",
                "--poster-style",
                "navy_serif",
            "--visual-density",
            self.args.visual_density,
                "--section-title-numbering",
                "off",
                "--header-seed",
                str(self.args.header_seed),
                "--enable-vlm-layout-review",
                "--enable-visual-legibility-review",
                "--enable-block-vlm-review",
                "--enable-adaptive-column-width",
            ]
        )
        command.append(
            "--enable-generated-background"
            if self.args.enable_generated_background
            else "--disable-generated-background"
        )
        command.append(
            "--enable-generated-teaser"
            if self.args.enable_generated_teaser
            else "--disable-generated-teaser"
        )
        if local_affiliation_logo:
            command.extend(["--aff-logo", str(local_affiliation_logo)])
        return command

    def environment(self, output_root: Path) -> dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "PYTHONPATH": str(REPO_ROOT),
                "PYTHONUNBUFFERED": "1",
                "PAPER2POSTER_GENERATED_BACKGROUND": "1" if self.args.enable_generated_background else "0",
                "PAPER2POSTER_GENERATED_TEASER": "1" if self.args.enable_generated_teaser else "0",
                "PAPER2POSTER_ALLOW_GENERATIVE_FALLBACK": "0",
                "PAPER2POSTER_OUTPUT_ROOT": str(output_root),
                "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
        if self.args.parser_max_chars > 0:
            env["PAPER2POSTER_PARSER_MAX_CHARS"] = str(self.args.parser_max_chars)
        if self.args.keypoint_limit > 0:
            env["PAPER2POSTER_KEYPOINT_LIMIT"] = str(self.args.keypoint_limit)
        return env

    def run_process(self, command: list[str], output_root: Path, log_path: Path) -> tuple[int, bool]:
        timed_out = False
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
            log_file.write(f"command: {' '.join(command)}\n")
            log_file.write(f"started_at: {now_iso()}\n\n")
            log_file.flush()
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=self.environment(output_root),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                return_code = process.wait(timeout=self.args.job_timeout_minutes * 60)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    return_code = process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    return_code = process.wait(timeout=15)
                log_file.write(f"\nTIMED OUT after {self.args.job_timeout_minutes} minutes\n")
        return return_code, timed_out

    def load_timing(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def collect_generative_asset_reports(
        self,
        pipeline_output: Path,
        job: Job,
        attempt_number: int,
    ) -> tuple[dict[str, dict[str, Any]], Path]:
        content_dir = pipeline_output / "content"
        teaser_report = self.load_timing(content_dir / "generated_teaser_report.json")
        background_report = self.load_timing(content_dir / "background_image_report.json")
        assets: dict[str, dict[str, Any]] = {}
        if self.args.enable_generated_teaser:
            assets["teaser"] = summarize_generative_asset_report(teaser_report, "teaser")
        if self.args.enable_generated_background:
            assets["background"] = summarize_generative_asset_report(background_report, "background")

        report_path = self.asset_reports_dir / job.subset / f"{job.job_id}_attempt_{attempt_number}.json"
        atomic_json(
            report_path,
            {
                "job_id": job.job_id,
                "subset": job.subset,
                "attempt": attempt_number,
                "captured_at": now_iso(),
                "assets": assets,
                "raw_reports": {
                    "teaser": teaser_report,
                    "background": background_report,
                },
            },
        )
        return assets, report_path

    def locate_final_png(self, output_dir: Path, paper_name: str) -> Path | None:
        exact = output_dir / f"{paper_name}.png"
        if valid_png(exact)[0]:
            return exact
        candidates = [
            path
            for path in sorted(output_dir.glob("*.png"))
            if "draft" not in path.stem.lower() and valid_png(path)[0]
        ]
        return candidates[-1] if candidates else None

    def run_job(self, job: Job) -> dict[str, Any]:
        existing = self.read_record(job)
        if self.is_complete(job):
            return existing

        attempts = list(existing.get("attempts") or [])
        if len(attempts) >= self.args.max_attempts:
            return existing

        with self.summary_lock:
            self.running.add(job.job_id)
            self.write_summary()

        record: dict[str, Any] = {
            "job_id": job.job_id,
            "subset": job.subset,
            "index": job.index,
            "paper_name": job.paper_path.parent.name,
            "paper_path": str(job.paper_path),
            "conference": job.conference,
            "model": self.args.model,
            "status": "running",
            "started_at": existing.get("started_at") or now_iso(),
            "attempts": attempts,
        }

        try:
            while len(record["attempts"]) < self.args.max_attempts:
                self.wait_for_resources()
                attempt_number = len(record["attempts"]) + 1
                attempt_dir = self.work_dir / job.subset / job.job_id / f"attempt_{attempt_number}"
                if attempt_dir.exists():
                    shutil.rmtree(attempt_dir)
                attempt_dir.mkdir(parents=True)
                log_path = self.logs_dir / job.subset / f"{job.job_id}_attempt_{attempt_number}.log"
                token_path = self.tokens_dir / job.subset / f"{job.job_id}_attempt_{attempt_number}.json"

                print(
                    f"[{now_iso()}] START {job.subset} {job.index:03d} "
                    f"attempt={attempt_number} {job.paper_path.parent.name}",
                    flush=True,
                )
                attempt_started_at = now_iso()
                started = time.monotonic()
                output_root = attempt_dir / "output"
                return_code, timed_out = self.run_process(self.command_for(job), output_root, log_path)
                elapsed = round(time.monotonic() - started, 2)

                pipeline_output = output_root / job.paper_path.parent.name
                timing_path = pipeline_output / "timing_cost_log.json"
                timing = self.load_timing(timing_path)
                if timing:
                    atomic_json(token_path, timing)

                overall = timing.get("overall") or {}
                final_png = self.locate_final_png(pipeline_output, job.paper_path.parent.name)
                png_ok = final_png is not None
                attempt_status, quality_gate_accepted = classify_attempt(return_code, png_ok)
                final_quality_gate = self.load_timing(pipeline_output / "content" / "final_quality_gate.json")
                if final_quality_gate:
                    quality_gate_accepted = bool(final_quality_gate.get("accepted"))
                generative_assets, asset_report_path = self.collect_generative_asset_reports(
                    pipeline_output,
                    job,
                    attempt_number,
                )
                attempt = {
                    "attempt": attempt_number,
                    "started_at": attempt_started_at,
                    "runtime_seconds": elapsed,
                    "return_code": return_code,
                    "timed_out": timed_out,
                    "status": attempt_status,
                    "quality_gate_accepted": quality_gate_accepted,
                    "quality_gate_failures": final_quality_gate.get("failures", []),
                    "input_tokens": int(overall.get("total_input_tokens", 0) or 0),
                    "output_tokens": int(overall.get("total_output_tokens", 0) or 0),
                    "total_tokens": int(overall.get("total_tokens", 0) or 0),
                    "api_calls": int(overall.get("total_api_calls", 0) or 0),
                    "runtime_log": str(log_path),
                    "token_log": str(token_path) if timing else None,
                    "generative_assets": generative_assets,
                    "generative_asset_report": str(asset_report_path),
                }
                record["attempts"].append(attempt)

                if final_png is not None:
                    destination = self.poster_path(job)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(final_png, destination)
                    png_valid, dimensions = valid_png(destination)
                    if png_valid:
                        record["status"] = "success"
                        record["quality_gate_accepted"] = quality_gate_accepted
                        record["quality_gate_failures"] = final_quality_gate.get("failures", [])
                        record["generative_assets"] = generative_assets
                        record["generative_asset_report"] = str(asset_report_path)
                        record["poster_png"] = str(destination)
                        record["poster_dimensions"] = dimensions
                    else:
                        record["status"] = "failed"
                        attempt["status"] = "failed"
                        attempt["reason"] = "copied PNG failed validation"

                shutil.rmtree(attempt_dir, ignore_errors=True)
                if record.get("status") == "success":
                    break
                if attempt_number < self.args.max_attempts:
                    time.sleep(self.args.retry_delay_seconds)

            if record.get("status") != "success":
                record["status"] = "failed"
            record["finished_at"] = now_iso()
            record["cumulative_input_tokens"] = sum(item["input_tokens"] for item in record["attempts"])
            record["cumulative_output_tokens"] = sum(item["output_tokens"] for item in record["attempts"])
            record["cumulative_total_tokens"] = sum(item["total_tokens"] for item in record["attempts"])
            record["cumulative_api_calls"] = sum(item["api_calls"] for item in record["attempts"])
            atomic_json(self.record_path(job), record)
            print(
                f"[{now_iso()}] {record['status'].upper()} {job.subset} {job.index:03d} "
                f"input={record['cumulative_input_tokens']} output={record['cumulative_output_tokens']} "
                f"{job.paper_path.parent.name}",
                flush=True,
            )
            return record
        finally:
            with self.summary_lock:
                self.running.discard(job.job_id)
                self.write_summary()

    def all_records(self) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        for job in self.jobs:
            record = self.read_record(job)
            if record:
                records[job.job_id] = record
        return records

    def write_summary(self) -> None:
        records = self.all_records()
        rows = []
        for job in self.jobs:
            record = records.get(job.job_id, {})
            status = "running" if job.job_id in self.running else record.get("status", "pending")
            rows.append(
                {
                    "subset": job.subset,
                    "index": job.index,
                    "job_id": job.job_id,
                    "paper_name": job.paper_path.parent.name,
                    "status": status,
                    "attempts": len(record.get("attempts") or []),
                    "input_tokens": int(record.get("cumulative_input_tokens", 0) or 0),
                    "output_tokens": int(record.get("cumulative_output_tokens", 0) or 0),
                    "total_tokens": int(record.get("cumulative_total_tokens", 0) or 0),
                    "runtime_seconds": round(
                        sum(float(item.get("runtime_seconds", 0)) for item in record.get("attempts") or []),
                        2,
                    ),
                    "poster_png": record.get("poster_png", ""),
                    "quality_gate_accepted": record.get("quality_gate_accepted", ""),
                    "teaser_source": ((record.get("generative_assets") or {}).get("teaser") or {}).get("asset_source", ""),
                    "teaser_needs_regeneration": ((record.get("generative_assets") or {}).get("teaser") or {}).get(
                        "needs_regeneration", ""
                    ),
                    "background_source": ((record.get("generative_assets") or {}).get("background") or {}).get(
                        "asset_source", ""
                    ),
                    "background_needs_regeneration": ((record.get("generative_assets") or {}).get("background") or {}).get(
                        "needs_regeneration", ""
                    ),
                }
            )

        csv_path = self.run_dir / "summary.csv"
        temporary = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(csv_path)

        counts = {status: sum(row["status"] == status for row in rows) for status in ("pending", "running", "success", "failed")}
        quality_rejected = sum(
            row["status"] == "success" and row["quality_gate_accepted"] is False
            for row in rows
        )
        teaser_needs_regeneration = sum(row["teaser_needs_regeneration"] is True for row in rows)
        background_needs_regeneration = sum(row["background_needs_regeneration"] is True for row in rows)
        subset_summary = {}
        for subset in self.args.subsets:
            subset_rows = [row for row in rows if row["subset"] == subset]
            subset_summary[subset] = {
                "total": len(subset_rows),
                "success": sum(row["status"] == "success" for row in subset_rows),
                "failed": sum(row["status"] == "failed" for row in subset_rows),
                "input_tokens": sum(row["input_tokens"] for row in subset_rows),
                "output_tokens": sum(row["output_tokens"] for row in subset_rows),
            }
        atomic_json(
            self.run_dir / "summary.json",
            {
                "updated_at": now_iso(),
                "total_papers": len(rows),
                **counts,
                "quality_rejected": quality_rejected,
                "teaser_needs_regeneration": teaser_needs_regeneration,
                "background_needs_regeneration": background_needs_regeneration,
                "input_tokens": sum(row["input_tokens"] for row in rows),
                "output_tokens": sum(row["output_tokens"] for row in rows),
                "total_tokens": sum(row["total_tokens"] for row in rows),
                "subsets": subset_summary,
            },
        )

    def run(self) -> int:
        self.write_manifest()
        with self.summary_lock:
            self.write_summary()
        pending = [job for job in self.jobs if not self.is_complete(job) and len(self.read_record(job).get("attempts") or []) < self.args.max_attempts]
        print(
            f"[{now_iso()}] benchmark papers={len(self.jobs)} pending={len(pending)} "
            f"concurrency={self.args.concurrency} run_dir={self.run_dir}",
            flush=True,
        )
        if self.args.dry_run:
            return 0

        with ThreadPoolExecutor(max_workers=self.args.concurrency, thread_name_prefix="poster") as executor:
            futures = {executor.submit(self.run_job, job): job for job in pending}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"[{now_iso()}] RUNNER ERROR {job.job_id}: {exc}", flush=True)

        with self.summary_lock:
            self.write_summary()
        summary = json.loads((self.run_dir / "summary.json").read_text(encoding="utf-8"))
        print(f"[{now_iso()}] COMPLETE {json.dumps(summary, ensure_ascii=False)}", flush=True)
        return 0 if summary.get("failed", 0) == 0 else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PosterMELD benchmark subsets with bounded concurrency.")
    parser.add_argument("--benchmark-root", type=Path, default=REPO_ROOT / "Benchmark")
    parser.add_argument("--subsets", nargs="+", choices=SUPPORTED_SUBSETS, default=list(DEFAULT_SUBSETS))
    parser.add_argument("--run-dir", type=Path, default=REPO_ROOT / "output" / "benchmark_aaai2026_cvpr2025")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--job-timeout-minutes", type=int, default=45)
    parser.add_argument("--retry-delay-seconds", type=int, default=20)
    parser.add_argument("--launch-stagger-seconds", type=float, default=5.0)
    parser.add_argument("--min-free-disk-gb", type=float, default=8.0)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--layout-template", default="auto")
    parser.add_argument("--visual-density", choices=("lean", "balanced", "rich"), default="balanced")
    parser.add_argument("--parser-max-chars", type=int, default=0)
    parser.add_argument("--keypoint-limit", type=int, default=0)
    parser.add_argument("--header-seed", type=int, default=20260710)
    parser.add_argument("--max-papers", type=int, default=0, help="Limit this run to N papers; zero means all.")
    parser.add_argument("--round-robin-subsets", action="store_true", help="Interleave papers from each selected subset.")
    parser.add_argument("--enable-generated-background", action="store_true")
    parser.add_argument("--enable-generated-teaser", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1")
    if args.parser_max_chars < 0:
        parser.error("--parser-max-chars cannot be negative")
    if args.keypoint_limit < 0:
        parser.error("--keypoint-limit cannot be negative")
    return args


def main() -> int:
    args = parse_args()
    jobs = discover_jobs(args.benchmark_root.resolve(), args.subsets)
    if args.round_robin_subsets:
        grouped = {subset: [job for job in jobs if job.subset == subset] for subset in args.subsets}
        jobs = [
            grouped[subset][index]
            for index in range(max((len(items) for items in grouped.values()), default=0))
            for subset in args.subsets
            if index < len(grouped[subset])
        ]
    if args.max_papers > 0:
        jobs = jobs[: args.max_papers]
    if not jobs:
        print("No benchmark papers found", file=sys.stderr)
        return 1

    args.run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.run_dir / "runner.lock"
    with lock_path.open("w", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"Another benchmark runner already holds {lock_path}", file=sys.stderr)
            return 1
        lock_file.write(str(os.getpid()))
        lock_file.flush()
        return BatchRunner(args, jobs).run()


if __name__ == "__main__":
    raise SystemExit(main())
