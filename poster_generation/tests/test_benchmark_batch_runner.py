import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw

from scripts.build_legacy_teaser_regeneration_manifest import build_inventory
from scripts.repair_benchmark_generated_assets import remove_repaired_asset_failures
from scripts.run_benchmark_batch import (
    BatchRunner,
    Job,
    classify_attempt,
    discover_jobs,
    find_local_affiliation_logo,
    has_no_affiliation_logo_marker,
    summarize_generative_asset_report,
)
from scripts.watch_benchmark_quota import HARD_QUOTA_PATTERN, find_quota_error, snapshot_log_offsets


def test_classify_attempt_retains_png_when_quality_gate_rejects() -> None:
    assert classify_attempt(return_code=1, png_available=True) == ("success", False)
    assert classify_attempt(return_code=0, png_available=True) == ("success", True)
    assert classify_attempt(return_code=1, png_available=False) == ("failed", False)


def test_resume_skips_valid_poster_with_generated_assets_pending_repair(tmp_path: Path) -> None:
    paper_path = tmp_path / "benchmark" / "pairs" / "paper" / "paper.pdf"
    paper_path.parent.mkdir(parents=True)
    paper_path.write_bytes(b"pdf")
    job = Job("pairs", 1, paper_path, "job-1", "")
    runner = BatchRunner(SimpleNamespace(run_dir=tmp_path / "run"), [job])
    runner.poster_path(job).parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 16), "white").save(runner.poster_path(job))
    runner.record_path(job).parent.mkdir(parents=True, exist_ok=True)
    runner.record_path(job).write_text(
        json.dumps(
            {
                "status": "success",
                "generative_assets": {"teaser": {"needs_regeneration": True}},
            }
        ),
        encoding="utf-8",
    )

    assert runner.is_complete(job) is True


def test_discover_jobs_supports_remaining_benchmark_subsets(tmp_path: Path) -> None:
    for subset in ("neurips2025", "p2peval", "pairs"):
        paper_dir = tmp_path / subset / f"{subset}-paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "paper.pdf").write_bytes(b"pdf")

    jobs = discover_jobs(tmp_path, ["neurips2025", "p2peval", "pairs"])

    assert [job.subset for job in jobs] == ["neurips2025", "p2peval", "pairs"]
    assert [job.conference for job in jobs] == ["NeurIPS 2025", "", ""]


def test_find_local_affiliation_logo_prefers_named_paper_asset(tmp_path: Path) -> None:
    paper_path = tmp_path / "paper.pdf"
    paper_path.write_bytes(b"pdf")
    fallback_dir = tmp_path / "affiliation_logos"
    fallback_dir.mkdir()
    (fallback_dir / "fallback.png").write_bytes(b"fallback")
    preferred = tmp_path / "affiliation_logo.png"
    preferred.write_bytes(b"preferred")

    assert find_local_affiliation_logo(paper_path) == preferred.resolve()


def test_find_local_affiliation_logo_uses_single_logo_directory_asset(tmp_path: Path) -> None:
    paper_path = tmp_path / "paper.pdf"
    paper_path.write_bytes(b"pdf")
    logo_dir = tmp_path / "affiliation_logos"
    logo_dir.mkdir()
    expected = logo_dir / "institution.png"
    expected.write_bytes(b"logo")

    assert find_local_affiliation_logo(paper_path) == expected.resolve()


def test_find_local_affiliation_logo_rejects_ambiguous_directory(tmp_path: Path) -> None:
    paper_path = tmp_path / "paper.pdf"
    paper_path.write_bytes(b"pdf")
    logo_dir = tmp_path / "affiliation_logos"
    logo_dir.mkdir()
    (logo_dir / "one.png").write_bytes(b"one")
    (logo_dir / "two.png").write_bytes(b"two")

    assert find_local_affiliation_logo(paper_path) is None


def test_no_affiliation_logo_marker(tmp_path: Path) -> None:
    paper_path = tmp_path / "paper.pdf"
    paper_path.write_bytes(b"pdf")
    assert not has_no_affiliation_logo_marker(paper_path)

    (tmp_path / "no_affiliation_logo").write_text("independent researcher\n", encoding="utf-8")
    assert has_no_affiliation_logo_marker(paper_path)


def test_quota_watcher_recognizes_image_provider_balance_wording() -> None:
    messages = [
        "Your account balance is insufficient; please recharge before continuing",
        "You exceeded your current quota, please check your plan and billing details",
        "生图账户余额不足，请充值后重试",
    ]

    assert all(HARD_QUOTA_PATTERN.search(message) for message in messages)


def test_quota_watcher_resume_ignores_existing_errors(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "paper.log"
    log_path.parent.mkdir()
    log_path.write_text("Error: insufficient quota\n", encoding="utf-8")
    offsets = snapshot_log_offsets(tmp_path)

    assert find_quota_error(tmp_path, offsets) is None

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("Error: account balance is insufficient\n")

    match = find_quota_error(tmp_path, offsets)
    assert match is not None
    assert match[0] == log_path


def test_summarize_generative_asset_report_marks_fallback_for_regeneration() -> None:
    summary = summarize_generative_asset_report(
        {
            "enabled": True,
            "applied": True,
            "asset_source": "procedural",
            "used_procedural_fallback": True,
            "fallback_reason": "readable_text_artifacts",
            "generation_attempt_count": 3,
        },
        "teaser",
    )

    assert summary["asset_source"] == "procedural"
    assert summary["needs_regeneration"] is True
    assert summary["reason"] == "readable_text_artifacts"
    assert summary["generation_attempt_count"] == 3


def test_summarize_generative_asset_report_preserves_valid_image_api_asset() -> None:
    summary = summarize_generative_asset_report(
        {
            "enabled": True,
            "applied": True,
            "asset_source": "image_api",
            "used_procedural_fallback": False,
            "needs_regeneration": False,
            "generation_attempt_count": 2,
        },
        "background",
    )

    assert summary["asset_source"] == "image_api"
    assert summary["needs_regeneration"] is False
    assert summary["generation_attempt_count"] == 2


def test_asset_repair_ignores_malformed_quality_failure_entries() -> None:
    record = {
        "quality_gate_failures": [
            None,
            {"category": "generated_asset", "asset": "teaser"},
            {"category": "occupancy"},
        ]
    }

    remove_repaired_asset_failures(record, {"teaser"})

    assert record["quality_gate_failures"] == [{"category": "occupancy"}]


def test_legacy_teaser_inventory_separates_procedural_signature(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    records_dir = run_dir / "records" / "aaai2026"
    posters_dir = run_dir / "posters" / "aaai2026"
    records_dir.mkdir(parents=True)
    posters_dir.mkdir(parents=True)

    reference_path = posters_dir / "reference.png"
    distinct_path = posters_dir / "distinct.png"
    reference = Image.new("RGB", (1000, 500), "white")
    draw = ImageDraw.Draw(reference)
    for x in range(60, 250, 20):
        draw.rectangle((x, 125, x + 12, 140), fill=(80, 110, 150))
    draw.line((70, 180, 120, 130, 170, 190, 230, 145), fill=(20, 40, 90), width=4)
    reference.save(reference_path)
    distinct = Image.new("RGB", (1000, 500), "white")
    ImageDraw.Draw(distinct).ellipse((60, 115, 250, 220), fill=(30, 30, 30))
    distinct.save(distinct_path)

    records = [
        ("reference", reference_path, 1),
        ("distinct", distinct_path, 2),
    ]
    for job_id, poster_path, index in records:
        (records_dir / f"{job_id}.json").write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "subset": "aaai2026",
                    "index": index,
                    "paper_name": job_id,
                    "model": "gpt-4o",
                    "status": "success",
                    "poster_png": str(poster_path),
                }
            ),
            encoding="utf-8",
        )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "jobs": [
                    {"job_id": "reference", "subset": "aaai2026", "index": 1},
                    {"job_id": "distinct", "subset": "aaai2026", "index": 2},
                    {"job_id": "pending", "subset": "aaai2026", "index": 3},
                ]
            }
        ),
        encoding="utf-8",
    )

    inventory = build_inventory(run_dir, "reference", threshold=0.75)
    actions = {item["job_id"]: item["action"] for item in inventory["items"]}

    assert actions == {"reference": "regenerate", "distinct": "preserve"}
    assert inventory["counts"] == {
        "total_benchmark_jobs": 3,
        "completed_posters_audited": 2,
        "preserve_successful": 1,
        "regenerate_required": 1,
        "unprocessed": 1,
        "run_after_recharge": 2,
    }
    assert [item["job_id"] for item in inventory["unprocessed_jobs"]] == ["pending"]
