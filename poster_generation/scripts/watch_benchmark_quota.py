#!/usr/bin/env python3
"""Stop a benchmark run when an API reports exhausted credit or hard quota."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


HARD_QUOTA_PATTERN = re.compile(
    r"(?:"
    r"insufficient[_ -]?quota|"
    r"quota\s+(?:exceeded|exhausted)|"
    r"exceeded\s+(?:your\s+)?(?:current\s+)?quota|"
    r"billing\s+hard\s+limit|"
    r"insufficient\s+(?:account\s+)?balance|"
    r"(?:account\s+)?balance\s+(?:is\s+)?(?:insufficient|exhausted|depleted)|"
    r"credits?\s+(?:exhausted|depleted)|"
    r"credit\s+balance|"
    r"payment\s+required|"
    r"(?:402\s+client\s+error|http(?:/\S+)?\s+402)|"
    r"余额不足|额度不足|账户欠费|余额已用完|资源包.*不足|需要充值"
    r")",
    re.IGNORECASE,
)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def target_session_exists(session: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def quota_log_paths(run_dir: Path) -> list[Path]:
    return [run_dir / "launcher.log", *sorted((run_dir / "logs").rglob("*.log"))]


def snapshot_log_offsets(run_dir: Path) -> dict[Path, int]:
    """Record existing log sizes so a resumed watcher ignores old quota errors."""
    return {path: path.stat().st_size for path in quota_log_paths(run_dir) if path.is_file()}


def find_quota_error(run_dir: Path, offsets: dict[Path, int] | None = None) -> tuple[Path, int, str] | None:
    for path in quota_log_paths(run_dir):
        if not path.is_file():
            continue
        try:
            start = offsets.get(path, 0) if offsets is not None else 0
            current_size = path.stat().st_size
            if current_size < start:
                start = 0
            with path.open("rb") as handle:
                handle.seek(start)
                payload = handle.read()
                if offsets is not None:
                    offsets[path] = handle.tell()
            for line_number, raw_line in enumerate(payload.splitlines(), start=1):
                line = raw_line.decode("utf-8", errors="replace")
                if HARD_QUOTA_PATTERN.search(line):
                    return path, line_number, line.strip()[:1000]
        except OSError:
            continue
    return None


def matching_pids(pattern: str) -> list[int]:
    result = subprocess.run(
        ["pgrep", "-f", pattern],
        capture_output=True,
        text=True,
        check=False,
    )
    return [int(value) for value in result.stdout.split() if value.isdigit() and int(value) != os.getpid()]


def terminate_run(run_dir: Path, session: str) -> None:
    pipeline_pattern = r"src\.workflow\.pipeline .*\/Benchmark\/"
    runner_pattern = rf"scripts\/run_benchmark_batch\.py .*{re.escape(str(run_dir))}"
    pipeline_pids = matching_pids(pipeline_pattern)
    runner_pids = matching_pids(runner_pattern)

    for pid in pipeline_pids:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    for pid in runner_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    subprocess.run(
        ["tmux", "kill-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument(
        "--ignore-existing",
        action="store_true",
        help="Ignore quota errors already present in logs when the watcher starts.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    marker_path = run_dir / "QUOTA_STOPPED.json"
    offsets = snapshot_log_offsets(run_dir) if args.ignore_existing else None
    print(f"[{now_iso()}] quota watcher started for {args.session}", flush=True)
    while target_session_exists(args.session):
        match = find_quota_error(run_dir, offsets)
        if match:
            path, line_number, line = match
            payload = {
                "stopped_at": now_iso(),
                "reason": "hard API quota or balance error detected",
                "source_log": str(path),
                "line_number": line_number,
                "matched_line": line,
                "benchmark_session": args.session,
            }
            marker_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(payload, ensure_ascii=False), flush=True)
            terminate_run(run_dir, args.session)
            return 2
        time.sleep(max(args.poll_seconds, 1.0))

    print(f"[{now_iso()}] benchmark session ended without a hard quota error", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
