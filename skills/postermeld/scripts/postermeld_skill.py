#!/usr/bin/env python3
"""Thin installer and launcher for the complete PosterMELD runtime."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


DEFAULT_REPOSITORY = "https://github.com/Shannon4Science/PosterMELD.git"
DEFAULT_REF = "main"
FULL_QUALITY_ARGS = [
    "--layout-template",
    "auto",
    "--visual-density",
    "rich",
    "--enable-affiliation-logos",
    "--affiliation-logo-mode",
    "single",
    "--enable-generated-teaser",
    "--enable-generated-background",
    "--enable-vlm-layout-review",
    "--enable-visual-legibility-review",
    "--enable-block-vlm-review",
]
TEXT_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "ZHIPU_API_KEY",
    "MOONSHOT_API_KEY",
    "MINIMAX_API_KEY",
    "ALIBABA_API_KEY",
)


def cache_root() -> Path:
    configured = os.getenv("POSTERMELD_SKILL_CACHE")
    return Path(configured).expanduser() if configured else Path.home() / ".cache" / "postermeld-skill"


def is_runtime(path: Path) -> bool:
    return (
        (path / "pyproject.toml").is_file()
        and (path / "src" / "workflow" / "pipeline.py").is_file()
        and (path / "config" / "poster_config.yaml").is_file()
    )


def candidate_runtimes(explicit: str | None = None) -> Iterable[Path]:
    if explicit:
        yield Path(explicit).expanduser()
        return
    if os.getenv("POSTERMELD_RUNTIME_DIR"):
        yield Path(os.environ["POSTERMELD_RUNTIME_DIR"]).expanduser()
        return

    roots = [Path.cwd(), Path(__file__).resolve()]
    seen: set[Path] = set()
    for root in roots:
        for parent in (root, *root.parents):
            for candidate in (parent, parent / "poster_generation"):
                resolved = candidate.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    yield resolved

    yield cache_root() / "runtime" / "poster_generation"


def find_runtime(explicit: str | None = None) -> Path | None:
    for candidate in candidate_runtimes(explicit):
        if is_runtime(candidate):
            return candidate.resolve()
    return None


def python_candidates(runtime: Path) -> Iterable[Path]:
    if os.getenv("POSTERMELD_PYTHON"):
        yield Path(os.environ["POSTERMELD_PYTHON"]).expanduser()
    yield runtime / ".venv" / "bin" / "python"
    yield runtime.parent / ".venv" / "bin" / "python"
    if platform.system() == "Windows":
        yield runtime / ".venv" / "Scripts" / "python.exe"
        yield runtime.parent / ".venv" / "Scripts" / "python.exe"


def runtime_python(runtime: Path) -> Path | None:
    for candidate in python_candidates(runtime):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            probe = subprocess.run(
                [str(candidate), "-c", "import src.workflow.pipeline"],
                cwd=runtime,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if probe.returncode == 0:
                # Preserve the virtual-environment path. Resolving the symlink can
                # bypass pyvenv.cfg on installations whose bin/python is linked to
                # a shared interpreter.
                return candidate.absolute()
    return None


def venv_python(venv: Path) -> Path:
    if platform.system() == "Windows":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def read_env_file(path: Path | None) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path or not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith(("'", '"')) and value.endswith(value[0]):
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def resolve_env_file(runtime: Path, explicit: str | None = None) -> Path | None:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if os.getenv("POSTERMELD_ENV_FILE"):
        candidates.append(Path(os.environ["POSTERMELD_ENV_FILE"]).expanduser())
    candidates.extend([Path.cwd() / ".env", runtime / ".env", runtime.parent / ".env"])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def merged_environment(runtime: Path, env_file: str | None = None) -> tuple[dict[str, str], Path | None]:
    resolved = resolve_env_file(runtime, env_file)
    loaded = read_env_file(resolved)
    environment = os.environ.copy()
    for key, value in loaded.items():
        environment.setdefault(key, value)
    return environment, resolved


def executable(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    if name == "soffice" and platform.system() == "Darwin":
        mac_path = Path("/Applications/LibreOffice.app/Contents/MacOS/soffice")
        if mac_path.is_file():
            return str(mac_path)
    return None


def configured(environment: dict[str, str], key: str) -> bool:
    value = str(environment.get(key, "")).strip()
    if not value:
        return False
    normalized = value.lower()
    placeholders = ("your_", "your-", "replace", "change-me", "placeholder", "example", "<")
    return not normalized.startswith(placeholders) and normalized not in {"none", "null", "..."}


def doctor_report(runtime_arg: str | None, env_file: str | None) -> dict[str, object]:
    runtime = find_runtime(runtime_arg)
    if not runtime:
        return {
            "runtime_ready": False,
            "runtime_dir": None,
            "python_ready": False,
            "full_quality_ready": False,
            "message": "PosterMELD runtime is not installed.",
        }

    python_path = runtime_python(runtime)
    environment, resolved_env = merged_environment(runtime, env_file)
    text_ready = any(configured(environment, key) for key in TEXT_KEYS)
    services = {
        "text_model": text_ready,
        "vlm_review": configured(environment, "VLM_API_KEY"),
        "image_generation": configured(environment, "IMAGE_API_KEY"),
        "mineru_parsing": configured(environment, "MINERU_API_KEY"),
        "libreoffice": bool(executable("soffice")),
    }
    required = ("text_model", "vlm_review", "image_generation", "libreoffice")
    full_quality_ready = bool(python_path) and all(services[name] for name in required)
    strict_ready = full_quality_ready and services["mineru_parsing"]
    return {
        "runtime_ready": True,
        "runtime_dir": str(runtime),
        "python_ready": bool(python_path),
        "python": str(python_path) if python_path else None,
        "env_file": str(resolved_env) if resolved_env else None,
        "services": services,
        "full_quality_ready": full_quality_ready,
        "strict_quality_ready": strict_ready,
        "missing_full_quality_services": [name for name in required if not services[name]],
        "missing_strict_services": [name for name, ready in services.items() if not ready],
    }


def print_doctor(report: dict[str, object]) -> None:
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report.get("runtime_ready"):
        print("Run: postermeld_skill.py install", file=sys.stderr)
        return
    if not report.get("python_ready"):
        print("The runtime exists but its Python environment is not ready. Run install.", file=sys.stderr)
    if not report.get("full_quality_ready"):
        print(
            "Full-quality generation is unavailable. Configure every missing service before running; "
            "the skill will not substitute a reduced renderer.",
            file=sys.stderr,
        )
    if not report.get("strict_quality_ready"):
        print("MinerU is not configured; the runtime may use its parser fallback.", file=sys.stderr)


def install_runtime(args: argparse.Namespace) -> int:
    existing = find_runtime(args.runtime)
    if existing and not args.cached_copy:
        runtime = existing
        print(f"Using existing PosterMELD runtime: {runtime}")
    else:
        destination = cache_root() / "runtime"
        if destination.exists():
            runtime = destination / "poster_generation"
            if not is_runtime(runtime):
                print(f"Cache directory is incomplete: {destination}", file=sys.stderr)
                return 2
            print(f"Using cached PosterMELD runtime: {runtime}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            command = [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                args.ref,
                args.repository,
                str(destination),
            ]
            print(f"Installing PosterMELD runtime from {args.repository} ({args.ref})")
            result = subprocess.run(command, check=False)
            if result.returncode != 0:
                return result.returncode
            runtime = destination / "poster_generation"

    if not is_runtime(runtime):
        print(f"Invalid PosterMELD runtime: {runtime}", file=sys.stderr)
        return 2
    if runtime_python(runtime):
        print(f"PosterMELD runtime is ready: {runtime}")
        return 0

    uv = executable("uv")
    if uv:
        command = [uv, "sync", "--project", str(runtime)]
    else:
        python311 = executable("python3.11")
        if not python311:
            print("Python 3.11 or uv is required to install PosterMELD.", file=sys.stderr)
            return 2
        venv = runtime / ".venv"
        create = subprocess.run([python311, "-m", "venv", str(venv)], check=False)
        if create.returncode != 0:
            return create.returncode
        python_path = venv_python(venv)
        command = [
            str(python_path),
            "-m",
            "pip",
            "install",
            "-r",
            str(runtime / "requirements.txt"),
        ]
    print("Installing the complete PosterMELD dependency environment. This is a one-time operation.")
    result = subprocess.run(command, cwd=runtime, check=False)
    if result.returncode != 0:
        return result.returncode

    if not uv:
        editable = subprocess.run(
            [str(venv_python(runtime / ".venv")), "-m", "pip", "install", "-e", ".", "--no-deps"],
            cwd=runtime,
            check=False,
        )
        if editable.returncode != 0:
            return editable.returncode

    ready_python = runtime_python(runtime)
    if not ready_python:
        print("Installation completed, but the PosterMELD module cannot be imported.", file=sys.stderr)
        return 2
    print(f"PosterMELD runtime is ready: {runtime}")
    return 0


def pipeline_command(runtime: Path, python_path: Path, paper: Path, extra: list[str]) -> list[str]:
    return [
        str(python_path),
        "-m",
        "src.workflow.pipeline",
        str(paper),
        *FULL_QUALITY_ARGS,
        *extra,
    ]


def run_pipeline(args: argparse.Namespace) -> int:
    runtime = find_runtime(args.runtime)
    if not runtime:
        print("PosterMELD runtime is missing. Run the install command first.", file=sys.stderr)
        return 2
    python_path = runtime_python(runtime)
    if not python_path:
        print("PosterMELD Python environment is missing. Run the install command first.", file=sys.stderr)
        return 2

    paper = Path(args.paper).expanduser().resolve()
    if not paper.is_file() or paper.suffix.lower() != ".pdf":
        print(f"Paper PDF not found: {paper}", file=sys.stderr)
        return 2

    environment, resolved_env = merged_environment(runtime, args.env_file)
    report = doctor_report(str(runtime), args.env_file)
    readiness_key = "strict_quality_ready" if args.strict else "full_quality_ready"
    if not report.get(readiness_key):
        print_doctor(report)
        print("Generation stopped before any model call because full-quality requirements are incomplete.", file=sys.stderr)
        return 3

    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else Path.cwd() / "postermeld-output"
    output_root.mkdir(parents=True, exist_ok=True)
    environment["PAPER2POSTER_OUTPUT_ROOT"] = str(output_root)
    extra = list(args.pipeline_args)
    if extra and extra[0] == "--":
        extra = extra[1:]
    command = pipeline_command(runtime, python_path, paper, extra)
    expected_output = output_root / (paper.parent.name or paper.stem)

    if args.dry_run:
        print(json.dumps({
            "runtime": str(runtime),
            "env_file": str(resolved_env) if resolved_env else None,
            "output_root": str(output_root),
            "command": command,
        }, indent=2, ensure_ascii=False))
        return 0

    expected_output.mkdir(parents=True, exist_ok=True)
    log_path = expected_output / "skill_run.log"
    started = time.time()
    print(f"Starting full PosterMELD pipeline for: {paper}")
    print(f"Output directory: {expected_output}")
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=runtime,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
                log_file.flush()
            return_code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            return_code = process.wait()

    outputs = sorted(
        str(path)
        for path in expected_output.glob("*")
        if path.suffix.lower() in {".png", ".pptx"}
    )
    run_report = {
        "paper": str(paper),
        "runtime": str(runtime),
        "env_file": str(resolved_env) if resolved_env else None,
        "output_root": str(output_root),
        "output_dir": str(expected_output),
        "log_path": str(log_path),
        "return_code": return_code,
        "started_at_unix": started,
        "elapsed_seconds": round(time.time() - started, 2),
        "full_quality_defaults": FULL_QUALITY_ARGS,
        "pipeline_args": extra,
        "outputs": outputs,
    }
    (expected_output / "skill_run_report.json").write_text(
        json.dumps(run_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if return_code == 0:
        print(json.dumps(run_report, indent=2, ensure_ascii=False))
    else:
        print(f"PosterMELD failed with exit code {return_code}. See {log_path}", file=sys.stderr)
    return return_code


def list_templates(args: argparse.Namespace) -> int:
    runtime = find_runtime(args.runtime)
    if not runtime:
        print("PosterMELD runtime is missing. Run install first.", file=sys.stderr)
        return 2
    python_path = runtime_python(runtime)
    if not python_path:
        print("PosterMELD Python environment is missing. Run install first.", file=sys.stderr)
        return 2
    return subprocess.run(
        [str(python_path), "-m", "src.workflow.pipeline", "--list-layout-templates"],
        cwd=runtime,
        check=False,
    ).returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Install and invoke the complete PosterMELD pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    install = subparsers.add_parser("install", help="Locate or install the full PosterMELD runtime.")
    install.add_argument("--runtime", help="Use a local poster_generation directory.")
    install.add_argument("--repository", default=os.getenv("POSTERMELD_REPOSITORY", DEFAULT_REPOSITORY))
    install.add_argument("--ref", default=os.getenv("POSTERMELD_RUNTIME_REF", DEFAULT_REF))
    install.add_argument("--cached-copy", action="store_true", help="Install into the skill cache even when a local runtime exists.")
    install.set_defaults(func=install_runtime)

    doctor = subparsers.add_parser("doctor", help="Check runtime and full-quality service readiness.")
    doctor.add_argument("--runtime")
    doctor.add_argument("--env-file")
    doctor.add_argument("--strict", action="store_true", help="Also require MinerU precise parsing.")

    run = subparsers.add_parser("run", help="Run the full PosterMELD pipeline.")
    run.add_argument("--paper", required=True)
    run.add_argument("--runtime")
    run.add_argument("--env-file")
    run.add_argument("--output-root")
    run.add_argument("--strict", action="store_true", help="Require MinerU in addition to all visual services.")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument(
        "pipeline_args",
        nargs=argparse.REMAINDER,
        help="Arguments after -- are passed to the PosterMELD CLI and override defaults.",
    )
    run.set_defaults(func=run_pipeline)

    templates = subparsers.add_parser("templates", help="List templates from the complete runtime.")
    templates.add_argument("--runtime")
    templates.set_defaults(func=list_templates)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "doctor":
        report = doctor_report(args.runtime, args.env_file)
        print_doctor(report)
        key = "strict_quality_ready" if args.strict else "full_quality_ready"
        return 0 if report.get(key) else 3
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
