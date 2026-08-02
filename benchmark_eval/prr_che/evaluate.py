from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from common.api import ApiPool, load_api_keys, response_text, response_to_dict
from common.io import (
    ManifestItem,
    append_jsonl,
    atomic_write_json,
    encode_image,
    image_exists,
    load_manifest,
    now,
    read_json,
    result_path,
)
from prr_che.parsing import parse_che, parse_prr


EVENT_LOCK = threading.Lock()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def valid_prr(stage: Any) -> bool:
    return bool(
        isinstance(stage, dict)
        and stage.get("status") == "success"
        and isinstance((stage.get("parsed") or {}).get("print_ready"), bool)
    )


def valid_che(stage: Any) -> bool:
    parsed = stage.get("parsed") if isinstance(stage, dict) else None
    return bool(
        isinstance(stage, dict)
        and stage.get("status") == "success"
        and isinstance(parsed, dict)
        and isinstance(parsed.get("che_score"), (int, float))
    )


def pipeline_complete(record: Any) -> bool:
    if not isinstance(record, dict) or not valid_prr(record.get("prr")):
        return False
    if record["prr"]["parsed"]["print_ready"] is False:
        return (record.get("che") or {}).get("status") == "not_applicable"
    return valid_che(record.get("che"))


def missing_record(item: ManifestItem, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "updated_at": now(),
        "pipeline_status": "success_not_print_ready",
        "model": args.model,
        "item": item.to_dict(),
        "prr": {
            "status": "success",
            "origin": "deterministic_missing_image",
            "raw_response": None,
            "parsed": {
                "assessability": "insufficient",
                "print_ready": False,
                "reason": "No poster image exists for this manifest item.",
                "warnings": [],
                "checks": {},
            },
            "attempts": [],
        },
        "che": {
            "status": "not_applicable",
            "reason": "PRR print_ready is false",
            "parsed": None,
        },
    }


def call_stage(
    *,
    item: ManifestItem,
    stage_name: str,
    prompt: str,
    parser: Callable[[str], dict[str, Any]],
    image_b64: str,
    key_slot: int,
    pool: ApiPool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    stage_started = time.time()
    for attempt_number in range(1, args.max_attempts + 1):
        attempt_started = time.time()
        attempt: dict[str, Any] = {
            "attempt": attempt_number,
            "started_at": now(),
            "key_slot": key_slot,
            "key_fingerprint": pool.fingerprints[key_slot],
            "status": "running",
        }
        try:
            request: dict[str, Any] = {
                "model": args.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}",
                                    "detail": args.vision_detail,
                                },
                            },
                        ],
                    }
                ],
                "response_format": {"type": "json_object"},
            }
            if args.max_output_tokens is not None:
                request["max_tokens"] = args.max_output_tokens
            response = pool.create(key_slot, **request)
            raw = response_text(response)
            metadata = response_to_dict(response)
            finish_reason = response.choices[0].finish_reason
            if finish_reason == "length":
                raise RuntimeError("Server ended the response with finish_reason=length")
            parsed = parser(raw)
            attempt.update(
                {
                    "status": "success",
                    "finished_at": now(),
                    "elapsed_seconds": time.time() - attempt_started,
                    "raw_response": raw,
                    "response_metadata": metadata,
                    "finish_reason": finish_reason,
                }
            )
            attempts.append(attempt)
            return {
                "status": "success",
                "origin": "api",
                "raw_response": raw,
                "parsed": parsed,
                "response_metadata": metadata,
                "finish_reason": finish_reason,
                "attempts": attempts,
                "elapsed_seconds": time.time() - stage_started,
                "updated_at": now(),
            }
        except Exception as exc:
            attempt.update(
                {
                    "status": "failed",
                    "finished_at": now(),
                    "elapsed_seconds": time.time() - attempt_started,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
            attempts.append(attempt)
            append_jsonl(
                args.event_log,
                {
                    "time": now(),
                    "event": "stage_attempt_failed",
                    "item_id": item.id,
                    "stage": stage_name,
                    "key_slot": key_slot,
                    "attempt": attempt_number,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "will_retry": attempt_number < args.max_attempts,
                },
                EVENT_LOCK,
            )
            if attempt_number < args.max_attempts:
                time.sleep(args.retry_delay)
    return {
        "status": "failed",
        "origin": "api",
        "raw_response": attempts[-1].get("raw_response") if attempts else None,
        "parsed": None,
        "attempts": attempts,
        "error": attempts[-1]["error"] if attempts else {"message": "unknown error"},
        "elapsed_seconds": time.time() - stage_started,
        "updated_at": now(),
    }


def evaluate_one(
    item: ManifestItem,
    key_slot: int,
    prr_prompt: str,
    che_prompt: str,
    pool: ApiPool,
    args: argparse.Namespace,
) -> str:
    destination = result_path(args.output_dir, item, "prr_che.json")
    existing = read_json(destination, {}) if not args.force else {}
    if pipeline_complete(existing):
        return "skipped"
    if not image_exists(item):
        atomic_write_json(destination, missing_record(item, args))
        return "success_not_print_ready"

    record = existing if isinstance(existing, dict) else {}
    record.update(
        {
            "schema_version": 1,
            "updated_at": now(),
            "pipeline_status": "running",
            "model": args.model,
            "api_base_url": args.base_url,
            "item": item.to_dict(),
            "prompt_files": {
                "prr": str(args.prr_prompt.resolve()),
                "che": str(args.che_prompt.resolve()),
            },
            "prompt_sha256": {
                "prr": sha256_text(prr_prompt),
                "che": sha256_text(che_prompt),
            },
        }
    )
    started = time.time()
    try:
        image_b64, image_metadata = encode_image(
            item.poster_path,
            max_dimension=args.max_image_dimension,
            max_bytes=args.max_image_bytes,
        )
        record["image"] = image_metadata
        if args.force or not valid_prr(record.get("prr")):
            record["prr"] = call_stage(
                item=item,
                stage_name="prr",
                prompt=prr_prompt,
                parser=parse_prr,
                image_b64=image_b64,
                key_slot=key_slot,
                pool=pool,
                args=args,
            )
            atomic_write_json(destination, record)

        if not valid_prr(record.get("prr")):
            record["pipeline_status"] = "failed_prr"
        elif record["prr"]["parsed"]["print_ready"] is False:
            record["che"] = {
                "status": "not_applicable",
                "reason": "PRR print_ready is false",
                "parsed": None,
            }
            record["pipeline_status"] = "success_not_print_ready"
        else:
            if args.force or not valid_che(record.get("che")):
                record["che"] = call_stage(
                    item=item,
                    stage_name="che",
                    prompt=che_prompt,
                    parser=parse_che,
                    image_b64=image_b64,
                    key_slot=key_slot,
                    pool=pool,
                    args=args,
                )
            record["pipeline_status"] = "success" if valid_che(record.get("che")) else "failed_che"
    except Exception as exc:
        record["pipeline_status"] = "failed_local"
        record["local_error"] = {"type": type(exc).__name__, "message": str(exc)}
    record["updated_at"] = now()
    record["elapsed_seconds_latest_run"] = time.time() - started
    atomic_write_json(destination, record)
    append_jsonl(
        args.event_log,
        {
            "time": now(),
            "event": "item_complete",
            "item_id": item.id,
            "method": item.method,
            "pipeline_status": record["pipeline_status"],
        },
        EVENT_LOCK,
    )
    return str(record["pipeline_status"])


def main() -> int:
    prompt_root = Path(__file__).resolve().parent / "prompts"
    parser = argparse.ArgumentParser(description="Run cascaded PRR and conditional CHE evaluation.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="outputs/prr_che")
    parser.add_argument("--prr-prompt", type=Path, default=prompt_root / "print_ready_prompt.txt")
    parser.add_argument("--che-prompt", type=Path, default=prompt_root / "che_prompt.txt")
    parser.add_argument("--api-key-file", default=None)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--workers-per-key", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--retry-delay", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--max-image-dimension", type=int, default=2600)
    parser.add_argument("--max-image-bytes", type=int, default=6 * 1024 * 1024)
    parser.add_argument("--vision-detail", choices=("low", "high", "auto"), default="high")
    parser.add_argument("--event-log", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.output_dir = Path(args.output_dir).resolve()
    args.event_log = args.event_log or args.output_dir / "events.jsonl"
    args.prr_prompt = args.prr_prompt.resolve()
    args.che_prompt = args.che_prompt.resolve()
    items = load_manifest(args.manifest)
    if args.limit is not None:
        items = items[: max(0, args.limit)]
    prr_prompt = args.prr_prompt.read_text(encoding="utf-8").strip()
    che_prompt = args.che_prompt.read_text(encoding="utf-8").strip()
    if args.dry_run:
        print(f"validated={len(items)} prr_prompt_sha256={sha256_text(prr_prompt)} che_prompt_sha256={sha256_text(che_prompt)}")
        return 0

    keys = load_api_keys(args.api_key_file)
    pool = ApiPool(keys, args.base_url, args.timeout, args.workers_per_key)
    append_jsonl(
        args.event_log,
        {
            "time": now(),
            "event": "run_start",
            "model": args.model,
            "items": len(items),
            "key_count": len(keys),
            "key_fingerprints": pool.fingerprints,
            "workers_per_key": args.workers_per_key,
            "max_attempts": args.max_attempts,
            "retry_delay": args.retry_delay,
            "max_output_tokens": args.max_output_tokens,
        },
        EVENT_LOCK,
    )
    counts: Counter[str] = Counter()
    started = time.time()
    workers = min(pool.global_workers, max(1, len(items)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                evaluate_one,
                item,
                index % len(keys),
                prr_prompt,
                che_prompt,
                pool,
                args,
            ): item
            for index, item in enumerate(items)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            try:
                counts[future.result()] += 1
            except Exception as exc:
                counts["failed_worker"] += 1
                append_jsonl(
                    args.event_log,
                    {
                        "time": now(),
                        "event": "worker_failed",
                        "item_id": futures[future].id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    EVENT_LOCK,
                )
            if completed % 10 == 0 or completed == len(futures):
                elapsed = time.time() - started
                rate = completed / elapsed if elapsed else 0.0
                eta = (len(futures) - completed) / rate if rate else 0.0
                print(f"progress={completed}/{len(futures)} rate={rate:.3f}/s eta={eta:.0f}s statuses={dict(counts)}", flush=True)

    incomplete = []
    for item in items:
        if not pipeline_complete(read_json(result_path(args.output_dir, item, "prr_che.json"), {})):
            incomplete.append(item.id)
    final = {
        "time": now(),
        "event": "run_finish",
        "status": "complete" if not incomplete else "incomplete",
        "valid": len(items) - len(incomplete),
        "expected": len(items),
        "incomplete": incomplete,
        "counts": dict(counts),
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(args.output_dir / "final_status.json", final)
    append_jsonl(args.event_log, final, EVENT_LOCK)
    return 0 if not incomplete else 2


if __name__ == "__main__":
    raise SystemExit(main())
