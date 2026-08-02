from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path
from typing import Any

import torch
from bert_score import BERTScorer
from transformers.utils import logging as transformers_logging

from common.io import ManifestItem, append_jsonl, atomic_write_json, keypoint_reference, load_manifest, now
from keypoint_bertscore.state import (
    BERTSCORE_MODEL_NAME,
    BERTSCORE_NUM_LAYERS,
    OCR_MODEL_NAME,
    ocr_json_path,
    ocr_text_path,
    score_path,
    successful,
)


def distributed_context() -> tuple[int, int, int]:
    return (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("LOCAL_RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
    )


def terminal_record(item: ManifestItem, status: str, reason: str, model_path: str) -> dict[str, Any]:
    return {
        "status": status,
        "created_at": now(),
        "item": item.to_dict(),
        "model": BERTSCORE_MODEL_NAME,
        "model_path": model_path,
        "bert_precision": 0.0,
        "bert_recall": 0.0,
        "bert_f1": 0.0,
        "included_as_zero_in_full_benchmark": True,
        "reason": reason,
    }


def load_pair(output_dir: Path, item: ManifestItem) -> tuple[str, str, dict[str, Any]]:
    candidate = ocr_text_path(output_dir, item).read_text(encoding="utf-8").strip()
    if not candidate:
        raise ValueError("empty MinerU OCR text")
    if not item.annotation_path:
        raise ValueError("annotation_path is missing")
    reference, annotation = keypoint_reference(item.annotation_path)
    return candidate, reference, annotation


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute BERTScore between poster OCR and ordered paper keypoints.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="outputs/keypoint_bertscore")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--outer-batch-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-dir", default="logs/keypoint_bertscore")
    args = parser.parse_args()

    rank, local_rank, world_size = distributed_context()
    transformers_logging.set_verbosity_error()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for BERTScore in this evaluation pipeline")
    torch.cuda.set_device(local_rank)
    output_dir = Path(args.output_dir).resolve()
    items = load_manifest(args.manifest)
    if args.limit is not None:
        items = items[: max(0, args.limit)]
    pending = [item for item in items if args.force or not successful(score_path(output_dir, item), BERTSCORE_MODEL_NAME)]
    shard = pending[rank::world_size]
    model_path = str(Path(args.model_path).resolve())
    log_path = Path(args.log_dir).resolve() / f"bertscore_rank{rank}.jsonl"
    append_jsonl(log_path, {"event": "start", "time": now(), "rank": rank, "world_size": world_size, "pending": len(shard)})

    score_items: list[ManifestItem] = []
    for item in shard:
        destination = score_path(output_dir, item)
        if not item.poster_path or not Path(item.poster_path).is_file():
            atomic_write_json(destination, terminal_record(item, "missing_poster", "Poster image is missing", model_path))
        elif not successful(ocr_json_path(output_dir, item), OCR_MODEL_NAME):
            atomic_write_json(destination, terminal_record(item, "ocr_failed", "No successful MinerU OCR result", model_path))
        else:
            score_items.append(item)

    if not score_items:
        append_jsonl(log_path, {"event": "finish", "time": now(), "rank": rank, "success": 0, "scorable": 0, "all_tasks": len(shard)})
        print(f"BERTScore rank {rank}: no scorable pending items", flush=True)
        return 0

    scorer = BERTScorer(
        model_type=model_path,
        num_layers=BERTSCORE_NUM_LAYERS,
        batch_size=args.batch_size,
        nthreads=4,
        all_layers=False,
        idf=False,
        device=f"cuda:{local_rank}",
        lang=None,
        rescale_with_baseline=False,
        use_fast_tokenizer=False,
    )

    current_batch_size = args.batch_size
    completed = 0
    for cursor in range(0, len(score_items), args.outer_batch_size):
        batch = score_items[cursor : cursor + args.outer_batch_size]
        valid_items: list[ManifestItem] = []
        candidates: list[str] = []
        references: list[str] = []
        annotations: list[dict[str, Any]] = []
        for item in batch:
            try:
                candidate, reference, annotation = load_pair(output_dir, item)
                valid_items.append(item)
                candidates.append(candidate)
                references.append(reference)
                annotations.append(annotation)
            except Exception as exc:
                atomic_write_json(score_path(output_dir, item), terminal_record(item, "invalid_input", str(exc), model_path))

        started = time.time()
        while valid_items:
            try:
                precision, recall, f1 = scorer.score(candidates, references, batch_size=current_batch_size)
                break
            except torch.cuda.OutOfMemoryError:
                if current_batch_size <= 1:
                    raise
                current_batch_size = max(1, current_batch_size // 2)
                torch.cuda.empty_cache()
                gc.collect()
                append_jsonl(log_path, {"event": "oom_reduce_batch", "time": now(), "rank": rank, "new_batch_size": current_batch_size})
        else:
            precision = recall = f1 = torch.empty(0)

        elapsed = time.time() - started
        for item, candidate, reference, annotation, p, r, f in zip(
            valid_items, candidates, references, annotations, precision, recall, f1
        ):
            atomic_write_json(
                score_path(output_dir, item),
                {
                    "status": "success",
                    "created_at": now(),
                    "item": item.to_dict(),
                    "ocr_json_path": str(ocr_json_path(output_dir, item)),
                    "ocr_text_path": str(ocr_text_path(output_dir, item)),
                    "model": BERTSCORE_MODEL_NAME,
                    "model_path": model_path,
                    "num_layers": BERTSCORE_NUM_LAYERS,
                    "idf": False,
                    "rescale_with_baseline": False,
                    "use_fast_tokenizer": False,
                    "bert_precision": float(p),
                    "bert_recall": float(r),
                    "bert_f1": float(f),
                    "candidate_character_count": len(candidate),
                    "reference_character_count": len(reference),
                    "reference_keypoint_count": len(annotation["paper_poster_keypoints"]),
                    "elapsed_seconds_estimate": elapsed / max(1, len(valid_items)),
                    "included_as_zero_in_full_benchmark": False,
                },
            )
            completed += 1
        append_jsonl(
            log_path,
            {
                "event": "batch_complete",
                "time": now(),
                "rank": rank,
                "cursor": cursor + len(batch),
                "total": len(score_items),
                "success": completed,
                "batch_size": current_batch_size,
            },
        )

    append_jsonl(
        log_path,
        {
            "event": "finish",
            "time": now(),
            "rank": rank,
            "success": completed,
            "scorable": len(score_items),
            "all_tasks": len(shard),
            "final_batch_size": current_batch_size,
        },
    )
    print(f"BERTScore rank {rank}: {completed}/{len(score_items)} successful; final batch size={current_batch_size}", flush=True)
    return 0 if completed == len(score_items) else 2


if __name__ == "__main__":
    raise SystemExit(main())
