from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

from loguru import logger
from mineru_vl_utils import MinerUClient, MinerULogitsProcessor
from mineru_vl_utils.post_process import json2md
from PIL import Image
from vllm import LLM

from common.io import ManifestItem, append_jsonl, atomic_write_json, atomic_write_text, load_manifest, now
from keypoint_bertscore.state import OCR_MODEL_NAME, ocr_json_path, ocr_text_path, successful


Image.MAX_IMAGE_PIXELS = None


def save_failure(item: ManifestItem, output_dir: Path, rank: int, exc: BaseException, elapsed: float) -> None:
    atomic_write_json(
        ocr_json_path(output_dir, item),
        {
            "status": "failed",
            "created_at": now(),
            "model": OCR_MODEL_NAME,
            "item": item.to_dict(),
            "rank": rank,
            "elapsed_seconds": elapsed,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        },
    )


def load_image(item: ManifestItem) -> tuple[Image.Image, tuple[int, int]]:
    if not item.poster_path:
        raise FileNotFoundError(f"No poster image for {item.id}")
    with Image.open(item.poster_path) as source:
        dimensions = source.size
        image = source.convert("RGB")
        image.load()
    return image, dimensions


def process_batch(
    client: MinerUClient,
    items: list[ManifestItem],
    output_dir: Path,
    rank: int,
    log_path: Path,
    model_path: str,
) -> int:
    images: list[Image.Image] = []
    dimensions: list[tuple[int, int]] = []
    started = time.time()
    try:
        for item in items:
            image, size = load_image(item)
            images.append(image)
            dimensions.append(size)
        blocks_per_image = client.batch_two_step_extract(images, image_analysis=False)
        if len(blocks_per_image) != len(items):
            raise RuntimeError(f"MinerU returned {len(blocks_per_image)} results for {len(items)} images")
        elapsed_per_item = (time.time() - started) / max(1, len(items))
        completed = 0
        for item, size, blocks in zip(items, dimensions, blocks_per_image):
            try:
                serializable_blocks = [dict(block) for block in blocks]
                markdown = json2md(serializable_blocks).strip()
                if not markdown:
                    raise ValueError(f"MinerU produced empty OCR text for {item.id}")
                text_path = ocr_text_path(output_dir, item)
                atomic_write_text(text_path, markdown + "\n")
                atomic_write_json(
                    ocr_json_path(output_dir, item),
                    {
                        "status": "success",
                        "created_at": now(),
                        "model": OCR_MODEL_NAME,
                        "model_path": model_path,
                        "item": item.to_dict(),
                        "rank": rank,
                        "image_width": size[0],
                        "image_height": size[1],
                        "elapsed_seconds_estimate": elapsed_per_item,
                        "block_count": len(serializable_blocks),
                        "character_count": len(markdown),
                        "text_path": str(text_path),
                        "blocks": serializable_blocks,
                    },
                )
                completed += 1
            except Exception as exc:
                save_failure(item, output_dir, rank, exc, time.time() - started)
                append_jsonl(
                    log_path,
                    {"event": "item_failed", "time": now(), "rank": rank, "item_id": item.id, "error": str(exc)},
                )
        return completed
    finally:
        for image in images:
            image.close()


def process_with_isolation(
    client: MinerUClient,
    items: list[ManifestItem],
    output_dir: Path,
    rank: int,
    log_path: Path,
    model_path: str,
) -> int:
    try:
        return process_batch(client, items, output_dir, rank, log_path, model_path)
    except BaseException as exc:
        if len(items) > 1:
            midpoint = len(items) // 2
            return process_with_isolation(client, items[:midpoint], output_dir, rank, log_path, model_path) + process_with_isolation(
                client, items[midpoint:], output_dir, rank, log_path, model_path
            )
        save_failure(items[0], output_dir, rank, exc, 0.0)
        append_jsonl(
            log_path,
            {"event": "item_failed", "time": now(), "rank": rank, "item_id": items[0].id, "error": str(exc)},
        )
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="OCR poster images with one MinerU vLLM engine per GPU.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="outputs/keypoint_bertscore")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--page-batch-size", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=65536)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-dir", default="logs/keypoint_bertscore")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    if args.rank < 0 or args.rank >= args.world_size:
        raise ValueError(f"Invalid rank {args.rank} for world size {args.world_size}")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must select one GPU for each OCR rank")

    output_dir = Path(args.output_dir).resolve()
    all_items = load_manifest(args.manifest)
    available = [item for item in all_items if item.poster_path and Path(item.poster_path).is_file()]
    if args.limit is not None:
        available = available[: max(0, args.limit)]
    pending = [
        item
        for item in available
        if args.force or not successful(ocr_json_path(output_dir, item), OCR_MODEL_NAME)
    ]
    shard = pending[args.rank :: args.world_size]
    log_path = Path(args.log_dir).resolve() / f"ocr_rank{args.rank}.jsonl"
    append_jsonl(
        log_path,
        {
            "event": "start",
            "time": now(),
            "rank": args.rank,
            "world_size": args.world_size,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "pending": len(shard),
        },
    )
    if not shard:
        print(f"OCR rank {args.rank}: no pending items", flush=True)
        return 0

    model_path = str(Path(args.model_path).resolve())
    load_started = time.time()
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        limit_mm_per_prompt={"image": 1},
        logits_processors=[MinerULogitsProcessor],
        trust_remote_code=False,
        disable_log_stats=True,
        seed=2027 + args.rank,
    )
    client = MinerUClient(
        backend="vllm-engine",
        vllm_llm=llm,
        batch_size=0,
        use_tqdm=False,
        image_analysis=False,
    )
    append_jsonl(
        log_path,
        {
            "event": "model_loaded",
            "time": now(),
            "rank": args.rank,
            "elapsed_seconds": time.time() - load_started,
            "page_batch_size": args.page_batch_size,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        },
    )

    completed = 0
    for cursor in range(0, len(shard), args.page_batch_size):
        batch = shard[cursor : cursor + args.page_batch_size]
        started = time.time()
        count = process_with_isolation(client, batch, output_dir, args.rank, log_path, model_path)
        completed += count
        append_jsonl(
            log_path,
            {
                "event": "batch_complete",
                "time": now(),
                "rank": args.rank,
                "batch_size": len(batch),
                "success": count,
                "completed": completed,
                "cursor": cursor + len(batch),
                "total": len(shard),
                "elapsed_seconds": time.time() - started,
            },
        )
    append_jsonl(log_path, {"event": "finish", "time": now(), "rank": args.rank, "success": completed, "tasks": len(shard)})
    print(f"OCR rank {args.rank}: {completed}/{len(shard)} successful", flush=True)
    return 0 if completed == len(shard) else 2


if __name__ == "__main__":
    raise SystemExit(main())
