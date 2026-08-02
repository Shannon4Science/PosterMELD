"""Batch runner for poster block extraction.

Examples:
    python batch_process.py --input ./data/posters --output ./output
    python batch_process.py --input ./data/posters --output ./output --concurrency 5
    python batch_process.py --input ./data/posters --output ./output --limit 5
    python batch_process.py --input ./data/posters --output ./output --overwrite
    python batch_process.py --input ./data/posters --output ./output --visualize
"""

import os
import sys
import json
import time
import argparse
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add this package root to the import path for local execution.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import load_config, run_pipeline


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
CHECKPOINT_FILE = "checkpoint.json"


def find_images(folder: str) -> list:
    """Recursively find poster image files."""
    images = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTS:
                images.append(os.path.join(root, f))
    return sorted(images)


def load_checkpoint(output_dir: str) -> dict:
    """Load checkpoint state as {filename: status metadata}."""
    path = os.path.join(output_dir, CHECKPOINT_FILE)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(output_dir: str, checkpoint: dict):
    """Persist checkpoint state."""
    path = os.path.join(output_dir, CHECKPOINT_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)


def format_time(seconds: float) -> str:
    """Format elapsed seconds for console logs."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60)}s"
    else:
        return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60)}m"


def process_single(image_path: str, config: dict, output_dir: str) -> tuple:
    """Process one image and return (poster_name, status, error_msg)."""
    poster_name = os.path.basename(image_path)
    try:
        run_pipeline(image_path, config, output_dir)
        return (poster_name, "success", None)
    except Exception as e:
        traceback.print_exc()
        return (poster_name, "failed", str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Batch poster block extraction with slice extraction and VLM merging"
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Input folder containing poster images",
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="Output folder for JSON and visualization files",
    )
    parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="Path to a configuration YAML file",
    )
    parser.add_argument(
        "--extractor", type=str, default=None, choices=["mineru", "ppocr"],
        help="Slice extractor to use",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Reprocess files even if they are marked successful in the checkpoint",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of files to process",
    )
    parser.add_argument(
        "--concurrency", type=int, default=None,
        help="Number of concurrent workers",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Write raw and merged visualization images",
    )
    parser.add_argument(
        "--continue-on-error", action="store_true", default=True,
        help="Continue processing after individual image failures",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config or os.path.join(script_dir, "config.yaml")
    config = load_config(config_path)

    if args.extractor:
        config["extractor"] = args.extractor
    if args.visualize:
        config.setdefault("output", {})["visualize"] = True

    concurrency = args.concurrency or config.get("batch", {}).get("concurrency", 10)

    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)

    if not os.path.isdir(input_dir):
        print(f"Error: input folder does not exist - {input_dir}")
        sys.exit(1)

    all_images = find_images(input_dir)
    if not all_images:
        print(f"Error: no image files found in {input_dir}")
        sys.exit(1)

    os.makedirs(os.path.join(output_dir, "blocks_json"), exist_ok=True)
    if config.get("output", {}).get("visualize", False):
        os.makedirs(os.path.join(output_dir, "visualized"), exist_ok=True)

    if args.overwrite:
        checkpoint = {}
    else:
        checkpoint = load_checkpoint(output_dir)

    def _key(p):
        return os.path.basename(p)

    skipped = [p for p in all_images if checkpoint.get(_key(p), {}).get("status") == "success"]
    to_process = [p for p in all_images if checkpoint.get(_key(p), {}).get("status") != "success"]

    if args.limit is not None:
        to_process = to_process[: args.limit]

    total = len(to_process)
    print(f"\n{'='*70}")
    print("Batch processing")
    print(f"{'='*70}")
    print(f"Input folder:      {input_dir}")
    print(f"Output folder:     {output_dir}")
    print(f"Images found:      {len(all_images)}")
    print(f"Skipped completed: {len(skipped)}")
    print(f"To process:        {total}")
    print(f"Concurrency:       {concurrency}")
    if args.limit is not None:
        print(f"Limit:             {args.limit}")
    print(f"{'='*70}\n")

    if total == 0:
        print("No files need processing.")
        return

    start_time = time.time()
    success_count = 0
    failure_count = 0
    failures = []
    checkpoint_lock = threading.Lock()
    completed_count = 0

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_path = {
            executor.submit(process_single, img_path, config, output_dir): img_path
            for img_path in to_process
        }

        for future in as_completed(future_to_path):
            poster_name, status, err_msg = future.result()
            completed_count += 1

            elapsed = time.time() - start_time
            avg_time = elapsed / completed_count
            eta = avg_time * (total - completed_count)

            with checkpoint_lock:
                if status == "success":
                    success_count += 1
                    checkpoint[poster_name] = {
                        "status": "success",
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                else:
                    failure_count += 1
                    failures.append((poster_name, err_msg))
                    checkpoint[poster_name] = {
                        "status": "failed",
                        "error": err_msg,
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                save_checkpoint(output_dir, checkpoint)

            print(f"[{completed_count}/{total}] {poster_name}  {status}"
                  f"  (elapsed: {format_time(elapsed)}, eta: {format_time(eta)})")

    total_elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print("Batch complete")
    print(f"{'='*70}")
    print(f"Elapsed:     {format_time(total_elapsed)}")
    print(f"Concurrency: {concurrency}")
    print(f"Success:     {success_count}")
    print(f"Failed:      {failure_count}")
    if success_count > 0:
        print(f"Average:     {format_time(total_elapsed / success_count)}/image")
    print(f"{'='*70}")

    if failures:
        print("\nFailures:")
        for name, err in failures:
            print(f"  - {name}: {err[:100]}")

        log_path = os.path.join(output_dir, "failures.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"Batch processing failures ({time.strftime('%Y-%m-%d %H:%M:%S')})\n")
            f.write(f"Input:  {input_dir}\n")
            f.write(f"Output: {output_dir}\n\n")
            for name, err in failures:
                f.write(f"[{name}]\n{err}\n\n")
        print(f"\nFailure log saved to: {log_path}")


if __name__ == "__main__":
    main()
