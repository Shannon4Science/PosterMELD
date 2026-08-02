"""Single-poster pipeline for block extraction, VLM merging, and export.

Examples:
    python pipeline.py --image path/to/poster.png
    python pipeline.py --image path/to/poster.png --config config.yaml
    python pipeline.py --image path/to/poster.png --extractor ppocr
"""

import os
import sys
import argparse
import yaml
from PIL import Image

# Add this package root to the import path for local execution.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.extract_mineru import MinerUExtractor
from src.vlm_merge import VLMMerger
from src.export import export_results
from src.visualize import visualize_blocks, visualize_raw_blocks


def load_config(config_path: str) -> dict:
    """Load a YAML configuration file and expand environment-variable placeholders."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return _expand_env_values(config)


def _expand_env_values(value):
    if isinstance(value, dict):
        return {k: _expand_env_values(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_values(v) for v in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def build_extractor(config: dict):
    """Create the configured slice extractor."""
    extractor_name = config.get("extractor", "mineru").lower()

    if extractor_name == "mineru":
        cfg = config["mineru"]
        return MinerUExtractor(
            token=cfg["token"],
            base_url=cfg["base_url"],
            model_version=cfg["model_version"],
            is_ocr=cfg["is_ocr"],
            enable_table=cfg["enable_table"],
            enable_formula=cfg["enable_formula"],
            language=cfg["language"],
            poll_interval=cfg["poll_interval"],
            max_poll_time=cfg["max_poll_time"],
        )

    elif extractor_name == "ppocr":
        # Delay the PaddleOCR import so MinerU-only runs do not require it.
        from src.extract_ppocr import PPOCRExtractor
        cfg = config.get("ppocr", {})
        return PPOCRExtractor(
            lang=cfg.get("lang", "en"),
            use_angle_cls=cfg.get("use_angle_cls", True),
            det_db_thresh=cfg.get("det_db_thresh", 0.3),
            min_image_area_ratio=cfg.get("min_image_area_ratio", 0.005),
        )

    else:
        raise ValueError(
            f"Unknown extractor '{extractor_name}'. Supported values: 'mineru', 'ppocr'"
        )


def run_pipeline(image_path: str, config: dict, output_dir: str = None):
    """Run the full block extraction pipeline for one poster image."""
    image_path = os.path.abspath(image_path)
    poster_name = os.path.splitext(os.path.basename(image_path))[0]

    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            config["paths"]["output_dir"],
        )
    os.makedirs(os.path.join(output_dir, "blocks_json"), exist_ok=True)

    enable_visualize = config.get("output", {}).get("visualize", False)
    if enable_visualize:
        os.makedirs(os.path.join(output_dir, "visualized"), exist_ok=True)

    with Image.open(image_path) as img:
        image_size = img.size  # (width, height)
    extractor_name = config.get("extractor", "mineru").lower()
    print(f"\n{'='*60}")
    print(f"Poster:    {poster_name}")
    print(f"Size:      {image_size[0]} x {image_size[1]}")
    print(f"Extractor: {extractor_name}")
    print(f"{'='*60}")

    # ========== Step 1: Slice extraction ==========
    print(f"\n[Step 1] Running {extractor_name} slice extraction...")
    extractor = build_extractor(config)
    raw_blocks = extractor.extract(image_path)
    print(f"  Extracted {len(raw_blocks)} fine-grained slices")

    raw_vis_path = None
    if enable_visualize:
        raw_vis_path = os.path.join(output_dir, "visualized", f"{poster_name}_raw.png")
        visualize_raw_blocks(image_path, raw_blocks, raw_vis_path)

    # ========== Step 2: VLM semantic merging ==========
    print("\n[Step 2] Running VLM semantic merging...")
    llm_cfg = config["llm"]
    merger = VLMMerger(
        api_key=llm_cfg["api_key"],
        base_url=llm_cfg["base_url"],
        model=llm_cfg["vlm_model"],
        max_tokens=llm_cfg["max_tokens"],
        temperature=llm_cfg["temperature"],
    )
    merged_blocks = merger.merge(image_path, raw_blocks, image_size)
    print(f"  Merged into {len(merged_blocks)} semantic blocks")

    print("\n  Merge summary:")
    for i, block in enumerate(merged_blocks):
        print(f"    [{i}] {block['label']} ({block['type']}) "
              f"- {block['num_sub_blocks']} slices")

    # ========== Step 3: Export ==========
    print("\n[Step 3] Exporting results...")

    json_path = export_results(
        poster_path=image_path,
        blocks=merged_blocks,
        image_size=image_size,
        output_dir=os.path.join(output_dir, "blocks_json"),
        filename=poster_name,
    )

    vis_path = None
    if enable_visualize:
        vis_path = os.path.join(output_dir, "visualized", f"{poster_name}_merged.png")
        visualize_blocks(image_path, merged_blocks, vis_path)

    print(f"\n{'='*60}")
    print("Pipeline complete")
    print(f"  JSON:       {json_path}")
    if enable_visualize:
        print(f"  Raw slices: {raw_vis_path}")
        print(f"  Merged:     {vis_path}")
    print(f"{'='*60}\n")

    return merged_blocks


def main():
    parser = argparse.ArgumentParser(description="Poster block extraction pipeline")
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to a poster image",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a configuration YAML file",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--extractor", type=str, default=None, choices=["mineru", "ppocr"],
        help="Slice extractor to use",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Write raw and merged visualization images",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config or os.path.join(script_dir, "config.yaml")
    config = load_config(config_path)

    if args.extractor:
        config["extractor"] = args.extractor

    if args.visualize:
        config.setdefault("output", {})["visualize"] = True

    if args.image:
        image_path = args.image
    else:
        image_path = os.path.join(script_dir, config["paths"]["example_poster"])

    if not os.path.exists(image_path):
        print(f"Error: image not found - {image_path}")
        sys.exit(1)

    run_pipeline(image_path, config, args.output)


if __name__ == "__main__":
    main()
