"""
export.py   JSON 
"""

import os
import json
from typing import List, Dict


def export_results(poster_path: str, blocks: List[Dict], image_size: tuple,
                   output_dir: str, filename: str = None) -> str:
    """
     block  JSON 

    Args:
        poster_path:  poster 
        blocks:  block 
        image_size:  (width, height)
        output_dir: 
        filename:  poster 

    Returns:
        
    """
    os.makedirs(output_dir, exist_ok=True)

    if filename is None:
        filename = os.path.splitext(os.path.basename(poster_path))[0]

    result = {
        "poster_path": os.path.abspath(poster_path),
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "coord_system": "normalized [0, 1000]",
        "num_blocks": len(blocks),
        "blocks": [],
    }

    for i, block in enumerate(blocks):
        block_data = {
            "id": i,
            "label": block.get("label", "unknown"),
            "type": block["type"],
            "bbox": [round(c, 1) for c in block["bbox"]],
            "polygon": block.get("polygon"),
            "content": block.get("content", ""),
            "description": block.get("description", ""),
            "num_sub_blocks": block.get("num_sub_blocks", 1),
            "member_ids": block.get("member_ids", []),
        }
        result["blocks"].append(block_data)

    output_path = os.path.join(output_dir, f"{filename}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[Export]  {len(blocks)}  block  {output_path}")
    return output_path
