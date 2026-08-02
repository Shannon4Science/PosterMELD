"""VLM-based semantic merging for fine-grained poster slices."""

import os
import re
import json
import base64
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union
import requests
from typing import List, Dict


class VLMMerger:
    """Merge fine-grained OCR/layout slices into semantic poster blocks."""

    MAX_RETRIES = 5

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1",
                 model: str = "gpt-4o", max_tokens: int = 8192,
                 temperature: float = 0.1):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def merge(self, poster_image_path: str, raw_blocks: List[Dict],
              image_size: tuple = None) -> List[Dict]:
        """Merge raw slices into semantically coherent poster blocks."""
        if not raw_blocks:
            print("[VLM] No slices to merge")
            return []

        # 
        img_base64 = self._encode_image(poster_image_path)

        # 
        blocks_desc = self._format_blocks(raw_blocks)

        #  prompt
        prompt = self._build_prompt(len(raw_blocks), blocks_desc)

        #  VLM MAX_RETRIES 
        merged_blocks = None
        last_error = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                print(f"[VLM] Attempt {attempt}/{self.MAX_RETRIES}, model: {self.model}")
                response = self._call_vlm(img_base64, prompt, self.model)
                merged_blocks = self._parse_response(response, raw_blocks, image_size)
                break
            except Exception as e:
                last_error = e
                print(f"[VLM] Attempt {attempt} failed: {e}")

        if merged_blocks is None:
            raise RuntimeError(
                f"VLM model {self.model} failed after {self.MAX_RETRIES} attempts: {last_error}"
            )

        #  block  L 
        merged_blocks = self._compute_polygons(merged_blocks)

        print(f"[VLM] Merge complete: {len(raw_blocks)} slices -> {len(merged_blocks)} blocks")
        return merged_blocks

    def _encode_image(self, image_path: str) -> str:
        """Encode an image as base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _format_blocks(self, blocks: List[Dict]) -> str:
        """Format slice metadata for the VLM prompt."""
        lines = []
        for b in blocks:
            bbox = b["bbox"]
            content = b.get("content", "")
            #  80 
            if len(content) > 80:
                content = content[:80] + "..."
            content = content.replace("\n", " ").strip()

            line = (
                f"  [{b['id']}] type={b['type']}, "
                f"bbox=({bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f})"
            )
            if content:
                line += f', content="{content}"'
            lines.append(line)
        return "\n".join(lines)

    def _build_prompt(self, num_blocks: int, blocks_desc: str) -> str:
        """Build the VLM prompt."""
        return f"""You are an expert in academic poster structure analysis.

I provide one academic poster image and a list of fine-grained OCR/layout slices extracted from the poster. These slices are too small to serve as layout units. Your task is to merge them into a small set of semantically independent poster blocks.

Each output block should correspond to one complete content region, such as:
- Title: paper title, authors, affiliations, logos, top banner, and other header elements. Do not split title, authors, affiliations, and logos into separate blocks.
- Abstract, Introduction, or Background
- Method or Approach
- Experiments, Evaluation, or Results
- Conclusion
- References
- Other independent regions, such as Acknowledgement or QR Code

Slice coordinates use normalized poster coordinates in [0, 1000]. The origin is the top-left corner and (1000, 1000) is the bottom-right corner.

Slice list ({num_blocks} slices):
{blocks_desc}

Return only strict JSON with this schema:
```json
{{
    "blocks": [
        {{
            "label": "semantic label such as Title, Method, or Results",
            "type": "text, image, or mixed",
            "member_ids": [0, 1, 2],
            "description": "one-sentence block description"
        }}
    ]
}}
```

Rules:
1. Every slice must belong to exactly one output block.
2. Adjacent slices from the same section should be merged.
3. Slices from different sections should not be merged only because they are spatially close.
4. Image slices and their related captions or explanatory text may form mixed blocks.
5. A typical academic poster should contain roughly 5 to 10 semantic blocks.
6. All top-header elements must belong to one Title block."""

    def _call_vlm(self, img_base64: str, prompt: str, model: str) -> str:
        """ VLM API"""
        # 
        img_url = f"data:image/png;base64,{img_base64}"

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": img_url},
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        content = data["choices"][0]["message"]["content"]
        print(f"[VLM] Model {model} returned {len(content)} characters")
        return content

    def _parse_response(self, response: str, raw_blocks: List[Dict],
                        image_size: tuple = None) -> List[Dict]:
        """Parse the VLM JSON response."""
        # Extract JSON that may be wrapped in a markdown code block.
        json_str = self._extract_json(response)
        result = json.loads(json_str)

        all_assigned_ids = set()
        merged_blocks = []

        for block_info in result["blocks"]:
            member_ids = block_info["member_ids"]
            all_assigned_ids.update(member_ids)

            members = []
            for mid in member_ids:
                found = [b for b in raw_blocks if b["id"] == mid]
                if found:
                    members.append(found[0])

            if not members:
                continue

            x1 = min(m["bbox"][0] for m in members)
            y1 = min(m["bbox"][1] for m in members)
            x2 = max(m["bbox"][2] for m in members)
            y2 = max(m["bbox"][3] for m in members)

            texts = [m.get("content", "") for m in members if m.get("content")]

            has_image = any(m["type"] == "image" for m in members)
            has_text = any(m["type"] in ("text", "table", "equation") for m in members)
            if has_image and has_text:
                block_type = "mixed"
            elif has_image:
                block_type = "image"
            else:
                block_type = "text"

            merged_blocks.append({
                "label": block_info["label"],
                "type": block_type,
                "bbox": [x1, y1, x2, y2],
                "content": "\n".join(texts),
                "description": block_info.get("description", ""),
                "member_ids": member_ids,
                "num_sub_blocks": len(members),
            })

        all_ids = {b["id"] for b in raw_blocks}
        missing_ids = all_ids - all_assigned_ids
        if missing_ids:
            print(f"[VLM] Warning: {len(missing_ids)} slices were not assigned: {missing_ids}")
            for mid in missing_ids:
                block = [b for b in raw_blocks if b["id"] == mid][0]
                merged_blocks.append({
                    "label": "Unassigned",
                    "type": block["type"],
                    "bbox": block["bbox"],
                    "content": block.get("content", ""),
                    "description": "Slice was not assigned by the VLM",
                    "member_ids": [mid],
                    "num_sub_blocks": 1,
                })

        return merged_blocks

    def _compute_polygons(self, blocks: List[Dict]) -> List[Dict]:
        """Compute non-overlapping polygon boundaries for merged blocks."""
        if len(blocks) < 2:
            for block in blocks:
                bx = block["bbox"]
                block["polygon"] = {
                    "exterior": [[bx[0], bx[1]], [bx[2], bx[1]],
                                 [bx[2], bx[3]], [bx[0], bx[3]]],
                    "holes": [],
                }
            return blocks

        shapes = []
        for block in blocks:
            bx = block["bbox"]
            shapes.append(shapely_box(bx[0], bx[1], bx[2], bx[3]))

        for i, block in enumerate(blocks):
            area_i = shapes[i].area
            current_shape = shapes[i]

            cutouts = []
            for j, other in enumerate(blocks):
                if i == j:
                    continue
                if not current_shape.intersects(shapes[j]):
                    continue
                area_j = shapes[j].area
                if area_i > area_j:
                    cutouts.append(shapes[j])

            if cutouts:
                # Slightly expand cutouts to avoid narrow residual holes.
                gap_tolerance = 20
                expanded_cutouts = []
                for cutout in cutouts:
                    expanded = cutout.buffer(gap_tolerance, join_style=2)
                    expanded = expanded.intersection(current_shape)
                    expanded_cutouts.append(expanded)

                cut_union = unary_union(expanded_cutouts)
                result_shape = current_shape.difference(cut_union)

                polygon_data = self._shapely_to_polygon(result_shape)
                block["polygon"] = polygon_data

                cut_labels = [blocks[j]["label"] for j in range(len(blocks))
                              if j != i and shapes[j].area < area_i
                              and current_shape.intersects(shapes[j])]
                n_ext = len(polygon_data["exterior"])
                n_holes = len(polygon_data["holes"])
                shape_desc = f"{n_ext}-vertex polygon"
                if n_holes > 0:
                    shape_desc += f" with {n_holes} holes"
                print(f"[VLM] Post-processed block[{i}]({block['label']}) "
                      f"as {shape_desc}; subtracted {', '.join(cut_labels)}")
            else:
                bx = block["bbox"]
                block["polygon"] = {
                    "exterior": [[bx[0], bx[1]], [bx[2], bx[1]],
                                 [bx[2], bx[3]], [bx[0], bx[3]]],
                    "holes": [],
                }

        return blocks

    @staticmethod
    def _shapely_to_polygon(shape) -> dict:
        """Convert a shapely geometry into the exported polygon schema."""
        from shapely.geometry import Polygon, MultiPolygon

        if isinstance(shape, MultiPolygon):
            shape = max(shape.geoms, key=lambda g: g.area)

        if isinstance(shape, Polygon):
            exterior = [[round(x, 1), round(y, 1)]
                        for x, y in list(shape.exterior.coords)[:-1]]
            holes = []
            for interior in shape.interiors:
                hole = [[round(x, 1), round(y, 1)]
                        for x, y in list(interior.coords)[:-1]]
                holes.append(hole)
            return {"exterior": exterior, "holes": holes}

        # fallback
        env = shape.envelope
        coords = [[round(x, 1), round(y, 1)]
                  for x, y in list(env.exterior.coords)[:-1]]
        return {"exterior": coords, "holes": []}

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract a JSON object from a VLM response."""
        # Try a fenced json block first.
        pattern = r"```json\s*([\s\S]*?)\s*```"
        match = re.search(pattern, text)
        if match:
            return match.group(1)

        # Try any fenced code block.
        pattern = r"```\s*([\s\S]*?)\s*```"
        match = re.search(pattern, text)
        if match:
            return match.group(1)

        # Fall back to the outermost JSON-looking object.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return text[start:end + 1]

        raise ValueError(f"Could not extract JSON from VLM output:\n{text[:500]}")
