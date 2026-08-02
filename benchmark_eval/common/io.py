from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


@dataclass(frozen=True)
class ManifestItem:
    id: str
    method: str
    subset: str
    paper_name: str
    poster_path: str | None
    reference_poster_path: str | None
    annotation_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def safe_id(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if clean:
        return clean
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def atomic_write_text(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, destination)


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def append_jsonl(path: str | Path, value: Any, lock: threading.Lock | None = None) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
    if lock is None:
        with destination.open("a", encoding="utf-8") as handle:
            handle.write(line)
        return
    with lock:
        with destination.open("a", encoding="utf-8") as handle:
            handle.write(line)


def read_json(path: str | Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _resolve_path(value: Any, base: Path) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return str(path.resolve())


def _manifest_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"Manifest line {line_number} must be a JSON object")
            rows.append(value)
        return rows

    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        rows = value.get("items", value.get("tasks"))
        if isinstance(rows, list):
            return rows
    raise ValueError("A JSON manifest must be a list or contain an items/tasks list")


def load_manifest(path: str | Path) -> list[ManifestItem]:
    manifest_path = Path(path).expanduser().resolve()
    rows = _manifest_rows(manifest_path)
    items: list[ManifestItem] = []
    seen_ids: set[str] = set()
    seen_safe_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        item_id = str(row.get("id", row.get("task_id", ""))).strip()
        method = str(row.get("method", "")).strip()
        if not item_id or not method:
            raise ValueError(f"Manifest item {index} requires non-empty id and method")
        encoded_id = safe_id(item_id)
        if item_id in seen_ids:
            raise ValueError(f"Duplicate manifest id: {item_id}")
        if encoded_id in seen_safe_ids:
            raise ValueError(f"Manifest ids collide after path normalization: {item_id}")
        seen_ids.add(item_id)
        seen_safe_ids.add(encoded_id)
        items.append(
            ManifestItem(
                id=item_id,
                method=method,
                subset=str(row.get("subset", "")).strip(),
                paper_name=str(row.get("paper_name", row.get("paper", item_id))).strip(),
                poster_path=_resolve_path(row.get("poster_path", row.get("image_path")), manifest_path.parent),
                reference_poster_path=_resolve_path(row.get("reference_poster_path"), manifest_path.parent),
                annotation_path=_resolve_path(row.get("annotation_path"), manifest_path.parent),
            )
        )
    if not items:
        raise ValueError(f"Manifest contains no items: {manifest_path}")
    return items


def result_path(output_dir: str | Path, item: ManifestItem, filename: str) -> Path:
    return Path(output_dir).resolve() / safe_id(item.id) / filename


def image_exists(item: ManifestItem) -> bool:
    return bool(item.poster_path and Path(item.poster_path).is_file())


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL | re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("The response JSON must be an object")
    return parsed


def encode_image(
    path: str | Path,
    max_dimension: int = 2600,
    max_bytes: int = 6 * 1024 * 1024,
) -> tuple[str, dict[str, Any]]:
    source_path = Path(path)
    started = time.time()
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(source_path) as source:
        original_size = source.size
        image = source.convert("RGB")
        image.load()
    if max(image.size) > max_dimension:
        scale = max_dimension / max(image.size)
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )

    quality = 92
    while True:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True, progressive=True)
        encoded = buffer.getvalue()
        if len(encoded) <= max_bytes:
            break
        if quality > 72:
            quality -= 8
            continue
        new_size = (
            max(1, round(image.width * 0.85)),
            max(1, round(image.height * 0.85)),
        )
        if new_size == image.size:
            image.close()
            raise RuntimeError(f"Could not encode image below {max_bytes} bytes")
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        quality = 84

    metadata = {
        "source_path": str(source_path.resolve()),
        "source_bytes": source_path.stat().st_size,
        "original_width": original_size[0],
        "original_height": original_size[1],
        "sent_width": image.width,
        "sent_height": image.height,
        "sent_format": "jpeg",
        "jpeg_quality": quality,
        "sent_bytes": len(encoded),
        "sent_sha256": hashlib.sha256(encoded).hexdigest(),
        "encode_seconds": time.time() - started,
    }
    image.close()
    return base64.b64encode(encoded).decode("ascii"), metadata


def keypoint_reference(annotation_path: str | Path) -> tuple[str, dict[str, Any]]:
    annotation = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    points = annotation.get("paper_poster_keypoints")
    if not isinstance(points, list) or not points:
        raise ValueError(f"No paper_poster_keypoints in {annotation_path}")
    if not all(isinstance(point, dict) for point in points):
        raise ValueError(f"Invalid keypoint list in {annotation_path}")
    order = annotation.get("reading_order")
    by_id = {point.get("id"): point for point in points}
    if isinstance(order, list) and len(order) == len(points) and set(order) == set(by_id):
        points = [by_id[point_id] for point_id in order]
    texts = [str(point.get("key_point", "")).strip() for point in points]
    if not all(texts):
        raise ValueError(f"One or more keypoints are empty in {annotation_path}")
    return "\n".join(texts), annotation


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None
