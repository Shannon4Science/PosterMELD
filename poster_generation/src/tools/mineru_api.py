"""
MinerU precise parsing API adapter.

The adapter keeps MinerU-specific upload, polling, and output unpacking out of
the parser so the rest of the pipeline can keep consuming the existing
raw.md/figures/tables contract.
"""

from __future__ import annotations

import json
import os
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


@dataclass
class MinerUExtraction:
    raw_text: str
    extract_dir: Path
    zip_path: Path
    content_list_path: Optional[Path]
    content_items: List[Dict[str, Any]]
    pdf_path: Optional[Path] = None
    report: Dict[str, Any] = field(default_factory=dict)


class MinerUClient:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://mineru.net",
        model_version: str = "vlm",
        language: str = "en",
        enable_table: bool = True,
        enable_formula: bool = True,
        is_ocr: bool = False,
        request_timeout: float = 60.0,
        poll_interval: float = 5.0,
        poll_timeout: float = 900.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_version = model_version
        self.language = language
        self.enable_table = enable_table
        self.enable_formula = enable_formula
        self.is_ocr = is_ocr
        self.request_timeout = request_timeout
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout

    @classmethod
    def from_env(cls, config: Dict[str, Any] | None = None) -> "MinerUClient":
        config = config or {}
        api_key = os.getenv("MINERU_API_KEY", "").strip()
        if not api_key:
            raise ValueError("MINERU_API_KEY is not set")

        return cls(
            api_key,
            base_url=os.getenv("MINERU_BASE_URL", str(config.get("base_url") or "https://mineru.net")),
            model_version=os.getenv("MINERU_MODEL_VERSION", str(config.get("model_version") or "vlm")),
            language=os.getenv("MINERU_LANGUAGE", str(config.get("language") or "en")),
            enable_table=_env_bool("MINERU_ENABLE_TABLE", bool(config.get("enable_table", True))),
            enable_formula=_env_bool("MINERU_ENABLE_FORMULA", bool(config.get("enable_formula", True))),
            is_ocr=_env_bool("MINERU_IS_OCR", bool(config.get("is_ocr", False))),
            request_timeout=float(os.getenv("MINERU_REQUEST_TIMEOUT_SECONDS", config.get("request_timeout_seconds", 60))),
            poll_interval=float(os.getenv("MINERU_POLL_INTERVAL_SECONDS", config.get("poll_interval_seconds", 5))),
            poll_timeout=float(os.getenv("MINERU_POLL_TIMEOUT_SECONDS", config.get("poll_timeout_seconds", 900))),
        )

    def parse_pdf(self, pdf_path: str | Path, content_dir: str | Path) -> MinerUExtraction:
        pdf_path = Path(pdf_path)
        content_dir = Path(content_dir)
        extract_dir = content_dir / "mineru_raw"
        extract_dir.mkdir(parents=True, exist_ok=True)

        report: Dict[str, Any] = {
            "backend": "mineru",
            "model_version": self.model_version,
            "base_url": self.base_url,
            "pdf_name": pdf_path.name,
            "fallback_used": False,
        }

        batch = self._request_file_urls(pdf_path)
        batch_id = batch["batch_id"]
        upload_url = batch["upload_url"]
        report["batch_id"] = batch_id

        self._upload_pdf(upload_url, pdf_path)
        result = self._poll_result(batch_id, pdf_path.name)
        report.update({
            "state": result.get("state") or result.get("status"),
            "full_zip_url_present": bool(result.get("full_zip_url")),
            "err_msg": result.get("err_msg") or result.get("message") or "",
        })

        full_zip_url = result.get("full_zip_url")
        if not full_zip_url:
            raise RuntimeError(f"MinerU result missing full_zip_url for batch {batch_id}")

        zip_path = extract_dir / "mineru_result.zip"
        self._download_file(full_zip_url, zip_path)
        _safe_extract_zip(zip_path, extract_dir)

        raw_md_path = _find_first(extract_dir, ["full.md", "*_full.md", "*.md"])
        if raw_md_path is None:
            raise RuntimeError("MinerU output zip did not contain a markdown file")
        raw_text = raw_md_path.read_text(encoding="utf-8", errors="replace")
        (content_dir / "raw.md").write_text(raw_text, encoding="utf-8")

        content_list_path = _find_first(extract_dir, ["*_content_list.json", "content_list.json"])
        content_items: List[Dict[str, Any]] = []
        if content_list_path is not None:
            with content_list_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                content_items = [item for item in loaded if isinstance(item, dict)]

        report.update({
            "zip_path": str(zip_path),
            "extract_dir": str(extract_dir),
            "markdown_path": str(raw_md_path),
            "content_list_path": str(content_list_path) if content_list_path else None,
            "content_item_count": len(content_items),
        })

        return MinerUExtraction(
            raw_text=raw_text,
            extract_dir=extract_dir,
            zip_path=zip_path,
            content_list_path=content_list_path,
            content_items=content_items,
            pdf_path=pdf_path,
            report=report,
        )

    def _request_file_urls(self, pdf_path: Path) -> Dict[str, str]:
        endpoint = f"{self.base_url}/api/v4/file-urls/batch"
        payload = {
            "enable_formula": self.enable_formula,
            "enable_table": self.enable_table,
            "language": self.language,
            "model_version": self.model_version,
            "files": [
                {
                    "name": pdf_path.name,
                    "is_ocr": self.is_ocr,
                    "data_id": pdf_path.stem[:64],
                }
            ],
        }
        response = requests.post(endpoint, headers=self._headers(), json=payload, timeout=self.request_timeout)
        self._raise_for_status(response)
        data = self._response_data(response.json())
        batch_id = data.get("batch_id")
        file_urls = data.get("file_urls") or data.get("fileUrls") or []
        upload_url = _first_upload_url(file_urls)
        if not batch_id or not upload_url:
            raise RuntimeError("MinerU file-url response missing batch_id or upload URL")
        return {"batch_id": str(batch_id), "upload_url": upload_url}

    def _upload_pdf(self, upload_url: str, pdf_path: Path) -> None:
        with pdf_path.open("rb") as f:
            response = requests.put(
                upload_url,
                data=f,
                timeout=self.request_timeout,
            )
        self._raise_for_status(response)

    def _poll_result(self, batch_id: str, pdf_name: str) -> Dict[str, Any]:
        endpoint = f"{self.base_url}/api/v4/extract-results/batch/{batch_id}"
        deadline = time.time() + self.poll_timeout
        last_payload: Dict[str, Any] = {}

        while time.time() < deadline:
            response = requests.get(endpoint, headers=self._headers(), timeout=self.request_timeout)
            self._raise_for_status(response)
            last_payload = response.json()
            data = self._response_data(last_payload)
            result = _select_result_for_file(data, pdf_name)
            state = str(result.get("state") or result.get("status") or data.get("state") or "").lower()

            if result.get("full_zip_url"):
                return result
            if state in {"done", "success", "completed"} and result:
                return result
            if state in {"failed", "fail", "error"}:
                raise RuntimeError(result.get("err_msg") or result.get("message") or f"MinerU batch {batch_id} failed")

            time.sleep(self.poll_interval)

        raise TimeoutError(f"MinerU batch {batch_id} did not finish within {self.poll_timeout:.0f}s: {last_payload}")

    def _download_file(self, url: str, path: Path) -> None:
        response = requests.get(url, timeout=self.request_timeout)
        self._raise_for_status(response)
        path.write_bytes(response.content)

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _response_data(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise RuntimeError("MinerU response is not a JSON object")
        code = payload.get("code")
        if code not in (None, 0, "0", 200, "200"):
            message = payload.get("msg") or payload.get("message") or payload
            raise RuntimeError(f"MinerU API error: {message}")
        data = payload.get("data", payload)
        return data if isinstance(data, dict) else {}

    def _raise_for_status(self, response: requests.Response) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            body = response.text[:1000]
            raise requests.HTTPError(f"{exc}; body={body}") from exc


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _first_upload_url(file_urls: Any) -> Optional[str]:
    if isinstance(file_urls, list) and file_urls:
        first = file_urls[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            return first.get("url") or first.get("upload_url") or first.get("file_url")
    return None


def _select_result_for_file(data: Dict[str, Any], pdf_name: str) -> Dict[str, Any]:
    candidates = (
        data.get("extract_result")
        or data.get("extract_results")
        or data.get("results")
        or data.get("files")
        or []
    )
    if isinstance(candidates, dict):
        candidates = [candidates]
    if isinstance(candidates, list):
        dict_candidates = [item for item in candidates if isinstance(item, dict)]
        for item in dict_candidates:
            name = str(item.get("file_name") or item.get("name") or item.get("filename") or "")
            if name == pdf_name:
                return item
        if dict_candidates:
            return dict_candidates[0]
    return data


def _find_first(root: Path, patterns: List[str]) -> Optional[Path]:
    for pattern in patterns:
        matches = sorted(root.rglob(pattern))
        if matches:
            return matches[0]
    return None


def _safe_extract_zip(zip_path: Path, target_dir: Path) -> None:
    target_dir = target_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            destination = (target_dir / member.filename).resolve()
            if not str(destination).startswith(str(target_dir)):
                raise RuntimeError(f"Unsafe path in MinerU zip: {member.filename}")
        zf.extractall(target_dir)
