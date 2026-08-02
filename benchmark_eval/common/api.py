from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv


load_dotenv(override=False)


def load_api_keys(key_file: str | None = None) -> tuple[str, ...]:
    if key_file:
        keys = tuple(
            line.strip()
            for line in Path(key_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    else:
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        keys = (key,) if key else ()
    if not keys:
        raise RuntimeError("Set OPENAI_API_KEY or pass --api-key-file")
    if len(set(keys)) != len(keys):
        raise ValueError("API key file contains duplicate keys")
    return keys


def key_fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def response_to_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    return json.loads(json.dumps(response, default=str))


def response_text(response: Any) -> str:
    content = response.choices[0].message.content
    if isinstance(content, str) and content.strip():
        return content
    reasoning = getattr(response.choices[0].message, "reasoning", None)
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning
    return ""


class ApiPool:
    """OpenAI-compatible clients with a strict in-flight limit per API key."""

    def __init__(
        self,
        keys: tuple[str, ...],
        base_url: str,
        timeout: float,
        workers_per_key: int,
    ) -> None:
        if workers_per_key < 1:
            raise ValueError("workers_per_key must be positive")
        self.keys = keys
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.workers_per_key = workers_per_key
        self.semaphores = [threading.BoundedSemaphore(workers_per_key) for _ in keys]
        self.local = threading.local()

    @property
    def global_workers(self) -> int:
        return len(self.keys) * self.workers_per_key

    @property
    def fingerprints(self) -> list[str]:
        return [key_fingerprint(key) for key in self.keys]

    def client(self, slot: int) -> OpenAI:
        clients = getattr(self.local, "clients", None)
        if clients is None:
            clients = {}
            self.local.clients = clients
        if slot not in clients:
            clients[slot] = OpenAI(
                api_key=self.keys[slot],
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=0,
            )
        return clients[slot]

    def create(self, slot: int, **kwargs: Any) -> Any:
        with self.semaphores[slot]:
            return self.client(slot).chat.completions.create(**kwargs)
