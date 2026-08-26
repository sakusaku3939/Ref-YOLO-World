#!/usr/bin/env python3
"""Shared file and Hugging Face rows helpers for dataset preparation."""

from __future__ import annotations

import json
import random
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


HF_ROWS_API = "https://datasets-server.huggingface.co/rows"
REQUEST_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Ref-YOLO-World/dataset-preparation",
}
RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers=REQUEST_HEADERS)


def _retry_after(exc: urllib.error.HTTPError) -> float | None:
    value = exc.headers.get("Retry-After") if exc.headers else None
    if value is None:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        return None


def _fetch_page(
    dataset: str,
    split: str,
    offset: int,
    length: int,
    max_retries: int,
    backoff_sec: float,
) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": "default",
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    url = f"{HF_ROWS_API}?{query}"
    attempt = 0
    while True:
        retry_after = None
        try:
            with urllib.request.urlopen(request(url), timeout=60) as response:
                payload = json.load(response)
            return [item["row"] for item in payload.get("rows", [])]
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRYABLE_HTTP_CODES or attempt >= max_retries:
                raise
            retry_after = _retry_after(exc)
        except (TimeoutError, urllib.error.URLError):
            if attempt >= max_retries:
                raise

        delay = retry_after
        if delay is None:
            delay = backoff_sec * (2**attempt)
            delay += random.uniform(0.0, min(backoff_sec, 1.0))
        print(
            f"retrying {dataset}:{split} offset={offset} "
            f"attempt={attempt + 1}/{max_retries} sleep={delay:.1f}s"
        )
        time.sleep(delay)
        attempt += 1


def fetch_hf_rows(
    dataset: str,
    split: str,
    *,
    chunk_size: int = 100,
    sleep_sec: float = 0.1,
    max_retries: int = 8,
    backoff_sec: float = 5.0,
) -> list[dict[str, Any]]:
    if not 1 <= chunk_size <= 100:
        raise ValueError("chunk_size must be between 1 and 100")
    rows = []
    offset = 0
    while True:
        page = _fetch_page(
            dataset,
            split,
            offset,
            chunk_size,
            max_retries,
            backoff_sec,
        )
        rows.extend(page)
        if len(page) < chunk_size:
            return rows
        offset += len(page)
        if sleep_sec > 0:
            time.sleep(sleep_sec)


def download_file(url: str, destination: Path, *, force: bool = False) -> None:
    if destination.is_file() and not force:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    try:
        with urllib.request.urlopen(request(url), timeout=60) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
