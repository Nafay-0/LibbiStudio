#!/usr/bin/env python3
"""
RunwayML basic Photo (Text-to-Image) tester
- Loads secrets from .env
- Creates a text_to_image task
- Polls /v1/tasks/{id} until SUCCEEDED/FAILED (min 5s poll)
- Prints output URLs
- Optionally downloads outputs into ./runway_outputs/

Install:
  pip install requests python-dotenv

.env (same folder):
  RUNWAYML_API_SECRET=your_api_key_here
  RUNWAY_API_BASE=https://api.dev.runwayml.com
  RUNWAY_API_VERSION=2024-11-06
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv


DEFAULT_BASE_URL = "https://api.dev.runwayml.com"
DEFAULT_VERSION = "2024-11-06"


# -----------------------------
# Utility
# -----------------------------
def pretty(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def require_env(name: str) -> str:
    v = (os.getenv(name) or "").strip()
    if not v:
        raise SystemExit(f"Missing {name}. Put it in .env or environment.")
    return v


def runway_headers(api_key: str, version: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "X-Runway-Version": version,
        "Content-Type": "application/json",
    }


def safe_json(resp: requests.Response) -> Any:
    try:
        return resp.json()
    except Exception:
        return resp.text


def request_with_429_retries(
    method: str,
    url: str,
    *,
    headers: dict,
    json_body: dict | None = None,
    params: dict | None = None,
    timeout_s: int = 60,
    max_retries: int = 6,
) -> dict:
    """
    Retries on 429 with exponential backoff + jitter.
    Respects Retry-After header if provided.
    """
    for attempt in range(max_retries + 1):
        resp = requests.request(
            method=method.upper(),
            url=url,
            headers=headers,
            json=json_body,
            params=params,
            timeout=timeout_s,
        )

        if resp.status_code != 429:
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}: {safe_json(resp)}")
            return resp.json()

        # 429 handling
        print("\n[429 Too Many Requests]")
        print("Headers:", dict(resp.headers))
        print("Body:", safe_json(resp))

        if attempt == max_retries:
            break

        ra = resp.headers.get("Retry-After")
        if ra:
            try:
                sleep_s = float(ra)
            except ValueError:
                sleep_s = 2.0
        else:
            sleep_s = min(60.0, (2 ** attempt)) + random.random()

        print(f"Retrying in {sleep_s:.2f}s (attempt {attempt+1}/{max_retries})...")
        time.sleep(sleep_s)

    raise RuntimeError("Still receiving 429 after retries (rate limit or credits/quota).")


def extract_output_urls(task: dict) -> List[str]:
    out = task.get("output")
    if isinstance(out, list):
        return [u for u in out if isinstance(u, str) and u.startswith("http")]
    return []


def download_urls(urls: List[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, url in enumerate(urls, start=1):
        # Try to infer extension from url
        suffix = ".bin"
        lower = url.lower()
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
            if ext in lower:
                suffix = ext
                break

        path = out_dir / f"output_{i}{suffix}"
        print(f"Downloading {url} -> {path}")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)


# -----------------------------
# Runway calls
# -----------------------------
def create_text_to_image(
    base_url: str,
    api_key: str,
    version: str,
    *,
    model: str,
    prompt: str,
    ratio: str,
) -> str:
    payload = {
        "model": model,
        "promptText": prompt,
        "ratio": ratio,
    }
    resp = request_with_429_retries(
        "POST",
        f"{base_url}/v1/text_to_image",
        headers=runway_headers(api_key, version),
        json_body=payload,
    )
    task_id = resp.get("id")
    if not task_id:
        raise RuntimeError(f"Create response missing id: {resp}")
    return task_id


def get_task(base_url: str, api_key: str, version: str, task_id: str) -> dict:
    return request_with_429_retries(
        "GET",
        f"{base_url}/v1/tasks/{task_id}",
        headers=runway_headers(api_key, version),
    )


def wait_for_task(
    base_url: str,
    api_key: str,
    version: str,
    task_id: str,
    *,
    poll_s: float = 5.0,
    timeout_s: int = 1200,
) -> dict:
    poll_s = max(5.0, poll_s)  # Runway docs recommend not expecting more frequent updates than ~5s
    start = time.time()

    while True:
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for task {task_id} after {timeout_s}s")

        task = get_task(base_url, api_key, version, task_id)
        status = str(task.get("status", "")).upper()
        print(f"status={status}")

        if status == "SUCCEEDED":
            return task
        if status == "FAILED":
            raise RuntimeError(f"Task FAILED: {task}")

        time.sleep(poll_s)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    load_dotenv()  # load .env into environment

    api_key = require_env("RUNWAYML_API_SECRET")
    base_url = (os.getenv("RUNWAY_API_BASE") or DEFAULT_BASE_URL).strip()
    version = (os.getenv("RUNWAY_API_VERSION") or DEFAULT_VERSION).strip()

    ap = argparse.ArgumentParser(description="RunwayML Text-to-Image (photo) quick test")
    ap.add_argument("--prompt", required=True, help="Text prompt")
    ap.add_argument("--ratio", default="1920:1080", help="Aspect ratio like 1920:1080")
    ap.add_argument("--model", default="gen4_image", help="Model name for text_to_image")
    ap.add_argument("--poll", type=float, default=5.0, help="Polling seconds (min 5s)")
    ap.add_argument("--timeout", type=int, default=1200, help="Timeout seconds")
    ap.add_argument("--download", action="store_true", help="Download output images to ./runway_outputs/")
    args = ap.parse_args()

    print("Base URL:", base_url)
    print("Version:", version)
    print("Model:", args.model)

    task_id = create_text_to_image(
        base_url,
        api_key,
        version,
        model=args.model,
        prompt=args.prompt,
        ratio=args.ratio,
    )
    print("Task created:", task_id)

    task = wait_for_task(
        base_url,
        api_key,
        version,
        task_id,
        poll_s=args.poll,
        timeout_s=args.timeout,
    )

    print("\nFinal task JSON:")
    pretty(task)

    urls = extract_output_urls(task)
    if urls:
        print("\nOutput URLs:")
        for u in urls:
            print(u)

        if args.download:
            download_urls(urls, Path("runway_outputs") / task_id)
            print("\nDownloaded to:", Path("runway_outputs") / task_id)
    else:
        print("\nNo output URLs found in task.output. Inspect the JSON above.")


if __name__ == "__main__":
    main()
