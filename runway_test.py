#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

DEFAULT_BASE_URL = "https://api.dev.runwayml.com"
DEFAULT_VERSION = "2024-11-06"


# -----------------------------
# Helpers
# -----------------------------
def pretty(x: Any) -> None:
    print(json.dumps(x, indent=2, ensure_ascii=False))


def require_env(name: str) -> str:
    v = (os.getenv(name) or "").strip()
    if not v:
        raise SystemExit(f"Missing {name}. Put it in .env or environment.")
    return v


def runway_headers(api_key: str, version: str, *, json_content: bool = True) -> Dict[str, str]:
    h = {
        "Authorization": f"Bearer {api_key}",
        "X-Runway-Version": version,
    }
    if json_content:
        h["Content-Type"] = "application/json"
    return h


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
    Respects Retry-After header if present.
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
            if resp.status_code == 204:
                return {"ok": True}
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


def get_task(base_url: str, api_key: str, version: str, task_id: str) -> dict:
    return request_with_429_retries(
        "GET",
        f"{base_url}/v1/tasks/{task_id}",
        headers=runway_headers(api_key, version, json_content=True),
    )


def wait_task(base_url: str, api_key: str, version: str, task_id: str, *, poll_s: float = 5.0, timeout_s: int = 1800) -> dict:
    poll_s = max(5.0, poll_s)  # Runway tasks usually don't update more frequently than ~5s
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
            raise RuntimeError(f"Task FAILED:\n{json.dumps(task, indent=2)}")

        time.sleep(poll_s)


def create_task(base_url: str, api_key: str, version: str, endpoint: str, payload: dict) -> str:
    resp = request_with_429_retries(
        "POST",
        f"{base_url}{endpoint}",
        headers=runway_headers(api_key, version, json_content=True),
        json_body=payload,
    )
    task_id = resp.get("id")
    if not task_id:
        raise RuntimeError(f"Create response missing id: {resp}")
    return task_id


def cancel_or_delete_task(base_url: str, api_key: str, version: str, task_id: str) -> dict:
    return request_with_429_retries(
        "DELETE",
        f"{base_url}/v1/tasks/{task_id}",
        headers=runway_headers(api_key, version, json_content=False),
        max_retries=3,
    )


# -----------------------------
# Commands
# -----------------------------
def cmd_text2image(args: argparse.Namespace) -> None:
    api_key = require_env("RUNWAYML_API_SECRET")
    base_url = (os.getenv("RUNWAY_API_BASE") or DEFAULT_BASE_URL).strip()
    version = (os.getenv("RUNWAY_API_VERSION") or DEFAULT_VERSION).strip()

    payload = {
        "model": args.model,
        "promptText": args.prompt,
        "ratio": args.ratio,
    }

    task_id = create_task(base_url, api_key, version, "/v1/text_to_image", payload)
    print("Task:", task_id)
    task = wait_task(base_url, api_key, version, task_id, poll_s=args.poll, timeout_s=args.timeout)

    print("\nFinal task:")
    pretty(task)

    urls = extract_output_urls(task)
    if urls:
        print("\nOutput URLs:")
        for u in urls:
            print(u)


def cmd_image2video(args: argparse.Namespace) -> None:
    api_key = require_env("RUNWAYML_API_SECRET")
    base_url = (os.getenv("RUNWAY_API_BASE") or DEFAULT_BASE_URL).strip()
    version = (os.getenv("RUNWAY_API_VERSION") or DEFAULT_VERSION).strip()

    payload = {
        "model": args.model,
        "promptImage": args.image,   # accepts https://... or runway://... or data URI (depending on Runway)
        "promptText": args.prompt,
        "ratio": args.ratio,
        "duration": args.duration,
    }

    task_id = create_task(base_url, api_key, version, "/v1/image_to_video", payload)
    print("Task:", task_id)
    task = wait_task(base_url, api_key, version, task_id, poll_s=args.poll, timeout_s=args.timeout)

    print("\nFinal task:")
    pretty(task)

    urls = extract_output_urls(task)
    if urls:
        print("\nOutput URLs:")
        for u in urls:
            print(u)


def cmd_text2video(args: argparse.Namespace) -> None:
    api_key = require_env("RUNWAYML_API_SECRET")
    base_url = (os.getenv("RUNWAY_API_BASE") or DEFAULT_BASE_URL).strip()
    version = (os.getenv("RUNWAY_API_VERSION") or DEFAULT_VERSION).strip()

    payload = {
        "model": args.model,
        "promptText": args.prompt,
        "ratio": args.ratio,
        "duration": args.duration,
        "audio": args.audio,
    }

    task_id = create_task(base_url, api_key, version, "/v1/text_to_video", payload)
    print("Task:", task_id)
    task = wait_task(base_url, api_key, version, task_id, poll_s=args.poll, timeout_s=args.timeout)

    print("\nFinal task:")
    pretty(task)

    urls = extract_output_urls(task)
    if urls:
        print("\nOutput URLs:")
        for u in urls:
            print(u)


def cmd_task(args: argparse.Namespace) -> None:
    api_key = require_env("RUNWAYML_API_SECRET")
    base_url = (os.getenv("RUNWAY_API_BASE") or DEFAULT_BASE_URL).strip()
    version = (os.getenv("RUNWAY_API_VERSION") or DEFAULT_VERSION).strip()
    pretty(get_task(base_url, api_key, version, args.id))


def cmd_cancel(args: argparse.Namespace) -> None:
    api_key = require_env("RUNWAYML_API_SECRET")
    base_url = (os.getenv("RUNWAY_API_BASE") or DEFAULT_BASE_URL).strip()
    version = (os.getenv("RUNWAY_API_VERSION") or DEFAULT_VERSION).strip()
    pretty(cancel_or_delete_task(base_url, api_key, version, args.id))


def cmd_org(args: argparse.Namespace) -> None:
    api_key = require_env("RUNWAYML_API_SECRET")
    base_url = (os.getenv("RUNWAY_API_BASE") or DEFAULT_BASE_URL).strip()
    version = (os.getenv("RUNWAY_API_VERSION") or DEFAULT_VERSION).strip()

    resp = request_with_429_retries(
        "GET",
        f"{base_url}/v1/organization",
        headers=runway_headers(api_key, version, json_content=True),
    )
    pretty(resp)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RunwayML API test script (text2image, image2video, text2video)")
    sub = p.add_subparsers(required=True)

    # text2image
    s = sub.add_parser("text2image", help="Create text-to-image task and wait for output.")
    s.add_argument("--prompt", required=True)
    s.add_argument("--ratio", default="1920:1080")
    s.add_argument("--model", default="gen4_image")
    s.add_argument("--poll", type=float, default=5.0)
    s.add_argument("--timeout", type=int, default=1200)
    s.set_defaults(func=cmd_text2image)

    # image2video
    s = sub.add_parser("image2video", help="Create image-to-video task and wait for output.")
    s.add_argument("--image", required=True, help="Image URL or runway://... (if supported)")
    s.add_argument("--prompt", required=True)
    s.add_argument("--ratio", default="1280:720")
    s.add_argument("--duration", type=int, default=5)
    s.add_argument("--model", default="gen4_turbo")
    s.add_argument("--poll", type=float, default=5.0)
    s.add_argument("--timeout", type=int, default=1800)
    s.set_defaults(func=cmd_image2video)

    # text2video
    s = sub.add_parser("text2video", help="Create text-to-video task and wait for output.")
    s.add_argument("--prompt", required=True)
    s.add_argument("--ratio", default="1280:720")
    s.add_argument("--duration", type=int, default=8)
    s.add_argument("--audio", action=argparse.BooleanOptionalAction, default=True)
    s.add_argument("--model", default="veo3.1")
    s.add_argument("--poll", type=float, default=5.0)
    s.add_argument("--timeout", type=int, default=1800)
    s.set_defaults(func=cmd_text2video)

    # optional utilities
    s = sub.add_parser("task", help="Fetch a task by id.")
    s.add_argument("--id", required=True)
    s.set_defaults(func=cmd_task)

    s = sub.add_parser("cancel", help="Cancel/delete a task by id.")
    s.add_argument("--id", required=True)
    s.set_defaults(func=cmd_cancel)

    s = sub.add_parser("org", help="Get organization info.")
    s.set_defaults(func=cmd_org)

    return p


def main() -> None:
    load_dotenv()  # loads .env
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
