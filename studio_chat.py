#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------
# Config
# -----------------------------
ASSETS_DIR = Path("assets")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_RUNWAY_BASE = "https://api.dev.runwayml.com"
DEFAULT_RUNWAY_VERSION = "2024-11-06"

# Per Runway docs: don’t expect updates more frequent than once every 5 seconds
MIN_POLL_SECONDS = 5.0


# Runway ratio handling
# - The Runway API expects *pixel dimension* strings for `ratio` (e.g. "1920:1080").
# - LLMs/users often provide *aspect ratios* (e.g. "16:9"). We normalize those here.
RUNWAY_ALLOWED_RATIOS: Tuple[str, ...] = (
    "1024:1024",
    "1080:1080",
    "1168:880",
    "1360:768",
    "1440:1080",
    "1080:1440",
    "1808:768",
    "1920:1080",
    "1080:1920",
    "2112:912",
    "1280:720",
    "720:1280",
    "720:720",
    "960:720",
    "720:960",
    "1680:720",
)

RUNWAY_FRIENDLY_RATIO_MAP: Dict[str, str] = {
    # common aspect ratios -> nearest/standard allowed size
    "1:1": "1024:1024",
    "square": "1024:1024",
    "4:3": "1440:1080",
    "3:4": "1080:1440",
    "16:9": "1920:1080",
    "9:16": "1080:1920",
    "21:9": "2112:912",
    "2:1": "1680:720",
    "landscape": "1920:1080",
    "portrait": "1080:1920",
}


def _ratio_to_float(r: str) -> Optional[float]:
    """
    Parses "W:H" into W/H.
    Returns None if not parseable or invalid.
    """
    try:
        a, b = r.split(":", 1)
        w = float(a)
        h = float(b)
        if w <= 0 or h <= 0:
            return None
        return w / h
    except Exception:
        return None


def normalize_runway_ratio(ratio: Any, *, default: str) -> str:
    """
    Normalize user/LLM-provided ratios into a Runway-accepted pixel ratio string.

    Accepts:
    - pixel ratios: "1920:1080"
    - aspect ratios: "16:9", "1:1", "4:3", "9:16"
    - dimension forms: "1920x1080", "1920×1080", "1920*1080"
    - words: "landscape", "portrait", "square", "default"
    If unknown, falls back to a closest allowed ratio (or the default).
    """
    if ratio is None:
        raw = ""
    else:
        raw = str(ratio)

    s = raw.strip().lower()
    if not s or s in {"default", "auto"}:
        s = default

    # normalize separators / whitespace
    s = s.replace(" ", "").replace("×", "x").replace("*", "x")

    # Map friendly aspect ratios / keywords
    mapped = RUNWAY_FRIENDLY_RATIO_MAP.get(s)
    if mapped:
        return mapped

    # Convert "WxH" -> "W:H"
    if ":" not in s and "x" in s:
        parts = s.split("x")
        if len(parts) == 2:
            try:
                w = int(float(parts[0]))
                h = int(float(parts[1]))
                s = f"{w}:{h}"
            except Exception:
                pass

    # Already allowed?
    allowed_lower = {a.lower() for a in RUNWAY_ALLOWED_RATIOS}
    if s in allowed_lower:
        # return canonical casing from allowed list
        for a in RUNWAY_ALLOWED_RATIOS:
            if a.lower() == s:
                return a
        return s  # fallback (shouldn't happen)

    # Try choosing closest allowed by aspect ratio distance
    target = _ratio_to_float(s)
    if target is not None:
        best = None
        best_err = float("inf")
        for a in RUNWAY_ALLOWED_RATIOS:
            ar = _ratio_to_float(a)
            if ar is None:
                continue
            err = abs(ar - target)
            if err < best_err:
                best_err = err
                best = a
        if best:
            # If it's not exact, still prefer "closest" to avoid a hard failure.
            print(f"[ratio] Normalized '{raw}' -> '{best}'")
            return best

    # Hard fallback: default (must still be allowed; if not, pick the first allowed)
    if default.lower() in allowed_lower:
        print(f"[ratio] Unrecognized '{raw}'. Using default '{default}'.")
        for a in RUNWAY_ALLOWED_RATIOS:
            if a.lower() == default.lower():
                return a
        return default

    print(f"[ratio] Unrecognized '{raw}' and default '{default}' not in allowed list. Using '{RUNWAY_ALLOWED_RATIOS[0]}'.")
    return RUNWAY_ALLOWED_RATIOS[0]


# -----------------------------
# Small state store
# -----------------------------
@dataclass
class Asset:
    kind: str              # "image" or "video"
    local_path: str
    source_url: Optional[str]
    created_at: str
    runway_task_id: Optional[str]


class AssetStore:
    def __init__(self, path: Path):
        self.path = path
        self.assets: List[Asset] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.assets = [Asset(**x) for x in data.get("assets", [])]
            except Exception:
                self.assets = []

    def _save(self) -> None:
        self.path.write_text(json.dumps({"assets": [asdict(a) for a in self.assets]}, indent=2), encoding="utf-8")

    def add(self, asset: Asset) -> None:
        self.assets.append(asset)
        self._save()

    def list_summary(self, limit: int = 12) -> str:
        if not self.assets:
            return "No saved assets yet."
        items = self.assets[-limit:]
        lines = []
        for i, a in enumerate(items, start=max(1, len(self.assets) - len(items) + 1)):
            lines.append(f"{i}) {a.kind} | {a.local_path} | task={a.runway_task_id or '-'}")
        return "\n".join(lines)

    def get(self, index_1based: int) -> Asset:
        if index_1based < 1 or index_1based > len(self.assets):
            raise ValueError("Asset index out of range.")
        return self.assets[index_1based - 1]

    def last_image(self) -> Optional[Asset]:
        for a in reversed(self.assets):
            if a.kind == "image":
                return a
        return None


# -----------------------------
# Runway REST client
# -----------------------------
class RunwayClient:
    def __init__(self, api_key: str, base_url: str, version: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.version = version

    def _headers(self, json_content: bool = True) -> Dict[str, str]:
        h = {
            "Authorization": f"Bearer {self.api_key}",
            "X-Runway-Version": self.version,
        }
        if json_content:
            h["Content-Type"] = "application/json"
        return h

    def _safe_json(self, resp: requests.Response) -> Any:
        try:
            return resp.json()
        except Exception:
            return resp.text

    def _request_with_429_retries(
        self,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
        params: dict | None = None,
        timeout_s: int = 60,
        max_retries: int = 6,
        json_content: bool = True,
    ) -> Any:
        url = f"{self.base_url}{path}"
        for attempt in range(max_retries + 1):
            resp = requests.request(
                method=method.upper(),
                url=url,
                headers=self._headers(json_content=json_content),
                json=json_body,
                params=params,
                timeout=timeout_s,
            )

            if resp.status_code != 429:
                if resp.status_code >= 400:
                    raise RuntimeError(f"Runway HTTP {resp.status_code}: {self._safe_json(resp)}")
                if resp.status_code == 204:
                    return {"ok": True}
                return resp.json()

            # 429
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

            print(f"[Runway 429] sleeping {sleep_s:.2f}s then retrying...")
            time.sleep(sleep_s)

        raise RuntimeError("Runway: still receiving 429 after retries (rate limit or credits/quota).")

    def create_text_to_image(self, model: str, prompt: str, ratio: str) -> str:
        payload = {"model": model, "promptText": prompt, "ratio": ratio}
        resp = self._request_with_429_retries("POST", "/v1/text_to_image", json_body=payload)
        task_id = resp.get("id")
        if not task_id:
            raise RuntimeError(f"Runway create_text_to_image missing id: {resp}")
        return task_id

    def create_text_to_video(self, model: str, prompt: str, ratio: str, duration: int, audio: bool) -> str:
        payload = {"model": model, "promptText": prompt, "ratio": ratio, "duration": duration, "audio": audio}
        resp = self._request_with_429_retries("POST", "/v1/text_to_video", json_body=payload)
        task_id = resp.get("id")
        if not task_id:
            raise RuntimeError(f"Runway create_text_to_video missing id: {resp}")
        return task_id

    def create_image_to_video(self, model: str, prompt_image_uri: str, prompt: str, ratio: str, duration: int) -> str:
        payload = {"model": model, "promptImage": prompt_image_uri, "promptText": prompt, "ratio": ratio, "duration": duration}
        resp = self._request_with_429_retries("POST", "/v1/image_to_video", json_body=payload)
        task_id = resp.get("id")
        if not task_id:
            raise RuntimeError(f"Runway create_image_to_video missing id: {resp}")
        return task_id

    def get_task(self, task_id: str) -> dict:
        # GET /v1/tasks/{id}
        return self._request_with_429_retries("GET", f"/v1/tasks/{task_id}")

    def wait_task(self, task_id: str, poll_s: float = 5.0, timeout_s: int = 1800) -> dict:
        poll_s = max(MIN_POLL_SECONDS, poll_s)
        start = time.time()
        while True:
            if time.time() - start > timeout_s:
                raise TimeoutError(f"Timed out waiting for Runway task {task_id} after {timeout_s}s")

            task = self.get_task(task_id)
            status = str(task.get("status", "")).upper()
            print(f"runway.status={status}")

            if status == "SUCCEEDED":
                return task
            if status == "FAILED":
                raise RuntimeError(f"Runway task FAILED: {json.dumps(task, indent=2)}")

            time.sleep(poll_s)

    def create_ephemeral_upload(self, filename: str) -> dict:
        # POST /v1/uploads {"filename":..., "type":"ephemeral"}
        payload = {"filename": filename, "type": "ephemeral"}
        return self._request_with_429_retries("POST", "/v1/uploads", json_body=payload)

    def upload_file_and_get_runway_uri(self, file_path: Path) -> str:
        init = self.create_ephemeral_upload(file_path.name)
        upload_url = init.get("uploadUrl")
        fields = init.get("fields") or {}
        runway_uri = init.get("runwayUri")
        if not upload_url or not isinstance(fields, dict) or not runway_uri:
            raise RuntimeError(f"Unexpected upload init response: {init}")

        with file_path.open("rb") as f:
            files = {"file": (file_path.name, f)}
            resp = requests.post(upload_url, data=fields, files=files, timeout=180)
            resp.raise_for_status()

        return runway_uri


# -----------------------------
# Download helpers
# -----------------------------
def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def guess_ext_from_url(url: str, default: str) -> str:
    lower = url.lower()
    for ext in [".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".mov", ".webm"]:
        if ext in lower:
            return ext
    return default


def download_outputs(task: dict, kind: str) -> List[Path]:
    outs = task.get("output")
    if not isinstance(outs, list) or not outs:
        return []

    saved: List[Path] = []
    task_id = task.get("id", "task")
    out_dir = ASSETS_DIR / task_id
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, url in enumerate([u for u in outs if isinstance(u, str) and u.startswith("http")], start=1):
        ext = guess_ext_from_url(url, ".bin" if kind == "image" else ".mp4")
        path = out_dir / f"{kind}_{i}{ext}"
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        path.write_bytes(r.content)
        saved.append(path)

    return saved


def file_link(p: Path) -> str:
    return f"file://{p.resolve()}"


# -----------------------------
# OpenAI tool wiring
# -----------------------------
def tool_schemas() -> List[dict]:
    return [
        {
            "type": "function",
            "name": "runway_text_to_image",
            "description": "Generate an image from a text prompt via Runway. Saves outputs locally and returns local file paths + output URLs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "ratio": {
                        "type": "string",
                        "description": 'Aspect ratio like "16:9"/"1:1"/"9:16"/"4:3" OR pixel ratio like "1920:1080". (Will be normalized to a Runway-supported value.)',
                    },
                    "model": {"type": "string", "description": "Example: gen4_image"},
                },
                "required": ["prompt"],
            },
        },
        {
            "type": "function",
            "name": "runway_text_to_video",
            "description": "Generate a video from a text prompt via Runway. Saves outputs locally and returns local file paths + output URLs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "ratio": {
                        "type": "string",
                        "description": 'Aspect ratio like "16:9"/"1:1"/"9:16"/"4:3" OR pixel ratio like "1280:720". (Will be normalized to a Runway-supported value.)',
                    },
                    "duration": {"type": "integer", "description": "Seconds"},
                    "audio": {"type": "boolean"},
                    "model": {"type": "string", "description": "Example: veo3.1"},
                },
                "required": ["prompt"],
            },
        },
        {
            "type": "function",
            "name": "runway_image_to_video",
            "description": "Generate a video from an image (local path or URL) + text prompt via Runway. If local path is provided, the file is uploaded ephemerally first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image": {"type": "string", "description": "Local file path OR https URL OR runway:// URI"},
                    "prompt": {"type": "string"},
                    "ratio": {
                        "type": "string",
                        "description": 'Aspect ratio like "16:9"/"1:1"/"9:16"/"4:3" OR pixel ratio like "1280:720". (Will be normalized to a Runway-supported value.)',
                    },
                    "duration": {"type": "integer"},
                    "model": {"type": "string", "description": "Example: gen4_turbo"},
                },
                "required": ["image", "prompt"],
            },
        },
        {
            "type": "function",
            "name": "list_saved_assets",
            "description": "List locally saved images/videos so the user can pick one by index.",
            "parameters": {"type": "object", "properties": {}},
        },
    ]


def system_instructions() -> str:
    return (
        "You are a helpful media-generation assistant.\n"
        "Your job is to guide the user through creating:\n"
        "1) text-to-image, 2) text-to-video, 3) image-to-video.\n\n"
        "Behavior rules:\n"
        "- Start by asking what the user wants to do (text-to-image, text-to-video, image-to-video).\n"
        "- If image-to-video: ask for an image (URL/runway URI/local path) OR offer to list saved assets.\n"
        '- When asking for a ratio, accept "16:9", "1:1", "9:16", "4:3" (or pixel sizes like "1920:1080").\n'
        "- When you decide to generate, call the appropriate tool.\n"
        "- After tool results come back, show the user: (a) local file link(s) and (b) remote output URL(s).\n"
        "- Then ask what they want to do next.\n"
        "- Keep responses concise and terminal-friendly.\n"
    )


# -----------------------------
# Tool implementations
# -----------------------------
def runway_text_to_image_impl(runway: RunwayClient, store: AssetStore, args: dict) -> dict:
    prompt = args["prompt"]
    ratio = normalize_runway_ratio(args.get("ratio"), default="1920:1080")
    model = args.get("model", "gen4_image")

    task_id = runway.create_text_to_image(model=model, prompt=prompt, ratio=ratio)
    task = runway.wait_task(task_id, poll_s=5.0, timeout_s=1800)
    saved = download_outputs(task, kind="image")
    urls = [u for u in (task.get("output") or []) if isinstance(u, str)]

    for p in saved:
        store.add(Asset(kind="image", local_path=str(p), source_url=urls[0] if urls else None, created_at=now_iso(), runway_task_id=task_id))

    return {
        "task_id": task_id,
        "saved_files": [str(p) for p in saved],
        "output_urls": urls,
    }


def runway_text_to_video_impl(runway: RunwayClient, store: AssetStore, args: dict) -> dict:
    prompt = args["prompt"]
    ratio = normalize_runway_ratio(args.get("ratio"), default="1280:720")
    duration = int(args.get("duration", 8))
    audio = bool(args.get("audio", True))
    model = args.get("model", "veo3.1")

    task_id = runway.create_text_to_video(model=model, prompt=prompt, ratio=ratio, duration=duration, audio=audio)
    task = runway.wait_task(task_id, poll_s=5.0, timeout_s=3600)
    saved = download_outputs(task, kind="video")
    urls = [u for u in (task.get("output") or []) if isinstance(u, str)]

    for p in saved:
        store.add(Asset(kind="video", local_path=str(p), source_url=urls[0] if urls else None, created_at=now_iso(), runway_task_id=task_id))

    return {
        "task_id": task_id,
        "saved_files": [str(p) for p in saved],
        "output_urls": urls,
    }


def runway_image_to_video_impl(runway: RunwayClient, store: AssetStore, args: dict) -> dict:
    image = args["image"]
    prompt = args["prompt"]
    ratio = normalize_runway_ratio(args.get("ratio"), default="1280:720")
    duration = int(args.get("duration", 5))
    model = args.get("model", "gen4_turbo")

    # Resolve image input
    prompt_image_uri = image
    p = Path(image)
    if p.exists() and p.is_file():
        prompt_image_uri = runway.upload_file_and_get_runway_uri(p)

    task_id = runway.create_image_to_video(model=model, prompt_image_uri=prompt_image_uri, prompt=prompt, ratio=ratio, duration=duration)
    task = runway.wait_task(task_id, poll_s=5.0, timeout_s=3600)
    saved = download_outputs(task, kind="video")
    urls = [u for u in (task.get("output") or []) if isinstance(u, str)]

    for sp in saved:
        store.add(Asset(kind="video", local_path=str(sp), source_url=urls[0] if urls else None, created_at=now_iso(), runway_task_id=task_id))

    return {
        "task_id": task_id,
        "used_prompt_image_uri": prompt_image_uri,
        "saved_files": [str(x) for x in saved],
        "output_urls": urls,
    }


def list_saved_assets_impl(store: AssetStore) -> dict:
    return {"assets": store.list_summary(limit=20)}


# -----------------------------
# Chat loop (OpenAI tool calling)
# -----------------------------
def main() -> None:
    load_dotenv()

    # Load env
    runway_key = (os.getenv("RUNWAYML_API_SECRET") or "").strip()
    if not runway_key:
        raise SystemExit("Missing RUNWAYML_API_SECRET in .env")

    runway_base = (os.getenv("RUNWAY_API_BASE") or DEFAULT_RUNWAY_BASE).strip()
    runway_ver = (os.getenv("RUNWAY_API_VERSION") or DEFAULT_RUNWAY_VERSION).strip()

    model = (os.getenv("OPENAI_MODEL") or "gpt-5").strip()

    # Init clients
    oai = OpenAI()
    runway = RunwayClient(api_key=runway_key, base_url=runway_base, version=runway_ver)
    store = AssetStore(ASSETS_DIR / "catalog.json")

    tools = tool_schemas()
    input_list: List[dict] = []

    print("\nMedia Assistant (OpenAI tool calling + Runway)")
    print("Type 'quit' to exit.\n")

    # Kick off with an initial user message to prompt the assistant to start
    input_list.append({"role": "user", "content": "Start. Ask me what you can do."})

    while True:
        # Ask model (with tools)
        response = oai.responses.create(
            model=model,
            tools=tools,
            instructions=system_instructions(),
            input=input_list,
        )

        # Append model outputs to running input list (per OpenAI function calling flow)
        input_list += response.output

        # Execute any tool calls
        tool_called = False
        for item in response.output:
            if getattr(item, "type", None) == "function_call":
                tool_called = True
                name = item.name
                args = json.loads(item.arguments) if isinstance(item.arguments, str) else (item.arguments or {})

                if name == "runway_text_to_image":
                    result = runway_text_to_image_impl(runway, store, args)
                elif name == "runway_text_to_video":
                    result = runway_text_to_video_impl(runway, store, args)
                elif name == "runway_image_to_video":
                    result = runway_image_to_video_impl(runway, store, args)
                elif name == "list_saved_assets":
                    result = list_saved_assets_impl(store)
                else:
                    result = {"error": f"Unknown tool: {name}"}

                # Provide function call output back to the model
                input_list.append(
                    {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps(result),
                    }
                )

        # If tools were called, loop once more to let the model respond with final text
        if tool_called:
            response2 = oai.responses.create(
                model=model,
                tools=tools,
                instructions=system_instructions(),
                input=input_list,
            )
            input_list += response2.output
            print("\n" + (response2.output_text or "").strip() + "\n")
        else:
            print("\n" + (response.output_text or "").strip() + "\n")

        # Print a quick local asset catalog for convenience
        print("Saved assets:")
        print(store.list_summary(limit=10))
        print()

        # Terminal user input
        user_msg = input("You: ").strip()
        if user_msg.lower() in {"q", "quit", "exit"}:
            break

        # Also allow a shortcut: "use last image"
        if user_msg.lower() == "use last image":
            last = store.last_image()
            if last:
                user_msg = f"Use this image for image-to-video: {last.local_path}"
            else:
                user_msg = "List saved assets."

        input_list.append({"role": "user", "content": user_msg})


if __name__ == "__main__":
    load_dotenv()
    main()
