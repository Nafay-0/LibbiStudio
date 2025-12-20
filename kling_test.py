from __future__ import annotations

import os
import time
import random
import pathlib
import base64
from typing import Any, Optional
from dotenv import load_dotenv

import requests  # pip install requests
import jwt       # pip install PyJWT   (IMPORTANT: not the "jwt" package)

DEFAULT_BASE_URL = "https://api.klingai.com"

# Defaults are intentionally configurable via env vars because Kling’s endpoints/models
# can vary by account/region and evolve over time. Reference:
# https://app.klingai.com/global/dev/document-api/apiReference/model/imageToVideo
BASE_URL = (os.environ.get("KLING_API_BASE") or DEFAULT_BASE_URL).strip().rstrip("/")

DEFAULT_I2V_CREATE_PATH = os.environ.get("KLING_I2V_CREATE_PATH", "/v1/videos/image2video").strip() or "/v1/videos/image2video"
DEFAULT_I2V_STATUS_PATH_TEMPLATE = (
    os.environ.get("KLING_I2V_STATUS_PATH_TEMPLATE", "/v1/videos/image2video/{task_id}").strip()
    or "/v1/videos/image2video/{task_id}"
)
# Per Kling docs, prefer model_name (model remains forward-compatible).
DEFAULT_I2V_MODEL_NAME = (os.environ.get("KLING_I2V_MODEL_NAME") or "kling-v1").strip()
DEFAULT_I2V_MODEL_LEGACY = (os.environ.get("KLING_I2V_MODEL") or "").strip()


# -----------------------------
# JWT snippet (updated for PyJWT)
# -----------------------------
load_dotenv()
def encode_jwt_token(ak: str, sk: str) -> str:
    headers = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "iss": ak,
        "exp": int(time.time()) + 1800,  # now + 30min
        "nbf": int(time.time()) - 5,     # now - 5s
    }
    token = jwt.encode(payload, sk, algorithm="HS256", headers=headers)
    if isinstance(token, bytes):  # older PyJWT may return bytes
        token = token.decode("utf-8")
    return token


def auth_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def safe_json(resp: requests.Response) -> dict | str:
    try:
        return resp.json()
    except Exception:
        return resp.text


def post_with_retries(
    url: str,
    *,
    headers: dict,
    payload: dict,
    max_retries: int = 6,
    timeout_s: int = 60,
) -> requests.Response:
    """
    Retries on 429 with exponential backoff + jitter.
    Uses Retry-After header if present.
    """
    for attempt in range(max_retries + 1):
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)

        if resp.status_code != 429:
            if resp.status_code >= 400:
                raise requests.HTTPError(f"HTTP {resp.status_code}: {safe_json(resp)}", response=resp)
            return resp

        print("\n[429 Too Many Requests]")
        print("Headers:", dict(resp.headers))
        print("Body:", safe_json(resp))

        retry_after = resp.headers.get("Retry-After")
        if retry_after is not None:
            try:
                sleep_s = float(retry_after)
            except ValueError:
                sleep_s = 2.0
        else:
            sleep_s = min(60.0, (2 ** attempt)) + random.random()

        if attempt == max_retries:
            break

        print(f"Retrying in {sleep_s:.2f}s (attempt {attempt + 1}/{max_retries})...")
        time.sleep(sleep_s)

    raise RuntimeError("Still receiving 429 after all retries (rate limit or quota/credits).")


def get_with_retries(
    url: str,
    *,
    headers: dict,
    max_retries: int = 4,
    timeout_s: int = 60,
) -> requests.Response:
    for attempt in range(max_retries + 1):
        resp = requests.get(url, headers=headers, timeout=timeout_s)

        if resp.status_code != 429:
            if resp.status_code >= 400:
                raise requests.HTTPError(f"HTTP {resp.status_code}: {safe_json(resp)}", response=resp)
            return resp

        retry_after = resp.headers.get("Retry-After")
        if retry_after is not None:
            try:
                sleep_s = float(retry_after)
            except ValueError:
                sleep_s = 2.0
        else:
            sleep_s = min(30.0, (2 ** attempt)) + random.random()

        print(f"GET 429. Sleeping {sleep_s:.2f}s then retrying...")
        time.sleep(sleep_s)

    raise RuntimeError("GET still receiving 429 after retries.")


def extract_first_url(obj: Any) -> Optional[str]:
    """Best-effort: find a URL in nested dict/list response."""
    if isinstance(obj, dict):
        for k in ["url", "image_url", "video_url", "download_url", "src"]:
            v = obj.get(k)
            if isinstance(v, str) and v.startswith("http"):
                return v
        for v in obj.values():
            found = extract_first_url(v)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = extract_first_url(item)
            if found:
                return found
    return None


def wait_for_task(task_id: str, token: str, *, status_path_template: str, timeout_s: int = 900, poll_s: int = 5) -> dict:
    start = time.time()
    status_url = f"{BASE_URL}{status_path_template.format(task_id=task_id)}"

    while True:
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for task {task_id} after {timeout_s}s")

        resp = get_with_retries(status_url, headers=auth_headers(token))
        body = resp.json()

        data = body.get("data", body)
        status = None
        if isinstance(data, dict):
            status = data.get("task_status") or data.get("status")

        if isinstance(status, str):
            s = status.lower()
            if s in {"succeed", "succeeded", "success", "completed", "complete", "done"}:
                return body
            if s in {"failed", "fail", "error"}:
                raise RuntimeError(f"Task failed: {body}")

        time.sleep(poll_s)


def _read_image_as_base64_no_prefix(path: pathlib.Path) -> str:
    """
    Kling Image-to-Video supports passing `image` as:
    - an image URL, OR
    - raw Base64 string (NO 'data:image/...;base64,' prefix).
    Docs: https://app.klingai.com/global/dev/document-api/apiReference/model/imageToVideo
    """
    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def resolve_image_param(image: str) -> str:
    """
    Accept:
    - https://... URL (passed through)
    - local file path (encoded as raw base64, without data: prefix)
    - raw base64 (best-effort: if it doesn't look like a path or URL)
    """
    s = (image or "").strip()
    if not s:
        raise ValueError("image is required (URL, local path, or raw base64).")

    if s.startswith("http://") or s.startswith("https://"):
        return s

    p = pathlib.Path(s)
    if p.exists() and p.is_file():
        return _read_image_as_base64_no_prefix(p)

    # Fallback: treat as already-base64
    return s


def find_default_image() -> Optional[str]:
    """
    Convenience: try to find a reasonable local PNG/JPG in the repo to use as a default.
    """
    # 1) A commonly produced Runway output file in this repo
    runway_dir = pathlib.Path("runway_outputs")
    if runway_dir.exists():
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for p in runway_dir.rglob(ext):
                if p.is_file():
                    return str(p)

    # 2) Any asset image in assets/
    assets_dir = pathlib.Path("assets")
    if assets_dir.exists():
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for p in assets_dir.rglob(ext):
                if p.is_file():
                    return str(p)

    return None


def main() -> None:
    # Prefer env vars; do NOT paste secrets into chat
    ak = os.environ.get("KLING_ACCESS_KEY", "").strip()
    sk = os.environ.get("KLING_SECRET_KEY", "").strip()


    if not ak or not sk:
        raise SystemExit(
            "Set env vars first:\n"
            '  export KLING_ACCESS_KEY="..."\n'
            '  export KLING_SECRET_KEY="..."\n'
        )

    # Generate + print token (debug)
    authorization = encode_jwt_token(ak, sk)
    print("API_TOKEN:", authorization)

    # --- Create image-to-video task ---
    # Endpoint/model reference:
    # https://app.klingai.com/global/dev/document-api/apiReference/model/imageToVideo
    create_url = f"{BASE_URL}{DEFAULT_I2V_CREATE_PATH}"

    # Image can be URL or local path (we'll base64-encode local files without prefix).
    image_in = (os.environ.get("KLING_I2V_IMAGE") or "").strip()
    if not image_in:
        found = find_default_image()
        if found:
            image_in = found
            print(f"Using default image: {image_in}")
        else:
            raise SystemExit(
                "Set KLING_I2V_IMAGE to an https URL or local image path (.png/.jpg/.jpeg).\n"
                "Example:\n"
                '  export KLING_I2V_IMAGE="runway_outputs/.../output_1.png"\n'
            )
    image_param = resolve_image_param(image_in)

    payload = {
        "model_name": DEFAULT_I2V_MODEL_NAME,
        "image": image_param,
        # prompt is optional per docs; but usually gives better results
        "prompt": (os.environ.get("KLING_I2V_PROMPT") or "The camera slowly pushes in; subtle motion; cinematic").strip(),
        "negative_prompt": (os.environ.get("KLING_I2V_NEGATIVE_PROMPT") or "blurry, low quality, artifacts").strip(),
        # duration enum values: "5", "10" (docs list string)
        "duration": (os.environ.get("KLING_I2V_DURATION") or "5").strip(),
        # mode: std (default) or pro
        "mode": (os.environ.get("KLING_I2V_MODE") or "std").strip(),
        # cfg_scale: float in [0,1] (not supported by kling-v2.x per docs)
        "cfg_scale": float(os.environ.get("KLING_I2V_CFG_SCALE") or "0.5"),
    }

    # Optional: end-frame control (image_tail)
    image_tail_in = (os.environ.get("KLING_I2V_IMAGE_TAIL") or "").strip()
    if image_tail_in:
        payload["image_tail"] = resolve_image_param(image_tail_in)

    if DEFAULT_I2V_MODEL_LEGACY:
        payload["model"] = DEFAULT_I2V_MODEL_LEGACY

    create_resp = post_with_retries(
        create_url,
        headers=auth_headers(authorization),
        payload=payload,
    ).json()
    print("\nCreate response:", create_resp)

    task_id = create_resp.get("data", {}).get("task_id")
    if not task_id:
        raise RuntimeError(f"Could not find task_id in response:\n{create_resp}")

    # --- Poll until done ---
    done_resp = wait_for_task(
        task_id,
        authorization,
        status_path_template=DEFAULT_I2V_STATUS_PATH_TEMPLATE,
        timeout_s=1800,
        poll_s=5,
    )
    print("\nDone response:", done_resp)

    # --- Download output ---
    out_url = extract_first_url(done_resp)
    if not out_url:
        print("\nCould not auto-find an output URL. Check Done response for the result fields.")
        return

    out_dir = pathlib.Path("kling_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Guess extension (optional)
    lower = out_url.lower()
    if ".mp4" in lower:
        ext = ".mp4"
    elif ".mov" in lower:
        ext = ".mov"
    elif ".webm" in lower:
        ext = ".webm"
    else:
        ext = ".mp4"
    out_path = out_dir / f"{task_id}{ext}"

    file_resp = requests.get(out_url, timeout=120)
    file_resp.raise_for_status()
    out_path.write_bytes(file_resp.content)

    print(f"\nDownloaded output to: {out_path}")
    print(f"From URL: {out_url}")


if __name__ == "__main__":
    main()
