from __future__ import annotations

import os
import time
import random
import pathlib
from typing import Any, Optional
from dotenv import load_dotenv

import requests  # pip install requests
import jwt       # pip install PyJWT   (IMPORTANT: not the "jwt" package)

BASE_URL = "https://api.klingai.com"

# Defaults are intentionally configurable via env vars because Kling’s endpoints/models
# can vary by account/region and evolve over time. Reference:
# https://app.klingai.com/global/dev/document-api/apiReference/model/textToVideo
DEFAULT_T2V_CREATE_PATH = os.environ.get("KLING_T2V_CREATE_PATH", "/v1/videos/text2video").strip() or "/v1/videos/text2video"
DEFAULT_T2V_STATUS_PATH_TEMPLATE = (
    os.environ.get("KLING_T2V_STATUS_PATH_TEMPLATE", "/v1/videos/text2video/{task_id}").strip()
    or "/v1/videos/text2video/{task_id}"
)
# Per Kling docs, the newer field name is `model_name` (model remains forward-compatible).
# Default per docs is "kling-v1".
DEFAULT_T2V_MODEL_NAME = (os.environ.get("KLING_T2V_MODEL_NAME") or "kling-v1").strip()
DEFAULT_T2V_MODEL_LEGACY = (os.environ.get("KLING_T2V_MODEL") or "").strip()


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

    # --- Create text-to-video task ---
    # Endpoint/model reference:
    # https://app.klingai.com/global/dev/document-api/apiReference/model/textToVideo
    create_url = f"{BASE_URL}{DEFAULT_T2V_CREATE_PATH}"

    payload = {
        # Per docs, prefer model_name. (model is forward-compatible; leaving it out is OK.)
        "model_name": DEFAULT_T2V_MODEL_NAME,
        "prompt": "A cinematic shot of rainy Helsinki street at night, neon reflections, shallow depth of field",
        "negative_prompt": "blurry, low quality, artifacts",
        # Per docs:
        # - aspect_ratio enum: "16:9" | "9:16" | "1:1"
        # - duration enum: "5" | "10"  (docs list it as string)
        "aspect_ratio": "16:9",
        "duration": "5",
        # Mode: "std" (default) or "pro"
        "mode": "std",
        # Sound: "on" | "off" (only supported in v2.6+)
        "sound": "off",
    }
    if DEFAULT_T2V_MODEL_LEGACY:
        payload["model"] = DEFAULT_T2V_MODEL_LEGACY

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
        status_path_template=DEFAULT_T2V_STATUS_PATH_TEMPLATE,
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
