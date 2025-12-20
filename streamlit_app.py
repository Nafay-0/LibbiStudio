from __future__ import annotations

import base64
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
from dotenv import load_dotenv

try:
    import jwt  # PyJWT
except Exception:  # pragma: no cover
    jwt = None


# -----------------------------
# Config
# -----------------------------
ROOT_DIR = Path(__file__).resolve().parent
DOTENV_PATH = ROOT_DIR / ".env"
OUTPUT_DIR = ROOT_DIR / "kling_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUNWAY_OUTPUT_DIR = ROOT_DIR / "runway_outputs" / "streamlit"
RUNWAY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_BASE_URL = "https://api.klingai.com"
DEFAULT_RUNWAY_BASE_URL = "https://api.dev.runwayml.com"
DEFAULT_RUNWAY_VERSION = "2024-11-06"

# Docs:
# - Text-to-Video: https://app.klingai.com/global/dev/document-api/apiReference/model/textToVideo
# - Image-to-Video: https://app.klingai.com/global/dev/document-api/apiReference/model/imageToVideo
T2V_CREATE_PATH = "/v1/videos/text2video"
T2V_GET_PATH_TEMPLATE = "/v1/videos/text2video/{task_id}"
I2V_CREATE_PATH = "/v1/videos/image2video"
I2V_GET_PATH_TEMPLATE = "/v1/videos/image2video/{task_id}"

T2V_MODEL_OPTIONS = [
    "kling-v1",
    "kling-v1-6",
    "kling-v2-master",
    "kling-v2-1-master",
    "kling-v2-5-turbo",
    "kling-v2-6",
]

I2V_MODEL_OPTIONS = [
    "kling-v1",
    "kling-v1-5",
    "kling-v1-6",
    "kling-v2-master",
    "kling-v2-1",
    "kling-v2-1-master",
    "kling-v2-5-turbo",
    "kling-v2-6",
]

RUNWAY_ALLOWED_RATIOS: List[str] = [
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
]

RUNWAY_T2V_MODELS = [
    # Keep in sync with Runway API validation errors / docs.
    # Example allowed set observed from API error:
    # gen3a_turbo, gen4.5, veo3, veo3.1, veo3.1_fast
    "gen3a_turbo",
    "gen4.5",
    "veo3",
    "veo3.1",
    "veo3.1_fast",
]

RUNWAY_I2V_MODELS = [
    # Image-to-video typically supports Runway's gen* models.
    "gen3a_turbo",
    "gen4.5",
]


# -----------------------------
# Kling + Runway helpers
# -----------------------------
def safe_json(resp: requests.Response) -> Any:
    try:
        return resp.json()
    except Exception:
        return resp.text


def encode_jwt_token(ak: str, sk: str) -> str:
    if jwt is None:
        raise RuntimeError('PyJWT is not installed. Install it: pip install "PyJWT"')
    headers = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "iss": ak,
        "exp": int(time.time()) + 1800,  # now + 30min
        "nbf": int(time.time()) - 5,  # now - 5s
    }
    token = jwt.encode(payload, sk, algorithm="HS256", headers=headers)
    if isinstance(token, bytes):
        token = token.decode("utf-8")
    return token


def auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def request_with_429_retries(
    method: str,
    url: str,
    *,
    headers: dict,
    json_body: dict | None = None,
    timeout_s: int = 60,
    max_retries: int = 6,
) -> dict:
    for attempt in range(max_retries + 1):
        resp = requests.request(method=method.upper(), url=url, headers=headers, json=json_body, timeout=timeout_s)

        if resp.status_code != 429:
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}: {safe_json(resp)}")
            if resp.status_code == 204:
                return {"ok": True}
            return resp.json()

        if attempt == max_retries:
            break

        ra = resp.headers.get("Retry-After")
        if ra:
            try:
                sleep_s = float(ra)
            except ValueError:
                sleep_s = 2.0
        else:
            sleep_s = min(60.0, (2**attempt)) + random.random()
        time.sleep(sleep_s)

    raise RuntimeError("Still receiving 429 after retries (rate limit or credits/quota).")


def extract_video_urls(resp: dict) -> List[str]:
    data = resp.get("data") or {}
    task_result = data.get("task_result") or {}
    videos = task_result.get("videos") or []
    out: List[str] = []
    if isinstance(videos, list):
        for v in videos:
            if isinstance(v, dict):
                url = v.get("url")
                if isinstance(url, str) and url.startswith("http"):
                    out.append(url)
    return out


def download_video(url: str, *, out_path: Path) -> None:
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    out_path.write_bytes(r.content)


def file_bytes_to_base64_no_prefix(data: bytes) -> str:
    # Kling requires raw base64 string WITHOUT "data:image/...;base64," prefix
    return base64.b64encode(data).decode("utf-8")


@dataclass
class KlingEnv:
    access_key: str
    secret_key: str
    base_url: str


def load_kling_env() -> KlingEnv:
    # Avoid python-dotenv find_dotenv() edge cases by providing explicit path.
    load_dotenv(dotenv_path=DOTENV_PATH, override=False)
    ak = (os.getenv("KLING_ACCESS_KEY") or "").strip()
    sk = (os.getenv("KLING_SECRET_KEY") or "").strip()
    base_url = (os.getenv("KLING_API_BASE") or DEFAULT_BASE_URL).strip().rstrip("/")
    return KlingEnv(access_key=ak, secret_key=sk, base_url=base_url)


@dataclass
class RunwayEnv:
    api_secret: str
    base_url: str
    version: str


def load_runway_env() -> RunwayEnv:
    load_dotenv(dotenv_path=DOTENV_PATH, override=False)
    api_secret = (os.getenv("RUNWAYML_API_SECRET") or "").strip()
    base_url = (os.getenv("RUNWAY_API_BASE") or DEFAULT_RUNWAY_BASE_URL).strip().rstrip("/")
    version = (os.getenv("RUNWAY_API_VERSION") or DEFAULT_RUNWAY_VERSION).strip()
    return RunwayEnv(api_secret=api_secret, base_url=base_url, version=version)


def runway_headers(api_key: str, version: str, *, json_content: bool = True) -> Dict[str, str]:
    h = {"Authorization": f"Bearer {api_key}", "X-Runway-Version": version}
    if json_content:
        h["Content-Type"] = "application/json"
    return h


def runway_request_with_429_retries(
    method: str,
    url: str,
    *,
    headers: dict,
    json_body: dict | None = None,
    params: dict | None = None,
    timeout_s: int = 60,
    max_retries: int = 6,
) -> dict:
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
                raise RuntimeError(f"Runway HTTP {resp.status_code}: {safe_json(resp)}")
            if resp.status_code == 204:
                return {"ok": True}
            return resp.json()

        if attempt == max_retries:
            break

        ra = resp.headers.get("Retry-After")
        if ra:
            try:
                sleep_s = float(ra)
            except ValueError:
                sleep_s = 2.0
        else:
            sleep_s = min(60.0, (2**attempt)) + random.random()
        time.sleep(sleep_s)

    raise RuntimeError("Runway: still receiving 429 after retries (rate limit or credits/quota).")


def runway_get_task(base_url: str, api_key: str, version: str, task_id: str) -> dict:
    return runway_request_with_429_retries(
        "GET",
        f"{base_url}/v1/tasks/{task_id}",
        headers=runway_headers(api_key, version, json_content=True),
    )


def runway_wait_task(
    base_url: str,
    api_key: str,
    version: str,
    task_id: str,
    *,
    poll_s: float,
    timeout_s: int,
    status_placeholder: Any,
) -> dict:
    poll_s = max(5.0, poll_s)
    start = time.time()
    while True:
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for Runway task {task_id} after {timeout_s}s")
        task = runway_get_task(base_url, api_key, version, task_id)
        status = str(task.get("status", "")).upper()
        status_placeholder.info(f"status={status}")
        if status == "SUCCEEDED":
            return task
        if status == "FAILED":
            raise RuntimeError(f"Runway task FAILED: {task}")
        time.sleep(poll_s)


def runway_extract_output_urls(task: dict) -> List[str]:
    out = task.get("output")
    if isinstance(out, list):
        return [u for u in out if isinstance(u, str) and u.startswith("http")]
    return []


def runway_create_ephemeral_upload(base_url: str, api_key: str, version: str, filename: str) -> dict:
    payload = {"filename": filename, "type": "ephemeral"}
    return runway_request_with_429_retries(
        "POST",
        f"{base_url}/v1/uploads",
        headers=runway_headers(api_key, version, json_content=True),
        json_body=payload,
    )


def runway_upload_file_and_get_uri(base_url: str, api_key: str, version: str, filename: str, data: bytes) -> str:
    init = runway_create_ephemeral_upload(base_url, api_key, version, filename)
    upload_url = init.get("uploadUrl")
    fields = init.get("fields") or {}
    runway_uri = init.get("runwayUri")
    if not upload_url or not isinstance(fields, dict) or not runway_uri:
        raise RuntimeError(f"Unexpected Runway upload init response: {init}")

    files = {"file": (filename, data)}
    resp = requests.post(upload_url, data=fields, files=files, timeout=180)
    resp.raise_for_status()
    return str(runway_uri)


def runway_download_first_video(task_id: str, urls: List[str]) -> Path:
    if not urls:
        raise RuntimeError("No output URLs to download.")
    url = urls[0]
    ext = ".mp4"
    lower = url.lower()
    if ".mov" in lower:
        ext = ".mov"
    elif ".webm" in lower:
        ext = ".webm"

    out_dir = RUNWAY_OUTPUT_DIR / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"video_1{ext}"
    download_video(url, out_path=out_path)
    return out_path


def poll_task(
    *,
    base_url: str,
    get_path_template: str,
    task_id: str,
    token: str,
    timeout_s: int,
    poll_s: float,
    status_placeholder: Any,
) -> dict:
    start = time.time()
    while True:
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for task {task_id} after {timeout_s}s")

        resp = request_with_429_retries(
            "GET",
            f"{base_url}{get_path_template.format(task_id=task_id)}",
            headers=auth_headers(token),
            timeout_s=60,
            max_retries=4,
        )
        data = resp.get("data") or {}
        status = str(data.get("task_status") or "").lower()
        status_msg = str(data.get("task_status_msg") or "").strip()

        status_placeholder.info(f"status={status}" + (f" | {status_msg}" if status_msg else ""))

        if status == "succeed":
            return resp
        if status == "failed":
            raise RuntimeError(f"Task failed: {resp}")

        time.sleep(poll_s)


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="LibbiStudio – Kling Video UI", layout="wide")
st.title("LibbiStudio – Kling Video Generation (Streamlit)")

env = load_kling_env()
runway_env = load_runway_env()
with st.sidebar:
    st.header("Credentials")
    st.caption("Loaded from `.env` (or environment variables).")

    st.subheader("Kling")
    st.text_input("Kling Base URL", value=env.base_url, key="kling_base_url")
    st.text_input("KLING_ACCESS_KEY", value=env.access_key, type="password", key="kling_ak")
    st.text_input("KLING_SECRET_KEY", value=env.secret_key, type="password", key="kling_sk")

    st.subheader("Runway")
    st.text_input("Runway Base URL", value=runway_env.base_url, key="runway_base_url")
    st.text_input("Runway Version", value=runway_env.version, key="runway_version")
    st.text_input("RUNWAYML_API_SECRET", value=runway_env.api_secret, type="password", key="runway_api_secret")

    st.caption("Docs:")
    st.markdown("- [Kling Text to Video](https://app.klingai.com/global/dev/document-api/apiReference/model/textToVideo)")
    st.markdown("- [Kling Image to Video](https://app.klingai.com/global/dev/document-api/apiReference/model/imageToVideo)")
    st.markdown("- Runway docs: set `RUNWAYML_API_SECRET` + version/base from your account")

base_url = (st.session_state.get("kling_base_url") or DEFAULT_BASE_URL).strip().rstrip("/")
ak = (st.session_state.get("kling_ak") or "").strip()
sk = (st.session_state.get("kling_sk") or "").strip()

runway_base_url = (st.session_state.get("runway_base_url") or DEFAULT_RUNWAY_BASE_URL).strip().rstrip("/")
runway_version = (st.session_state.get("runway_version") or DEFAULT_RUNWAY_VERSION).strip()
runway_api_secret = (st.session_state.get("runway_api_secret") or "").strip()

kling_available = bool(ak and sk)
runway_available = bool(runway_api_secret)

if not kling_available and not runway_available:
    st.warning("Configure Kling and/or Runway credentials in `.env` (or via the sidebar) to run generations.")
    st.stop()


tab_t2v, tab_i2v = st.tabs(["Text → Video", "Image → Video"])


with tab_t2v:
    st.subheader("Text → Video")
    provider_options = [p for p in ["Kling", "Runway"] if (p == "Kling" and kling_available) or (p == "Runway" and runway_available)]
    provider = st.selectbox("Provider", options=provider_options, index=0, key="t2v_provider")

    col1, col2 = st.columns([2, 1])
    with col1:
        prompt = st.text_area(
            "Prompt",
            value="A cinematic rainy Helsinki street at night, neon reflections.",
            height=120,
            key="t2v_prompt",
        )
        negative_prompt = st.text_area(
            "Negative prompt (optional)",
            value="blurry, low quality, artifacts",
            height=80,
            key="t2v_negative_prompt",
        )

    with col2:
        if provider == "Kling":
            model_name = st.selectbox("Model", options=T2V_MODEL_OPTIONS, index=0, key="t2v_kling_model")
            aspect_ratio = st.selectbox("Aspect ratio", options=["16:9", "9:16", "1:1"], index=0, key="t2v_kling_ratio")
            duration = st.selectbox("Duration (seconds)", options=[5, 10], index=0, key="t2v_kling_duration")
            mode = st.selectbox("Mode", options=["std", "pro"], index=0, key="t2v_kling_mode")
            sound = st.selectbox("Sound", options=["off", "on"], index=0, help='Only supported by newer models (per docs, v2.6+).', key="t2v_kling_sound")
        else:
            # If we changed the allowed model list, Streamlit may keep an old value in session_state.
            # That can cause us to send an invalid model to Runway (400). Clear it defensively.
            if "t2v_runway_model" in st.session_state and st.session_state["t2v_runway_model"] not in RUNWAY_T2V_MODELS:
                del st.session_state["t2v_runway_model"]
            model_name = st.selectbox("Model", options=RUNWAY_T2V_MODELS, index=0, key="t2v_runway_model")
            ratio = st.selectbox("Ratio (pixel)", options=RUNWAY_ALLOWED_RATIOS, index=RUNWAY_ALLOWED_RATIOS.index("1920:1080"), key="t2v_runway_ratio")
            duration = st.number_input("Duration (seconds)", min_value=1, max_value=30, value=8, step=1, key="t2v_runway_duration")
            audio = st.checkbox("Audio", value=False, key="t2v_runway_audio")

    if st.button("Generate video (Text → Video)", type="primary", key="t2v_generate"):
        if provider == "Kling":
            with st.spinner("Creating task (Kling)..."):
                token = encode_jwt_token(ak, sk)
                payload: Dict[str, Any] = {
                    "model_name": model_name,
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "duration": str(int(duration)),  # docs use string enum values
                    "mode": mode,
                    "sound": sound,
                }
                if negative_prompt.strip():
                    payload["negative_prompt"] = negative_prompt.strip()

                create_resp = request_with_429_retries(
                    "POST",
                    f"{base_url}{T2V_CREATE_PATH}",
                    headers=auth_headers(token),
                    json_body=payload,
                    timeout_s=60,
                )
                task_id = (create_resp.get("data") or {}).get("task_id")
                if not task_id:
                    st.error(f"Could not find data.task_id in response: {create_resp}")
                    st.stop()
                task_id = str(task_id)
                st.success(f"Created task: {task_id}")

            status_ph = st.empty()
            with st.spinner("Waiting for completion..."):
                done = poll_task(
                    base_url=base_url,
                    get_path_template=T2V_GET_PATH_TEMPLATE,
                    task_id=task_id,
                    token=token,
                    timeout_s=1800,
                    poll_s=5.0,
                    status_placeholder=status_ph,
                )

            urls = extract_video_urls(done)
            if not urls:
                st.warning("Task succeeded but no video URL found in response.")
                st.json(done)
                st.stop()

            url = urls[0]
            out_path = OUTPUT_DIR / f"{task_id}.mp4"
            with st.spinner("Downloading MP4..."):
                download_video(url, out_path=out_path)

            st.success(f"Saved: {out_path}")
            st.video(str(out_path))
            st.download_button(
                "Download MP4",
                data=out_path.read_bytes(),
                file_name=out_path.name,
                mime="video/mp4",
                key="t2v_kling_dl",
            )
            st.caption(f"Remote URL: `{url}`")
        else:
            try:
                with st.spinner("Creating task (Runway)..."):
                    payload = {
                        "model": model_name,
                        "promptText": prompt,
                        "ratio": ratio,
                        "duration": int(duration),
                        "audio": bool(audio),
                    }
                    task_id = runway_request_with_429_retries(
                        "POST",
                        f"{runway_base_url}/v1/text_to_video",
                        headers=runway_headers(runway_api_secret, runway_version, json_content=True),
                        json_body=payload,
                    ).get("id")
                    if not task_id:
                        st.error("Runway create response missing id.")
                        st.stop()
                    task_id = str(task_id)
                    st.success(f"Created task: {task_id}")

                status_ph = st.empty()
                with st.spinner("Waiting for completion..."):
                    task = runway_wait_task(
                        runway_base_url,
                        runway_api_secret,
                        runway_version,
                        task_id,
                        poll_s=5.0,
                        timeout_s=3600,
                        status_placeholder=status_ph,
                    )

                urls = runway_extract_output_urls(task)
                if not urls:
                    st.warning("Task succeeded but no output URL found in response.")
                    st.json(task)
                    st.stop()

                out_path = runway_download_first_video(task_id, urls)
                st.success(f"Saved: {out_path}")
                st.video(str(out_path))
                st.download_button(
                    "Download video",
                    data=out_path.read_bytes(),
                    file_name=out_path.name,
                    mime="video/mp4",
                    key="t2v_runway_dl",
                )
                st.caption(f"Remote URL: `{urls[0]}`")
            except Exception as e:
                st.error(str(e))
                st.stop()


with tab_i2v:
    st.subheader("Image → Video")
    provider_options = [p for p in ["Kling", "Runway"] if (p == "Kling" and kling_available) or (p == "Runway" and runway_available)]
    provider = st.selectbox("Provider", options=provider_options, index=0, key="i2v_provider")

    left, right = st.columns([2, 1])
    with left:
        uploaded = st.file_uploader("Upload image (.png/.jpg/.jpeg)", type=["png", "jpg", "jpeg"], key="i2v_upload")
        image_url = st.text_input("…or Image URL (optional)", value="", placeholder="https://example.com/image.png", key="i2v_image_url")
        prompt2 = st.text_area(
            "Prompt (optional)",
            value="The camera slowly pushes in; subtle motion; cinematic.",
            height=100,
            key="i2v_prompt",
        )
        negative2 = st.text_area(
            "Negative prompt (optional)",
            value="blurry, low quality, artifacts",
            height=80,
            key="i2v_negative_prompt",
        )

        if uploaded is not None:
            st.image(uploaded, caption="Input image", use_container_width=True)

    with right:
        if provider == "Kling":
            model_name2 = st.selectbox("Model", options=I2V_MODEL_OPTIONS, index=0, key="i2v_kling_model")
            duration2 = st.selectbox("Duration (seconds)", options=[5, 10], index=0, key="i2v_kling_duration")
            mode2 = st.selectbox("Mode", options=["std", "pro"], index=0, key="i2v_kling_mode")
            cfg_scale2 = st.slider("cfg_scale", min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="i2v_kling_cfg")
        else:
            # Clear stale invalid model persisted in session_state after model list updates.
            if "i2v_runway_model" in st.session_state and st.session_state["i2v_runway_model"] not in RUNWAY_I2V_MODELS:
                del st.session_state["i2v_runway_model"]
            model_name2 = st.selectbox("Model", options=RUNWAY_I2V_MODELS, index=0, key="i2v_runway_model")
            ratio2 = st.selectbox("Ratio (pixel)", options=RUNWAY_ALLOWED_RATIOS, index=RUNWAY_ALLOWED_RATIOS.index("1280:720"), key="i2v_runway_ratio")
            duration2 = st.number_input("Duration (seconds)", min_value=1, max_value=30, value=5, step=1, key="i2v_runway_duration")

    if st.button("Generate video (Image → Video)", type="primary", key="i2v_generate"):
        if uploaded is None and not image_url.strip():
            st.error("Upload an image or provide an image URL.")
            st.stop()

        if provider == "Kling":
            with st.spinner("Preparing image (Kling)..."):
                if uploaded is not None:
                    img_b64 = file_bytes_to_base64_no_prefix(uploaded.getvalue())
                    image_param = img_b64
                else:
                    image_param = image_url.strip()

            with st.spinner("Creating task (Kling)..."):
                token = encode_jwt_token(ak, sk)
                payload2: Dict[str, Any] = {
                    "model_name": model_name2,
                    "image": image_param,
                    "duration": str(int(duration2)),
                    "mode": mode2,
                    "cfg_scale": float(cfg_scale2),
                }
                if prompt2.strip():
                    payload2["prompt"] = prompt2.strip()
                if negative2.strip():
                    payload2["negative_prompt"] = negative2.strip()

                create_resp2 = request_with_429_retries(
                    "POST",
                    f"{base_url}{I2V_CREATE_PATH}",
                    headers=auth_headers(token),
                    json_body=payload2,
                    timeout_s=60,
                )
                task_id2 = (create_resp2.get("data") or {}).get("task_id")
                if not task_id2:
                    st.error(f"Could not find data.task_id in response: {create_resp2}")
                    st.stop()
                task_id2 = str(task_id2)
                st.success(f"Created task: {task_id2}")

            status_ph2 = st.empty()
            with st.spinner("Waiting for completion..."):
                done2 = poll_task(
                    base_url=base_url,
                    get_path_template=I2V_GET_PATH_TEMPLATE,
                    task_id=task_id2,
                    token=token,
                    timeout_s=1800,
                    poll_s=5.0,
                    status_placeholder=status_ph2,
                )

            urls2 = extract_video_urls(done2)
            if not urls2:
                st.warning("Task succeeded but no video URL found in response.")
                st.json(done2)
                st.stop()

            url2 = urls2[0]
            out_path2 = OUTPUT_DIR / f"{task_id2}.mp4"
            with st.spinner("Downloading MP4..."):
                download_video(url2, out_path=out_path2)

            st.success(f"Saved: {out_path2}")
            st.video(str(out_path2))
            st.download_button(
                "Download MP4",
                data=out_path2.read_bytes(),
                file_name=out_path2.name,
                mime="video/mp4",
                key="i2v_kling_dl",
            )
            st.caption(f"Remote URL: `{url2}`")
        else:
            try:
                with st.spinner("Preparing image (Runway)..."):
                    if uploaded is not None:
                        runway_uri = runway_upload_file_and_get_uri(
                            runway_base_url,
                            runway_api_secret,
                            runway_version,
                            filename=getattr(uploaded, "name", "image.png") or "image.png",
                            data=uploaded.getvalue(),
                        )
                        prompt_image = runway_uri
                    else:
                        prompt_image = image_url.strip()

                with st.spinner("Creating task (Runway)..."):
                    payload = {
                        "model": model_name2,
                        "promptImage": prompt_image,
                        "promptText": prompt2.strip() or "Animate this image.",
                        "ratio": ratio2,
                        "duration": int(duration2),
                    }
                    task_id = runway_request_with_429_retries(
                        "POST",
                        f"{runway_base_url}/v1/image_to_video",
                        headers=runway_headers(runway_api_secret, runway_version, json_content=True),
                        json_body=payload,
                    ).get("id")
                    if not task_id:
                        st.error("Runway create response missing id.")
                        st.stop()
                    task_id = str(task_id)
                    st.success(f"Created task: {task_id}")

                status_ph = st.empty()
                with st.spinner("Waiting for completion..."):
                    task = runway_wait_task(
                        runway_base_url,
                        runway_api_secret,
                        runway_version,
                        task_id,
                        poll_s=5.0,
                        timeout_s=3600,
                        status_placeholder=status_ph,
                    )

                urls = runway_extract_output_urls(task)
                if not urls:
                    st.warning("Task succeeded but no output URL found in response.")
                    st.json(task)
                    st.stop()

                out_path = runway_download_first_video(task_id, urls)
                st.success(f"Saved: {out_path}")
                st.video(str(out_path))
                st.download_button(
                    "Download video",
                    data=out_path.read_bytes(),
                    file_name=out_path.name,
                    mime="video/mp4",
                    key="i2v_runway_dl",
                )
                st.caption(f"Remote URL: `{urls[0]}`")
            except Exception as e:
                st.error(str(e))
                st.stop()


