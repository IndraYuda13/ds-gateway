"""Vision support: extract images from OpenAI-format messages and upload
them to the upstream DeepSeek file API, returning ref_file_ids."""

import base64
import re
import urllib.request
from typing import Any, Dict, List, Tuple

MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB safety cap
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36")

_EXT_BY_CT = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/gif": "gif",
    "image/webp": "webp",
    "image/bmp": "bmp",
    "image/svg+xml": "svg",
}


def _filename_from_url(url: str, content_type: str, idx: int) -> str:
    m = re.search(r"/([^/?]+\.[A-Za-z0-9]{2,5})(?:\?|$)", url)
    if m:
        return m.group(1)
    ext = _EXT_BY_CT.get((content_type or "").split(";")[0].strip(), "png")
    return f"image_{idx}.{ext}"


def _download(url: str) -> Tuple[bytes, str]:
    req = urllib.request.Request(url, headers={"user-agent": _UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        ct = resp.headers.get_content_type() or "image/png"
        data = resp.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError("image too large")
        return data, ct


def _decode_data_uri(url: str) -> Tuple[bytes, str]:
    m = re.match(r"data:([^;,]+)(;base64)?,(.*)$", url, re.DOTALL)
    if not m:
        raise ValueError("malformed data URI")
    ct = m.group(1) or "image/png"
    if not m.group(2):
        raise ValueError("only base64 data URIs are supported")
    data = base64.b64decode(m.group(3))
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("image too large")
    return data, ct


def _part_image_url(part: Dict[str, Any]) -> str:
    """Extract the URL from an OpenAI image part (several shapes)."""
    iu = part.get("image_url")
    if isinstance(iu, dict):
        return iu.get("url") or ""
    if isinstance(iu, str):
        return iu
    # Anthropic-style / other shapes
    if part.get("type") in ("input_image", "image") and isinstance(part.get("source"), dict):
        src = part["source"]
        if src.get("type") == "base64":
            mt = src.get("media_type", "image/png")
            return f"data:{mt};base64,{src.get('data', '')}"
        return src.get("url") or ""
    return part.get("url") or ""


def extract_images(messages: List[Any]) -> List[Tuple[bytes, str, str]]:
    """Return [(bytes, filename, content_type)] for images in the LAST user message."""
    last_user = None
    for m in messages or []:
        role = (getattr(m, "role", None) or "").lower()
        if role == "user":
            last_user = m
    if last_user is None:
        return []

    content = getattr(last_user, "content", None)
    if not isinstance(content, list):
        return []

    out: List[Tuple[bytes, str, str]] = []
    for idx, part in enumerate(content):
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype not in ("image_url", "input_image", "image"):
            continue
        url = _part_image_url(part)
        if not url:
            continue
        try:
            if url.startswith("data:"):
                data, ct = _decode_data_uri(url)
            elif url.startswith("http://") or url.startswith("https://"):
                data, ct = _download(url)
            else:
                continue
        except Exception as e:
            print(f"[Vision] skipping image {idx}: {e}")
            continue
        out.append((data, _filename_from_url(url, ct, idx), ct))
    return out
