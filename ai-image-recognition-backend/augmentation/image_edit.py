"""
阶段2：根据接口风格调用图像编辑模型（图+文本 -> 图）。
当前支持：
1. DashScope 多模态生成接口
2. OpenAI Images Edit 兼容接口
"""
from __future__ import annotations

import base64
import imghdr
from typing import Any, Dict, List, Optional

import requests

from .config import (
    get_api_key,
    normalize_api_style,
    OPENAI_IMAGE_EDIT_SUFFIX,
    QWEN_IMAGE_EDIT_MODEL,
    QWEN_IMAGE_EDIT_URL,
)


class ImageEditConfigError(RuntimeError):
    """图像编辑配置错误，例如 API Key 或模型不合法。"""


class ImageEditRequestError(RuntimeError):
    """图像编辑请求失败。"""


def edit_images_with_provider(
    image_bytes: bytes,
    description: str,
    *,
    n: int = 1,
    timeout: int = 120,
    api_key: Optional[str] = None,
    image_model: Optional[str] = None,
    api_style: Optional[str] = None,
    base_url: Optional[str] = None,
    image_url: Optional[str] = None,
) -> List[bytes]:
    """
    根据接口风格调用图像编辑模型，对单张图片进行编辑并返回图像字节列表。
    """
    api_key = get_api_key(api_key)
    if not api_key:
        raise ImageEditConfigError("未提供 API Key，无法调用图像编辑模型")

    text = (description or "").strip() or "不进行任何修改"
    n = max(1, min(6, int(n or 1)))
    resolved_style = normalize_api_style(api_style)
    model_name = (image_model or QWEN_IMAGE_EDIT_MODEL).strip() or QWEN_IMAGE_EDIT_MODEL
    if resolved_style == "dashscope_multimodal":
        return _edit_images_with_dashscope(
            image_bytes=image_bytes,
            description=text,
            n=n,
            timeout=timeout,
            api_key=api_key,
            image_model=model_name,
            image_url=image_url,
        )
    if resolved_style == "openai_images_edit":
        return _edit_images_with_openai_images(
            image_bytes=image_bytes,
            description=text,
            n=n,
            timeout=timeout,
            api_key=api_key,
            image_model=model_name,
            base_url=base_url,
            image_url=image_url,
        )
    raise ImageEditConfigError(f"不支持的图像接口风格: {resolved_style}")


def edit_images_with_qwen(*args: Any, **kwargs: Any) -> List[bytes]:
    """兼容旧调用名。"""
    return edit_images_with_provider(*args, **kwargs)


def _edit_images_with_dashscope(
    *,
    image_bytes: bytes,
    description: str,
    n: int,
    timeout: int,
    api_key: str,
    image_model: str,
    image_url: Optional[str] = None,
) -> List[bytes]:
    request_url = (image_url or QWEN_IMAGE_EDIT_URL).strip() or QWEN_IMAGE_EDIT_URL
    payload = _build_dashscope_payload(image_bytes, description, n=n, image_model=image_model)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    resp = requests.post(request_url, json=payload, headers=headers, timeout=timeout)
    data = _parse_json_response(resp)
    if data.get("code"):
        _raise_provider_error(f"{data.get('code')} {data.get('message')}")

    image_urls = _extract_dashscope_image_urls(data)
    if not image_urls:
        raise ImageEditRequestError("图像编辑响应中未找到输出图片 URL")
    return _download_image_urls(image_urls, timeout=timeout)


def _edit_images_with_openai_images(
    *,
    image_bytes: bytes,
    description: str,
    n: int,
    timeout: int,
    api_key: str,
    image_model: str,
    base_url: Optional[str] = None,
    image_url: Optional[str] = None,
) -> List[bytes]:
    request_url = (image_url or "").strip()
    normalized_base_url = (base_url or "").strip().rstrip("/")
    if not request_url and normalized_base_url:
        request_url = f"{normalized_base_url}{OPENAI_IMAGE_EDIT_SUFFIX}"
    if not request_url:
        raise ImageEditConfigError("OpenAI Images Edit 方式需要配置图像接口 URL，或至少填写 API Base URL。")

    mime = _guess_mime(image_bytes)
    file_ext = _guess_extension(mime)
    files = {
        "image": (f"input.{file_ext}", image_bytes, mime),
    }
    data = {
        "model": image_model,
        "prompt": description,
        "n": str(n),
        "response_format": "b64_json",
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    resp = requests.post(request_url, data=data, files=files, headers=headers, timeout=timeout)
    payload = _parse_json_response(resp)
    if isinstance(payload.get("error"), dict):
        error = payload["error"]
        _raise_provider_error(str(error.get("message") or error.get("code") or payload["error"]))
    items = (payload or {}).get("data") or []
    if not items:
        raise ImageEditRequestError("图像编辑响应中未找到结果数据")

    outputs: List[bytes] = []
    remote_urls: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        b64_json = item.get("b64_json")
        if b64_json:
            try:
                outputs.append(base64.b64decode(b64_json))
                continue
            except Exception as exc:
                raise ImageEditRequestError(f"解析图像响应失败: {exc}") from exc
        image_result_url = item.get("url")
        if image_result_url:
            remote_urls.append(str(image_result_url))
    if remote_urls:
        outputs.extend(_download_image_urls(remote_urls, timeout=timeout))
    if not outputs:
        raise ImageEditRequestError("图像编辑响应中既没有 b64_json，也没有可下载的图片 URL")
    return outputs


def _build_dashscope_payload(
    image_bytes: bytes,
    text: str,
    *,
    n: int,
    image_model: Optional[str] = None,
) -> Dict[str, Any]:
    mime = _guess_mime(image_bytes)
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    image_data = f"data:{mime};base64,{b64}"
    return {
        "model": (image_model or QWEN_IMAGE_EDIT_MODEL).strip() or QWEN_IMAGE_EDIT_MODEL,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"image": image_data},
                        {"text": text},
                    ],
                }
            ]
        },
        "parameters": {
            "n": n,
            "watermark": False,
            "prompt_extend": True,
        },
    }


def _extract_dashscope_image_urls(data: Dict[str, Any]) -> List[str]:
    choices = (((data or {}).get("output") or {}).get("choices") or [])
    if not choices:
        return []
    content = (((choices[0] or {}).get("message") or {}).get("content") or [])
    urls: List[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("image"):
            urls.append(str(item["image"]))
    return urls


def _parse_json_response(resp: requests.Response) -> Dict[str, Any]:
    if resp.status_code != 200:
        detail = (resp.text or "")[:300]
        if resp.status_code in (400, 401, 403):
            raise ImageEditConfigError(f"图像编辑请求失败: HTTP {resp.status_code} {detail}")
        raise ImageEditRequestError(f"图像编辑请求失败: HTTP {resp.status_code} {detail}")
    try:
        return resp.json()
    except Exception as exc:
        raise ImageEditRequestError(f"图像编辑响应不是合法 JSON: {exc}") from exc


def _raise_provider_error(message: str) -> None:
    lowered = (message or "").lower()
    if any(token in lowered for token in ("model", "api key", "auth", "unauthorized", "forbidden")):
        raise ImageEditConfigError(f"图像编辑请求失败: {message}")
    raise ImageEditRequestError(f"图像编辑请求失败: {message}")


def _download_image_urls(urls: List[str], *, timeout: int) -> List[bytes]:
    outputs: List[bytes] = []
    for image_url in urls:
        img_resp = requests.get(image_url, timeout=timeout)
        if img_resp.status_code != 200:
            raise ImageEditRequestError(f"下载编辑结果失败: HTTP {img_resp.status_code}")
        outputs.append(img_resp.content)
    return outputs


def _guess_mime(image_bytes: bytes) -> str:
    kind = imghdr.what(None, h=image_bytes) or ""
    if kind == "png":
        return "image/png"
    if kind in ("jpg", "jpeg"):
        return "image/jpeg"
    if kind == "webp":
        return "image/webp"
    if kind == "gif":
        return "image/gif"
    if kind == "bmp":
        return "image/bmp"
    return "image/jpeg"


def _guess_extension(mime: str) -> str:
    mapping = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/webp": "webp",
        "image/gif": "gif",
        "image/bmp": "bmp",
    }
    return mapping.get(mime, "jpg")
