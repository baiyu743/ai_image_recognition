"""
智能体数据增广 - API 配置
优先从请求传入值读取，否则从环境变量 / 项目配置文件读取。
"""
import os
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

# 后端项目根目录（augmentation 包的上层目录）
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DOTENV_FILES = (
    _BACKEND_ROOT / ".env.local",
    _BACKEND_ROOT / ".env.production",
    _BACKEND_ROOT / ".env",
)
_API_KEY_FILE = _BACKEND_ROOT / ".augmentation_api_key"
# Windows 下可能被保存为 .augmentation_api_key.txt
_API_KEY_FILE_ALT = _BACKEND_ROOT / ".augmentation_api_key.txt"


def _load_env_from_project_files() -> list[str]:
    """
    从项目根目录自动加载 .env 文件（不覆盖已存在的环境变量）。
    支持 .env / .env.production / .env.local。
    """
    loaded = []
    for path in _DOTENV_FILES:
        if not path.exists():
            continue
        try:
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if not key:
                    continue
                # 去掉包裹引号
                if (value.startswith('"') and value.endswith('"')) or (
                    value.startswith("'") and value.endswith("'")
                ):
                    value = value[1:-1]
                # 不覆盖进程中已设置的变量
                os.environ.setdefault(key, value)
            loaded.append(path.name)
        except Exception:
            # 环境加载失败时保持静默，避免影响服务启动
            pass
    return loaded


_LOADED_ENV_FILES = _load_env_from_project_files()

# 千问 API 兼容地址
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen-turbo"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_IMAGE_EDIT_SUFFIX = "/images/edits"

# 两阶段增广流水线配置
# 阶段1：文本规范化（Chat Completions）
QWEN_TEXT_MODEL = os.environ.get("AUGMENTATION_TEXT_MODEL", QWEN_MODEL).strip() or QWEN_MODEL
# 阶段2：图像编辑（图+文本 -> 图）
QWEN_IMAGE_EDIT_MODEL = os.environ.get("AUGMENTATION_IMAGE_MODEL", "qwen-image-2.0").strip() or "qwen-image-2.0"
QWEN_IMAGE_EDIT_URL = os.environ.get(
    "AUGMENTATION_IMAGE_URL",
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
).strip() or "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
DEFAULT_PROVIDER_PRESET = os.environ.get("AUGMENTATION_PROVIDER_PRESET", "dashscope").strip() or "dashscope"
DEFAULT_API_STYLE = os.environ.get("AUGMENTATION_API_STYLE", "dashscope_multimodal").strip() or "dashscope_multimodal"

API_STYLE_OPTIONS = (
    "dashscope_multimodal",
    "openai_images_edit",
)
PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "dashscope": {
        "label": "DashScope",
        "api_style": "dashscope_multimodal",
        "base_url": QWEN_BASE_URL,
        "image_url": QWEN_IMAGE_EDIT_URL,
        "text_model": QWEN_TEXT_MODEL,
        "image_model": QWEN_IMAGE_EDIT_MODEL,
    },
    "openai_compatible": {
        "label": "OpenAI Compatible",
        "api_style": "openai_images_edit",
        "base_url": OPENAI_BASE_URL,
        "image_url": "",
        "text_model": "",
        "image_model": "",
    },
    "custom": {
        "label": "Custom",
        "api_style": "openai_images_edit",
        "base_url": "",
        "image_url": "",
        "text_model": "",
        "image_model": "",
    },
}

SUPPORTED_IMAGE_MODELS = (
    "qwen-image-2.0",
    "qwen-image-2.0-pro",
    "qwen-image-edit",
    "qwen-image-edit-max",
)
_IMAGE_MODEL_HINTS = (
    "image",
    "images",
    "image-edit",
    "gpt-image",
    "wanx",
    "flux",
    "diffusion",
    "recraft",
    "seedream",
    "ideogram",
    "sdxl",
    "stable-diffusion",
)
_TEXT_ONLY_MODEL_HINTS = (
    "turbo",
    "chat",
    "reason",
    "coder",
    "embedding",
    "rerank",
    "vl",
    "tts",
    "whisper",
)


def _read_key_from_file(path: Path) -> str:
    """从文件中读取第一行非空、非注释的内容作为 Key。"""
    try:
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    except Exception:
        pass
    return ""


def get_api_key(override: Optional[str] = None) -> str:
    """获取 API Key：优先使用请求传入值，否则读环境变量/本地文件。"""
    key = (override or "").strip()
    if key:
        return key
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if key:
        return key
    for path in (_API_KEY_FILE, _API_KEY_FILE_ALT):
        if path.exists():
            key = _read_key_from_file(path)
            if key:
                return key
    return ""


def is_configured(override: Optional[str] = None) -> bool:
    """是否已配置 API Key。"""
    return bool(get_api_key(override=override))


def _normalize_url(url: Optional[str]) -> str:
    return (url or "").strip().rstrip("/")


def _join_url(base_url: Optional[str], suffix: str) -> str:
    base = _normalize_url(base_url)
    if not base:
        return ""
    return f"{base}{suffix}"


def _hostname_of(url: Optional[str]) -> str:
    raw = _normalize_url(url)
    if not raw:
        return ""
    try:
        return (urlparse(raw).hostname or "").lower()
    except Exception:
        return ""


def _path_of(url: Optional[str]) -> str:
    raw = _normalize_url(url)
    if not raw:
        return ""
    try:
        return (urlparse(raw).path or "").rstrip("/") or "/"
    except Exception:
        return ""


def _is_dashscope_host(url: Optional[str]) -> bool:
    return _hostname_of(url).endswith("dashscope.aliyuncs.com")


def _is_dashscope_base_like_url(url: Optional[str]) -> bool:
    if not _is_dashscope_host(url):
        return False
    return _path_of(url) in ("/", "/api/v1", "/compatible-mode/v1")


def _normalize_dashscope_multimodal_image_url(
    image_url: Optional[str],
    *,
    base_url: Optional[str] = None,
) -> tuple[str, list[str]]:
    resolved_image_url = _normalize_url(image_url)
    notes: list[str] = []

    if _is_dashscope_base_like_url(resolved_image_url):
        notes.append("检测到你填写的是 DashScope 的基础地址，系统已自动改为官方图像接口地址。")
        return _normalize_url(QWEN_IMAGE_EDIT_URL), notes

    if not resolved_image_url and _is_dashscope_base_like_url(base_url):
        notes.append("检测到你填写的是 DashScope 的文本 Base URL，系统已自动补全官方图像接口地址。")
        return _normalize_url(QWEN_IMAGE_EDIT_URL), notes

    return resolved_image_url, notes


def _normalize_dashscope_text_base_url(base_url: Optional[str]) -> tuple[str, list[str]]:
    resolved_base_url = _normalize_url(base_url)
    notes: list[str] = []
    if _is_dashscope_host(resolved_base_url) and _path_of(resolved_base_url) in ("/", "/api/v1"):
        notes.append("检测到你填写的是 DashScope 的基础地址，系统已自动改为兼容模式文本 Base URL。")
        return _normalize_url(QWEN_BASE_URL), notes
    return resolved_base_url, notes


def normalize_api_style(api_style: Optional[str] = None) -> str:
    raw = (api_style or DEFAULT_API_STYLE).strip().lower()
    if raw in API_STYLE_OPTIONS:
        return raw
    if raw in ("dashscope", "qwen", "dashscope-image"):
        return "dashscope_multimodal"
    if raw in ("openai", "openai_compatible", "images_edit", "image_edit"):
        return "openai_images_edit"
    return raw or DEFAULT_API_STYLE


def resolve_provider_config(
    *,
    provider_preset: Optional[str] = None,
    api_style: Optional[str] = None,
    base_url: Optional[str] = None,
    image_url: Optional[str] = None,
    text_model: Optional[str] = None,
    image_model: Optional[str] = None,
) -> dict[str, Any]:
    preset_key = (provider_preset or DEFAULT_PROVIDER_PRESET).strip() or DEFAULT_PROVIDER_PRESET
    preset = PROVIDER_PRESETS.get(preset_key, PROVIDER_PRESETS["custom"])
    resolved_api_style = normalize_api_style(api_style or preset.get("api_style") or DEFAULT_API_STYLE)
    resolved_base_url = _normalize_url(base_url or preset.get("base_url") or "")
    resolved_image_url = _normalize_url(image_url or preset.get("image_url") or "")
    config_notes: list[str] = []

    if resolved_api_style == "dashscope_multimodal":
        resolved_base_url, base_notes = _normalize_dashscope_text_base_url(resolved_base_url)
        config_notes.extend(base_notes)

    if not resolved_image_url:
        if resolved_api_style == "dashscope_multimodal":
            resolved_image_url = _normalize_url(preset.get("image_url") or QWEN_IMAGE_EDIT_URL)
        elif resolved_api_style == "openai_images_edit":
            resolved_image_url = _join_url(resolved_base_url, OPENAI_IMAGE_EDIT_SUFFIX)

    if resolved_api_style == "dashscope_multimodal":
        resolved_image_url, dashscope_notes = _normalize_dashscope_multimodal_image_url(
            resolved_image_url,
            base_url=resolved_base_url,
        )
        config_notes.extend(dashscope_notes)

    return {
        "provider_preset": preset_key,
        "provider_label": preset.get("label", preset_key),
        "api_style": resolved_api_style,
        "base_url": resolved_base_url,
        "image_url": resolved_image_url,
        "text_model": (text_model if text_model is not None else preset.get("text_model", "")).strip(),
        "image_model": (image_model if image_model is not None else preset.get("image_model", "")).strip(),
        "notes": config_notes,
    }


def validate_image_model(
    model_name: Optional[str] = None,
    *,
    api_style: Optional[str] = None,
    base_url: Optional[str] = None,
    image_url: Optional[str] = None,
) -> dict[str, Any]:
    """
    检查模型名是否看起来具备生图/改图能力。
    这里只做服务端前置校验，避免明显的文本模型被拿来做图像增广。
    """
    raw_name = (model_name if model_name is not None else QWEN_IMAGE_EDIT_MODEL or "").strip()
    resolved_api_style = normalize_api_style(api_style)
    resolved_base_url = _normalize_url(base_url)
    resolved_image_url = _normalize_url(image_url)
    name = raw_name.lower()
    if not raw_name:
        return {
            "supported": False,
            "status": "info",
            "message": "请先填写要用于 AI 增广的图像模型名称。",
            "model": raw_name,
        }

    if resolved_api_style not in API_STYLE_OPTIONS:
        return {
            "supported": False,
            "status": "error",
            "message": f"当前图像接口风格 {resolved_api_style} 暂不支持，请改用 DashScope 多模态接口或 OpenAI Images Edit 接口。",
            "model": raw_name,
        }

    if resolved_api_style == "openai_images_edit" and not (resolved_image_url or resolved_base_url):
        return {
            "supported": False,
            "status": "info",
            "message": "OpenAI Images Edit 方式需要填写 API Base URL 或图像接口 URL。",
            "model": raw_name,
        }

    for supported in SUPPORTED_IMAGE_MODELS:
        if name == supported or name.startswith(f"{supported}-"):
            return {
                "supported": True,
                "status": "success",
                "message": f"当前模型 {raw_name} 可用于 AI 图像增广。",
                "model": raw_name,
            }

    if any(hint in name for hint in _IMAGE_MODEL_HINTS):
        return {
            "supported": True,
            "status": "success",
            "message": f"当前模型 {raw_name} 看起来具备图像生成/编辑能力，将按 AI 增广处理。",
            "model": raw_name,
            "inferred": True,
        }

    if any(hint in name for hint in _TEXT_ONLY_MODEL_HINTS):
        return {
            "supported": False,
            "status": "error",
            "message": (
                f"当前模型 {raw_name} 看起来更像文本/理解模型，不支持生图。"
                f"请改用 {', '.join(SUPPORTED_IMAGE_MODELS)} 这类图像模型。"
            ),
            "model": raw_name,
        }

    return {
        "supported": True,
        "status": "warning",
        "message": (
            f"已完成 {resolved_api_style} 接口配置，但暂时无法仅凭模型名确认 {raw_name} 是否具备增广能力。"
            f"请确认该模型支持图像编辑 / 图像增广；若接口返回不支持，系统会直接提示。"
        ),
        "model": raw_name,
        "uncertain": True,
    }


def get_image_model_output_limit(model_name: Optional[str] = None) -> int:
    """部分模型仅支持单张输出。"""
    name = (model_name or QWEN_IMAGE_EDIT_MODEL or "").strip().lower()
    if name == "qwen-image-edit":
        return 1
    return 6


def get_diagnostic(
    image_model: Optional[str] = None,
    *,
    provider_preset: Optional[str] = None,
    api_style: Optional[str] = None,
    base_url: Optional[str] = None,
    image_url: Optional[str] = None,
    text_model: Optional[str] = None,
) -> dict:
    """返回 Key 配置诊断信息（不包含 Key 内容），便于排查未调用模型的问题。"""
    resolved = resolve_provider_config(
        provider_preset=provider_preset,
        api_style=api_style,
        base_url=base_url,
        image_url=image_url,
        text_model=text_model,
        image_model=image_model,
    )
    return {
        "api_configured": is_configured(),
        "loaded_env_files": _LOADED_ENV_FILES,
        "env_var_set": bool(os.environ.get("DASHSCOPE_API_KEY", "").strip()),
        "text_model": QWEN_TEXT_MODEL,
        "image_edit_model": QWEN_IMAGE_EDIT_MODEL,
        "image_edit_url": QWEN_IMAGE_EDIT_URL,
        "provider_presets": [
            {
                "value": key,
                "label": value["label"],
                "api_style": value["api_style"],
                "base_url": value["base_url"],
                "image_url": value["image_url"],
            }
            for key, value in PROVIDER_PRESETS.items()
        ],
        "api_style_options": list(API_STYLE_OPTIONS),
        "resolved_provider_config": resolved,
        "supported_image_models": list(SUPPORTED_IMAGE_MODELS),
        "image_model_validation": validate_image_model(
            resolved["image_model"],
            api_style=resolved["api_style"],
            base_url=resolved["base_url"],
            image_url=resolved["image_url"],
        ),
        "key_file_primary_exists": _API_KEY_FILE.exists(),
        "key_file_primary_path": str(_API_KEY_FILE),
        "key_file_alt_exists": _API_KEY_FILE_ALT.exists(),
    }
