"""
智能体数据增广 API：接收图片+指令，返回增广后的图片列表。
"""
import base64
import json
import re
import time
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any

from .config import (
    get_diagnostic,
    get_image_model_output_limit,
    is_configured,
    normalize_api_style,
    QWEN_IMAGE_EDIT_MODEL,
    QWEN_TEXT_MODEL,
    resolve_provider_config,
    validate_image_model,
)
from .llm import normalize_instruction_to_description
from .apply import apply_augmentation
from .image_edit import edit_images_with_provider, ImageEditConfigError, ImageEditRequestError

router = APIRouter(
    prefix="/api/augmentation",
    tags=["augmentation"],
    responses={404: {"description": "Not found"}},
)

CLASSIC_AUGMENTATION_PRESETS: Dict[str, Dict[str, Any]] = {
    "brightness_up": {
        "label": "亮度增强",
        "description": "提高亮度，适合模拟光照增强。",
        "params": {"brightness": 1.2},
    },
    "contrast_up": {
        "label": "对比度增强",
        "description": "提高对比度，增强目标边缘。",
        "params": {"contrast": 1.3},
    },
    "noise_light": {
        "label": "轻微噪声",
        "description": "添加轻微高斯噪声，模拟采集噪点。",
        "params": {"noise_std": 0.02},
    },
    "blur_light": {
        "label": "轻微模糊",
        "description": "添加轻微模糊，模拟失焦场景。",
        "params": {"blur": 3},
    },
    "flip_horizontal": {
        "label": "水平翻转",
        "description": "左右翻转图片。",
        "params": {"flip_horizontal": True},
    },
    "rotate_ccw_90": {
        "label": "逆时针旋转90度",
        "description": "将图片逆时针旋转 90 度。",
        "params": {"rotate": 90},
    },
}


@router.get("/status")
async def augmentation_status(
    image_model: Optional[str] = None,
    provider_preset: Optional[str] = None,
    api_style: Optional[str] = None,
    base_url: Optional[str] = None,
    image_url: Optional[str] = None,
    text_model: Optional[str] = None,
):
    """
    检查 API Key 是否已配置。若 api_configured 为 false，则不会调用千问，用量不会增加。
    用于排查「模型用量未显示调用」的问题。
    """
    diagnostic = get_diagnostic(
        image_model=image_model,
        provider_preset=provider_preset,
        api_style=api_style,
        base_url=base_url,
        image_url=image_url,
        text_model=text_model,
    )
    diagnostic["classic_presets"] = [
        {
            "value": key,
            "label": item["label"],
            "description": item["description"],
        }
        for key, item in CLASSIC_AUGMENTATION_PRESETS.items()
    ]
    return JSONResponse(content=diagnostic)


def _params_to_slug(params: Dict[str, Any]) -> str:
    """将增广参数转为见名知义的短标签，用于文件名。"""
    parts = []
    if _truthy(params.get("flip_horizontal")):
        parts.append("水平翻转")
    if _truthy(params.get("flip_vertical")):
        parts.append("垂直翻转")
    r = params.get("rotate")
    if r is not None and float(r) != 0:
        a = int(round(float(r)))
        if a == 90:
            parts.append("逆时针90度")
        elif a == -90:
            parts.append("顺时针90度")
        elif a == 180:
            parts.append("旋转180度")
        else:
            parts.append(f"旋转{a}度")
    if params.get("noise_std") and float(params["noise_std"]) > 0:
        parts.append("加噪")
    if params.get("brightness") and float(params["brightness"]) != 1.0:
        parts.append("调亮度")
    if params.get("contrast") and float(params["contrast"]) != 1.0:
        parts.append("调对比度")
    if params.get("saturation") and float(params["saturation"]) != 1.0:
        parts.append("调饱和度")
    if params.get("blur") and int(params["blur"]) > 0:
        parts.append("模糊")
    if not parts:
        return "增广"
    return "_".join(parts)


def _truthy(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "是")
    return bool(v)


def _slugify_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "增广"
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[，。；：、,.!！?？()（）\[\]{}]+", "_", text)
    text = re.sub(r'[<>:"/\\|?*]+', "_", text)
    text = re.sub(r"_+", "_", text).strip("._ ")
    # 限制长度，避免文件名过长
    return (text[:24] or "增广")


def _parse_requested_count(instruction: str) -> int:
    """
    从用户指令中提取期望生成张数，默认 1，最大 6。
    示例：生成2张 / 来三张 / 出两张图
    """
    s = (instruction or "").strip()
    if not s:
        return 1
    # 阿拉伯数字：2张、3个结果
    m = re.search(r"(\d+)\s*(?:张|个|幅)", s)
    if m:
        try:
            return max(1, min(6, int(m.group(1))))
        except Exception:
            pass
    # 中文数字
    zh_map = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6}
    m2 = re.search(r"(一|两|二|三|四|五|六)\s*(?:张|个|幅)", s)
    if m2:
        return zh_map.get(m2.group(1), 1)
    return 1


def _sanitize_filename(original_name: str, slug_source: Any, index: int = 0) -> str:
    """生成见名知义且不重复的增广文件名，统一加 aug_ 前缀以区分原图。"""
    base = re.sub(r"\.[^.]+$", "", original_name or "image")
    if isinstance(slug_source, dict):
        slug = _params_to_slug(slug_source)
    else:
        slug = _slugify_text(str(slug_source or "增广"))
    # 去掉可能影响文件系统的字符
    slug = re.sub(r'[<>:"/\\|?*]', "_", slug)
    slug = slug.strip("._ ") or "增广"
    # 加短时间戳避免同一张图同一种增广多次时重名
    unique = int(time.time() * 1000) % 100000
    # 统一 aug_ 前缀，便于在文件列表和导出中区分原图与增广图
    if index > 0:
        return f"aug_{base}_{slug}_{unique}_{index}.jpg"
    return f"aug_{base}_{slug}_{unique}.jpg"


def _parse_classic_options(raw_options: Optional[str]) -> List[str]:
    if not raw_options:
        return []
    try:
        parsed = json.loads(raw_options)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"常规增广选项格式错误: {exc}") from exc
    if not isinstance(parsed, list):
        raise HTTPException(status_code=400, detail="常规增广选项必须是数组")

    options: List[str] = []
    invalid: List[str] = []
    for item in parsed:
        key = str(item or "").strip()
        if not key:
            continue
        if key not in CLASSIC_AUGMENTATION_PRESETS:
            invalid.append(key)
            continue
        if key not in options:
            options.append(key)
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"存在不支持的常规增广选项: {', '.join(invalid)}",
        )
    return options


async def _run_classic_augmentation(
    images: List[UploadFile],
    selected_options: List[str],
    instruction: str,
) -> JSONResponse:
    if not selected_options:
        raise HTTPException(status_code=400, detail="请至少选择一个常规增广选项")

    results = []
    for i, f in enumerate(images):
        try:
            raw = await f.read()
            for j, option_key in enumerate(selected_options):
                preset = CLASSIC_AUGMENTATION_PRESETS[option_key]
                aug_bytes = apply_augmentation(raw, preset["params"])
                b64 = base64.b64encode(aug_bytes).decode("utf-8")
                name = _sanitize_filename(
                    f.filename or "image.jpg",
                    preset["label"],
                    index=(i + 1) * 100 + (j + 1),
                )
                results.append({
                    "filename": name,
                    "image_base64": b64,
                    "method": "classic",
                    "source_index": i,
                    "variant_index": j,
                    "classic_option": option_key,
                    "classic_label": preset["label"],
                })
        except Exception as e:
            results.append({
                "filename": f.filename or "image.jpg",
                "error": str(e),
                "image_base64": None,
            })

    return JSONResponse(content={
        "success": True,
        "mode_used": "classic",
        "instruction": instruction,
        "classic_options_used": [
            {
                "value": key,
                "label": CLASSIC_AUGMENTATION_PRESETS[key]["label"],
                "description": CLASSIC_AUGMENTATION_PRESETS[key]["description"],
            }
            for key in selected_options
        ],
        "augmented": results,
        "requested_count": len(selected_options),
        "effective_count": len(selected_options),
        "fallback_count": 0,
        "tips": ["常规增广不依赖 API Key，每个勾选项会为每张原图生成 1 张结果。"],
        "api_configured": is_configured(),
    })


async def _run_ai_augmentation(
    images: List[UploadFile],
    instruction: str,
    api_key: Optional[str],
    provider_preset: Optional[str],
    api_style: Optional[str],
    base_url: Optional[str],
    image_url: Optional[str],
    text_model: Optional[str],
    image_model: Optional[str],
) -> JSONResponse:
    raw_instruction = (instruction or "").strip()
    if not raw_instruction:
        raise HTTPException(status_code=400, detail="AI 增广请先输入增广指令")
    if not is_configured(api_key):
        raise HTTPException(status_code=400, detail="AI 增广需要填写 API Key（或由服务端预先配置）")

    provider_config = resolve_provider_config(
        provider_preset=provider_preset,
        api_style=api_style,
        base_url=base_url,
        image_url=image_url,
        text_model=text_model,
        image_model=image_model,
    )
    resolved_model = provider_config["image_model"] or ""
    validation = validate_image_model(
        resolved_model,
        api_style=provider_config["api_style"],
        base_url=provider_config["base_url"],
        image_url=provider_config["image_url"],
    )
    if validation.get("status") in ("error", "info"):
        raise HTTPException(status_code=400, detail=validation.get("message") or "当前模型不支持图像增广")

    description = normalize_instruction_to_description(
        raw_instruction,
        api_key=api_key,
        text_model=provider_config["text_model"] or None,
        base_url=provider_config["base_url"] or None,
    )
    requested_n = _parse_requested_count(raw_instruction)
    output_limit = get_image_model_output_limit(resolved_model)
    effective_n = min(requested_n, output_limit)
    tips: List[str] = []
    if validation.get("status") == "warning":
        tips.append(validation["message"])
    if not provider_config["text_model"]:
        tips.append("未填写文本模型，本次已跳过 LLM 指令规范化，改用关键词回退描述。")
    tips.extend(provider_config.get("notes") or [])
    tips.append(
        f"当前按 {provider_config['provider_label']} / {provider_config['api_style']} 方式调用图像接口。"
    )
    if effective_n < requested_n:
        tips.append(
            f"当前模型 {resolved_model} 仅支持单张输出，已按 1 张处理。若需多张请切换到 qwen-image-2.0 / qwen-image-2.0-pro / qwen-image-edit-max。"
        )

    results = []
    for i, f in enumerate(images):
        try:
            raw = await f.read()
            aug_bytes_list = edit_images_with_provider(
                raw,
                description,
                n=effective_n,
                api_key=api_key,
                image_model=resolved_model,
                api_style=provider_config["api_style"],
                base_url=provider_config["base_url"] or None,
                image_url=provider_config["image_url"] or None,
            )
            if len(aug_bytes_list) < effective_n:
                tips.append(
                    f"第 {i + 1} 张输入图请求 {effective_n} 张，模型实际返回 {len(aug_bytes_list)} 张。"
                )

            for j, aug_bytes in enumerate(aug_bytes_list):
                b64 = base64.b64encode(aug_bytes).decode("utf-8")
                name = _sanitize_filename(
                    f.filename or "image.jpg",
                    description,
                    index=(i + 1) * 100 + (j + 1),
                )
                results.append({
                    "filename": name,
                    "image_base64": b64,
                    "method": "ai-image",
                    "source_index": i,
                    "variant_index": j,
                })
        except ImageEditConfigError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ImageEditRequestError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except Exception as e:
            results.append({
                "filename": f.filename or "image.jpg",
                "error": str(e),
                "image_base64": None,
            })

    return JSONResponse(content={
        "success": True,
        "mode_used": "ai",
        "instruction": instruction,
        "description_used": description,
        "requested_count": requested_n,
        "effective_count": effective_n,
        "text_model_used": provider_config["text_model"] or "",
        "image_model_used": resolved_model,
        "api_style_used": provider_config["api_style"],
        "base_url_used": provider_config["base_url"],
        "image_url_used": provider_config["image_url"],
        "provider_preset_used": provider_config["provider_preset"],
        "image_model_validation": validation,
        "augmented": results,
        "fallback_count": 0,
        "tips": tips,
        "api_configured": is_configured(api_key),
        "user_supplied_api_key": bool((api_key or "").strip()),
    })


@router.post("/run")
async def run_augmentation(
    instruction: str = Form("", description="AI 增广指令，如：增加光照变化、添加轻微噪声"),
    mode: str = Form("ai", description="增广模式：ai / classic"),
    api_key: Optional[str] = Form(None, description="用户自填 API Key"),
    provider_preset: Optional[str] = Form(None, description="供应商预设"),
    api_style: Optional[str] = Form(None, description="图像接口风格"),
    base_url: Optional[str] = Form(None, description="文本模型 API Base URL"),
    image_url: Optional[str] = Form(None, description="图像接口 URL"),
    text_model: Optional[str] = Form(None, description="文本模型名"),
    image_model: Optional[str] = Form(None, description="图像增广模型名"),
    classic_options: Optional[str] = Form(None, description="常规增广选项 JSON 数组"),
    images: List[UploadFile] = File(..., description="要增广的图片"),
):
    """
    根据模式对选中的图片执行 AI 增广或常规增广，返回增广后的图片（base64）及文件名。
    """
    if not images:
        raise HTTPException(status_code=400, detail="请至少上传一张图片")

    normalized_mode = (mode or "ai").strip().lower()
    if normalized_mode == "classic":
        selected_options = _parse_classic_options(classic_options)
        return await _run_classic_augmentation(images, selected_options, instruction)
    if normalized_mode == "ai":
        return await _run_ai_augmentation(
            images,
            instruction,
            api_key,
            provider_preset,
            normalize_api_style(api_style),
            base_url,
            image_url,
            text_model,
            image_model,
        )
    raise HTTPException(status_code=400, detail=f"不支持的增广模式: {mode}")
