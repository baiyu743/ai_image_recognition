"""
模型注册与存储：用户上传模型、可选目录模型、下载到本地。
"""
import json
import os
import uuid
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any

# 用户模型与下载目录：固定为 backend 根目录下，不依赖进程 cwd
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
USER_MODELS_DIR = _BACKEND_ROOT / "user_models"
DOWNLOADED_MODELS_DIR = _BACKEND_ROOT / "downloaded_models"
METADATA_FILENAME = "user_models_metadata.json"
CATALOG_FILENAME = "model_catalog.json"

ALLOWED_EXTENSIONS = {".pt", ".pth", ".onnx"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500MB


def _ensure_dirs():
    USER_MODELS_DIR.resolve().mkdir(parents=True, exist_ok=True)
    DOWNLOADED_MODELS_DIR.resolve().mkdir(parents=True, exist_ok=True)


def _metadata_path() -> Path:
    return USER_MODELS_DIR / METADATA_FILENAME


def _catalog_path() -> Path:
    p = Path(__file__).resolve().parent / CATALOG_FILENAME
    return p


def load_user_metadata() -> List[Dict[str, Any]]:
    _ensure_dirs()
    path = _metadata_path()
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_user_metadata(entries: List[Dict[str, Any]]) -> None:
    _ensure_dirs()
    with open(_metadata_path(), "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def get_user_model_path(user_model_id: str) -> Optional[Path]:
    """user_model_id 格式: user:uuid"""
    if not user_model_id.startswith("user:"):
        return None
    uid = user_model_id.replace("user:", "", 1)
    entries = load_user_metadata()
    for e in entries:
        if e.get("id") == user_model_id and os.path.exists(e.get("path", "")):
            return Path(e["path"])
    return None


def add_user_model(file_path: Path, name: str, task: str) -> Dict[str, Any]:
    _ensure_dirs()
    uid = str(uuid.uuid4())
    ext = file_path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"不支持的文件格式，仅支持: {', '.join(ALLOWED_EXTENSIONS)}")
    dest_name = f"user_{uid}{ext}"
    dest = USER_MODELS_DIR / dest_name
    shutil.copy2(file_path, dest)
    entry = {
        "id": f"user:{uid}",
        "name": name or dest_name,
        "task": task,
        "path": str(dest.resolve()),
    }
    entries = load_user_metadata()
    entries.append(entry)
    save_user_metadata(entries)
    return entry


def load_catalog() -> List[Dict[str, Any]]:
    """可选模型目录：可下载的预置模型列表。"""
    path = _catalog_path()
    if not path.exists():
        return _default_catalog()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return _default_catalog()


def _default_catalog() -> List[Dict[str, Any]]:
    """默认可选模型（与现有内置对应，并增加可下载项）。"""
    return [
        # 检测
        {"id": "yolov8n", "name": "YOLOv8-nano", "task": "detection", "source": "ultralytics:yolov8n.pt", "size": "6MB", "accuracy": "37.3% mAP", "speed": "极快"},
        {"id": "yolov8s", "name": "YOLOv8-small", "task": "detection", "source": "ultralytics:yolov8s.pt", "size": "22MB", "accuracy": "44.9% mAP", "speed": "快速"},
        {"id": "yolov8m", "name": "YOLOv8-medium", "task": "detection", "source": "ultralytics:yolov8m.pt", "size": "52MB", "accuracy": "50.2% mAP", "speed": "中等"},
        {"id": "yolov8l", "name": "YOLOv8-large", "task": "detection", "source": "ultralytics:yolov8l.pt", "size": "87MB", "accuracy": "52.9% mAP", "speed": "慢速"},
        # 分割
        {"id": "yolov8n-seg", "name": "YOLOv8-nano-Seg", "task": "segmentation", "source": "ultralytics:yolov8n-seg.pt", "size": "7MB", "accuracy": "中高", "speed": "快速"},
        {"id": "yolov8s-seg", "name": "YOLOv8-small-Seg", "task": "segmentation", "source": "ultralytics:yolov8s-seg.pt", "size": "23MB", "accuracy": "中高", "speed": "中等"},
        {"id": "yolov8m-seg", "name": "YOLOv8-medium-Seg", "task": "segmentation", "source": "ultralytics:yolov8m-seg.pt", "size": "72MB", "accuracy": "高", "speed": "中等"},
        {"id": "yolov8l-seg", "name": "YOLOv8-large-Seg", "task": "segmentation", "source": "ultralytics:yolov8l-seg.pt", "size": "150MB", "accuracy": "高", "speed": "慢速"},
        # 分类
        {"id": "yolov8n-cls", "name": "YOLOv8-nano-Cls", "task": "classification", "source": "ultralytics:yolov8n-cls.pt", "size": "6MB", "accuracy": "通用", "speed": "极快"},
        {"id": "yolov8s-cls", "name": "YOLOv8-small-Cls", "task": "classification", "source": "ultralytics:yolov8s-cls.pt", "size": "14MB", "accuracy": "高", "speed": "快速"},
        {"id": "yolov8m-cls", "name": "YOLOv8-medium-Cls", "task": "classification", "source": "ultralytics:yolov8m-cls.pt", "size": "31MB", "accuracy": "高", "speed": "中等"},
        {"id": "yolov8l-cls", "name": "YOLOv8-large-Cls", "task": "classification", "source": "ultralytics:yolov8l-cls.pt", "size": "68MB", "accuracy": "极高", "speed": "慢速"},
        {"id": "yolov8x-cls", "name": "YOLOv8-xlarge-Cls", "task": "classification", "source": "ultralytics:yolov8x-cls.pt", "size": "99MB", "accuracy": "极高", "speed": "慢速"},
    ]


def get_downloaded_model_path(model_id: str) -> Optional[Path]:
    """已下载的目录模型路径（返回绝对路径，供 FileResponse 等使用）。"""
    _ensure_dirs()
    base = DOWNLOADED_MODELS_DIR.resolve()
    for ext in (".pt", ".pth", ""):
        p = base / f"{model_id}{ext}"
        if p.exists():
            return p.resolve()
    return None


def download_catalog_model(model_id: str) -> Path:
    """将目录中的模型下载到本地，返回绝对路径。"""
    _ensure_dirs()
    base = DOWNLOADED_MODELS_DIR.resolve()
    dest = base / f"{model_id}.pt"
    if dest.exists():
        # 已存在时也要先校验，避免损坏文件一直被当成“可用”
        try:
            from ultralytics import YOLO
            YOLO(str(dest))
            return dest
        except (EOFError, OSError, RuntimeError):
            dest.unlink(missing_ok=True)
    catalog = load_catalog()
    entry = next((e for e in catalog if e.get("id") == model_id), None)
    if not entry:
        raise ValueError(f"未知模型 id: {model_id}")
    source = entry.get("source", "")
    if not source:
        raise ValueError("该模型无下载源")
    if source.startswith("ultralytics:"):
        filename = source.split(":", 1)[1]
        from ultralytics import YOLO
        orig_cwd = os.getcwd()
        try:
            os.chdir(str(base))
            model = YOLO(filename)
            if Path(filename).exists():
                src = Path(filename).resolve()
                if src != dest:
                    shutil.move(str(src), str(dest))
            else:
                model.save(str(dest))
        except (OSError, IOError) as e:
            raise ValueError(f"下载或写入失败: {e}") from e
        finally:
            os.chdir(orig_cwd)
        # 校验：能成功加载才视为下载完整
        try:
            YOLO(str(dest))
        except (EOFError, OSError, RuntimeError) as e:
            if dest.exists():
                dest.unlink(missing_ok=True)
            if isinstance(e, EOFError) or "Ran out of input" in str(e):
                raise ValueError("模型下载不完整或已损坏，请重试") from e
            raise ValueError(f"模型校验失败: {e}") from e
        return dest
    raise ValueError("暂仅支持 ultralytics 预训练模型下载")


def is_catalog_model_local(model_id: str) -> bool:
    return get_downloaded_model_path(model_id) is not None
