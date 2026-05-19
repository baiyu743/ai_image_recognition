from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey
from sqlalchemy.orm import Session, relationship
from datetime import datetime
import uuid
import os
import json
import re
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml
import base64
from io import BytesIO
from PIL import Image as PILImage
import threading
import time
import torch
import anyio
from pydantic import BaseModel

# 导入数据库设置
from database import Base, engine, get_db

# 导入AI模型服务
from ai_models import ai_service

# 数据库模型
class Image(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    annotations = relationship("AutoAnnotation", back_populates="image")
    manual_annotations = relationship("ManualAnnotation", back_populates="image")

class AutoAnnotation(Base):
    __tablename__ = "auto_annotations"
    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("images.id"))
    tool_type = Column(String)
    annotation_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    image = relationship("Image", back_populates="annotations")

class ManualAnnotation(Base):
    __tablename__ = "manual_annotations"
    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("images.id"))
    tool_type = Column(String)
    annotation_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    image = relationship("Image", back_populates="manual_annotations")

# 导入其他模块的模型，确保它们在创建表之前被加载
from training import models as training_models
from evaluation import models as evaluation_models
from visiofirm import models as visiofirm_models
from settings import models as settings_models

# 创建数据库表
Base.metadata.create_all(bind=engine)

# 创建FastAPI应用
app = FastAPI(
    title="AI Auto-Annotation Service",
    description="An API for automatic and manual image annotation.",
    version="1.0.0",
    openapi_url="/api/v1/openapi.json"
)

PROJECT_ID_PATTERN = re.compile(r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$", re.I)

def _projects_root() -> str:
    base = os.getenv("PROJECTS_ROOT", "projects")
    return os.path.abspath(base)

def _ensure_projects_root():
    os.makedirs(_projects_root(), exist_ok=True)

def _require_safe_project_id(project_id: str) -> str:
    if not PROJECT_ID_PATTERN.match(project_id or ""):
        raise HTTPException(status_code=400, detail="Invalid project_id")
    return project_id

def _project_dir(project_id: str) -> str:
    project_id = _require_safe_project_id(project_id)
    return os.path.join(_projects_root(), project_id)

def _project_meta_path(project_id: str) -> str:
    return os.path.join(_project_dir(project_id), "meta.json")

def _read_project_meta(project_id: str) -> Dict[str, Any]:
    meta_path = _project_meta_path(project_id)
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="Project not found")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_project_meta(project_id: str, meta: Dict[str, Any]):
    meta_path = _project_meta_path(project_id)
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def _ensure_project_dirs(project_id: str) -> Dict[str, str]:
    base = _project_dir(project_id)
    paths = {
        "base": base,
        "data": os.path.join(base, "data"),
        "images_train": os.path.join(base, "data", "images", "train"),
        "images_val": os.path.join(base, "data", "images", "val"),
        "labels_train": os.path.join(base, "data", "labels", "train"),
        "labels_val": os.path.join(base, "data", "labels", "val"),
        "annotations": os.path.join(base, "annotations"),
        "staging": os.path.join(base, "staging"),
        "staging_images": os.path.join(base, "staging", "images"),
        "training": os.path.join(base, "training"),
        "evaluation": os.path.join(base, "evaluation"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:4173",
        "http://aivisioniterate.top",
        "https://aivisioniterate.top"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/projects")
async def create_project(name: str = Form(...)):
    _ensure_projects_root()
    project_id = str(uuid.uuid4())
    _ensure_project_dirs(project_id)
    now = datetime.utcnow().isoformat()
    meta = {
        "project_id": project_id,
        "name": name,
        "created_at": now,
        "updated_at": now,
        "dataset_yaml_path": None,
        "last_training_task_id": None,
        "last_training_output_dir": None,
        "last_evaluation_id": None,
        "last_evaluation_output_dir": None,
    }
    _write_project_meta(project_id, meta)
    return JSONResponse(content=meta)

@app.get("/api/projects")
async def list_projects():
    _ensure_projects_root()
    root = _projects_root()
    projects = []
    for entry in os.listdir(root):
        if not PROJECT_ID_PATTERN.match(entry):
            continue
        meta_path = os.path.join(root, entry, "meta.json")
        if not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                projects.append(json.load(f))
        except Exception:
            continue
    projects.sort(key=lambda x: x.get("updated_at") or x.get("created_at") or "", reverse=True)
    return JSONResponse(content={"projects": projects})

@app.get("/api/projects/{project_id}")
async def get_project(project_id: str):
    meta = _read_project_meta(project_id)
    paths = _ensure_project_dirs(project_id)
    return JSONResponse(content={
        **meta,
        "paths": {
            "base": paths["base"],
            "dataset_yaml_path": meta.get("dataset_yaml_path"),
            "training_dir": paths["training"],
            "evaluation_dir": paths["evaluation"],
        }
    })

@app.put("/api/projects/{project_id}")
async def update_project(project_id: str, name: str = Form(...)):
    meta = _read_project_meta(project_id)
    meta["name"] = name
    meta["updated_at"] = datetime.utcnow().isoformat()
    _write_project_meta(project_id, meta)
    return JSONResponse(content=meta)

@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    meta = _read_project_meta(project_id)
    project_path = _project_dir(project_id)
    if not os.path.isdir(project_path):
        raise HTTPException(status_code=404, detail="Project not found")

    shutil.rmtree(project_path)
    return JSONResponse(content={
        "success": True,
        "project_id": project_id,
        "name": meta.get("name"),
    })

def _clamp01(x: float) -> float:
    if x < 0:
        return 0.0
    if x > 1:
        return 1.0
    return x

@app.post("/api/projects/{project_id}/dataset/from-annotations")
async def create_dataset_from_annotations(
    project_id: str,
    images: List[UploadFile] = File(...),
    annotations: str = Form(...),
    categories: Optional[str] = Form(None),
    val_ratio: float = Form(0.2),
    splits: Optional[str] = Form(None),
):
    project_id = _require_safe_project_id(project_id)
    paths = _ensure_project_dirs(project_id)
    try:
        ann_map = json.loads(annotations) if annotations else {}
        if not isinstance(ann_map, dict):
            raise ValueError("annotations must be a JSON object mapping image filename to annotation list")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid annotations JSON")

    dataset = _load_or_init_dataset_yaml(project_id)
    dataset_yaml_path = dataset["dataset_yaml_path"]
    category_list: List[str] = list(dataset.get("names") or [])
    if categories:
        try:
            parsed = json.loads(categories)
            if isinstance(parsed, list):
                for x in parsed:
                    s = str(x).strip()
                    if s and s not in category_list:
                        category_list.append(s)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid categories JSON")

    splits_map: Dict[str, str] = {}
    if splits:
        try:
            parsed_splits = json.loads(splits)
            if isinstance(parsed_splits, dict):
                splits_map = {str(k): str(v) for k, v in parsed_splits.items()}
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid splits JSON")

    image_count = 0
    bbox_count = 0

    n_total = len(images)
    n_val = 0
    if not splits_map:
        n_val = int(round(n_total * float(val_ratio))) if n_total else 0
        n_val = max(0, min(n_total, n_val))

    for idx, upload in enumerate(images):
        filename = os.path.basename(upload.filename or f"image_{idx}.jpg")
        raw = await upload.read()
        split_value = (splits_map.get(filename) or "").lower().strip()
        if split_value in ("val", "valid", "validation", "test", "testing"):
            is_val = True
        elif split_value in ("train", "training"):
            is_val = False
        else:
            is_val = idx < n_val
        img_dir = paths["images_val"] if is_val else paths["images_train"]
        lbl_dir = paths["labels_val"] if is_val else paths["labels_train"]

        img_path = os.path.join(img_dir, filename)
        with open(img_path, "wb") as f:
            f.write(raw)

        label_path = os.path.join(lbl_dir, Path(filename).with_suffix(".txt").name)
        anns = ann_map.get(filename, [])
        lines: List[str] = []
        if isinstance(anns, list):
            for ann in anns:
                if not isinstance(ann, dict):
                    continue
                if ann.get("type") not in ("bbox", "bounding_box", "obb"):
                    continue
                bbox = ann.get("bbox") or {}
                try:
                    x = float(bbox.get("x", 0))
                    y = float(bbox.get("y", 0))
                    w = float(bbox.get("width", 0))
                    h = float(bbox.get("height", 0))
                except Exception:
                    continue
                label = (ann.get("label") or "unknown").strip() or "unknown"
                cls_id = _ensure_name_index(category_list, label)
                cx = _clamp01((x + w / 2.0) / 100.0)
                cy = _clamp01((y + h / 2.0) / 100.0)
                bw = _clamp01(w / 100.0)
                bh = _clamp01(h / 100.0)
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                bbox_count += 1

        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        image_count += 1

    _write_dataset_yaml(paths["data"], category_list, dataset_yaml_path)

    session_path = os.path.join(paths["annotations"], "session_annotations.json")
    try:
        if os.path.exists(session_path):
            with open(session_path, "r", encoding="utf-8") as f:
                prev = json.load(f) or {}
        else:
            prev = {}
    except Exception:
        prev = {}
    merged_annotations = {}
    try:
        if isinstance(prev.get("annotations"), dict):
            merged_annotations.update(prev.get("annotations"))
    except Exception:
        pass
    merged_annotations.update(ann_map)
    with open(session_path, "w", encoding="utf-8") as f:
        json.dump({"annotations": merged_annotations, "categories": category_list}, f, ensure_ascii=False, indent=2)

    meta = _read_project_meta(project_id)
    meta["dataset_yaml_path"] = dataset_yaml_path
    meta["updated_at"] = datetime.utcnow().isoformat()
    _write_project_meta(project_id, meta)

    return JSONResponse(content={
        "success": True,
        "project_id": project_id,
        "dataset_yaml_path": dataset_yaml_path,
        "image_count": image_count,
        "bbox_count": bbox_count,
        "class_count": len(category_list),
        "categories": category_list,
    })

class ProjectAutoAnnotateFileRequest(BaseModel):
    source: str = "staging"  # staging | dataset
    filename: str
    split: Optional[str] = None  # train | val（source=dataset 时可选）
    tool: str = "object_detection"
    model: Optional[str] = None
    return_annotated_image: int = 0

class ImportFromStagingItem(BaseModel):
    filename: str
    split: str = "train"

class ImportFromStagingRequest(BaseModel):
    items: List[ImportFromStagingItem]
    annotations_by_filename: Dict[str, Any] = {}
    categories: Optional[List[str]] = None
    move: bool = True

def _staging_uploads_dir(paths: Dict[str, str]) -> str:
    return os.path.join(paths["staging"], ".uploads")

UPLOAD_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$", re.I)

def _require_safe_upload_id(upload_id: str) -> str:
    if not UPLOAD_ID_PATTERN.match(upload_id or ""):
        raise HTTPException(status_code=400, detail="Invalid upload_id")
    return upload_id

def _unique_filename(dir_path: str, filename: str) -> str:
    filename = os.path.basename(filename or "")
    if not filename:
        return ""
    base, ext = os.path.splitext(filename)
    if not ext:
        ext = ".jpg"
    candidate = f"{base}{ext}"
    if not os.path.exists(os.path.join(dir_path, candidate)):
        return candidate
    for i in range(1, 1000000):
        candidate = f"{base}_{i}{ext}"
        if not os.path.exists(os.path.join(dir_path, candidate)):
            return candidate
    return f"{base}_{uuid.uuid4().hex}{ext}"

@app.post("/api/projects/{project_id}/staging/images")
async def upload_project_staging_images(
    project_id: str,
    images: List[UploadFile] = File(...),
    overwrite: int = Form(0),
):
    project_id = _require_safe_project_id(project_id)
    paths = _ensure_project_dirs(project_id)
    staging_dir = paths["staging_images"]
    do_overwrite = bool(int(overwrite or 0))

    limiter = anyio.CapacityLimiter(8)
    stored = []
    for idx, upload in enumerate(images):
        original = os.path.basename(upload.filename or f"image_{idx}.jpg")
        filename = original if do_overwrite else _unique_filename(staging_dir, original)
        if not filename:
            continue
        out_path = os.path.join(staging_dir, filename)

        def _copy_to_disk() -> None:
            with open(out_path, "wb") as f:
                shutil.copyfileobj(upload.file, f, length=1024 * 1024)

        try:
            await anyio.to_thread.run_sync(_copy_to_disk, limiter=limiter)
        finally:
            try:
                upload.file.close()
            except Exception:
                pass

        stored.append({
            "name": filename,
            "url_path": f"/api/projects/{project_id}/staging/images/{filename}",
        })
    return JSONResponse(content={"success": True, "count": len(stored), "images": stored})

@app.post("/api/projects/{project_id}/staging/uploads/init")
async def init_project_staging_upload(
    project_id: str,
    filename: str = Form(...),
    total_size: int = Form(...),
    chunk_size: int = Form(...),
    overwrite: int = Form(0),
    upload_id: Optional[str] = Form(None),
):
    project_id = _require_safe_project_id(project_id)
    paths = _ensure_project_dirs(project_id)
    uploads_dir = _staging_uploads_dir(paths)
    os.makedirs(uploads_dir, exist_ok=True)

    filename = os.path.basename(filename or "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    try:
        total_size = int(total_size)
        chunk_size = int(chunk_size)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid size")
    if total_size <= 0 or chunk_size <= 0:
        raise HTTPException(status_code=400, detail="Invalid size")

    do_overwrite = bool(int(overwrite or 0))
    if upload_id:
        upload_id = _require_safe_upload_id(upload_id)
    else:
        upload_id = uuid.uuid4().hex

    upload_dir = os.path.join(uploads_dir, upload_id)
    chunks_dir = os.path.join(upload_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    total_chunks = int((total_size + chunk_size - 1) // chunk_size)
    meta_path = os.path.join(upload_dir, "meta.json")
    meta = {
        "upload_id": upload_id,
        "filename": filename,
        "total_size": total_size,
        "chunk_size": chunk_size,
        "total_chunks": total_chunks,
        "created_at": datetime.utcnow().isoformat(),
    }

    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                prev = json.load(f) or {}
            if (
                prev.get("filename") == filename
                and int(prev.get("total_size") or 0) == total_size
                and int(prev.get("chunk_size") or 0) == chunk_size
            ):
                meta = prev
                total_chunks = int(meta.get("total_chunks") or total_chunks)
        except Exception:
            pass

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if do_overwrite:
        existing = os.path.join(paths["staging_images"], filename)
        if os.path.exists(existing):
            try:
                os.remove(existing)
            except Exception:
                pass

    received = []
    try:
        for fn in os.listdir(chunks_dir):
            if not fn.endswith(".part"):
                continue
            try:
                idx = int(fn.split(".")[0])
            except Exception:
                continue
            received.append(idx)
    except Exception:
        received = []
    received = sorted(list(set([x for x in received if 0 <= x < total_chunks])))

    return JSONResponse(content={
        "success": True,
        "upload_id": upload_id,
        "total_chunks": total_chunks,
        "received": received,
    })

@app.post("/api/projects/{project_id}/staging/uploads/{upload_id}/chunk")
async def upload_project_staging_chunk(
    project_id: str,
    upload_id: str,
    index: int = Form(...),
    chunk: UploadFile = File(...),
):
    project_id = _require_safe_project_id(project_id)
    upload_id = _require_safe_upload_id(upload_id)
    paths = _ensure_project_dirs(project_id)
    uploads_dir = _staging_uploads_dir(paths)
    upload_dir = os.path.join(uploads_dir, upload_id)
    meta_path = os.path.join(upload_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="Upload not found")
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f) or {}
    except Exception:
        raise HTTPException(status_code=500, detail="Upload meta corrupted")
    try:
        idx = int(index)
        total_chunks = int(meta.get("total_chunks") or 0)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid index")
    if idx < 0 or (total_chunks and idx >= total_chunks):
        raise HTTPException(status_code=400, detail="Invalid index")

    chunks_dir = os.path.join(upload_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    out_path = os.path.join(chunks_dir, f"{idx}.part")

    limiter = anyio.CapacityLimiter(16)

    def _copy_to_disk() -> None:
        with open(out_path, "wb") as f:
            shutil.copyfileobj(chunk.file, f, length=1024 * 1024)

    try:
        await anyio.to_thread.run_sync(_copy_to_disk, limiter=limiter)
    finally:
        try:
            chunk.file.close()
        except Exception:
            pass
    return JSONResponse(content={"success": True, "upload_id": upload_id, "index": idx})

@app.post("/api/projects/{project_id}/staging/uploads/{upload_id}/complete")
async def complete_project_staging_upload(project_id: str, upload_id: str):
    project_id = _require_safe_project_id(project_id)
    upload_id = _require_safe_upload_id(upload_id)
    paths = _ensure_project_dirs(project_id)
    uploads_dir = _staging_uploads_dir(paths)
    upload_dir = os.path.join(uploads_dir, upload_id)
    meta_path = os.path.join(upload_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="Upload not found")
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f) or {}
    except Exception:
        raise HTTPException(status_code=500, detail="Upload meta corrupted")

    filename = os.path.basename(str(meta.get("filename") or "")).strip()
    if not filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    try:
        total_size = int(meta.get("total_size") or 0)
        total_chunks = int(meta.get("total_chunks") or 0)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid meta")
    if total_size <= 0 or total_chunks <= 0:
        raise HTTPException(status_code=400, detail="Invalid meta")

    chunks_dir = os.path.join(upload_dir, "chunks")
    if not os.path.isdir(chunks_dir):
        raise HTTPException(status_code=400, detail="No chunks")

    final_dir = paths["staging_images"]
    os.makedirs(final_dir, exist_ok=True)
    final_name = _unique_filename(final_dir, filename)
    final_path = os.path.join(final_dir, final_name)
    tmp_path = os.path.join(final_dir, f".{upload_id}.tmp")

    def _assemble() -> None:
        total_written = 0
        with open(tmp_path, "wb") as out:
            for i in range(total_chunks):
                part_path = os.path.join(chunks_dir, f"{i}.part")
                if not os.path.exists(part_path):
                    raise RuntimeError(f"Missing chunk {i}")
                with open(part_path, "rb") as inp:
                    shutil.copyfileobj(inp, out, length=1024 * 1024)
                total_written = out.tell()
        if total_written != total_size:
            raise RuntimeError(f"Size mismatch: {total_written} != {total_size}")
        os.replace(tmp_path, final_path)

    try:
        await anyio.to_thread.run_sync(_assemble, limiter=anyio.CapacityLimiter(2))
    except RuntimeError as e:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise

    try:
        shutil.rmtree(upload_dir, ignore_errors=True)
    except Exception:
        pass

    return JSONResponse(content={
        "success": True,
        "name": final_name,
        "url_path": f"/api/projects/{project_id}/staging/images/{final_name}",
    })

@app.get("/api/projects/{project_id}/staging/images/{filename}")
async def get_project_staging_image(project_id: str, filename: str):
    project_id = _require_safe_project_id(project_id)
    filename = os.path.basename(filename or "")
    if not filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    paths = _ensure_project_dirs(project_id)
    img_path = _safe_join(paths["staging_images"], filename)
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(img_path)

@app.post("/api/projects/{project_id}/auto_annotate/file")
async def auto_annotate_project_file(req: ProjectAutoAnnotateFileRequest, project_id: str):
    project_id = _require_safe_project_id(project_id)
    paths = _ensure_project_dirs(project_id)
    filename = os.path.basename(req.filename or "")
    if not filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    tool = (req.tool or "").strip()
    if tool not in ("object_detection", "image_classification", "image_segmentation"):
        raise HTTPException(status_code=400, detail=f"Tool type '{tool}' is not supported.")

    file_path = None
    source = (req.source or "staging").lower().strip()
    if source == "staging":
        p = _safe_join(paths["staging_images"], filename)
        if os.path.exists(p):
            file_path = p
    elif source == "dataset":
        split = (req.split or "").lower().strip()
        candidates = []
        if split in ("train", "val"):
            candidates.append(_safe_join(paths["data"], "images", split, filename))
        else:
            candidates.append(_safe_join(paths["data"], "images", "train", filename))
            candidates.append(_safe_join(paths["data"], "images", "val", filename))
        for p in candidates:
            if os.path.exists(p):
                file_path = p
                break
    else:
        raise HTTPException(status_code=400, detail="Invalid source")

    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")

    annotations = []
    annotated_image_base64 = None
    if tool == "object_detection":
        with open(file_path, "rb") as f:
            image_bytes = f.read()
        include_image = bool(int(req.return_annotated_image or 0))
        annotations, annotated_image_base64 = ai_service.detect_objects_with_visualization(
            image_bytes,
            model_name=req.model or "YOLO",
            include_image=include_image,
        )
    elif tool == "image_classification":
        annotations = ai_service.classify_image(file_path, model_name=req.model)
    elif tool == "image_segmentation":
        annotations = ai_service.segment_objects(file_path, model_name=req.model)

    return JSONResponse(content={
        "annotations": annotations,
        "annotated_image": annotated_image_base64,
        "source": source,
        "filename": filename,
    })

@app.post("/api/projects/{project_id}/dataset/from-staging")
async def import_dataset_from_staging(project_id: str, req: ImportFromStagingRequest):
    project_id = _require_safe_project_id(project_id)
    paths = _ensure_project_dirs(project_id)

    dataset = _load_or_init_dataset_yaml(project_id)
    dataset_yaml_path = dataset["dataset_yaml_path"]
    category_list: List[str] = list(dataset.get("names") or [])
    if req.categories:
        for x in req.categories:
            s = str(x).strip()
            if s and s not in category_list:
                category_list.append(s)

    image_count = 0
    bbox_count = 0

    for it in req.items:
        filename = os.path.basename(it.filename or "")
        if not filename:
            continue
        split = (it.split or "train").lower().strip()
        split = "val" if split in ("val", "valid", "validation", "test", "testing") else "train"

        src_path = _safe_join(paths["staging_images"], filename)
        if not os.path.exists(src_path):
            continue

        img_dir = paths["images_val"] if split == "val" else paths["images_train"]
        lbl_dir = paths["labels_val"] if split == "val" else paths["labels_train"]
        dst_path = os.path.join(img_dir, filename)

        if os.path.exists(dst_path):
            if req.move:
                try:
                    os.remove(src_path)
                except Exception:
                    pass
        else:
            if req.move:
                os.replace(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

        anns = (req.annotations_by_filename or {}).get(filename, [])
        label_path = os.path.join(lbl_dir, Path(filename).with_suffix(".txt").name)
        lines: List[str] = []
        if isinstance(anns, list):
            for ann in anns:
                if not isinstance(ann, dict):
                    continue
                if ann.get("type") not in ("bbox", "bounding_box", "obb"):
                    continue
                bbox = ann.get("bbox") or {}
                try:
                    x = float(bbox.get("x", 0))
                    y = float(bbox.get("y", 0))
                    w = float(bbox.get("width", 0))
                    h = float(bbox.get("height", 0))
                except Exception:
                    continue
                label = (ann.get("label") or "unknown").strip() or "unknown"
                cls_id = _ensure_name_index(category_list, label)
                cx = _clamp01((x + w / 2.0) / 100.0)
                cy = _clamp01((y + h / 2.0) / 100.0)
                bw = _clamp01(w / 100.0)
                bh = _clamp01(h / 100.0)
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                bbox_count += 1
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        image_count += 1

    _write_dataset_yaml(paths["data"], category_list, dataset_yaml_path)

    session_path = os.path.join(paths["annotations"], "session_annotations.json")
    try:
        if os.path.exists(session_path):
            with open(session_path, "r", encoding="utf-8") as f:
                prev = json.load(f) or {}
        else:
            prev = {}
    except Exception:
        prev = {}
    merged_annotations = {}
    try:
        if isinstance(prev.get("annotations"), dict):
            merged_annotations.update(prev.get("annotations"))
    except Exception:
        pass
    if isinstance(req.annotations_by_filename, dict):
        merged_annotations.update(req.annotations_by_filename)
    with open(session_path, "w", encoding="utf-8") as f:
        json.dump({"annotations": merged_annotations, "categories": category_list}, f, ensure_ascii=False, indent=2)

    meta = _read_project_meta(project_id)
    meta["dataset_yaml_path"] = dataset_yaml_path
    meta["updated_at"] = datetime.utcnow().isoformat()
    _write_project_meta(project_id, meta)

    return JSONResponse(content={
        "success": True,
        "project_id": project_id,
        "dataset_yaml_path": dataset_yaml_path,
        "image_count": image_count,
        "bbox_count": bbox_count,
        "class_count": len(category_list),
        "categories": category_list,
    })

def _safe_join(base_dir: str, *parts: str) -> str:
    base = Path(base_dir).resolve()
    target = base.joinpath(*parts).resolve()
    if base != target and base not in target.parents:
        raise HTTPException(status_code=400, detail="Invalid path")
    return str(target)

def _load_project_dataset(project_id: str) -> Dict[str, Any]:
    meta = _read_project_meta(project_id)
    dataset_yaml_path = meta.get("dataset_yaml_path") or os.path.join(_project_dir(project_id), "data", "dataset.yaml")
    dataset_yaml_path = os.path.abspath(dataset_yaml_path)
    if not os.path.exists(dataset_yaml_path):
        raise HTTPException(status_code=404, detail="dataset.yaml not found for project")
    with open(dataset_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    data_root = cfg.get("path") or os.path.dirname(dataset_yaml_path)
    if not os.path.isabs(data_root):
        data_root = os.path.abspath(os.path.join(os.path.dirname(dataset_yaml_path), data_root))
    names = cfg.get("names") or []
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))]
    if not isinstance(names, list):
        names = []
    return {"dataset_yaml_path": dataset_yaml_path, "data_root": data_root, "names": names}

def _yolo_txt_to_annotations(label_path: str, names: List[str]) -> List[Dict[str, Any]]:
    if not os.path.exists(label_path):
        return []
    annotations = []
    with open(label_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(float(parts[0]))
                cx = float(parts[1])
                cy = float(parts[2])
                bw = float(parts[3])
                bh = float(parts[4])
            except Exception:
                continue
            label = names[cls_id] if 0 <= cls_id < len(names) else str(cls_id)
            x = (cx - bw / 2.0) * 100.0
            y = (cy - bh / 2.0) * 100.0
            annotations.append({
                "id": f"yolo-{i}",
                "type": "bbox",
                "label": label,
                "confidence": 1.0,
                "bbox": {
                    "x": float(max(0.0, min(100.0, x))),
                    "y": float(max(0.0, min(100.0, y))),
                    "width": float(max(0.0, min(100.0, bw * 100.0))),
                    "height": float(max(0.0, min(100.0, bh * 100.0))),
                }
            })
    return annotations

def _write_dataset_yaml(data_root: str, names: List[str], dataset_yaml_path: str) -> None:
    yaml_content = "\n".join([
        f"path: {data_root}",
        "train: images/train",
        "val: images/val",
        f"names: {json.dumps(names, ensure_ascii=False)}",
        "",
    ])
    with open(dataset_yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

def _load_or_init_dataset_yaml(project_id: str) -> Dict[str, Any]:
    meta = _read_project_meta(project_id)
    dataset_yaml_path = meta.get("dataset_yaml_path") or os.path.join(_project_dir(project_id), "data", "dataset.yaml")
    dataset_yaml_path = os.path.abspath(dataset_yaml_path)
    if not os.path.exists(dataset_yaml_path):
        paths = _ensure_project_dirs(project_id)
        data_root = paths["data"]
        names: List[str] = []
        _write_dataset_yaml(data_root, names, dataset_yaml_path)
        meta["dataset_yaml_path"] = dataset_yaml_path
        meta["updated_at"] = datetime.utcnow().isoformat()
        _write_project_meta(project_id, meta)
        return {"dataset_yaml_path": dataset_yaml_path, "data_root": data_root, "names": names}
    dataset = _load_project_dataset(project_id)
    return dataset

def _ensure_name_index(names: List[str], label: str) -> int:
    label = (label or "unknown").strip() or "unknown"
    if label in names:
        return names.index(label)
    names.append(label)
    return len(names) - 1

class BatchAnnotateRequest(BaseModel):
    tool: str = "object_detection"
    model: Optional[str] = None
    split: str = "all"  # train | val | all
    filenames: Optional[List[str]] = None
    only_unannotated: bool = True
    conf: float = 0.25
    iou: float = 0.7
    imgsz: int = 640
    max_det: int = 200

class BatchAnnotateTaskState(BaseModel):
    task_id: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    total: int = 0
    processed: int = 0
    bbox_count: int = 0
    last_image: Optional[str] = None
    error: Optional[str] = None

_batch_tasks_lock = threading.Lock()
_batch_tasks: Dict[str, BatchAnnotateTaskState] = {}

def _batch_task_update(task_id: str, **kwargs) -> None:
    with _batch_tasks_lock:
        st = _batch_tasks.get(task_id)
        if not st:
            return
        updated = st.model_copy(update=kwargs)
        _batch_tasks[task_id] = updated

def _list_project_images(data_root: str, split: str) -> List[str]:
    split = (split or "all").lower().strip()
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    splits = ["train", "val"] if split == "all" else [split]
    out = []
    for sp in splits:
        if sp not in ("train", "val"):
            continue
        img_dir = os.path.join(data_root, "images", sp)
        if not os.path.isdir(img_dir):
            continue
        for fn in sorted(os.listdir(img_dir)):
            if os.path.splitext(fn)[1].lower() in image_extensions:
                out.append(os.path.join(img_dir, fn))
    return out

def _infer_split_from_path(data_root: str, img_path: str) -> str:
    try:
        rel = os.path.relpath(img_path, data_root).replace("\\", "/")
        parts = rel.split("/")
        if len(parts) >= 2 and parts[0] == "images" and parts[1] in ("train", "val"):
            return parts[1]
    except Exception:
        pass
    return "train"

def _label_path_for_image(data_root: str, split: str, filename: str) -> str:
    return os.path.join(data_root, "labels", split, Path(filename).with_suffix(".txt").name)

def _batch_annotate_worker(project_id: str, task_id: str, req: BatchAnnotateRequest) -> None:
    try:
        _batch_task_update(task_id, status="running", started_at=datetime.utcnow().isoformat())
        dataset = _load_or_init_dataset_yaml(project_id)
        data_root = dataset["data_root"]
        names: List[str] = list(dataset["names"] or [])
        dataset_yaml_path = dataset["dataset_yaml_path"]

        all_images = _list_project_images(data_root, req.split)
        if req.filenames:
            allow = set([os.path.basename(x) for x in req.filenames if x])
            all_images = [p for p in all_images if os.path.basename(p) in allow]

        selected_paths = []
        for p in all_images:
            fn = os.path.basename(p)
            sp = _infer_split_from_path(data_root, p)
            lbl = _label_path_for_image(data_root, sp, fn)
            if req.only_unannotated and os.path.exists(lbl) and os.path.getsize(lbl) > 0:
                continue
            selected_paths.append(p)

        _batch_task_update(task_id, total=len(selected_paths))
        if not selected_paths:
            _batch_task_update(task_id, status="completed", finished_at=datetime.utcnow().isoformat())
            return

        model_name = (req.model or "YOLO").strip() if req.model else "YOLO"
        model, use_yolo, _resolved = ai_service._get_detection_model(model_name)
        if model is None or not use_yolo:
            model = ai_service._load_builtin_detection_model("YOLO")
        if model is None:
            raise RuntimeError("YOLO detection model unavailable")

        device = 0 if torch.cuda.is_available() else "cpu"
        half = bool(device != "cpu")

        bbox_total = 0
        processed = 0
        results = model.predict(
            source=selected_paths,
            stream=True,
            imgsz=req.imgsz,
            conf=req.conf,
            iou=req.iou,
            max_det=req.max_det,
            device=device,
            half=half,
            verbose=False,
        )

        for r in results:
            img_path = r.path
            filename = os.path.basename(img_path)
            sp = _infer_split_from_path(data_root, img_path)
            lbl_path = _label_path_for_image(data_root, sp, filename)
            os.makedirs(os.path.dirname(lbl_path), exist_ok=True)

            lines: List[str] = []
            boxes = getattr(r, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                xywhn = getattr(boxes, "xywhn", None)
                cls = getattr(boxes, "cls", None)
                if xywhn is not None and cls is not None:
                    xywhn_list = xywhn.tolist()
                    cls_list = [int(x) for x in cls.tolist()]
                    for (cx, cy, bw, bh), cls_id in zip(xywhn_list, cls_list):
                        label = r.names.get(cls_id, str(cls_id))
                        cls_out = _ensure_name_index(names, str(label))
                        lines.append(f"{cls_out} {float(cx):.6f} {float(cy):.6f} {float(bw):.6f} {float(bh):.6f}")
                    bbox_total += len(lines)

            with open(lbl_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            processed += 1
            if processed % 5 == 0 or processed == len(selected_paths):
                _batch_task_update(task_id, processed=processed, bbox_count=bbox_total, last_image=filename)

        if names != list(dataset["names"] or []):
            _write_dataset_yaml(data_root, names, dataset_yaml_path)
            meta = _read_project_meta(project_id)
            meta["dataset_yaml_path"] = dataset_yaml_path
            meta["updated_at"] = datetime.utcnow().isoformat()
            _write_project_meta(project_id, meta)

        _batch_task_update(task_id, status="completed", processed=processed, bbox_count=bbox_total, finished_at=datetime.utcnow().isoformat())
    except Exception as e:
        _batch_task_update(task_id, status="failed", error=str(e), finished_at=datetime.utcnow().isoformat())

@app.post("/api/projects/{project_id}/auto_annotate/batch")
async def create_batch_auto_annotate_task(project_id: str, request: BatchAnnotateRequest):
    project_id = _require_safe_project_id(project_id)
    task_id = f"anno_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    state = BatchAnnotateTaskState(
        task_id=task_id,
        status="pending",
        created_at=datetime.utcnow().isoformat(),
        total=0,
        processed=0,
        bbox_count=0,
    )
    with _batch_tasks_lock:
        _batch_tasks[task_id] = state
    th = threading.Thread(target=_batch_annotate_worker, args=(project_id, task_id, request), daemon=True)
    th.start()
    return JSONResponse(content={"success": True, "task_id": task_id})

@app.get("/api/projects/{project_id}/auto_annotate/tasks/{task_id}")
async def get_batch_auto_annotate_task(project_id: str, task_id: str):
    project_id = _require_safe_project_id(project_id)
    with _batch_tasks_lock:
        st = _batch_tasks.get(task_id)
    if not st:
        raise HTTPException(status_code=404, detail="task not found")
    return JSONResponse(content=st.model_dump())

@app.get("/api/projects/{project_id}/dataset/images/{split}/{filename}")
async def get_project_dataset_image(project_id: str, split: str, filename: str):
    project_id = _require_safe_project_id(project_id)
    if split not in ("train", "val"):
        raise HTTPException(status_code=400, detail="Invalid split")
    filename = os.path.basename(filename or "")
    if not filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    dataset = _load_project_dataset(project_id)
    img_path = _safe_join(dataset["data_root"], "images", split, filename)
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(img_path)

@app.get("/api/projects/{project_id}/dataset/state")
async def get_project_dataset_state(
    project_id: str,
    include_thumbnails: bool = False,
    thumb_size: int = 160,
    thumb_quality: int = 70,
):
    project_id = _require_safe_project_id(project_id)
    dataset = _load_or_init_dataset_yaml(project_id)
    data_root = dataset["data_root"]
    names = dataset["names"]

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = []
    max_thumbnails = 200
    thumb_size = int(max(32, min(512, thumb_size)))
    thumb_quality = int(max(30, min(95, thumb_quality)))
    for split in ("train", "val"):
        img_dir = os.path.join(data_root, "images", split)
        lbl_dir = os.path.join(data_root, "labels", split)
        if not os.path.isdir(img_dir):
            continue
        for fn in sorted(os.listdir(img_dir)):
            ext = os.path.splitext(fn)[1].lower()
            if ext not in image_extensions:
                continue
            label_path = os.path.join(lbl_dir, Path(fn).with_suffix(".txt").name)
            anns = _yolo_txt_to_annotations(label_path, names)
            thumbnail_data_url = None
            if include_thumbnails and len(images) < max_thumbnails:
                try:
                    img_path = os.path.join(img_dir, fn)
                    with PILImage.open(img_path) as im:
                        im = im.convert("RGB")
                        im.thumbnail((thumb_size, thumb_size))
                        buf = BytesIO()
                        im.save(buf, format="JPEG", quality=thumb_quality, optimize=True)
                        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                        thumbnail_data_url = f"data:image/jpeg;base64,{b64}"
                except Exception:
                    thumbnail_data_url = None
            images.append({
                "name": fn,
                "split": split,
                "url_path": f"/api/projects/{project_id}/dataset/images/{split}/{fn}",
                "annotations": anns,
                "thumbnail": thumbnail_data_url,
            })
    return JSONResponse(content={
        "project_id": project_id,
        "dataset_yaml_path": dataset["dataset_yaml_path"],
        "categories": names,
        "images": images
    })

@app.get("/api/projects/{project_id}/artifacts/evaluation-data")
async def get_latest_evaluation_data(project_id: str):
    project_id = _require_safe_project_id(project_id)
    training_dir = os.path.join(_project_dir(project_id), "training")
    if not os.path.exists(training_dir):
        raise HTTPException(status_code=404, detail="No training artifacts")
    newest_path = None
    newest_mtime = None
    for root, _, files in os.walk(training_dir):
        for fn in files:
            if fn != "evaluation_data.json":
                continue
            p = os.path.join(root, fn)
            try:
                mtime = os.path.getmtime(p)
            except Exception:
                continue
            if newest_mtime is None or mtime > newest_mtime:
                newest_mtime = mtime
                newest_path = p
    if not newest_path:
        raise HTTPException(status_code=404, detail="evaluation_data.json not found")
    with open(newest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(content={"path": newest_path, "data": data})

@app.post("/api/projects/{project_id}/artifacts/evaluation-data/regenerate")
async def regenerate_latest_evaluation_data(project_id: str):
    project_id = _require_safe_project_id(project_id)
    meta = _read_project_meta(project_id)
    dataset_yaml_path = meta.get("dataset_yaml_path")
    if not dataset_yaml_path or not os.path.exists(dataset_yaml_path):
        raise HTTPException(status_code=404, detail="dataset.yaml not found for project")

    training_dir = os.path.join(_project_dir(project_id), "training")
    if not os.path.exists(training_dir):
        raise HTTPException(status_code=404, detail="No training artifacts")

    best_pt = None
    last_out = meta.get("last_training_output_dir")
    if last_out and os.path.isdir(last_out):
        p_try = os.path.join(last_out, "weights", "best.pt")
        if os.path.exists(p_try):
            best_pt = p_try
    if not best_pt:
        newest_mtime = None
        for root, _, files in os.walk(training_dir):
            for fn in files:
                if fn != "best.pt":
                    continue
                p = os.path.join(root, fn)
                try:
                    mtime = os.path.getmtime(p)
                except Exception:
                    continue
                if newest_mtime is None or mtime > newest_mtime:
                    newest_mtime = mtime
                    best_pt = p
    if not best_pt:
        raise HTTPException(status_code=404, detail="best.pt not found under project training directory")

    try:
        from training.training_service import TrainingTask, TrainingConfig, TrainingType, training_service
        tmp_task = TrainingTask(
            task_id=f"regen_eval_{int(time.time())}",
            training_type=TrainingType.REGULAR,
            config=TrainingConfig(task="detect", data_path=dataset_yaml_path, epochs=1, project=training_dir, name=meta.get("name") or "project"),
            total_epochs=1
        )
        training_service._generate_evaluation_json(tmp_task, best_pt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to regenerate evaluation_data.json: {str(e)}")

    newest_path = None
    newest_mtime = None
    for root, _, files in os.walk(training_dir):
        for fn in files:
            if fn != "evaluation_data.json":
                continue
            p = os.path.join(root, fn)
            try:
                mtime = os.path.getmtime(p)
            except Exception:
                continue
            if newest_mtime is None or mtime > newest_mtime:
                newest_mtime = mtime
                newest_path = p
    if not newest_path:
        raise HTTPException(status_code=404, detail="evaluation_data.json not found after regeneration")
    with open(newest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(content={"success": True, "path": newest_path, "data": data})

@app.get("/api/projects/{project_id}/evaluations")
async def list_project_evaluations(project_id: str, limit: int = 50):
    project_id = _require_safe_project_id(project_id)
    limit = int(max(1, min(200, limit)))
    paths = _ensure_project_dirs(project_id)
    eval_root = paths["evaluation"]

    entries: List[Dict[str, Any]] = []
    if os.path.isdir(eval_root):
        for name in os.listdir(eval_root):
            if not name.startswith("eval_"):
                continue
            full_dir = os.path.join(eval_root, name)
            if not os.path.isdir(full_dir):
                continue
            evaluation_id = name[len("eval_"):]
            try:
                mtime = os.path.getmtime(full_dir)
            except Exception:
                mtime = None
            metrics_path = os.path.join(full_dir, "metrics.json")
            report_path = os.path.join(full_dir, "analysis_report.md")
            metrics = None
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        metrics = json.load(f)
                except Exception:
                    metrics = None
            has_report = os.path.exists(report_path)
            summary = None
            if isinstance(metrics, dict):
                summary = {
                    "mAP50": metrics.get("mAP50"),
                    "mAP50_95": metrics.get("mAP50_95"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                }
            entries.append({
                "evaluation_id": evaluation_id,
                "output_dir": full_dir,
                "metrics_path": metrics_path if os.path.exists(metrics_path) else None,
                "report_path": report_path if has_report else None,
                "created_at": datetime.utcfromtimestamp(mtime).isoformat() if mtime else None,
                "summary": summary,
            })
    entries.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    entries = entries[:limit]

    meta = _read_project_meta(project_id)
    return JSONResponse(content={
        "project_id": project_id,
        "last_evaluation_id": meta.get("last_evaluation_id"),
        "evaluations": entries,
    })

@app.get("/api/projects/{project_id}/evaluations/{evaluation_id}")
async def get_project_evaluation(project_id: str, evaluation_id: str):
    project_id = _require_safe_project_id(project_id)
    evaluation_id = str(evaluation_id or "").strip()
    if not re.match(r"^[0-9A-Za-z_\\-]+$", evaluation_id):
        raise HTTPException(status_code=400, detail="Invalid evaluation_id")
    paths = _ensure_project_dirs(project_id)
    eval_dir = _safe_join(paths["evaluation"], f"eval_{evaluation_id}")
    if not os.path.isdir(eval_dir):
        raise HTTPException(status_code=404, detail="Evaluation not found")

    metrics_path = os.path.join(eval_dir, "metrics.json")
    report_path = os.path.join(eval_dir, "analysis_report.md")
    metrics = None
    report_md = None
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            metrics = None
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                report_md = f.read()
        except Exception:
            report_md = None

    try:
        mtime = os.path.getmtime(eval_dir)
        created_at = datetime.utcfromtimestamp(mtime).isoformat()
    except Exception:
        created_at = None

    return JSONResponse(content={
        "project_id": project_id,
        "evaluation_id": evaluation_id,
        "output_dir": eval_dir,
        "created_at": created_at,
        "metrics": metrics,
        "report_md": report_md,
    })

# 导入并包含训练API路由
from training.api import router as training_router
app.include_router(training_router)

# 导入并包含评估API路由
from evaluation.routes import router as evaluation_router
app.include_router(evaluation_router)

# 导入并包含VisioFirm API路由
from visiofirm.routes import router as visiofirm_router
app.include_router(visiofirm_router)

# 导入并包含Settings API路由
from settings.api import router as settings_router
app.include_router(settings_router)

# 智能体数据增广 API：在 main 中直接注册，避免子模块导入失败导致 404
try:
    from augmentation.routes import router as augmentation_router
    app.include_router(augmentation_router)
except Exception as e:
    import logging
    _log = logging.getLogger("uvicorn.error")
    _log.warning("增广模块加载失败，/api/augmentation/run 将返回 503: %s", e)
    from typing import List
    @app.post("/api/augmentation/run")
    async def _augmentation_unavailable(
        instruction: str = Form(...),
        images: List[UploadFile] = File(...),
    ):
        return JSONResponse(
            status_code=503,
            content={"detail": "增广模块未加载。请：1) 在 ai-image-recognition-backend 目录下启动；2) pip install openai；3) 在 .env/.env.production 配置 DASHSCOPE_API_KEY（或配置 .augmentation_api_key）"}
        )

# 在应用启动时打印已注册路由，便于部署环境排障
@app.on_event("startup")
async def log_registered_routes():
    try:
        route_info = []
        for route in app.routes:
            path = getattr(route, "path", None)
            methods = getattr(route, "methods", None)
            if path:
                route_info.append({
                    "path": path,
                    "methods": sorted(list(methods)) if methods else []
                })
        print(f"已注册路由数量: {len(route_info)}")
        for r in route_info:
            print(f"路由: {r['path']} 方法: {','.join(r['methods'])}")
    except Exception as e:
        print(f"启动时打印路由失败: {e}")

@app.get("/api/tools")
async def get_tools():
    """返回可用的AI标注工具列表"""
    tools = [
        {"value": "object_detection", "label": "目标检测"},
        {"value": "image_classification", "label": "图像分类"},
        {"value": "image_segmentation", "label": "图像分割"}
    ]
    return JSONResponse(content=tools)

@app.post("/api/auto_annotate")
async def auto_annotate(
    image: UploadFile = File(...),
    tool: str = Form(...),
    model: Optional[str] = Form(None),
    return_annotated_image: int = Form(0),
    db: Session = Depends(get_db),
):
    try:
        # 1. 读取图片数据
        image_bytes = await image.read()
        
        # 2. 保存图片信息到数据库
        db_image = db.query(Image).filter(Image.filename == image.filename).first()
        if not db_image:
            db_image = Image(filename=image.filename)
            db.add(db_image)
            db.commit()
            db.refresh(db_image)

        # 3. 使用AI模型生成标注
        annotations = []
        annotated_image_base64 = None
        
        # 保存上传的图片到临时文件
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file.write(image_bytes)
            temp_file_path = temp_file.name
        
        try:
            if tool == "object_detection":
                include_image = bool(int(return_annotated_image or 0))
                annotations, annotated_image_base64 = ai_service.detect_objects_with_visualization(
                    image_bytes,
                    model_name=model or "YOLO",
                    include_image=include_image,
                )
            elif tool == "image_classification":
                # 调用分类方法
                classification_results = ai_service.classify_image(temp_file_path, model_name=model)
                annotations = classification_results
            elif tool == "image_segmentation":
                # 调用分割方法
                segmentation_results = ai_service.segment_objects(temp_file_path, model_name=model)
                annotations = segmentation_results
            else:
                raise HTTPException(status_code=400, detail=f"Tool type '{tool}' is not supported.")
        finally:
            # 清理临时文件
            import os
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        # 4. 保存所有标注到数据库
        for annotation in annotations:
            db_annotation = AutoAnnotation(
                image_id=db_image.id,
                tool_type=tool,
                annotation_data=annotation
            )
            db.add(db_annotation)
        
        db.commit()

        # 5. 封装响应数据
        response_data = {
            "annotations": annotations,
            "annotated_image": annotated_image_base64, # 添加标注后的图片
            "database_info": {
                "image_id": db_image.id,
                "annotation_count": len(annotations)
            }
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        db.rollback()
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error processing image: {str(e)}"}
        )

# 添加手动标注提交API

class AnnotationItem(BaseModel):
    id: Optional[str] = None
    type: str
    label: Optional[str] = None
    confidence: Optional[float] = 1.0
    bbox: Optional[Dict[str, float]] = None
    points: Optional[List[List[float]]] = None
    rectanglelabels: Optional[List[str]] = None
    polygonlabels: Optional[List[str]] = None
    keypointlabels: Optional[List[str]] = None
    x: Optional[float] = None
    y: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None

class SingleAnnotationRequest(BaseModel):
    imageName: str
    tool: str
    annotation: AnnotationItem

class MultipleAnnotationRequest(BaseModel):
    imageName: str
    tool: str
    annotations: List[AnnotationItem]

@app.post("/api/annotations/single")
async def submit_single_annotation(request: SingleAnnotationRequest, db: Session = Depends(get_db)):
    try:
        print(f"收到单标注保存请求: {request}")
        print(f"请求数据类型: imageName={type(request.imageName)}, tool={type(request.tool)}, annotation={type(request.annotation)}")
        print(f"标注数据: {request.annotation.dict()}")
        
        # 查找或创建图片记录
        db_image = db.query(Image).filter(Image.filename == request.imageName).first()
        if not db_image:
            db_image = Image(filename=request.imageName)
            db.add(db_image)
            db.commit()
            db.refresh(db_image)
        
        # 保存手动标注
        db_annotation = ManualAnnotation(
            image_id=db_image.id,
            tool_type=request.tool,
            annotation_data=request.annotation.dict()
        )
        db.add(db_annotation)
        db.commit()
        db.refresh(db_annotation)
        
        return JSONResponse(content={
            "success": True,
            "message": "Annotation saved successfully",
            "annotation_id": db_annotation.id,
            "image_id": db_image.id
        })
        
    except Exception as e:
        db.rollback()
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error saving annotation: {str(e)}"}
        )

@app.post("/api/annotations/batch")
async def submit_batch_annotations(request: MultipleAnnotationRequest, db: Session = Depends(get_db)):
    try:
        # 查找或创建图片记录
        db_image = db.query(Image).filter(Image.filename == request.imageName).first()
        if not db_image:
            db_image = Image(filename=request.imageName)
            db.add(db_image)
            db.commit()
            db.refresh(db_image)
        
        # 保存多个手动标注
        created_annotations = []
        for annotation in request.annotations:
            db_annotation = ManualAnnotation(
                image_id=db_image.id,
                tool_type=request.tool,
                annotation_data=annotation.dict()
            )
            db.add(db_annotation)
            created_annotations.append(db_annotation)
        
        db.commit()
        
        return JSONResponse(content={
            "success": True,
            "message": "Annotations saved successfully",
            "annotation_ids": [ann.id for ann in created_annotations],
            "image_id": db_image.id,
            "count": len(created_annotations)
        })
        
    except Exception as e:
        db.rollback()
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error saving annotations: {str(e)}"}
        )

@app.delete("/api/annotations/{annotation_id}")
async def delete_annotation(annotation_id: int, db: Session = Depends(get_db)):
    try:
        # 查找标注
        db_annotation = db.query(ManualAnnotation).filter(ManualAnnotation.id == annotation_id).first()
        if not db_annotation:
            raise HTTPException(status_code=404, detail="Annotation not found")
        
        # 删除标注
        db.delete(db_annotation)
        db.commit()
        
        return JSONResponse(content={
            "success": True,
            "message": "Annotation deleted successfully"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error deleting annotation: {str(e)}"}
        )

# 保留原有API以确保向后兼容性
@app.post("/api/annotations")
async def submit_annotation(annotation_data: dict, db: Session = Depends(get_db)):
    try:
        print(f"收到标注保存请求: {annotation_data}")
        
        # 根据图片名称查找图片
        image_name = annotation_data.get('imageName')
        tool_type = annotation_data.get('tool')
        annotation = annotation_data.get('annotation')
        
        if not all([image_name, tool_type, annotation]):
            print(f"缺少必需字段: image_name={image_name}, tool_type={tool_type}, annotation={annotation}")
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        # 查找或创建图片记录
        db_image = db.query(Image).filter(Image.filename == image_name).first()
        if not db_image:
            print(f"创建新图片记录: {image_name}")
            db_image = Image(filename=image_name)
            db.add(db_image)
            db.commit()
            db.refresh(db_image)
        
        # 保存手动标注
        print(f"保存标注到数据库，图片ID: {db_image.id}, 工具类型: {tool_type}")
        db_annotation = ManualAnnotation(
            image_id=db_image.id,
            tool_type=tool_type,
            annotation_data=annotation
        )
        db.add(db_annotation)
        db.commit()
        db.refresh(db_annotation)
        
        print(f"标注保存成功，ID: {db_annotation.id}")
        return JSONResponse(content={
            "success": True,
            "message": "Annotation saved successfully",
            "annotation_id": db_annotation.id,
            "image_id": db_image.id
        })
        
    except HTTPException as http_exc:
        print(f"HTTP错误: {http_exc.detail}")
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"保存标注时出错: {str(e)}")
        print(f"错误堆栈: {error_trace}")
        db.rollback()
        return JSONResponse(
            status_code=500,
            content={"success": False, "detail": f"Error saving annotation: {str(e)}"}
        )

# 获取图片的所有标注
@app.get("/api/images/{image_id}/annotations")
async def get_image_annotations(image_id: int, db: Session = Depends(get_db)):
    try:
        db_image = db.query(Image).filter(Image.id == image_id).first()
        if not db_image:
            raise HTTPException(status_code=404, detail="Image not found")
        
        auto_annotations = db.query(AutoAnnotation).filter(AutoAnnotation.image_id == image_id).all()
        manual_annotations = db.query(ManualAnnotation).filter(ManualAnnotation.image_id == image_id).all()
        
        return JSONResponse(content={
            "image_id": image_id,
            "filename": db_image.filename,
            "auto_annotations": [{
                "id": ann.id,
                "tool_type": ann.tool_type,
                "annotation_data": ann.annotation_data,
                "created_at": ann.created_at.isoformat()
            } for ann in auto_annotations],
            "manual_annotations": [{
                "id": ann.id,
                "tool_type": ann.tool_type,
                "annotation_data": ann.annotation_data,
                "created_at": ann.created_at.isoformat()
            } for ann in manual_annotations]
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error retrieving annotations: {str(e)}"}
        )

@app.get("/api/annotations/{image_name}")
async def get_annotations_by_image_name(image_name: str, db: Session = Depends(get_db)):
    try:
        db_image = db.query(Image).filter(Image.filename == image_name).first()
        if not db_image:
            return JSONResponse(content={
                "image_id": None,
                "filename": image_name,
                "auto_annotations": [],
                "manual_annotations": []
            })

        auto_annotations = db.query(AutoAnnotation).filter(AutoAnnotation.image_id == db_image.id).all()
        manual_annotations = db.query(ManualAnnotation).filter(ManualAnnotation.image_id == db_image.id).all()

        return JSONResponse(content={
            "image_id": db_image.id,
            "filename": db_image.filename,
            "auto_annotations": [{
                "id": ann.id,
                "tool_type": ann.tool_type,
                "annotation_data": ann.annotation_data,
                "created_at": ann.created_at.isoformat()
            } for ann in auto_annotations],
            "manual_annotations": [{
                "id": ann.id,
                "tool_type": ann.tool_type,
                "annotation_data": ann.annotation_data,
                "created_at": ann.created_at.isoformat()
            } for ann in manual_annotations]
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error retrieving annotations by name: {str(e)}"}
        )

@app.get("/")
def read_root():
    return {"message": "AI Auto-Annotation Backend is running."}

# 调试接口：返回已注册路由列表
@app.get("/api/routes")
async def list_routes():
    routes = []
    for route in app.routes:
        path = getattr(route, "path", None)
        name = getattr(route, "name", None)
        methods = getattr(route, "methods", None)
        if path:
            routes.append({
                "path": path,
                "name": name,
                "methods": sorted(list(methods)) if methods else []
            })
    return {"count": len(routes), "routes": routes}

@app.get("/api/images/list")
async def get_images_list(db: Session = Depends(get_db)):
    """
    获取所有已上传图片的列表
    """
    try:
        images = db.query(Image).all()
        
        # 为每个图片统计标注数量
        result = []
        for image in images:
            auto_count = db.query(AutoAnnotation).filter(AutoAnnotation.image_id == image.id).count()
            manual_count = db.query(ManualAnnotation).filter(ManualAnnotation.image_id == image.id).count()
            
            result.append({
                "id": image.id,
                "filename": image.filename,
                "uploaded_at": image.uploaded_at.isoformat() if image.uploaded_at else None,
                "auto_annotation_count": auto_count,
                "manual_annotation_count": manual_count
            })
        
        return JSONResponse(content={
            "success": True,
            "images": result
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error retrieving images list: {str(e)}"}
        )

@app.get("/api/health")
async def health_check():
    """
    健康检查端点
    """
    import datetime
    return {"status": "healthy", "timestamp": datetime.datetime.utcnow().isoformat()}

# 调试接口：返回版本与环境信息
@app.get("/api/version")
async def version_info():
    import os
    return {
        "app_title": app.title,
        "version": app.version,
        "openapi_url": app.openapi_url,
        "environment": os.environ.get("ENVIRONMENT", "unknown")
    }

@app.get("/pipeline-test")
def pipeline_test():
    """
    这是一个用于测试CI/CD流水线的API接口。
    """
    return {"message": "CI/CD pipeline test successful! The code has been updated."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
