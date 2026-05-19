from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Optional
import json
import os
import uuid
import tempfile
from pathlib import Path
from datetime import datetime
import time
from sqlalchemy.orm import Session
from fastapi import Depends

# 导入数据库和AI模型服务
from database import get_db
from ai_models import ai_service
from .models import VisioFirmAnnotation
from . import model_registry

router = APIRouter(
    prefix="/api/visiofirm",
    tags=["visiofirm"],
    responses={404: {"description": "Not found"}},
)

@router.post("/annotate")
async def annotate_image(
    image: UploadFile = File(...),
    tool: str = Form(...),
    model: Optional[str] = Form(None),
    categories: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    使用 Vision model iterate hub 进行图像自动标注
    
    - **file**: 要标注的图像文件
    - **tool_type**: 标注工具类型 (bbox, polygon, obb)
    - **model**: 可选的模型名称
    - **categories**: 可选的类别列表，JSON格式字符串
    """
    try:
        t0 = time.perf_counter()
        # 创建临时文件保存上传的图像
        temp_file_path = f"temp_{uuid.uuid4()}.jpg"
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await image.read())
        
        # 解析类别列表
        category_list = []
        if categories:
            try:
                category_list = json.loads(categories)
            except json.JSONDecodeError:
                category_list = []
        
        # 根据工具类型选择不同的标注方法
        annotations = []
        if tool == "object_detection":
            # 使用AI服务进行边界框检测
            detections = ai_service.detect_objects(temp_file_path, model_name=model)
            
            # 转换为前端需要的格式
            for det in detections:
                class_name = det["class_name"]
                if category_list and class_name not in category_list:
                    continue
                    
                # 从bbox_percent中获取百分比坐标
                x_percent, y_percent, w_percent, h_percent = det["bbox_percent"]
                
                annotations.append({
                    "type": "bbox",
                    "label": class_name,
                    "confidence": det["confidence"],
                    "bbox": {
                        "x": x_percent * 100,  # 转换为百分比
                        "y": y_percent * 100,  # 转换为百分比
                        "width": w_percent * 100,  # 转换为百分比
                        "height": h_percent * 100  # 转换为百分比
                    }
                })
        elif tool == "image_classification":
            # 使用AI服务进行图像分类
            classifications = ai_service.classify_image(temp_file_path, model_name=model)
            
            # 转换为前端需要的格式
            for cls in classifications:
                annotations.append({
                    "type": "classification",
                    "label": cls["class_name"],
                    "confidence": cls["confidence"]
                })

        elif tool == "image_segmentation":
            # 使用AI服务进行分割
            try:
                segments = ai_service.segment_objects(temp_file_path, model_name=model)
                
                # 转换为前端需要的格式
                for seg in segments:
                    class_name = seg["class_name"]
                    if category_list and class_name not in category_list:
                        continue
                        
                    annotations.append({
                        "type": "polygon",
                        "label": class_name,
                        "confidence": seg["confidence"],
                        "points": seg["points"]
                    })
            except ValueError:
                raise
            except Exception as seg_error:
                print(f"分割处理错误: {str(seg_error)}")
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"图像分割处理错误: {str(seg_error)}")
        else:
            raise HTTPException(status_code=400, detail=f"不支持的标注工具类型: {tool}")
        
        # 保存标注结果到数据库
        db_annotation = VisioFirmAnnotation(
            filename=image.filename,
            tool_type=tool,
            model=model,
            annotation_data=annotations,
            created_at=datetime.utcnow()
        )
        db.add(db_annotation)
        db.commit()
        
        # 删除临时文件
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        # 返回标注结果
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"annotate耗时: {elapsed_ms:.1f} ms, tool={tool}, model={model}, count={len(annotations)}")
        return JSONResponse(content={
            "success": True,
            "annotations": annotations,
            "message": f"使用VisioFirm AI成功标注了{len(annotations)}个对象"
        })
        
    except ValueError as e:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # 确保临时文件被删除
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        # 打印详细错误信息
        import traceback
        print(f"标注过程中出错: {str(e)}")
        traceback.print_exc()
        
        # 返回错误信息
        raise HTTPException(status_code=500, detail=f"标注过程中出错: {str(e)}")

# ---------- 内置模型 ID 与任务映射（供前端展示与筛选）----------
BUILTIN_MODELS = [
    {"id": "YOLO", "name": "YOLOv8-nano (内置)", "task": "detection", "source": "builtin", "isLocal": True, "description": "轻量级目标检测", "accuracy": "37.3% mAP", "speed": "极快"},
    {"id": "FasterRCNN", "name": "Faster R-CNN (内置)", "task": "detection", "source": "builtin", "isLocal": True, "description": "高精度目标检测", "accuracy": "42.1% mAP", "speed": "慢速"},
    {"id": "SSD", "name": "SSD (内置)", "task": "detection", "source": "builtin", "isLocal": True, "description": "SSD 目标检测", "accuracy": "中", "speed": "中等"},
    {"id": "ResNet", "name": "ResNet50 (内置)", "task": "classification", "source": "builtin", "isLocal": True, "description": "通用图像分类", "accuracy": "76.2%", "speed": "快速"},
    {"id": "EfficientNet", "name": "EfficientNet (内置)", "task": "classification", "source": "builtin", "isLocal": True, "description": "高效图像分类", "accuracy": "84.5%", "speed": "中等"},
    {"id": "YOLO-Seg", "name": "YOLOv8-Seg (内置)", "task": "segmentation", "source": "builtin", "isLocal": True, "description": "实例分割", "accuracy": "中高", "speed": "快速"},
    {"id": "MaskRCNN", "name": "Mask R-CNN (内置)", "task": "segmentation", "source": "builtin", "isLocal": True, "description": "实例分割", "accuracy": "中高", "speed": "中等"},
    {"id": "SAM", "name": "SAM 分割一切 (内置)", "task": "segmentation", "source": "builtin", "isLocal": True, "description": "需本地权重", "accuracy": "高", "speed": "中等"},
]


@router.get("/models")
async def get_available_models():
    """获取可用模型列表：内置 + 已下载目录模型 + 用户上传模型"""
    result = []
    # 1) 内置
    for m in BUILTIN_MODELS:
        result.append({**m})
    # 2) 目录模型（带 is_local 状态）
    catalog = model_registry.load_catalog()
    for e in catalog:
        is_local = model_registry.is_catalog_model_local(e["id"])
        result.append({
            "id": e["id"],
            "name": e.get("name", e["id"]),
            "task": e.get("task", "detection"),
            "source": "catalog",
            "isLocal": is_local,
            "description": e.get("description", ""),
            "accuracy": e.get("accuracy", ""),
            "speed": e.get("speed", ""),
            "size": e.get("size", ""),
        })
    # 3) 用户上传
    for e in model_registry.load_user_metadata():
        result.append({
            "id": e["id"],
            "name": e.get("name", e["id"]),
            "task": e.get("task", "detection"),
            "source": "user",
            "isLocal": True,
            "description": "用户上传模型",
        })
    return JSONResponse(content=result)


@router.get("/models/catalog")
async def get_models_catalog():
    """获取可下载模型目录（含是否已下载到本地）"""
    catalog = model_registry.load_catalog()
    out = []
    for e in catalog:
        out.append({
            **e,
            "isLocal": model_registry.is_catalog_model_local(e["id"]),
        })
    return JSONResponse(content=out)


@router.post("/models/upload")
async def upload_model(
    file: UploadFile = File(...),
    name: str = Form(...),
    task: str = Form(...),
):
    """上传用户自定义模型，导入后可直接用于标注。支持 .pt / .pth / .onnx"""
    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="请填写模型名称")
    task = (task or "detection").strip().lower()
    if task not in ("detection", "classification", "segmentation"):
        raise HTTPException(status_code=400, detail="task 须为 detection / classification / segmentation")
    ext = Path(file.filename or "").suffix.lower()
    if ext not in model_registry.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"仅支持格式: {', '.join(model_registry.ALLOWED_EXTENSIONS)}")
    content = await file.read()
    if len(content) > model_registry.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="模型文件不得超过 500MB")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        entry = model_registry.add_user_model(tmp_path, name.strip(), task)
        return JSONResponse(content={"success": True, "model": entry})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


@router.post("/models/download")
async def download_model_to_local(model_id: str = Form(...)):
    """将目录中的模型下载到服务器本地。成功后 is_local 为 true。"""
    if not model_id or not model_id.strip():
        raise HTTPException(status_code=400, detail="请提供 model_id")
    model_id = model_id.strip()
    try:
        path = model_registry.download_catalog_model(model_id)
        # 预热：下载后提前加载到内存，降低首次标注时延
        try:
            entry = next((e for e in model_registry.load_catalog() if e.get("id") == model_id), None)
            task = (entry or {}).get("task", "detection")
            if task == "segmentation":
                ai_service._get_segmentation_model(model_id)
            elif task == "classification":
                # 当前分类目录模型仍走统一分类逻辑，这里暂不强制预热
                pass
            else:
                ai_service._get_detection_model(model_id)
        except Exception as warmup_err:
            # 预热失败不影响下载成功结果，只记录日志
            print(f"模型预热失败: model_id={model_id}, err={warmup_err}")
        return JSONResponse(content={
            "success": True,
            "path": str(path),
            "model_id": model_id,
            "isLocal": True,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        msg = str(e) or "下载失败"
        raise HTTPException(status_code=500, detail=f"下载失败: {msg}")


@router.get("/models/{model_id}/file")
async def get_model_file(model_id: str):
    """返回已下载的模型文件流，供前端「另存为」到用户选择的路径。"""
    if not model_id or not model_id.strip():
        raise HTTPException(status_code=400, detail="请提供 model_id")
    model_id = model_id.strip()
    path = model_registry.get_downloaded_model_path(model_id)
    if not path:
        raise HTTPException(
            status_code=404,
            detail="该模型尚未下载到服务器，请先点击「下载到本地」将模型下载到服务器后再保存到电脑。",
        )
    if not path.exists():
        raise HTTPException(status_code=404, detail="模型文件不存在或已被删除。")
    return FileResponse(
        path=str(path),
        filename=path.name,
        media_type="application/octet-stream",
    )
