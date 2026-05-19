from fastapi import APIRouter, HTTPException, Depends, File, UploadFile
from typing import List, Optional
import shutil
import os
import uuid
import asyncio
from pathlib import Path
import sys
import threading
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import get_db
from settings.models import SystemSetting

from .training_service import (
    training_service, 
    TrainingConfig, 
    TrainingType, 
    TrainingStatus
)

class TrainingConfigRequest(BaseModel):
    """训练配置请求模型"""
    task: str = "detect"
    model_type: str = "s"
    data_path: str = ""
    epochs: int = 50
    imgsz: int = 640
    batch: int = 8
    project: str = "runs/train"
    name: Optional[str] = None
    project_id: Optional[str] = None
    resume_weights: Optional[str] = None
    patience: int = 15
    use_freeze_strategy: bool = True
    min_epochs_per_stage: int = 15

def apply_system_settings(config: TrainingConfigRequest, db: Session):
    try:
        setting = db.query(SystemSetting).filter(SystemSetting.key == "training_output_path").first()
        if setting and setting.value:
            # If current project path is default, use system setting
            if config.project == "runs/train":
                config.project = setting.value
    except Exception as e:
        print(f"Error applying system settings: {e}")

def apply_project_settings(config: TrainingConfigRequest):
    if not config.project_id:
        return
    import re
    if not re.match(r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$", config.project_id, re.I):
        raise HTTPException(status_code=400, detail="Invalid project_id")
    projects_root = os.path.abspath(os.getenv("PROJECTS_ROOT", "projects"))
    project_dir = os.path.join(projects_root, config.project_id)
    meta_path = os.path.join(project_dir, "meta.json")
    training_dir = os.path.join(project_dir, "training")
    os.makedirs(training_dir, exist_ok=True)
    config.project = training_dir
    if (not config.data_path) and os.path.exists(meta_path):
        try:
            import json
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            dataset_yaml = meta.get("dataset_yaml_path")
            if dataset_yaml and os.path.exists(dataset_yaml):
                config.data_path = dataset_yaml
        except Exception:
            pass

router = APIRouter(prefix="/api/training", tags=["training"])


class IncrementalTrainingRequest(TrainingConfigRequest):
    """增量训练请求模型"""
    base_model_path: str
    new_classes: Optional[List[str]] = None
    old_data_path: Optional[str] = None
    replay_ratio: float = 0.2


class DistillationTrainingRequest(TrainingConfigRequest):
    """蒸馏训练请求模型"""
    teacher_model_path: str
    base_model_path: Optional[str] = None # 学生模型路径，可选，默认同teacher
    
    # 蒸馏参数
    distill_temperature: float = 2.0
    distill_cls_weight: float = 1.0
    distill_reg_weight: float = 2.0
    distill_feat_weight: float = 5.0
    distill_mask_weight: float = 1.0
    distill_bg_weight: float = 0.05
    
    # 伪标签与一致性
    pseudo_conf_threshold: float = 0.7
    enable_consistency: bool = False
    consistency_weight: float = 1.0
    
    # 旧样本回放
    old_data_path: Optional[str] = None
    replay_ratio: float = 0.3
    replay_distill_boost: float = 1.0
    max_replay_samples: int = 1000


class TrainingTaskResponse(BaseModel):
    """训练任务响应模型"""
    task_id: str
    training_type: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: float
    current_epoch: int
    total_epochs: int
    error_message: Optional[str] = None
    result_path: Optional[str] = None
    metrics: dict = {}
    config: dict = {}


@router.post("/regular", response_model=dict)
async def start_regular_training(config: TrainingConfigRequest, db: Session = Depends(get_db)):
    """启动常规训练"""
    try:
        apply_system_settings(config, db)
        apply_project_settings(config)
        training_config = TrainingConfig(**config.model_dump())
        task_id = training_service.create_task(TrainingType.REGULAR, training_config)
        
        if training_service.start_task(task_id):
            return {"task_id": task_id, "message": "训练任务已启动"}
        else:
            raise HTTPException(status_code=400, detail="启动训练任务失败")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建训练任务失败: {str(e)}")


@router.post("/incremental", response_model=dict)
async def start_incremental_training(config: IncrementalTrainingRequest, db: Session = Depends(get_db)):
    """启动增量训练"""
    try:
        apply_system_settings(config, db)
        apply_project_settings(config)
        training_config = TrainingConfig(**config.dict())
        task_id = training_service.create_task(TrainingType.INCREMENTAL, training_config)
        
        if training_service.start_task(task_id):
            return {"task_id": task_id, "message": "增量训练任务已启动"}
        else:
            raise HTTPException(status_code=400, detail="启动增量训练任务失败")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建增量训练任务失败: {str(e)}")


@router.post("/freeze-strategy", response_model=dict)
async def start_freeze_strategy_training(config: TrainingConfigRequest, db: Session = Depends(get_db)):
    """启动冻结策略训练"""
    try:
        apply_system_settings(config, db)
        apply_project_settings(config)
        training_config = TrainingConfig(**config.dict())
        task_id = training_service.create_task(TrainingType.FREEZE_STRATEGY, training_config)
        
        if training_service.start_task(task_id):
            return {"task_id": task_id, "message": "冻结策略训练任务已启动"}
        else:
            raise HTTPException(status_code=400, detail="启动冻结策略训练任务失败")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建冻结策略训练任务失败: {str(e)}")


@router.post("/distillation", response_model=dict)
async def start_distillation_training(config: DistillationTrainingRequest, db: Session = Depends(get_db)):
    """启动蒸馏训练"""
    try:
        apply_system_settings(config, db)
        apply_project_settings(config)
        training_config = TrainingConfig(**config.dict())
        task_id = training_service.create_task(TrainingType.DISTILLATION, training_config)
        
        if training_service.start_task(task_id):
            return {"task_id": task_id, "message": "蒸馏训练任务已启动"}
        else:
            raise HTTPException(status_code=400, detail="启动蒸馏训练任务失败")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建蒸馏训练任务失败: {str(e)}")


@router.get("/tasks", response_model=List[TrainingTaskResponse])
async def get_all_tasks():
    """获取所有训练任务"""
    tasks = training_service.get_all_tasks()
    return [TrainingTaskResponse(**task.to_dict()) for task in tasks]


@router.get("/tasks/{task_id}", response_model=TrainingTaskResponse)
async def get_task(task_id: str):
    """获取指定训练任务"""
    task = training_service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    return TrainingTaskResponse(**task.to_dict())


@router.get("/tasks/{task_id}/logs")
async def get_task_logs(task_id: str):
    """获取训练任务日志"""
    task = training_service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    return {"logs": task.logs}


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    """取消训练任务"""
    if training_service.cancel_task(task_id):
        return {"message": "训练任务已取消"}
    else:
        raise HTTPException(status_code=400, detail="无法取消训练任务")


@router.get("/status")
async def get_training_status():
    """获取训练服务状态"""
    tasks = training_service.get_all_tasks()
    running_count = sum(1 for task in tasks if task.status == TrainingStatus.RUNNING)
    completed_count = sum(1 for task in tasks if task.status == TrainingStatus.COMPLETED)
    failed_count = sum(1 for task in tasks if task.status == TrainingStatus.FAILED)
    
    return {
        "total_tasks": len(tasks),
        "running_tasks": running_count,
        "completed_tasks": completed_count,
        "failed_tasks": failed_count,
        "service_status": "running"
    }


@router.post("/upload/config")
async def upload_training_config(file: UploadFile = File(...)):
    """上传训练配置文件"""
    try:
        # 确保uploads目录存在
        upload_dir = os.path.abspath("uploads/configs")
        os.makedirs(upload_dir, exist_ok=True)
        
        # 生成唯一文件名
        file_ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(upload_dir, unique_filename)
        
        # 保存文件
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {"path": file_path, "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传配置文件失败: {str(e)}")


@router.get("/server-files")
async def list_server_files(file_type: str = "yaml", root: str = "/root/autodl-tmp", limit: int = 200):
    """列出服务器上的数据集配置或模型文件（用于远程部署环境选择文件路径）"""
    if file_type not in {"yaml", "model"}:
        raise HTTPException(status_code=400, detail="file_type must be 'yaml' or 'model'")
    try:
        root_path = Path(root).expanduser().resolve()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid root path")

    allowed_roots = [Path("/root/autodl-tmp").resolve()]
    if not any(str(root_path).startswith(str(ar)) for ar in allowed_roots):
        raise HTTPException(status_code=403, detail="root path not allowed")

    if not root_path.exists() or not root_path.is_dir():
        return {"root": str(root_path), "files": []}

    patterns = ["*.yaml", "*.yml"] if file_type == "yaml" else ["*.pt", "*.pth", "*.onnx"]
    files = []
    scanned = 0
    max_scan = 20000

    for dirpath, _, filenames in os.walk(str(root_path)):
        for filename in filenames:
            scanned += 1
            if scanned > max_scan:
                break
            if not any(Path(filename).match(p) for p in patterns):
                continue
            full_path = Path(dirpath) / filename
            try:
                stat = full_path.stat()
            except Exception:
                continue
            files.append(
                {
                    "path": str(full_path),
                    "name": filename,
                    "mtime": int(stat.st_mtime),
                    "size": int(stat.st_size),
                }
            )
        if scanned > max_scan:
            break

    files.sort(key=lambda x: x["mtime"], reverse=True)
    return {"root": str(root_path), "files": files[: max(0, min(int(limit), 2000))]}


@router.get("/select-local-file")
async def select_local_file(file_type: str = "yaml"):
    """
    打开本地文件选择对话框 (仅限本地部署使用)
    file_type: 'yaml' (数据集配置) 或 'model' (模型权重)
    """
    def open_dialog():
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            if file_type == 'model':
                title = "选择模型权重文件"
                filetypes = [("Model files", "*.pt;*.pth;*.onnx"), ("All files", "*.*")]
            else:
                title = "选择数据集配置文件"
                filetypes = [("YAML files", "*.yaml;*.yml"), ("All files", "*.*")]

            file_path = filedialog.askopenfilename(
                title=title,
                filetypes=filetypes
            )
            root.destroy()
            return file_path
        except Exception as e:
            print(f"Error opening dialog: {e}")
            return ""

    if threading.current_thread() is not threading.main_thread():
        raise HTTPException(
            status_code=400,
            detail="file dialog must run on main thread (macOS limitation). Please input path or use /api/training/server-files",
        )

    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        raise HTTPException(
            status_code=400,
            detail="file dialog is not available in headless environment. Please input path or use /api/training/server-files",
        )

    path = open_dialog()
    if not path:
        raise HTTPException(
            status_code=400,
            detail="file dialog is not available in this environment, please input path or use /api/training/server-files",
        )
    return {"path": path}
