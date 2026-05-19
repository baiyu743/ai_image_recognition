import os
import uuid
import asyncio
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import torch
from ultralytics import YOLO

from .enhanced_training import train_model_with_enhanced_freeze, get_model_size_from_type
from .incremental_newtrain import IncrementalTrainer
try:
    from .distillation_trainer import DistillationTrainer
    from .seg_distillation_trainer import SegDistillationTrainer
    from .cls_distillation_trainer import ClsDistillationTrainer
    DISTILLATION_AVAILABLE = True
except ImportError:
    DISTILLATION_AVAILABLE = False
    print("Warning: Distillation trainers could not be imported.")


class TrainingStatus(Enum):
    """训练状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingType(Enum):
    """训练类型枚举"""
    REGULAR = "regular"
    INCREMENTAL = "incremental"
    FREEZE_STRATEGY = "freeze_strategy"
    DISTILLATION = "distillation"


@dataclass
class TrainingConfig:
    """训练配置"""
    task: str = "detect"  # detect, segment, classify
    model_type: str = "s"  # n, s, m, l, x
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
    # 增量训练特有参数
    base_model_path: Optional[str] = None
    new_classes: Optional[List[str]] = None
    old_data_path: Optional[str] = None # 用于增量训练的旧数据
    
    # 蒸馏训练参数
    teacher_model_path: Optional[str] = None
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
    replay_ratio: float = 0.0
    replay_distill_boost: float = 1.0
    max_replay_samples: int = 1000
    
    def to_dict(self):
        return asdict(self)


@dataclass
class TrainingTask:
    """训练任务"""
    task_id: str
    training_type: TrainingType
    config: TrainingConfig
    status: TrainingStatus = TrainingStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 0
    logs: List[str] = None
    error_message: Optional[str] = None
    result_path: Optional[str] = None
    metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.logs is None:
            self.logs = []
        if self.metrics is None:
            self.metrics = {}
    
    def to_dict(self):
        data = asdict(self)
        # 转换枚举为字符串
        data['training_type'] = self.training_type.value
        data['status'] = self.status.value
        # 转换datetime为字符串
        for field in ['created_at', 'started_at', 'completed_at']:
            if data[field]:
                data[field] = data[field].isoformat()
        return data


class TrainingService:
    """训练服务管理类"""
    
    def __init__(self):
        self.tasks: Dict[str, TrainingTask] = {}
        self.running_tasks: Dict[str, threading.Thread] = {}
        
    def create_task(self, training_type: TrainingType, config: TrainingConfig) -> str:
        """创建训练任务"""
        # 使用更易读的格式: Task_类型_YYYYMMDD_HHMMSS
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # 获取类型名称，如 regular, incremental
        type_str = training_type.value if hasattr(training_type, 'value') else str(training_type)
        task_id = f"Task_{type_str}_{timestamp}"
        
        # 防止极端情况下的ID冲突
        if task_id in self.tasks:
            import random
            task_id = f"{task_id}_{random.randint(100, 999)}"
            
        task = TrainingTask(
            task_id=task_id,
            training_type=training_type,
            config=config,
            total_epochs=config.epochs
        )
        if not task.config.name:
            task.config.name = task_id
        self.tasks[task_id] = task
        if task.config.project_id:
            try:
                projects_root = os.path.abspath(os.getenv("PROJECTS_ROOT", "projects"))
                meta_path = os.path.join(projects_root, task.config.project_id, "meta.json")
                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    meta["last_training_task_id"] = task_id
                    meta["last_training_output_dir"] = os.path.join(task.config.project, task.config.name or task_id)
                    meta["updated_at"] = datetime.utcnow().isoformat()
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        return task_id
    
    def get_task(self, task_id: str) -> Optional[TrainingTask]:
        """获取训练任务"""
        return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> List[TrainingTask]:
        """获取所有训练任务"""
        return list(self.tasks.values())
    
    def start_task(self, task_id: str) -> bool:
        """启动训练任务"""
        task = self.tasks.get(task_id)
        if not task or task.status != TrainingStatus.PENDING:
            return False
        
        task.status = TrainingStatus.RUNNING
        task.started_at = datetime.now()
        
        # 在新线程中执行训练
        thread = threading.Thread(target=self._run_training, args=(task,))
        thread.daemon = True
        self.running_tasks[task_id] = thread
        thread.start()
        
        return True
    
    def cancel_task(self, task_id: str) -> bool:
        """取消训练任务"""
        task = self.tasks.get(task_id)
        if not task or task.status not in [TrainingStatus.PENDING, TrainingStatus.RUNNING]:
            return False
        
        task.status = TrainingStatus.CANCELLED
        task.completed_at = datetime.now()
        
        # 如果任务正在运行，尝试停止线程（注意：Python线程无法强制停止）
        if task_id in self.running_tasks:
            del self.running_tasks[task_id]
        
        return True
    
    def _run_training(self, task: TrainingTask):
        """执行训练任务"""
        try:
            self._log_message(task, f"开始训练任务: {task.task_id}")
            try:
                cuda_available = torch.cuda.is_available()
                cuda_version = getattr(torch.version, "cuda", None)
                device_count = torch.cuda.device_count() if cuda_available else 0
                visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
                self._log_message(
                    task,
                    f"运行环境: torch={torch.__version__}, torch.cuda={cuda_version}, cuda_available={cuda_available}, device_count={device_count}, CUDA_VISIBLE_DEVICES={visible_devices}"
                )
            except Exception as e:
                self._log_message(task, f"读取CUDA信息失败: {str(e)}")
            
            if task.training_type == TrainingType.INCREMENTAL:
                result = self._run_incremental_training(task)
            elif task.training_type == TrainingType.FREEZE_STRATEGY:
                result = self._run_freeze_strategy_training(task)
            elif task.training_type == TrainingType.DISTILLATION:
                result = self._run_distillation_training(task)
            else:
                result = self._run_regular_training(task)
            
            if task.status == TrainingStatus.CANCELLED:
                self._log_message(task, "训练已取消")
                return

            if result:
                task.status = TrainingStatus.COMPLETED
                task.result_path = result.get('model_path')
                task.metrics = result.get('metrics', {})
                self._log_message(task, f"训练完成，模型保存至: {task.result_path}")
                
                # 生成评估数据文件
                if task.result_path and os.path.exists(task.result_path):
                    try:
                        self._generate_evaluation_json(task, task.result_path)
                    except Exception as e:
                        self._log_message(task, f"调用生成评估数据文件失败: {str(e)}")
                        import traceback
                        traceback.print_exc()
                else:
                    self._log_message(task, f"无法生成评估数据，因为模型路径不存在或为空: {task.result_path}")
            else:
                if task.status == TrainingStatus.CANCELLED:
                    self._log_message(task, "训练已取消")
                else:
                    task.status = TrainingStatus.FAILED
                    task.error_message = "训练失败，未返回结果"
                
        except Exception as e:
            if task.status == TrainingStatus.CANCELLED:
                self._log_message(task, "训练已取消")
            else:
                task.status = TrainingStatus.FAILED
                task.error_message = str(e)
                self._log_message(task, f"训练失败: {str(e)}")
        
        finally:
            task.completed_at = datetime.now()
            if task.status != TrainingStatus.CANCELLED:
                task.progress = 100.0
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
    
    def _run_incremental_training(self, task: TrainingTask) -> Optional[Dict]:
        """执行增量训练 (使用新的 IncrementalTrainer)"""
        config = task.config
        
        try:
            # 确保旧数据配置存在
            old_data_yaml = config.old_data_path
            if not old_data_yaml:
                 # 如果没有提供，尝试从base_model_path推断或者报错
                 # 这里简单起见，如果没提供就报错
                 raise ValueError("增量训练需要提供旧数据集配置 (old_data_path)")

            trainer = IncrementalTrainer(
                existing_model_path=config.base_model_path,
                old_data_yaml=old_data_yaml,
                new_data_yaml=config.data_path,
                task=config.task
            )
            
            # 分析变化
            trainer.analyze_changes()
            
            # 使用单阶段训练 (为了简单起见，或者可以根据config增加stage选项)
            # 这里我们使用单阶段，并支持 freeze
            trainer.train(
                project=config.project,
                name=config.name or f"inc_{task.task_id}",
                epochs=config.epochs,
                batch=config.batch,
                imgsz=config.imgsz,
                old_data_ratio=config.replay_ratio or 0.2, # 复用 replay_ratio 作为 old_data_ratio
                lr0=1e-4, # 默认值
                lrf=1e-5,
                freeze=0, # 默认不冻结，或者可以加参数
                plots=False,
                device='0' if torch.cuda.is_available() else 'cpu'
            )
            
            # 结果路径
            result_dir = os.path.join(config.project, config.name or f"inc_{task.task_id}")
            best_model_path = os.path.join(result_dir, "weights", "best.pt")
            
            return {
                "model_path": best_model_path,
                "metrics": {} # 需要从训练结果中解析
            }
            
        except Exception as e:
            self._log_message(task, f"增量训练出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _run_distillation_training(self, task: TrainingTask) -> Optional[Dict]:
        """执行蒸馏训练"""
        if not DISTILLATION_AVAILABLE:
            raise ImportError("蒸馏训练模块未安装或导入失败")
            
        config = task.config
        
        try:
            # 选择训练器
            if config.task == 'detect':
                TrainerClass = DistillationTrainer
            elif config.task == 'segment':
                TrainerClass = SegDistillationTrainer
            elif config.task == 'classify':
                TrainerClass = ClsDistillationTrainer
            else:
                raise ValueError(f"不支持的任务类型: {config.task}")
            
            # 加载教师模型
            teacher_yolo = YOLO(config.teacher_model_path)
            teacher_model = teacher_yolo.model
            
            # 加载学生模型 (可以是同一个，或者更小的)
            student_model = YOLO(config.base_model_path or config.teacher_model_path) # 如果没指定base，就用teacher作为起点
            
            # 准备参数
            train_args = {
                'epochs': config.epochs,
                'batch': config.batch,
                'imgsz': config.imgsz,
                'project': config.project,
                'name': config.name or f"distill_{task.task_id}",
                'device': '0' if torch.cuda.is_available() else 'cpu',
                'plots': True,
                
                # 蒸馏参数
                'temperature': config.distill_temperature,
                'distill_cls_weight': config.distill_cls_weight,
                'distill_reg_weight': config.distill_reg_weight,
                'distill_feat_weight': config.distill_feat_weight,
                # 'distill_mask_weight': config.distill_mask_weight, # 只有分割任务需要
                # 'distill_bg_weight': config.distill_bg_weight,   # 只有分类任务需要
                
                # 一致性参数
                'enable_consistency': config.enable_consistency,
                'consistency_weight': config.consistency_weight,
                'pseudo_conf_threshold': config.pseudo_conf_threshold,
                
                # 回放参数
                'replay_ratio': config.replay_ratio,
                'replay_distill_boost': config.replay_distill_boost,
                'max_replay_samples': config.max_replay_samples,
                'old_data_yaml': config.old_data_path,
            }

            if config.task == 'segment':
                train_args['distill_mask_weight'] = config.distill_mask_weight
            
            if config.task == 'classify':
                train_args['distill_bg_weight'] = config.distill_bg_weight
            
            # 启动训练
            student_model.train(
                trainer=TrainerClass,
                teacher_model=teacher_model,
                data=config.data_path,
                **train_args
            )
            
            # 结果路径
            result_dir = os.path.join(config.project, config.name or f"distill_{task.task_id}")
            best_model_path = os.path.join(result_dir, "weights", "best.pt")
            
            return {
                "model_path": best_model_path,
                "metrics": {} 
            }
            
        except Exception as e:
            self._log_message(task, f"蒸馏训练出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _run_freeze_strategy_training(self, task: TrainingTask) -> Optional[Dict]:
        """执行冻结策略训练"""
        config = task.config
        
        def progress_callback(epoch, total_epochs):
            if task.status == TrainingStatus.CANCELLED:
                return
            task.current_epoch = epoch
            task.progress = (epoch / total_epochs) * 100
            self._log_message(task, f"训练进度: {epoch}/{total_epochs} ({task.progress:.1f}%)")
        
        result = train_model_with_enhanced_freeze(
            task=config.task,
            model_type=config.model_type,
            data_path=config.data_path,
            epochs=config.epochs,
            imgsz=config.imgsz,
            batch=config.batch,
            project=config.project,
            name=config.name,
            resume_weights=config.resume_weights,
            patience=config.patience,
            use_freeze_strategy=config.use_freeze_strategy,
            min_epochs_per_stage=config.min_epochs_per_stage,
            progress_callback=progress_callback,
            cancel_callback=lambda: task.status == TrainingStatus.CANCELLED
        )
        
        return result
    
    def _run_regular_training(self, task: TrainingTask) -> Optional[Dict]:
        """执行常规训练"""
        config = task.config
        
        def progress_callback(epoch, total_epochs):
            if task.status == TrainingStatus.CANCELLED:
                return
            task.current_epoch = epoch
            task.progress = (epoch / total_epochs) * 100
            self._log_message(task, f"训练进度: {epoch}/{total_epochs} ({task.progress:.1f}%)")
        
        # 使用冻结策略训练但禁用冻结功能
        result = train_model_with_enhanced_freeze(
            task=config.task,
            model_type=config.model_type,
            data_path=config.data_path,
            epochs=config.epochs,
            imgsz=config.imgsz,
            batch=config.batch,
            project=config.project,
            name=config.name,
            resume_weights=config.resume_weights,
            patience=config.patience,
            use_freeze_strategy=False,  # 禁用冻结策略
            progress_callback=progress_callback,
            cancel_callback=lambda: task.status == TrainingStatus.CANCELLED
        )
        
        return result
    
    def _log_message(self, task: TrainingTask, message: str):
        """记录日志消息"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        task.logs.append(log_entry)
        print(log_entry)  # 同时输出到控制台

    def _generate_evaluation_json(self, task: TrainingTask, model_path: str):
        """生成用于评估优化的JSON文件"""
        try:
            import yaml
            import json
            from pathlib import Path
            
            self._log_message(task, "正在生成评估数据文件...")
            
            # 读取data.yaml获取验证集路径
            data_yaml_path = task.config.data_path
            if not os.path.isabs(data_yaml_path):
                 data_yaml_path = os.path.abspath(data_yaml_path)

            if not os.path.exists(data_yaml_path):
                self._log_message(task, f"data.yaml不存在: {data_yaml_path}")
                return

            with open(data_yaml_path, 'r', encoding='utf-8') as f:
                data_config = yaml.safe_load(f)
            
            val_path = data_config.get('val')
            path_root = data_config.get('path')
            names = data_config.get('names', {})
            
            # 确保names可以像字典一样访问
            if isinstance(names, list):
                names = {i: name for i, name in enumerate(names)}
            
            if not val_path:
                self._log_message(task, "未在data.yaml中找到val路径，跳过生成评估文件")
                return

            # 处理相对路径
            if not os.path.isabs(val_path):
                if path_root:
                    if not os.path.isabs(path_root):
                        # 如果 path 是相对的，假设相对于 data.yaml 目录
                        base_dir = os.path.dirname(data_yaml_path)
                        path_root = os.path.join(base_dir, path_root)
                    val_path = os.path.join(path_root, val_path)
                else:
                    base_dir = os.path.dirname(data_yaml_path)
                    val_path = os.path.join(base_dir, val_path)
            
            self._log_message(task, f"解析后的验证集路径: {val_path}")
            
            default_label_dir = None
            try:
                val_base = Path(val_path) if os.path.isdir(val_path) else Path(val_path).parent
                parts_val = list(val_base.parts)
                parts_val_rev = parts_val[::-1]
                if 'images' in parts_val_rev:
                    idx_rev = parts_val_rev.index('images')
                    idx = len(parts_val) - 1 - idx_rev
                    parts_val[idx] = 'labels'
                    default_label_dir = Path(*parts_val)
            except Exception:
                default_label_dir = None
            
            if default_label_dir:
                self._log_message(task, f"推断的标签目录: {default_label_dir}")

            label_index: Dict[str, str] = {}
            numeric_label_index: Dict[str, str] = {}
            label_dirs_to_scan: List[Path] = []
            try:
                if default_label_dir and default_label_dir.exists() and default_label_dir.is_dir():
                    label_dirs_to_scan.append(default_label_dir)
            except Exception:
                pass
            try:
                if default_label_dir and default_label_dir.name.lower() == "val":
                    p_train = default_label_dir.parent / "train"
                    if p_train.exists() and p_train.is_dir() and p_train not in label_dirs_to_scan:
                        label_dirs_to_scan.append(p_train)
            except Exception:
                pass
            try:
                p_val = Path(val_path)
                if p_val.exists():
                    if p_val.is_dir():
                        p_val_images_dir = p_val
                    else:
                        p_val_images_dir = p_val.parent
                    parts_val = list(p_val_images_dir.parts)
                    parts_val_rev = parts_val[::-1]
                    if 'images' in parts_val_rev:
                        idx_rev = parts_val_rev.index('images')
                        idx = len(parts_val) - 1 - idx_rev
                        parts_val[idx] = 'labels'
                        p_labels_dir = Path(*parts_val)
                        if p_labels_dir.exists() and p_labels_dir.is_dir() and p_labels_dir not in label_dirs_to_scan:
                            label_dirs_to_scan.append(p_labels_dir)
            except Exception:
                pass
            try:
                p_labels_root = Path(path_root) / "labels" if path_root else None
                if p_labels_root and p_labels_root.exists() and p_labels_root.is_dir() and p_labels_root not in label_dirs_to_scan:
                    label_dirs_to_scan.append(p_labels_root)
            except Exception:
                pass

            for d in label_dirs_to_scan:
                try:
                    for p in d.rglob("*.txt") if d.is_dir() else []:
                        if not p.is_file():
                            continue
                        stem = p.stem
                        label_index[stem] = str(p)
                        try:
                            import re
                            m = re.match(r"^(?:image|img)?(\d+)$", stem, re.I)
                            if m:
                                key = m.group(1)
                                if key not in numeric_label_index or len(stem) < len(Path(numeric_label_index[key]).stem):
                                    numeric_label_index[key] = str(p)
                        except Exception:
                            pass
                except Exception:
                    continue

            # 收集图片文件
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            image_files = []
            if os.path.isdir(val_path):
                for root, _, files in os.walk(val_path):
                    for file in files:
                        if os.path.splitext(file)[1].lower() in image_extensions:
                            image_files.append(os.path.join(root, file))
            elif os.path.isfile(val_path) and val_path.endswith('.txt'):
                 with open(val_path, 'r') as f:
                     for line in f:
                         line = line.strip()
                         if line:
                            if not os.path.isabs(line):
                                line = os.path.join(os.path.dirname(val_path), line)
                            image_files.append(line)
            
            if not image_files:
                self._log_message(task, f"未找到验证集图片: {val_path}")
                return

            self._log_message(task, f"找到 {len(image_files)} 张验证图片，开始预测...")

            # 加载模型
            model = YOLO(model_path)
            
            predictions = []
            ground_truths = []
            missing_label_images: List[str] = []
            found_label_images = 0
            
            # 运行预测
            # 使用stream=True以节省内存
            # conf=0.001 是常用的评估阈值
            results = model.predict(source=image_files, stream=True, conf=0.001, verbose=False)
            
            for src_path, result in zip(image_files, results):
                # 关键修复：某些情况下 result.path 会变成 image0.jpg/image1.jpg 等“合成路径”
                # 这会导致后续找不到 labels，从而 ground_truths=0
                img_path = src_path if isinstance(src_path, str) and src_path else getattr(result, "path", src_path)
                image_id = os.path.basename(str(img_path))
                
                # 预测框
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    # 关键修复：清洗类别名称，去除置信度后缀 (例如 "car (0.85)" -> "car")
                    raw_name = names.get(cls_id, str(cls_id))
                    import re
                    match = re.match(r"^(.+)\s\(\d+(\.\d+)?\)$", raw_name)
                    class_name = match.group(1) if match else raw_name
                    
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist() # [x1, y1, x2, y2]
                    
                    predictions.append({
                        "image_id": image_id,
                        "class_name": class_name,
                        "box": xyxy,
                        "confidence": conf
                    })
                
                # 处理真实标签 (Ground Truth)
                # 假设标准YOLO格式: 
                # /path/to/images/val/img1.jpg -> /path/to/labels/val/img1.txt
                # 或者 /path/to/val/img1.jpg -> /path/to/val/img1.txt (if images/labels not used)
                
                label_path = None
                p_img = Path(img_path)
                
                # 尝试策略1: 替换 images -> labels
                parts = list(p_img.parts)
                # 策略1: 替换 images -> labels
                # 您的数据结构: dataset/images/val/xx.jpg -> dataset/labels/val/xx.txt
                try:
                    parts_reversed = parts[::-1]
                    if 'images' in parts_reversed:
                        # 找到 'images' 的索引 (倒序)
                        idx_reversed = parts_reversed.index('images')
                        idx = len(parts) - 1 - idx_reversed
                        
                        new_parts = list(parts)
                        new_parts[idx] = 'labels'
                        p_label = Path(*new_parts).with_suffix('.txt')
                        
                        if p_label.exists():
                            label_path = str(p_label)
                        else:
                            # 尝试处理 windows 路径分隔符问题或大小写问题
                            # 如果 p_label 不存在，尝试递归查找？不，太慢。
                            # 打印一下调试信息 (仅在找不到时)
                            pass
                except:
                    pass

                # 策略3: 您的特殊结构 (dataset/images/val -> dataset/labels/train)
                # 某些情况下，验证集图片对应的标签可能被错误地放在了 labels/train 下？
                # 或者 dataset.yaml 中 val 指向了 images/val，但标签都在 labels/train？
                if not label_path:
                    try:
                        if 'images' in parts_reversed:
                            idx_reversed = parts_reversed.index('images')
                            idx = len(parts) - 1 - idx_reversed
                            new_parts = list(parts)
                            new_parts[idx] = 'labels'
                            # 强制将 val 替换为 train (针对只有 train 标签的情况)
                            if 'val' in new_parts:
                                val_idx = new_parts.index('val')
                                new_parts[val_idx] = 'train'
                            
                            p_label = Path(*new_parts).with_suffix('.txt')
                            if p_label.exists():
                                label_path = str(p_label)
                    except:
                        pass

                # 策略2: 同目录 txt
                if not label_path:
                    p_label = p_img.with_suffix('.txt')
                    if p_label.exists():
                        label_path = str(p_label)

                if not label_path:
                    try:
                        import re
                        stem = p_img.stem
                        candidates = []
                        m = re.match(r"^image(\d+)$", stem, re.I)
                        if m:
                            candidates.append(m.group(1))
                        m = re.match(r"^img(\d+)$", stem, re.I)
                        if m:
                            candidates.append(m.group(1))
                        if candidates:
                            label_dir = None
                            try:
                                if 'images' in parts_reversed:
                                    idx_reversed = parts_reversed.index('images')
                                    idx = len(parts) - 1 - idx_reversed
                                    new_parts = list(parts)
                                    new_parts[idx] = 'labels'
                                    label_dir = Path(*new_parts).parent
                            except Exception:
                                label_dir = None
                            if not label_dir:
                                label_dir = p_img.parent
                            label_dirs = [label_dir]
                            if default_label_dir and default_label_dir not in label_dirs:
                                label_dirs.append(default_label_dir)
                            for cand in candidates:
                                for d in label_dirs:
                                    p_try = (d / f"{cand}.txt")
                                    if p_try.exists():
                                        label_path = str(p_try)
                                        break
                                if label_path:
                                    break
                    except Exception:
                        pass

                if not label_path and default_label_dir:
                    try:
                        p_try = default_label_dir / Path(image_id).with_suffix(".txt").name
                        if p_try.exists():
                            label_path = str(p_try)
                    except Exception:
                        pass

                if not label_path and default_label_dir:
                    try:
                        p_try = default_label_dir / f"{p_img.stem}.txt"
                        if p_try.exists():
                            label_path = str(p_try)
                    except Exception:
                        pass
                
                if not label_path:
                    try:
                        p_from_index = label_index.get(p_img.stem)
                        if p_from_index and os.path.exists(p_from_index):
                            label_path = p_from_index
                    except Exception:
                        pass
                
                if not label_path:
                    try:
                        stem_id = Path(image_id).stem
                        p_from_index = label_index.get(stem_id)
                        if p_from_index and os.path.exists(p_from_index):
                            label_path = p_from_index
                    except Exception:
                        pass

                if not label_path:
                    try:
                        import re
                        m = re.match(r"^(?:image|img)?(\d+)$", p_img.stem, re.I)
                        if m:
                            p_from_index = numeric_label_index.get(m.group(1))
                            if p_from_index and os.path.exists(p_from_index):
                                label_path = p_from_index
                    except Exception:
                        pass
                
                if not label_path:
                    try:
                        import re
                        stem_id = Path(image_id).stem
                        m = re.match(r"^(?:image|img)?(\d+)$", stem_id, re.I)
                        if m:
                            p_from_index = numeric_label_index.get(m.group(1))
                            if p_from_index and os.path.exists(p_from_index):
                                label_path = p_from_index
                    except Exception:
                        pass
                
                if label_path:
                    found_label_images += 1
                    # 读取图片尺寸用于反归一化
                    h, w = result.orig_shape
                    
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cls_id = int(parts[0])
                                # 关键修复：清洗类别名称，去除置信度后缀
                                raw_name = names.get(cls_id, str(cls_id))
                                import re
                                match = re.match(r"^(.+)\s\(\d+(\.\d+)?\)$", raw_name)
                                class_name = match.group(1) if match else raw_name
                                
                                cx, cy, bw, bh = map(float, parts[1:5])
                                
                                # 反归一化
                                x1 = (cx - bw/2) * w
                                y1 = (cy - bh/2) * h
                                x2 = (cx + bw/2) * w
                                y2 = (cy + bh/2) * h
                                
                                ground_truths.append({
                                    "image_id": image_id,
                                    "class_name": class_name,
                                    "box": [x1, y1, x2, y2]
                                })
                else:
                    if len(missing_label_images) < 20:
                        missing_label_images.append(image_id)
            
            # 保存JSON文件
            output_dir = os.path.dirname(model_path)
            # 保存到 weights 的上一级，即 exp 目录
            exp_dir = os.path.dirname(output_dir) 
            json_path = os.path.join(exp_dir, "evaluation_data.json")
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "predictions": predictions,
                    "ground_truths": ground_truths,
                    "model_id": task.config.name or "unknown",
                    "task_id": task.task_id
                }, f, ensure_ascii=False, indent=2)
                
            if found_label_images == 0:
                self._log_message(task, f"警告：未找到任何验证集标签文件，ground_truths=0。val_path={val_path} label_dirs={ [str(d) for d in label_dirs_to_scan] } 示例缺失标签图片: {missing_label_images}")
            elif len(ground_truths) == 0:
                self._log_message(task, f"警告：验证集标签文件存在但内容为空，ground_truths=0。请检查：1) 验证集是否被划分到未标注图片；2) labels/val/*.txt 是否为空；3) 标注是否已导入到项目。")
            else:
                self._log_message(task, f"验证集标签文件命中: {found_label_images} 张，缺失标签: {len(missing_label_images)} 张")
            self._log_message(task, f"已生成评估数据文件: {json_path}")
            
        except Exception as e:
            self._log_message(task, f"生成评估数据文件失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e



# 全局训练服务实例
training_service = TrainingService()
