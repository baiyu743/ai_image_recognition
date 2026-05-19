from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Dict, Optional
import uuid
import datetime
import threading
import json
import os

from .llm_client import llm_client

from .models import EvaluationResultDB
from .metrics import EvaluationMetrics
from training.models import TrainingTaskDB
from database import get_db  # 假设main.py中有获取数据库会话的函数
from .llm_service import generate_llm_analysis
from settings.models import SystemSetting
import shutil

router = APIRouter()

# 数据模型
class Prediction(BaseModel):
    image_id: str
    class_name: str
    box: List[float]  # [x1, y1, x2, y2]
    confidence: float

class GroundTruth(BaseModel):
    image_id: str
    class_name: str
    box: List[float]  # [x1, y1, x2, y2]

class EvaluationRequest(BaseModel):
    model_id: str
    task_id: Optional[str] = None
    project_id: Optional[str] = None
    predictions: List[Prediction]
    ground_truths: List[GroundTruth]
    iou_threshold: float = 0.5

class EvaluationResponse(BaseModel):
    evaluation_id: str
    status: str
    message: str

# 评估服务类
class EvaluationService:
    def __init__(self):
        self.running_evaluations = {}
    
    def start_evaluation(self, db: Session, eval_request: EvaluationRequest) -> str:
        """开始评估任务"""
        # 使用时间戳 + 短随机数生成简短ID (例如: 20231027_143022_a1b2)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:4]
        evaluation_id = f"{timestamp}_{short_uuid}"
        
        # 创建评估记录
        eval_result = EvaluationResultDB(
            evaluation_id=evaluation_id,
            model_id=eval_request.model_id,
            task_id=eval_request.task_id,
            status="pending",
            config={
                "iou_threshold": eval_request.iou_threshold
            }
        )
        
        db.add(eval_result)
        db.commit()
        db.refresh(eval_result)
        
        # 在后台线程中执行评估
        # 注意：这里不应该传递db session到线程中，因为session不是线程安全的
        # 应该在线程内部创建新的session
        thread = threading.Thread(
            target=self._run_evaluation,
            args=(None, evaluation_id, eval_request)
        )
        thread.daemon = True
        thread.start()
        
        self.running_evaluations[evaluation_id] = thread
        
        return evaluation_id
    
    def _run_evaluation(self, db: Session, evaluation_id: str, eval_request: EvaluationRequest):
        """执行评估任务"""
        # 创建新的数据库会话，因为多线程中不应共享会话
        from database import SessionLocal
        db = SessionLocal()
        
        try:
            # 获取评估记录
            eval_result = db.query(EvaluationResultDB).filter(
                EvaluationResultDB.evaluation_id == evaluation_id
            ).first()
            
            if not eval_result:
                print(f"Evaluation {evaluation_id} not found in database")
                return
            
            # 更新状态为运行中
            eval_result.status = "running"
            db.commit()
            print(f"评估任务 {evaluation_id} 正在运行...")
            
            # 转换数据格式
            predictions = [
                {
                    'image_id': pred.image_id,
                    'class': pred.class_name,
                    'box': pred.box,
                    'confidence': pred.confidence
                }
                for pred in eval_request.predictions
            ]
            
            ground_truths = [
                {
                    'image_id': gt.image_id,
                    'class': gt.class_name,
                    'box': gt.box
                }
                for gt in eval_request.ground_truths
            ]
            
            # 1. 计算评估指标 (IOU, Precision, Recall)
            precision, recall = EvaluationMetrics.calculate_precision_recall(
                predictions, ground_truths, eval_request.iou_threshold
            )
            
            # 2. 计算mAP (mAP@0.5 和 mAP@0.5:0.95)
            map50, map50_95, class_maps = EvaluationMetrics.calculate_map(
                predictions, ground_truths
            )
            
            # 3. 计算F1分数
            f1_score = 2 * (precision['overall'] * recall['overall']) / \
                      (precision['overall'] + recall['overall'] + 1e-10)
            
            # 4. 生成PR曲线数据
            pr_curve_data = EvaluationMetrics.generate_pr_curve_data(
                predictions, ground_truths, eval_request.iou_threshold
            )
            
            # 5. 组织分类级详细指标
            class_metrics = {}
            all_classes = set(precision.keys()) - {'overall'}
            
            # 计算每个类别的AP
            for cls in all_classes:
                class_f1 = 2 * (precision[cls] * recall[cls]) / \
                          (precision[cls] + recall[cls] + 1e-10)
                
                class_metrics[cls] = {
                    'precision': precision[cls],
                    'recall': recall[cls],
                    'f1_score': class_f1,
                    'ap': class_maps.get(cls, 0.0)
                }
            
            # 5. 生成 LLM 优化建议
            metrics_summary = {
                'mAP50': map50,
                'mAP50_95': map50_95,
                'precision': precision['overall'],
                'recall': recall['overall'],
                'class_metrics': class_metrics
            }
            llm_analysis = generate_llm_analysis(metrics_summary, eval_request.model_id)

            # 6. 保存评估结果到文件 (读取系统设置)
            # 获取系统设置中的 output_path
            # 注意：这里需要一个新的 Session 或者复用当前的 db
            output_base_path = None
            if eval_request.project_id:
                import re
                if not re.match(r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$", eval_request.project_id, re.I):
                    raise ValueError("Invalid project_id")
                projects_root = os.path.abspath(os.getenv("PROJECTS_ROOT", "projects"))
                output_base_path = os.path.join(projects_root, eval_request.project_id, "evaluation")
                os.makedirs(output_base_path, exist_ok=True)
            if not output_base_path:
                setting = db.query(SystemSetting).filter(SystemSetting.key == "training_output_path").first()
                output_base_path = setting.value if setting else "runs/evaluation"
            
            # 确保是绝对路径或相对于当前工作目录
            if not os.path.isabs(output_base_path):
                output_base_path = os.path.abspath(output_base_path)
            
            # 创建评估结果目录
            eval_output_dir = os.path.join(output_base_path, f"eval_{evaluation_id}")
            os.makedirs(eval_output_dir, exist_ok=True)
            
            # 保存 metrics.json
            metrics_path = os.path.join(eval_output_dir, "metrics.json")
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_summary, f, ensure_ascii=False, indent=2)
            
            # 保存分析报告
            report_path = os.path.join(eval_output_dir, "analysis_report.md")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"# 模型评估报告: {eval_request.model_id}\n\n")
                f.write(llm_analysis)

            # 更新评估结果
            eval_result.status = "completed"
            eval_result.completed_at = datetime.datetime.utcnow()
            eval_result.precision = precision['overall']
            eval_result.recall = recall['overall']
            eval_result.f1_score = f1_score
            eval_result.mAP50 = map50
            eval_result.mAP50_95 = map50_95
            eval_result.class_metrics = class_metrics
            eval_result.pr_curve_data = pr_curve_data
            eval_result.llm_analysis = llm_analysis
            
            # 记录详细日志
            eval_result.logs = json.dumps([
                f"Evaluation completed successfully at {datetime.datetime.utcnow()}",
                f"Results saved to: {eval_output_dir}",
                f"mAP@0.5: {map50:.4f}",
                f"Precision: {precision['overall']:.4f}",
                f"Recall: {recall['overall']:.4f}",
                "LLM report generated"
            ])
            
            db.commit()
            print(f"评估任务 {evaluation_id} 已完成并保存")

            if eval_request.project_id:
                try:
                    projects_root = os.path.abspath(os.getenv("PROJECTS_ROOT", "projects"))
                    meta_path = os.path.join(projects_root, eval_request.project_id, "meta.json")
                    if os.path.exists(meta_path):
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        meta["last_evaluation_id"] = evaluation_id
                        meta["last_evaluation_output_dir"] = eval_output_dir
                        meta["updated_at"] = datetime.datetime.utcnow().isoformat()
                        with open(meta_path, "w", encoding="utf-8") as f:
                            json.dump(meta, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
            
        except Exception as e:
            print(f"Evaluation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 尝试重新获取对象以确保会话有效
            try:
                eval_result = db.query(EvaluationResultDB).filter(
                    EvaluationResultDB.evaluation_id == evaluation_id
                ).first()
                
                if eval_result:
                    eval_result.status = "failed"
                    eval_result.error_message = str(e)
                    eval_result.logs = json.dumps([
                        f"Evaluation failed at {datetime.datetime.utcnow()}",
                        f"Error: {str(e)}"
                    ])
                    db.commit()
            except Exception as inner_e:
                print(f"Failed to update error status: {str(inner_e)}")
                
        finally:
            # 必须关闭会话，释放连接
            db.close()
            if evaluation_id in self.running_evaluations:
                del self.running_evaluations[evaluation_id]
            db.close()

# 创建评估服务
evaluation_service = EvaluationService()

def _normalize_image_id(value: str) -> str:
    v = str(value or "").strip()
    try:
        return os.path.basename(v)
    except Exception:
        return v

def _normalize_xyxy(box: List[float]) -> List[float]:
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("box must be [x1, y1, x2, y2]")
    x1, y1, x2, y2 = [float(x) for x in box]
    x_min = min(x1, x2)
    x_max = max(x1, x2)
    y_min = min(y1, y2)
    y_max = max(y1, y2)
    return [x_min, y_min, x_max, y_max]

# API路由
@router.post("/api/models/{model_id}/evaluate", response_model=EvaluationResponse)
def evaluate_model(
    model_id: str,
    eval_request: EvaluationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """评估模型性能"""
    try:
        eval_request.model_id = model_id

        if not eval_request.predictions:
            raise HTTPException(status_code=400, detail="predictions is empty")
        if not eval_request.ground_truths:
            raise HTTPException(status_code=400, detail="ground_truths is empty (validation labels may be missing)")

        try:
            for p in eval_request.predictions:
                p.image_id = _normalize_image_id(p.image_id)
                p.box = _normalize_xyxy(p.box)
            for gt in eval_request.ground_truths:
                gt.image_id = _normalize_image_id(gt.image_id)
                gt.box = _normalize_xyxy(gt.box)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid evaluation data: {str(e)}")
        
        # 开始评估
        evaluation_id = evaluation_service.start_evaluation(db, eval_request)
        
        return {
            "evaluation_id": evaluation_id,
            "status": "pending",
            "message": "Evaluation started successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/models/{model_id}/evaluation/{evaluation_id}")
def get_evaluation_result(
    model_id: str,
    evaluation_id: str,
    db: Session = Depends(get_db)
):
    """获取评估结果"""
    eval_result = db.query(EvaluationResultDB).filter(
        EvaluationResultDB.evaluation_id == evaluation_id,
        EvaluationResultDB.model_id == model_id
    ).first()
    
    if not eval_result:
        raise HTTPException(status_code=404, detail="Evaluation result not found")
    
    # 生成PR曲线图像
    pr_curve_image = None
    if eval_result.status == "completed" and eval_result.pr_curve_data:
        pr_curve_image = EvaluationMetrics.plot_pr_curve(
            eval_result.pr_curve_data,
            f"PR Curve for Model {model_id}"
        )
    
    return {
        "evaluation_id": eval_result.evaluation_id,
        "model_id": eval_result.model_id,
        "task_id": eval_result.task_id,
        "status": eval_result.status,
        "created_at": eval_result.created_at,
        "completed_at": eval_result.completed_at,
        "metrics": {
            "mAP50": eval_result.mAP50,
            "mAP50_95": eval_result.mAP50_95,
            "precision": eval_result.precision,
            "recall": eval_result.recall,
            "f1_score": eval_result.f1_score,
            "class_metrics": eval_result.class_metrics
        },
        "pr_curve_data": eval_result.pr_curve_data,
        "pr_curve_image": pr_curve_image,
        "logs": eval_result.logs,
        "error_message": eval_result.error_message,
        "llm_analysis": eval_result.llm_analysis
    }

@router.get("/api/models/{model_id}/evaluations")
def get_model_evaluations(
    model_id: str,
    db: Session = Depends(get_db)
):
    """获取模型的所有评估记录"""
    evaluations = db.query(EvaluationResultDB).filter(
        EvaluationResultDB.model_id == model_id
    ).order_by(EvaluationResultDB.created_at.desc()).all()
    
    return [
        {
            "evaluation_id": eval_result.evaluation_id,
            "status": eval_result.status,
            "created_at": eval_result.created_at,
            "completed_at": eval_result.completed_at,
            "metrics": {
                "mAP50": eval_result.mAP50,
                "mAP50_95": eval_result.mAP50_95
            },
            "llm_analysis": eval_result.llm_analysis
        }
        for eval_result in evaluations
    ]
