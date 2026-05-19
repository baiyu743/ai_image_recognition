from sqlalchemy import Column, Integer, String, DateTime, JSON, Text, Float, ForeignKey
from database import Base
from datetime import datetime
from training.models import TrainingTaskDB

class EvaluationResultDB(Base):
    """评估结果数据库模型"""
    __tablename__ = "evaluation_results"
    
    id = Column(Integer, primary_key=True, index=True)
    evaluation_id = Column(String, unique=True, index=True)  # 评估ID
    model_id = Column(String, index=True)  # 关联的模型ID
    task_id = Column(String, ForeignKey(TrainingTaskDB.task_id), nullable=True)  # 可选关联的训练任务
    status = Column(String)  # pending, running, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # 总体评估指标
    mAP50 = Column(Float, nullable=True)  # mAP@0.5
    mAP50_95 = Column(Float, nullable=True)  # mAP@0.5:0.95
    precision = Column(Float, nullable=True)  # 总体精确率
    recall = Column(Float, nullable=True)  # 总体召回率
    f1_score = Column(Float, nullable=True)  # F1分数
    
    # 类别评估指标
    class_metrics = Column(JSON, nullable=True)  # 每个类别的详细指标
    
    # PR曲线数据
    pr_curve_data = Column(JSON, nullable=True)  # 用于绘制PR曲线的数据
    
    # 评估配置和日志
    config = Column(JSON)  # 评估配置
    logs = Column(Text, nullable=True)  # 评估日志
    error_message = Column(Text, nullable=True)  # 错误信息
    
    # LLM 分析结果
    llm_analysis = Column(Text, nullable=True)
