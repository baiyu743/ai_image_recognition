from sqlalchemy import Column, Integer, String, DateTime, JSON, Text, Float
from database import Base
from datetime import datetime

class TrainingTaskDB(Base):
    """训练任务数据库模型"""
    __tablename__ = "training_tasks"
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String, unique=True, index=True)
    training_type = Column(String)  # regular, incremental, freeze_strategy
    status = Column(String)  # pending, running, completed, failed, cancelled
    config = Column(JSON)  # 训练配置
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    progress = Column(Float, default=0.0)
    current_epoch = Column(Integer, default=0)
    total_epochs = Column(Integer, default=0)
    logs = Column(Text, nullable=True)  # JSON格式的日志列表
    error_message = Column(Text, nullable=True)
    result_path = Column(String, nullable=True)
    metrics = Column(JSON, nullable=True)


class TrainingLogDB(Base):
    """训练日志数据库模型"""
    __tablename__ = "training_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    level = Column(String, default="INFO")  # INFO, WARNING, ERROR
    message = Column(Text)
    epoch = Column(Integer, nullable=True)
    metrics = Column(JSON, nullable=True)