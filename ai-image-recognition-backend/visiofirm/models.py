from sqlalchemy import Column, Integer, String, DateTime, JSON
from datetime import datetime
from database import Base

class VisioFirmAnnotation(Base):
    """Vision model iterate hub 标注工具的标注结果模型"""
    __tablename__ = "visiofirm_annotations"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    tool_type = Column(String)  # object_detection, image_classification, image_segmentation
    model = Column(String, nullable=True)
    annotation_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
