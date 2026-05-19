from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List

from database import get_db
from .models import SystemSetting
import subprocess
import platform

router = APIRouter(prefix="/api/settings", tags=["settings"])

class SettingItem(BaseModel):
    key: str
    value: str
    description: Optional[str] = None

    class Config:
        orm_mode = True

@router.get("/select-path")
async def select_path_dialog():
    """Open a system dialog to select a directory and return the path"""
    try:
        if platform.system() == "Windows":
            # PowerShell command to open FolderBrowserDialog
            cmd = [
                "powershell",
                "-Command",
                "Add-Type -AssemblyName System.Windows.Forms; $f = New-Object System.Windows.Forms.FolderBrowserDialog; $f.ShowDialog() | Out-Null; $f.SelectedPath"
            ]
            # Use specific encoding for Windows to handle potential non-ASCII chars
            result = subprocess.run(cmd, capture_output=True, text=True)
            path = result.stdout.strip()
            return {"path": path}
        else:
            return {"path": "", "error": "Only supported on Windows"}
    except Exception as e:
        print(f"Error opening dialog: {e}")
        return {"path": "", "error": str(e)}

@router.get("/", response_model=List[SettingItem])
async def get_settings(db: Session = Depends(get_db)):
    settings = db.query(SystemSetting).all()
    # Ensure default training path exists if not in DB
    keys = [s.key for s in settings]
    if "training_output_path" not in keys:
        default_setting = SystemSetting(
            key="training_output_path", 
            value="runs/train", 
            description="模型训练结果输出路径"
        )
        db.add(default_setting)
        db.commit()
        db.refresh(default_setting)
        settings.append(default_setting)
    
    if "evaluation_save_path" not in keys:
        default_eval_setting = SystemSetting(
            key="evaluation_save_path", 
            value="runs/evaluation", 
            description="模型评估结果保存路径"
        )
        db.add(default_eval_setting)
        db.commit()
        db.refresh(default_eval_setting)
        settings.append(default_eval_setting)
    return settings

@router.post("/", response_model=SettingItem)
async def update_setting(item: SettingItem, db: Session = Depends(get_db)):
    setting = db.query(SystemSetting).filter(SystemSetting.key == item.key).first()
    if setting:
        setting.value = item.value
        if item.description:
            setting.description = item.description
    else:
        setting = SystemSetting(key=item.key, value=item.value, description=item.description)
        db.add(setting)
    
    db.commit()
    db.refresh(setting)
    return setting

@router.get("/{key}", response_model=SettingItem)
async def get_setting(key: str, db: Session = Depends(get_db)):
    setting = db.query(SystemSetting).filter(SystemSetting.key == key).first()
    if not setting:
        if key == "training_output_path":
            return SettingItem(key="training_output_path", value="runs/train", description="模型训练结果输出路径")
        raise HTTPException(status_code=404, detail="Setting not found")
    return setting
