"""
根据参数字典对图像进行增广（OpenCV + NumPy），返回新图像字节。
"""
import cv2
import numpy as np
from typing import Dict, Any


def _truthy(v: Any) -> bool:
    """将模型可能返回的 true/\"true\"/1 等统一视为 True。"""
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "是")
    return bool(v)


def apply_augmentation(image_bytes: bytes, params: Dict[str, Any]) -> bytes:
    """
    对图像字节应用增广，返回 JPEG 字节。
    params 来自 llm.parse_instruction_to_params，键为 brightness, contrast, noise_std 等。
    """
    if not params:
        return image_bytes

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return image_bytes
    img = _apply_one(img, params)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def _apply_one(img: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
    """顺序应用各项增广。"""
    # 亮度
    if "brightness" in p and p["brightness"] != 1.0:
        b = float(p["brightness"])
        b = max(0.5, min(2.0, b))
        img = cv2.convertScaleAbs(img, alpha=b, beta=0)

    # 对比度（简化：用 alpha）
    if "contrast" in p and p["contrast"] != 1.0:
        c = float(p["contrast"])
        c = max(0.5, min(2.0, c))
        img = cv2.convertScaleAbs(img, alpha=c, beta=128 * (1 - c))

    # 饱和度（转 HSV 再乘 S 通道）
    if "saturation" in p and p["saturation"] != 1.0:
        s = float(p["saturation"])
        s = max(0.0, min(2.0, s))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 高斯噪声
    if "noise_std" in p and float(p["noise_std"]) > 0:
        std = float(p["noise_std"])
        std = max(0, min(0.08, std))
        noise = np.random.randn(*img.shape).astype(np.float32) * (std * 255)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 旋转
    if "rotate" in p and float(p["rotate"]) != 0:
        angle = float(p["rotate"])
        angle = max(-180, min(180, angle))
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 水平翻转（兼容模型返回 true / "true" / 1）
    if _truthy(p.get("flip_horizontal")):
        img = cv2.flip(img, 1)

    # 垂直翻转
    if _truthy(p.get("flip_vertical")):
        img = cv2.flip(img, 0)

    # 高斯模糊
    if "blur" in p and int(p["blur"]) > 0:
        k = int(p["blur"])
        k = max(1, min(9, k))
        if k % 2 == 0:
            k += 1
        img = cv2.GaussianBlur(img, (k, k), 0)

    return img
