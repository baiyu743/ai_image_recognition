import os
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from transformers import pipeline
import io
import base64
import time

try:
    from visiofirm import model_registry
except ImportError:
    model_registry = None


class AIModelService:
    def __init__(self):
        # 模型改为按需加载，避免应用启动阶段因为大模型初始化或下载而阻塞。
        self.detection_models = {}
        self.segmentation_models = {}
        self.classification_models = {}
        self.sam_model = None
        # 推理速度优化参数（优先降低时延）
        self.det_imgsz = 640
        self.seg_imgsz = 640
        self.max_det = 200
        # 用户/目录模型懒加载缓存
        self._lazy_detection = {}
        self._lazy_segmentation = {}
        self._lazy_classification = {}
        self.fasterrcnn_classes = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        self._yolo_device = 0 if torch.cuda.is_available() else "cpu"
        self._yolo_half = bool(torch.cuda.is_available())
    
    def _load_models(self):
        """可选预热入口：仅预热本地 YOLO 权重，不再阻塞服务启动。"""
        self._load_builtin_detection_model("YOLO")
        self._load_builtin_segmentation_model("YOLO-Seg")

    def _load_builtin_detection_model(self, model_name):
        if model_name in self.detection_models:
            return self.detection_models[model_name]

        try:
            if model_name == "YOLO":
                self.detection_models["YOLO"] = self._safe_load_yolo('yolov8n.pt')
                print("✅ YOLO 目标检测模型加载成功")
                return self.detection_models["YOLO"]

            if model_name in ("FasterRCNN", "SSD"):
                import torchvision

                if model_name == "FasterRCNN":
                    weights_enum = getattr(torchvision.models.detection, "FasterRCNN_ResNet50_FPN_Weights", None)
                    weights = weights_enum.DEFAULT if weights_enum is not None else None
                    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                        weights=weights if weights_enum is not None else None,
                        pretrained=weights_enum is None,
                    )
                else:
                    weights_enum = getattr(torchvision.models.detection, "SSD300_VGG16_Weights", None)
                    weights = weights_enum.DEFAULT if weights_enum is not None else None
                    model = torchvision.models.detection.ssd300_vgg16(
                        weights=weights if weights_enum is not None else None,
                        pretrained=weights_enum is None,
                    )

                model.eval()
                self.detection_models[model_name] = model
                print(f"✅ {model_name} 目标检测模型加载成功")
                return model
        except Exception as e:
            print(f"❌ 加载内置检测模型失败({model_name}): {e}")
            if model_name in ("FasterRCNN", "SSD"):
                fallback = self._load_builtin_detection_model("YOLO")
                if fallback is not None:
                    self.detection_models[model_name] = fallback
                    print(f"⚠️ 使用 YOLO 作为 {model_name} 的回退模型")
                    return fallback

        return None

    def _load_builtin_segmentation_model(self, model_name):
        if model_name in self.segmentation_models:
            return self.segmentation_models[model_name]

        try:
            if model_name in ("YOLO-Seg", "MaskRCNN"):
                model = self._safe_load_yolo('yolov8n-seg.pt')
                self.segmentation_models[model_name] = model
                print(f"✅ {model_name} 分割模型加载成功")
                return model

            if model_name == "SAM":
                import importlib.util
                if importlib.util.find_spec("segment_anything") is None:
                    print("❌ 未安装segment_anything库，无法加载SAM模型")
                    return None

                from segment_anything import sam_model_registry, SamPredictor

                sam_checkpoint = "sam_vit_h_4b8939.pth"
                model_type = "vit_h"
                if not os.path.exists(sam_checkpoint):
                    print(f"❌ SAM模型文件不存在: {sam_checkpoint}")
                    return None

                print(f"正在加载SAM模型: {sam_checkpoint}")
                sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
                self.sam_model = SamPredictor(sam)
                print("✅ SAM模型加载成功")
                return self.sam_model
        except Exception as e:
            print(f"❌ 加载内置分割模型失败({model_name}): {e}")

        return None

    def _load_builtin_classification_model(self, model_name):
        if model_name in self.classification_models:
            return self.classification_models[model_name]

        try:
            import torchvision.models as models

            if model_name == "ResNet":
                weights_enum = getattr(models, "ResNet50_Weights", None)
                weights = weights_enum.DEFAULT if weights_enum is not None else None
                model = models.resnet50(
                    weights=weights if weights_enum is not None else None,
                    pretrained=weights_enum is None,
                )
            elif model_name == "EfficientNet":
                weights_enum = getattr(models, "EfficientNet_B0_Weights", None)
                weights = weights_enum.DEFAULT if weights_enum is not None else None
                model = models.efficientnet_b0(
                    weights=weights if weights_enum is not None else None,
                    pretrained=weights_enum is None,
                )
            else:
                return None

            model.eval()
            self.classification_models[model_name] = model
            print(f"✅ {model_name} 分类模型加载成功")
            return model
        except Exception as e:
            print(f"❌ 加载内置分类模型失败({model_name}): {e}")
            return None
    
    def detect_objects_with_visualization(self, image_bytes, model_name="YOLO", include_image: bool = False, max_visual_size: int = 1280, jpeg_quality: int = 80):
        """边界框检测，返回 (标注列表, 标注后的图片Base64编码)"""
        model, use_yolo, resolved_name = self._get_detection_model(model_name or "YOLO")
        if model is None or not use_yolo:
            model = self._load_builtin_detection_model("YOLO")
        if model is None:
            return [], None
        
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_w, img_h = image.size
            
            t0 = time.perf_counter()
            def _run_predict(conf: float):
                return model.predict(
                    source=image,
                    imgsz=self.det_imgsz,
                    conf=float(conf),
                    iou=0.7,
                    max_det=self.max_det,
                    device=self._yolo_device,
                    half=self._yolo_half,
                    verbose=False,
                )
            
            conf_primary = 0.25
            conf_fallback = 0.05
            results = _run_predict(conf_primary)
            
            annotations = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 获取边界框坐标
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = box.conf[0].item()
                        cls = int(box.cls[0].item())
                        
                        # 获取类别名称
                        class_name = model.names[cls]

                        # 转换为百分比坐标用于JSON返回
                        x_percent = (x1 / img_w) * 100
                        y_percent = (y1 / img_h) * 100
                        w_percent = ((x2 - x1) / img_w) * 100
                        h_percent = ((y2 - y1) / img_h) * 100
                        
                        annotation = {
                            "from_name": "tag",
                            "to_name": "img",
                            "type": "rectanglelabels",
                            "confidence": float(conf),
                            "value": {
                                "rectanglelabels": [class_name],
                                "x": x_percent,
                                "y": y_percent,
                                "width": w_percent,
                                "height": h_percent
                            }
                        }
                        annotations.append(annotation)
            
            if len(annotations) == 0 and conf_fallback < conf_primary:
                results = _run_predict(conf_fallback)
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            conf = box.conf[0].item()
                            cls = int(box.cls[0].item())
                            class_name = model.names[cls]
                            x_percent = (x1 / img_w) * 100
                            y_percent = (y1 / img_h) * 100
                            w_percent = ((x2 - x1) / img_w) * 100
                            h_percent = ((y2 - y1) / img_h) * 100
                            annotations.append({
                                "from_name": "tag",
                                "to_name": "img",
                                "type": "rectanglelabels",
                                "confidence": float(conf),
                                "value": {
                                    "rectanglelabels": [class_name],
                                    "x": x_percent,
                                    "y": y_percent,
                                    "width": w_percent,
                                    "height": h_percent
                                }
                            })
            
            annotated_image_data_url = None
            if include_image:
                try:
                    plotted = results[0].plot() if results else None
                    if plotted is not None:
                        h, w = plotted.shape[:2]
                        max_visual_size = int(max(256, min(4096, max_visual_size)))
                        jpeg_quality = int(max(40, min(95, jpeg_quality)))
                        scale = min(1.0, float(max_visual_size) / float(max(h, w)))
                        if scale < 1.0:
                            new_w = max(1, int(w * scale))
                            new_h = max(1, int(h * scale))
                            plotted = cv2.resize(plotted, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        ok, buffer = cv2.imencode('.jpg', plotted, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                        if ok:
                            annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')
                            annotated_image_data_url = f"data:image/jpeg;base64,{annotated_image_base64}"
                except Exception:
                    annotated_image_data_url = None

            dt_ms = (time.perf_counter() - t0) * 1000
            print(f"auto_annotate: detections={len(annotations)} include_image={include_image} device={self._yolo_device} time_ms={dt_ms:.1f} model={resolved_name or model_name}")
            return annotations, annotated_image_data_url
            
        except Exception as e:
            print(f"目标检测错误: {e}")
            return [], None
    
    def _get_segmentation_model(self, model_name):
        """解析 model_name：内置 / user:uuid / 目录 id，返回 (model, backend='yolo'|'sam'|None), key。"""
        if model_name and str(model_name).strip():
            mn = str(model_name).strip()
            # 用户上传模型（YOLO 格式）
            if model_registry and mn.startswith("user:"):
                path = model_registry.get_user_model_path(mn)
                if path and path.exists():
                    if mn not in self._lazy_segmentation:
                        self._lazy_segmentation[mn] = self._safe_load_yolo(path)
                    return self._lazy_segmentation[mn], "yolo", mn
            # 目录分割模型：仅当已下载到本地时才使用，不自动下载
            if model_registry and not mn.startswith("user:"):
                path = model_registry.get_downloaded_model_path(mn)
                if mn in self._lazy_segmentation:
                    if not path or not path.exists():
                        del self._lazy_segmentation[mn]
                    else:
                        return self._lazy_segmentation[mn], "yolo", mn
                if path and path.exists():
                    self._lazy_segmentation[mn] = self._safe_load_yolo(path)
                    return self._lazy_segmentation[mn], "yolo", mn
                catalog_ids = [e.get("id") for e in model_registry.load_catalog()]
                if mn in catalog_ids:
                    raise ValueError("该模型未下载到本地，请先在页面选择「下载到本地」后再使用")
            # 内置
            if mn in ("YOLO-Seg", "MaskRCNN", "SAM"):
                self._load_builtin_segmentation_model(mn)
            if mn in self.segmentation_models:
                return self.segmentation_models[mn], "yolo", mn
            if hasattr(self, 'sam_model') and self.sam_model and mn.upper() == "SAM":
                return self.sam_model, "sam", mn
        # 回退
        self._load_builtin_segmentation_model("YOLO-Seg")
        if self.segmentation_models and "YOLO-Seg" in self.segmentation_models:
            return self.segmentation_models["YOLO-Seg"], "yolo", "YOLO-Seg"
        if hasattr(self, 'sam_model') and self.sam_model:
            return self.sam_model, "sam", "SAM"
        return None, None, None

    def segment_objects(self, image_path, model_name=None):
        """多边形分割。支持内置、用户上传、目录已下载模型。"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            height, width = image.shape[:2]
            model, backend, key = self._get_segmentation_model(model_name)
            if model is None:
                return self._generate_mock_segments(width, height)
            if backend == "sam":
                return self._segment_with_sam(image, width, height)
            if backend == "yolo":
                return self._segment_with_yolo(image, width, height, key, model=model)
            return self._generate_mock_segments(width, height)
        except Exception as e:
            print(f"分割错误: {e}")
            return []
            
    def _segment_with_sam(self, image, width, height):
        """使用SAM模型进行分割"""
        try:
            # 转换为RGB格式
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 使用SAM模型生成掩码
            self.sam_model.set_image(image_rgb)
            
            # 自动生成提示点
            # 这里简化处理，使用图像中心点作为提示点
            center_x, center_y = width // 2, height // 2
            input_point = np.array([[center_x, center_y]])
            input_label = np.array([1])  # 1表示前景
            
            # 生成掩码
            masks, scores, _ = self.sam_model.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True
            )
            
            segments = []
            
            # 处理每个掩码
            for i, (mask, score) in enumerate(zip(masks, scores)):
                # 找到掩码的轮廓
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                # 简化轮廓点
                for contour in contours:
                    # 简化轮廓，减少点的数量
                    epsilon = 0.005 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # 转换为百分比坐标
                    points = []
                    for point in approx:
                        x, y = point[0]
                        points.append([(x / width) * 100, (y / height) * 100])
                    
                    # 只添加有足够点的轮廓
                    if len(points) >= 3:
                        segments.append({
                            "class_name": f"Segment_{i+1}",
                            "confidence": float(score),
                            "points": points
                        })
            
            return segments
            
        except Exception as e:
            print(f"SAM分割错误: {e}")
            return []
            
    def _segment_with_yolo(self, image, width, height, model_name="YOLO-Seg", model=None):
        """使用YOLO分割模型进行分割。类别与置信度从 result.boxes 取，轮廓从 result.masks.xy 取。"""
        try:
            if model is None:
                model = self.segmentation_models.get(model_name) or self._lazy_segmentation.get(model_name)
            if model is None:
                return self._generate_mock_segments(width, height)
            t0 = time.perf_counter()
            results = model(
                image,
                imgsz=self.seg_imgsz,
                conf=0.25,
                iou=0.7,
                max_det=self.max_det,
                verbose=False,
            )
            
            segments = []
            for result in results:
                masks = result.masks
                if masks is not None:
                    for i, mask in enumerate(masks):
                        # 获取掩码数据
                        mask_data = mask.data[0].cpu().numpy()
                        
                        # 获取类别和置信度
                        cls = int(mask.cls[0].item())
                        conf = mask.conf[0].item()
                        class_name = model.names[cls]
                        
                        # 找到掩码的轮廓
                        contours, _ = cv2.findContours(
                            (mask_data * 255).astype(np.uint8), 
                            cv2.RETR_EXTERNAL, 
                            cv2.CHAIN_APPROX_SIMPLE
                        )
                        
                        # 简化轮廓点
                        for contour in contours:
                            # 简化轮廓，减少点的数量
                            epsilon = 0.005 * cv2.arcLength(contour, True)
                            approx = cv2.approxPolyDP(contour, epsilon, True)
                            
                            # 转换为百分比坐标
                            points = []
                            for point in approx:
                                x, y = point[0]
                                points.append([(x / width) * 100, (y / height) * 100])
                            
                            # 只添加有足够点的轮廓
                            if len(points) >= 3:
                                segments.append({
                                    "class_name": class_name,
                                    "confidence": float(conf),
                                    "points": points
                                })
            
            dt_ms = (time.perf_counter() - t0) * 1000
            print(f"YOLO分割推理耗时: {dt_ms:.1f} ms, segments={len(segments)}")
            return segments
            
        except Exception as e:
            print(f"YOLO分割错误: {e}")
            return []
    
    def _generate_mock_segments(self, width, height):
        """生成模拟的分割结果"""
        segments = []
        
        # 模拟一个多边形
        points = [
            [width * 0.2, height * 0.2],
            [width * 0.8, height * 0.2],
            [width * 0.8, height * 0.8],
            [width * 0.2, height * 0.8]
        ]
        points_percent = [[(p[0] / width) * 100, (p[1] / height) * 100] for p in points]

        segments.append({
            "class_name": "Object",
            "confidence": 0.95,
            "points": points_percent
        })
        
        # 模拟另一个多边形
        points2 = [
            [width * 0.3, height * 0.3],
            [width * 0.7, height * 0.3],
            [width * 0.7, height * 0.6],
            [width * 0.3, height * 0.6]
        ]
        points2_percent = [[(p[0] / width) * 100, (p[1] / height) * 100] for p in points2]

        segments.append({
            "class_name": "Region",
            "confidence": 0.87,
            "points": points2_percent
        })
        
        return segments
            
    def _safe_load_yolo(self, path):
        """加载 YOLO 模型文件，若文件损坏则抛出明确的 ValueError。"""
        try:
            m = YOLO(str(path))
            try:
                m.fuse()
            except Exception:
                pass
            return m
        except (EOFError, OSError, RuntimeError) as e:
            if isinstance(e, EOFError) or "Ran out of input" in str(e):
                raise ValueError("模型文件已损坏或不完整，请删除后重新下载到本地") from e
            raise

    def _get_classification_model(self, model_name):
        """解析分类模型：内置 / 用户上传 / 已下载目录模型，返回 (model, backend, key)。"""
        if not model_name or not str(model_name).strip():
            model_name = "ResNet"
        model_name = str(model_name).strip()

        if model_registry and model_name.startswith("user:"):
            path = model_registry.get_user_model_path(model_name)
            if path and path.exists():
                if model_name not in self._lazy_classification:
                    self._lazy_classification[model_name] = self._safe_load_yolo(path)
                return self._lazy_classification[model_name], "yolo", model_name

        if model_registry and not model_name.startswith("user:"):
            catalog_entry = next((e for e in model_registry.load_catalog() if e.get("id") == model_name), None)
            is_catalog_classification = (catalog_entry or {}).get("task") == "classification"
            path = model_registry.get_downloaded_model_path(model_name) if is_catalog_classification else None
            if model_name in self._lazy_classification:
                if not path or not path.exists():
                    del self._lazy_classification[model_name]
                else:
                    return self._lazy_classification[model_name], "yolo", model_name
            if path and path.exists():
                self._lazy_classification[model_name] = self._safe_load_yolo(path)
                return self._lazy_classification[model_name], "yolo", model_name
            catalog_ids = [e.get("id") for e in model_registry.load_catalog() if e.get("task") == "classification"]
            if model_name in catalog_ids:
                raise ValueError("该模型未下载到本地，请先在页面选择「下载到本地」后再使用")

        if model_name in ("ResNet", "EfficientNet"):
            model = self.classification_models.get(model_name) or self._load_builtin_classification_model(model_name)
            if model is not None:
                return model, "torchvision", model_name

        fallback_name = "ResNet"
        model = self.classification_models.get(fallback_name) or self._load_builtin_classification_model(fallback_name)
        if model is not None:
            return model, "torchvision", fallback_name
        return None, None, model_name

    def _classify_with_torchvision(self, image_path, model):
        import torchvision.transforms as transforms

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        model.to(device)

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        image = Image.open(image_path).convert('RGB')
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

        with open('imagenet_classes.txt', 'r') as f:
            categories = [s.strip() for s in f.readlines()]

        top3_prob, top3_catid = torch.topk(probabilities, 3)
        classifications = []
        for i in range(top3_prob.size(0)):
            cat_id = int(top3_catid[i])
            classifications.append({
                "class_name": categories[cat_id],
                "confidence": float(top3_prob[i].item())
            })
        return classifications

    def _classify_with_yolo(self, image_path, model):
        results = model(image_path, verbose=False)
        classifications = []

        for result in results:
            probs = getattr(result, "probs", None)
            if probs is None:
                continue

            prob_tensor = getattr(probs, "data", None)
            if prob_tensor is None:
                continue

            if hasattr(probs, "top5"):
                top_indices = list(probs.top5[:3])
            else:
                topk = min(3, int(prob_tensor.shape[0]))
                _, indices = torch.topk(prob_tensor, topk)
                top_indices = [int(idx) for idx in indices.tolist()]

            names = getattr(result, "names", None) or getattr(model, "names", {})
            for idx in top_indices:
                class_name = names.get(int(idx), f"Class_{idx}") if isinstance(names, dict) else names[int(idx)]
                classifications.append({
                    "class_name": class_name,
                    "confidence": float(prob_tensor[int(idx)].item())
                })

            if classifications:
                break

        return classifications

    def _get_detection_model(self, model_name):
        """解析 model_name：内置 / user:uuid / 目录 id，返回 (model, use_yolo=True/False, py_name)。"""
        if not model_name or not str(model_name).strip():
            model_name = "YOLO"
        model_name = str(model_name).strip()
        # 用户上传模型
        if model_registry and model_name.startswith("user:"):
            path = model_registry.get_user_model_path(model_name)
            if path and path.exists():
                if model_name not in self._lazy_detection:
                    self._lazy_detection[model_name] = self._safe_load_yolo(path)
                return self._lazy_detection[model_name], True, model_name
        # 目录模型（检测类）：仅当用户已下载到本地时才使用，不自动下载
        if model_registry and not model_name.startswith("user:"):
            path = model_registry.get_downloaded_model_path(model_name)
            if model_name in self._lazy_detection:
                if not path or not path.exists():
                    del self._lazy_detection[model_name]
                else:
                    return self._lazy_detection[model_name], True, model_name
            if path and path.exists():
                self._lazy_detection[model_name] = self._safe_load_yolo(path)
                return self._lazy_detection[model_name], True, model_name
            catalog_ids = [e.get("id") for e in model_registry.load_catalog()]
            if model_name in catalog_ids:
                raise ValueError("该模型未下载到本地，请先在页面选择「下载到本地」后再使用")
        # 内置
        if model_name in ("YOLO", "FasterRCNN", "SSD"):
            self._load_builtin_detection_model(model_name)
        if model_name in self.detection_models:
            model = self.detection_models[model_name]
            use_pytorch = model_name in ["FasterRCNN", "SSD"] and isinstance(model, torch.nn.Module)
            return model, not use_pytorch, model_name
        # 回退
        model_name = "YOLO"
        self._load_builtin_detection_model(model_name)
        if model_name in self.detection_models:
            return self.detection_models[model_name], True, model_name
        return None, True, model_name

    def detect_objects(self, image_path, model_name=None):
        """边界框检测，返回检测结果列表"""
        model, use_yolo, resolved_name = self._get_detection_model(model_name)
        if model is None:
            return self._generate_mock_detections()
        if not self.detection_models and not self._lazy_detection:
            return self._generate_mock_detections()
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            height, width = image.shape[:2]
            if use_yolo:
                return self._detect_with_yolo(image, model)
            return self._detect_with_pytorch(image, model, resolved_name)
        except Exception as e:
            print(f"检测出错: {e}")
            return []
            
    def _detect_with_yolo(self, image, model):
        """使用YOLO模型进行目标检测"""
        height, width = image.shape[:2]
        t0 = time.perf_counter()
        results = model(
            image,
            imgsz=self.det_imgsz,
            conf=0.25,
            iou=0.7,
            max_det=self.max_det,
            verbose=False,
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    
                    # 获取类别名称
                    class_name = model.names[cls]
                    
                    # 转换为百分比坐标
                    x_percent = x1 / width
                    y_percent = y1 / height
                    w_percent = (x2 - x1) / width
                    h_percent = (y2 - y1) / height
                    
                    # 添加检测结果
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'bbox_percent': [x_percent, y_percent, w_percent, h_percent],
                        'confidence': float(conf),
                        'class_id': int(cls),
                        'class_name': class_name
                    })
        
        dt_ms = (time.perf_counter() - t0) * 1000
        print(f"YOLO检测推理耗时: {dt_ms:.1f} ms, detections={len(detections)}")
        return detections
        
    def _detect_with_pytorch(self, image, model, model_name):
        """使用PyTorch检测模型进行目标检测"""
        import torch
        import torchvision.transforms as transforms
        from PIL import Image
        import numpy as np
        
        # 大图自动缩放后再推理，显著降低 FasterRCNN/SSD 时延
        orig_height, orig_width = image.shape[:2]
        max_side = max(orig_height, orig_width)
        scale = 1.0
        infer_image = image
        if max_side > 1280:
            scale = 1280.0 / float(max_side)
            new_w = max(1, int(orig_width * scale))
            new_h = max(1, int(orig_height * scale))
            infer_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 转换图像格式
        image_rgb = cv2.cvtColor(infer_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # 转换为PyTorch张量
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img_tensor = transform(pil_image)
        
        # 进行推理
        t0 = time.perf_counter()
        with torch.no_grad():
            predictions = model([img_tensor])
        
        # 处理预测结果
        detections = []
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        
        # 设置置信度阈值
        threshold = 0.5
        
        # 获取原图尺寸（百分比坐标按原图计算）
        height, width = orig_height, orig_width
        
        for i, box in enumerate(boxes):
            if scores[i] >= threshold:
                x1, y1, x2, y2 = map(float, box)
                if scale != 1.0:
                    inv = 1.0 / scale
                    x1, y1, x2, y2 = x1 * inv, y1 * inv, x2 * inv, y2 * inv
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                label_id = int(labels[i])
                
                # 获取类别名称
                if model_name in ["FasterRCNN", "SSD"]:
                    class_name = self.fasterrcnn_classes[label_id]
                else:
                    class_name = f"Class_{label_id}"
                
                # 转换为百分比坐标
                x_percent = x1 / width
                y_percent = y1 / height
                w_percent = (x2 - x1) / width
                h_percent = (y2 - y1) / height
                
                # 添加检测结果
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'bbox_percent': [x_percent, y_percent, w_percent, h_percent],
                    'confidence': float(scores[i]),
                    'class_id': label_id,
                    'class_name': class_name
                })
        
        dt_ms = (time.perf_counter() - t0) * 1000
        print(f"{model_name}检测推理耗时: {dt_ms:.1f} ms, scale={scale:.3f}, detections={len(detections)}")
        return detections
    
    def detect_oriented_objects(self, image_path, model_name=None):
        """方向边界框(OBB)检测"""
        # 待集成实际模型，目前返回空
        return []
    
    def classify_image(self, image_path, model_name=None):
        """图像分类"""
        try:
            model, backend, resolved_name = self._get_classification_model(model_name)
            if model is None:
                return []

            if backend == "yolo":
                return self._classify_with_yolo(image_path, model)
            return self._classify_with_torchvision(image_path, model)
        except Exception as e:
            print(f"图像分类错误: model={model_name}, resolved={locals().get('resolved_name')}, err={e}")
            return []

    def detect_keypoints(self, image_bytes):
        """关键点检测"""
        # 待集成实际模型，目前返回空
        return []

# 全局模型实例
ai_service = AIModelService()
