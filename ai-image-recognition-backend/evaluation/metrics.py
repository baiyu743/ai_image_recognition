import matplotlib
matplotlib.use('Agg') 
import numpy as np
import json
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import io
import base64

class EvaluationMetrics:
    """评估指标计算工具类"""
    
    @staticmethod
    def calculate_precision_recall(  
        predictions: List[Dict], #模型预测结果列表
        ground_truths: List[Dict], #真实标注列表
        iou_threshold: float = 0.5   #IOU阈值
    ) -> Tuple[Dict, Dict]:
        """
        计算精确率和召回率
        """
        # 获取所有类别
        all_classes = set()
        for pred in predictions:
            all_classes.add(pred.get('class'))
        for gt in ground_truths:
            all_classes.add(gt.get('class'))
        all_classes = list(all_classes)
        
        # 初始化
        class_stats = {cls: {'tp': 0, 'fp': 0, 'fn': 0} for cls in all_classes}
        
        # 标记已匹配的真实标注
        matched_gt = {i: False for i in range(len(ground_truths))}
        
        # 按置信度排序预测结果
        predictions_sorted = sorted(predictions, key=lambda x: x.get('confidence', 0), reverse=True)
        
        # 计算TP, FP
        for pred in predictions_sorted:
            pred_cls = pred['class']
            pred_box = pred['box']
            best_iou = 0
            best_gt_idx = -1
            
            # 寻找最佳匹配的真实标注
            for gt_idx, gt in enumerate(ground_truths):
                if gt.get('image_id') == pred.get('image_id') and \
                   not matched_gt[gt_idx] and gt['class'] == pred_cls:
                    iou = EvaluationMetrics.calculate_iou(pred_box, gt['box'])
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            if best_gt_idx != -1:
                # 找到匹配的真实标注
                class_stats[pred_cls]['tp'] += 1
                matched_gt[best_gt_idx] = True
            else:
                # 未找到匹配的真实标注
                class_stats[pred_cls]['fp'] += 1
        
        # 计算FN
        for gt_idx, gt in enumerate(ground_truths):
            if not matched_gt[gt_idx]:
                gt_cls = gt['class']
                class_stats[gt_cls]['fn'] += 1
        
        # 计算精确率和召回率
        precision = {}
        recall = {}
        for cls in all_classes:
            tp = class_stats[cls]['tp']
            fp = class_stats[cls]['fp']
            fn = class_stats[cls]['fn']
            
            # 防止除以零
            precision[cls] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # 计算总体指标
        total_tp = sum(stats['tp'] for stats in class_stats.values())
        total_fp = sum(stats['fp'] for stats in class_stats.values())
        total_fn = sum(stats['fn'] for stats in class_stats.values())
        
        precision['overall'] = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall['overall'] = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        
        return precision, recall
    
    @staticmethod
    def calculate_iou(box1: List[float], box2: List[float]) -> float:
        """
        计算两个边界框的交并比(IOU)
        """
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2
        
        # 计算交集区域
        x1i = max(x1, x1g)
        y1i = max(y1, y1g)
        x2i = min(x2, x2g)
        y2i = min(y2, y2g)
        
        # 计算交集面积
        intersection = max(0, x2i - x1i) * max(0, y2i - y1i)
        
        # 计算两个框的面积
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2g - x1g) * (y2g - y1g)
        
        # 计算并集面积
        union = area1 + area2 - intersection
        
        # 计算IOU
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def calculate_ap(precision: List[float], recall: List[float]) -> float:
        """
        计算平均精度(AP)
        """
        # 确保列表按召回率排序
        sorted_indices = np.argsort(recall)
        precision = np.array(precision)[sorted_indices]
        recall = np.array(recall)[sorted_indices]
        
        # 确保增加一个(1.0, 0.0)点
        precision = np.concatenate(([0.0], precision, [0.0]))
        recall = np.concatenate(([0.0], recall, [1.0]))
        
        # 计算每个点的最大精度
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])
        
        # 计算PR曲线下的面积
        indices = np.where(recall[1:] != recall[:-1])[0] + 1
        ap = np.sum((recall[indices] - recall[indices - 1]) * precision[indices])
        
        return ap
    
    @staticmethod
    def calculate_map(
        predictions: List[Dict], #模型预测结果列表
        ground_truths: List[Dict], #真实标注列表
        iou_thresholds: Optional[List[float]] = None #IOU阈值列表
    ) -> Tuple[float, float, Dict]:
        """
        计算平均精度均值(mAP)      
            map50: mAP@0.5
            map50_95: mAP@0.5:0.95
            class_maps: 每个类别的mAP
        """
        if iou_thresholds is None:
            iou_thresholds = [0.5] + [i/100 for i in range(55, 96, 5)]  # 0.5, 0.55, ..., 0.95
        
        # 获取所有类别
        all_classes = set()
        for pred in predictions:
            all_classes.add(pred.get('class'))
        for gt in ground_truths:
            all_classes.add(gt.get('class'))
        all_classes = list(all_classes)
        
        # 按类别和IOU阈值计算AP
        class_aps = {cls: [] for cls in all_classes}
        
        for iou in iou_thresholds:
            # 按置信度排序所有预测
            preds_sorted = sorted(predictions, key=lambda x: x.get('confidence', 0), reverse=True)
            
            for cls in all_classes:
                # 筛选当前类别的预测和真实标注
                cls_preds = [p for p in preds_sorted if p['class'] == cls]
                cls_gts = [g for g in ground_truths if g['class'] == cls]
                
                # 计算每个预测的TP/FP
                matched_gts = set()
                precisions = []
                recalls = []
                tp_count = 0
                fp_count = 0
                
                for pred in cls_preds:
                    pred_box = pred['box']
                    best_iou = 0
                    best_gt_idx = -1
                    
                    # 寻找最佳匹配
                    for gt_idx, gt in enumerate(cls_gts):
                        if gt.get('image_id') == pred.get('image_id') and \
                           gt_idx not in matched_gts:
                            current_iou = EvaluationMetrics.calculate_iou(pred_box, gt['box'])
                            if current_iou > best_iou and current_iou >= iou:
                                best_iou = current_iou
                                best_gt_idx = gt_idx
                    
                    if best_gt_idx != -1:
                        tp_count += 1
                        matched_gts.add(best_gt_idx)
                    else:
                        fp_count += 1
                    
                    # 计算当前的精确率和召回率
                    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
                    recall = tp_count / len(cls_gts) if len(cls_gts) > 0 else 0.0
                    
                    precisions.append(precision)
                    recalls.append(recall)
                
                # 计算AP
                ap = EvaluationMetrics.calculate_ap(precisions, recalls)
                class_aps[cls].append(ap)
        
        # 计算每个类别的mAP
        class_maps = {}
        for cls in all_classes:
            class_maps[cls] = np.mean(class_aps[cls])
        
        # 计算mAP@0.5和mAP@0.5:0.95
        map50 = np.mean([class_aps[cls][0] for cls in all_classes])  # 第一个是0.5阈值
        map50_95 = np.mean([np.mean(aps) for aps in class_aps.values()])
        
        return map50, map50_95, class_maps
    
    @staticmethod
    def generate_pr_curve_data(
        predictions: List[Dict], 
        ground_truths: List[Dict], 
        iou_threshold: float = 0.5
    ) -> Dict:
        """
        生成PR曲线数据
        """
        # 获取所有类别
        all_classes = set()
        for pred in predictions:
            all_classes.add(pred.get('class'))
        for gt in ground_truths:
            all_classes.add(gt.get('class'))
        all_classes = list(all_classes)
        
        pr_data = {'overall': {'precision': [], 'recall': []}}
        
        # 计算总体PR曲线
        preds_sorted = sorted(predictions, key=lambda x: x.get('confidence', 0), reverse=True)
        matched_gts = set()
        tp_count = 0
        fp_count = 0
        total_gt = len(ground_truths)
        
        for pred in preds_sorted:
            pred_box = pred['box']
            best_iou = 0
            best_gt_idx = -1
            
            # 寻找最佳匹配
            for gt_idx, gt in enumerate(ground_truths):
                if gt_idx not in matched_gts and gt['class'] == pred['class']:
                    current_iou = EvaluationMetrics.calculate_iou(pred_box, gt['box'])
                    if current_iou > best_iou and current_iou >= iou_threshold:
                        best_iou = current_iou
                        best_gt_idx = gt_idx
            
            if best_gt_idx != -1:
                tp_count += 1
                matched_gts.add(best_gt_idx)
            else:
                fp_count += 1
            
            # 计算当前的精确率和召回率
            precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
            recall = tp_count / total_gt if total_gt > 0 else 0.0
            
            pr_data['overall']['precision'].append(precision)
            pr_data['overall']['recall'].append(recall)
        
        # 为每个类别计算PR曲线
        for cls in all_classes:
            pr_data[cls] = {'precision': [], 'recall': []}
            
            # 筛选当前类别的预测和真实标注
            cls_preds = [p for p in predictions if p['class'] == cls]
            cls_gts = [g for g in ground_truths if g['class'] == cls]
            
            # 按置信度排序
            cls_preds_sorted = sorted(cls_preds, key=lambda x: x.get('confidence', 0), reverse=True)
            
            matched_cls_gts = set()
            tp_count_cls = 0
            fp_count_cls = 0
            total_cls_gt = len(cls_gts)
            
            for pred in cls_preds_sorted:
                pred_box = pred['box']
                best_iou = 0
                best_gt_idx = -1
                
                # 寻找最佳匹配
                for gt_idx, gt in enumerate(cls_gts):
                    if gt_idx not in matched_cls_gts:
                        current_iou = EvaluationMetrics.calculate_iou(pred_box, gt['box'])
                        if current_iou > best_iou and current_iou >= iou_threshold:
                            best_iou = current_iou
                            best_gt_idx = gt_idx
                
                if best_gt_idx != -1:
                    tp_count_cls += 1
                    matched_cls_gts.add(best_gt_idx)
                else:
                    fp_count_cls += 1
                
                # 计算当前的精确率和召回率
                precision = tp_count_cls / (tp_count_cls + fp_count_cls) if (tp_count_cls + fp_count_cls) > 0 else 0.0
                recall = tp_count_cls / total_cls_gt if total_cls_gt > 0 else 0.0
                
                pr_data[cls]['precision'].append(precision)
                pr_data[cls]['recall'].append(recall)
        
        return pr_data
    
    @staticmethod
    def plot_pr_curve(pr_data: Dict, title: str = "Precision-Recall Curve") -> str:
        """
        绘制PR曲线并返回base64编码的图像
        """
        plt.figure(figsize=(10, 8))
        
        # 绘制总体PR曲线
        if 'overall' in pr_data:
            plt.plot(
                pr_data['overall']['recall'], 
                pr_data['overall']['precision'], 
                label='Overall', 
                linewidth=2
            )
        
        # 绘制每个类别的PR曲线
        for cls, data in pr_data.items():
            if cls != 'overall':
                plt.plot(
                    data['recall'], 
                    data['precision'], 
                    label=f'Class: {cls}', 
                    alpha=0.7,
                    linewidth=1.5
                )
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        # 保存图像到内存
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
        buffer.seek(0)
        
        # 转换为base64
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()
        plt.close()
        
        return image_base64
