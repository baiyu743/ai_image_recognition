
from typing import Dict, Any

def generate_llm_analysis(metrics: Dict[str, Any], model_id: str) -> str:
    """
    根据评估指标生成基于规则的“LLM”优化建议
    """
    map50 = metrics.get('mAP50', 0)
    map50_95 = metrics.get('mAP50_95', 0)
    precision = metrics.get('precision', 0)
    recall = metrics.get('recall', 0)
    class_metrics = metrics.get('class_metrics', {})

    analysis = []
    
    # 1. 总体评分与概述
    if map50 > 0.8:
        analysis.append(f"**总体评价**：模型 `{model_id}` 表现优秀，mAP@0.5 达到 {map50:.4f}。模型在大多数场景下能准确识别目标。")
    elif map50 > 0.5:
        analysis.append(f"**总体评价**：模型 `{model_id}` 表现良好，mAP@0.5 为 {map50:.4f}，但在复杂场景下仍有提升空间。")
    else:
        analysis.append(f"**总体评价**：模型 `{model_id}` 表现一般，mAP@0.5 仅为 {map50:.4f}，需要重点优化。")

    # 2. 精度与召回率分析
    if precision < 0.6 and recall > 0.8:
        analysis.append("**问题诊断**：模型存在较高的**误检率 (False Positives)**。虽然能找到大部分目标，但也把很多背景误判为目标。")
        analysis.append("**优化建议**：\n- 增加负样本（背景图片）进行训练。\n- 提高推理时的置信度阈值 (conf_threshold)。\n- 检查训练数据中是否有标注错误的背景框。")
    elif precision > 0.8 and recall < 0.6:
        analysis.append("**问题诊断**：模型存在较高的**漏检率 (False Negatives)**。模型非常谨慎，只在非常有把握时才预测，导致漏掉了许多目标。")
        analysis.append("**优化建议**：\n- 增加更多包含目标的训练数据，特别是小目标或遮挡目标。\n- 降低推理时的置信度阈值。\n- 尝试使用多尺度训练 (Multi-scale training) 或 Mosaic 数据增强。")
    elif precision < 0.6 and recall < 0.6:
        analysis.append("**问题诊断**：模型的精度和召回率都较低，说明模型尚未有效学习到特征。")
        analysis.append("**优化建议**：\n- 检查数据集标注质量，是否存在大量错标/漏标。\n- 检查模型容量是否不足，尝试使用更大的模型架构（如从 yolov8n 换到 yolov8s/m）。\n- 增加训练轮数 (Epochs)。")

    # 3. 类别分析
    if class_metrics:
        worst_classes = sorted(class_metrics.items(), key=lambda x: x[1]['ap'])[:3]
        best_classes = sorted(class_metrics.items(), key=lambda x: x[1]['ap'], reverse=True)[:3]
        
        worst_names = [name for name, _ in worst_classes]
        analysis.append(f"**类别分析**：\n- **表现最差的类别**：{', '.join(worst_names)}。建议针对这些类别收集更多特定数据，或使用过采样策略。")
        
    # 4. 边界框回归分析
    if map50_95 < map50 * 0.5:
        analysis.append("**定位精度分析**：mAP@0.5:0.95 显著低于 mAP@0.5，说明模型虽然能找到目标，但**边界框位置不够精确**。")
        analysis.append("**优化建议**：\n- 检查标注框是否贴合目标边缘。\n- 增加 IOU Loss 的权重。\n- 训练时开启更强的几何增强（如仿射变换）。")

    return "\n\n".join(analysis)
