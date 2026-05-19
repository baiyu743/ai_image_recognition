import os
import json
from typing import Dict, Optional
from dotenv import load_dotenv
# 替换为openai库（需安装：pip install openai>=1.0.0）
from openai import OpenAI

load_dotenv()

class LLMClient:
    """独立的自然语言模型客户端，专门处理评估结果的自然语言分析"""
    def __init__(
        self,
        api_key: Optional[str] = "sk-522443c8625c43c2b89defe519ad784c",
        # 阿里云DashScope支持的模型名
        model: str = "qwen-plus",
        base_url: Optional[str] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        
        # 初始化OpenAI客户端（对接阿里云DashScope）
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def generate_evaluation_analysis(self, eval_data: Dict) -> str:
        """核心方法：输入评估指标，输出自然语言分析报告"""
        prompt = f"""
        你是AI模型训练评估专家，现在需要分析目标检测模型的评估结果，输出以下4部分内容：
        1. 整体评价：总结模型核心性能（优势/短板）
        2. 关键问题：指出指标异常的点（如某类别召回率<0.6、mAP偏低等）
        3. 优化方向：分优先级给出可落地的优化策略
        4. 具体建议：细化到数据/模型/调参的具体动作

        评估数据：
        - 总体指标：
          mAP@0.5: {eval_data['mAP50']:.4f} | mAP@0.5:0.95: {eval_data['mAP50_95']:.4f}
          精确率: {eval_data['precision']:.4f} | 召回率: {eval_data['recall']:.4f} | F1: {eval_data['f1_score']:.4f}
        - 分类级指标：{json.dumps(eval_data['class_metrics'], indent=2, ensure_ascii=False)}

        规则：
        - 指标≥0.8强调优势，≤0.6重点指出问题
        - 优化建议需贴合目标检测场景（如数据增强、边界框回归、类别平衡等）
        - 语言专业但易懂，总字数控制在800字内
        """

        try:
            # 调用OpenAI兼容接口（阿里云DashScope）
            # 添加 60 秒超时设置，防止任务无限挂起
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是专注于目标检测模型评估的专家，输出结构化分析报告"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000,
                timeout=60.0 
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"【LLM分析生成失败】：{str(e)}\n\n手动分析建议：\n1. 整体mAP@0.5={eval_data['mAP50']:.4f}，需重点关注低分值类别\n2. 检查召回率/精确率失衡的类别，优先优化数据或模型参数\n3. 若mAP@0.5<0.7，建议增加训练数据量或调整IOU阈值"

# 单例实例
llm_client = LLMClient()
