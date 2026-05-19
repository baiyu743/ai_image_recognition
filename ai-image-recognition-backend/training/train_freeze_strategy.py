# enhanced_freeze_strategy.py

import torch
import matplotlib.pyplot as plt
import os
import logging
import shutil
from typing import List, Optional, Union
from pathlib import Path
import yaml


class EnhancedFreezeScheduler:
    def __init__(
            self,
            model,
            freeze_stages: List[str],
            output_dir="freeze_logs",
            patience=3,
            stage_weights_dir="stage_weights",
            auto_load_best=True,
            min_epochs_per_stage=10
    ):
        """
        增强版YOLOv8分阶段冻结训练调度器

        :param model: YOLOv8模型（Ultralytics YOLO）
        :param freeze_stages: list，每个元素为 layer/module 的字符串名，如 'model.model.0'
        :param output_dir: 保存图像和日志的目录
        :param patience: plateau 判定窗口
        :param stage_weights_dir: 每个阶段权重文件保存目录
        :param auto_load_best: 是否自动加载上个阶段的best.pt
        :param min_epochs_per_stage: 每个阶段最少训练轮数
        """
        self.model = model
        self.stages = freeze_stages
        self.current_stage = len(freeze_stages) - 1
        self.patience = patience
        self.min_epochs_per_stage = min_epochs_per_stage
        self.val_metrics = []
        self.output_dir = output_dir
        self.stage_weights_dir = stage_weights_dir
        self.auto_load_best = auto_load_best

        # 创建目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(stage_weights_dir, exist_ok=True)

        # 设置日志（强制UTF-8）
        self._setup_logging()

        # 验证所有模块路径的有效性
        self._validate_freeze_stages()

        # 阶段管理相关
        self.stage_epochs = [0 for _ in self.stages]  # 每个阶段已训练的轮数
        self.stage_best_metrics = [float('inf') for _ in self.stages]  # 每个阶段的最佳指标
        self.current_epoch = 0
        self.stage_changed = True  # 标记阶段是否刚切换

        # 初始化冻结策略
        self._freeze_all()
        self._unfreeze([self.stages[self.current_stage]])

        self.grad_norm_history = [[] for _ in self.stages]

        self.logger.info(f"EnhancedFreezeScheduler初始化完成，共{len(self.stages)}个阶段")
        self.logger.info(f"当前解冻阶段: {self.current_stage} - {self.stages[self.current_stage]}")

        # 尝试加载上个阶段的权重
        if self.auto_load_best and self.current_stage < len(self.stages) - 1:
            self._try_load_previous_stage_weights()

    def _setup_logging(self):
        """设置日志记录，强制UTF-8编码"""
        self.logger = logging.getLogger('EnhancedFreezeScheduler')
        self.logger.setLevel(logging.INFO)

        # 避免重复添加handler
        if not self.logger.handlers:
            # 文件handler - 强制UTF-8编码
            log_file = os.path.join(self.output_dir, 'enhanced_freeze_scheduler.log')
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setLevel(logging.INFO)

            # 控制台handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)

            # 格式化器
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)

            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    def _validate_freeze_stages(self):
        """验证所有冻结阶段的模块路径是否有效"""
        invalid_stages = []
        for stage in self.stages:
            if self._get_module_by_path(stage) is None:
                invalid_stages.append(stage)

        if invalid_stages:
            available_modules = self._get_available_modules()
            error_msg = (
                f"以下冻结阶段路径无效: {invalid_stages}\n"
                f"可用的模块路径示例:\n{available_modules}"
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        else:
            self.logger.info("所有冻结阶段路径验证通过")

    def _get_available_modules(self, max_depth=3, max_items=10) -> str:
        """获取可用的模块路径示例"""
        available = []
        
        def traverse(module, path, depth):
            if depth > max_depth or len(available) >= max_items:
                return
            for name, child in module.named_children():
                child_path = f"{path}.{name}" if path else name
                available.append(child_path)
                if len(available) >= max_items:
                    return
                traverse(child, child_path, depth + 1)
        
        traverse(self.model.model, "model.model", 0)
        return "\n".join(available[:max_items])

    def _get_module_by_path(self, module_path: str) -> Optional[torch.nn.Module]:
        """安全地通过路径获取模块"""
        try:
            # 分割路径
            parts = module_path.split('.')
            current = self.model
            
            for part in parts:
                if hasattr(current, part):
                    current = getattr(current, part)
                else:
                    return None
            
            return current
        except Exception as e:
            self.logger.warning(f"获取模块 {module_path} 时出错: {str(e)}")
            return None

    def _freeze_all(self):
        """冻结所有参数"""
        for param in self.model.parameters():
            param.requires_grad = False
        self.logger.info("已冻结所有模型参数")

    def _unfreeze(self, stage_names: List[str]):
        """解冻指定阶段的参数"""
        for stage_name in stage_names:
            self._set_requires_grad(stage_name, True)
        self.logger.info(f"已解冻阶段: {stage_names}")

    def _set_requires_grad(self, module_path: str, requires_grad: bool) -> bool:
        """设置指定模块的requires_grad属性"""
        module = self._get_module_by_path(module_path)
        if module is None:
            self.logger.warning(f"模块路径 {module_path} 无效，跳过")
            return False
        
        param_count = 0
        for param in module.parameters():
            param.requires_grad = requires_grad
            param_count += 1
        
        action = "解冻" if requires_grad else "冻结"
        self.logger.info(f"{action}模块 {module_path}，共 {param_count} 个参数")
        return True

    def _get_stage_weight_path(self, stage_idx: int, weight_type: str = "best") -> str:
        """获取阶段权重文件路径"""
        return os.path.join(self.stage_weights_dir, f"stage_{stage_idx}_{weight_type}.pt")

    def _try_load_previous_stage_weights(self):
        """尝试加载上个阶段的最佳权重"""
        if self.current_stage >= len(self.stages) - 1:
            return
        
        prev_stage_idx = self.current_stage + 1
        prev_weight_path = self._get_stage_weight_path(prev_stage_idx, "best")
        
        if os.path.exists(prev_weight_path):
            try:
                # 加载权重
                checkpoint = torch.load(prev_weight_path, map_location='cpu')
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
                # 恢复训练状态
                if 'scheduler_state' in checkpoint:
                    scheduler_state = checkpoint['scheduler_state']
                    self.val_metrics = scheduler_state.get('val_metrics', [])
                    self.stage_epochs = scheduler_state.get('stage_epochs', [0] * len(self.stages))
                    self.stage_best_metrics = scheduler_state.get('stage_best_metrics', [float('inf')] * len(self.stages))
                
                self.logger.info(f"成功加载上阶段权重: {prev_weight_path}")
                return True
            except Exception as e:
                self.logger.warning(f"加载上阶段权重失败: {str(e)}")
        
        return False

    def save_stage_weights(self, source_path: str, weight_type: str = "best"):
        """保存当前阶段的权重"""
        if not os.path.exists(source_path):
            self.logger.warning(f"源权重文件不存在: {source_path}")
            return
        
        stage_weight_path = self._get_stage_weight_path(self.current_stage, weight_type)
        
        try:
            # 加载源权重
            checkpoint = torch.load(source_path, map_location='cpu')
            
            # 添加调度器状态
            checkpoint['scheduler_state'] = {
                'current_stage': self.current_stage,
                'val_metrics': self.val_metrics,
                'stage_epochs': self.stage_epochs,
                'stage_best_metrics': self.stage_best_metrics,
                'current_epoch': self.current_epoch
            }
            
            # 保存
            torch.save(checkpoint, stage_weight_path)
            self.logger.info(f"阶段 {self.current_stage} 权重已保存: {stage_weight_path}")
            
        except Exception as e:
            self.logger.error(f"保存阶段权重失败: {str(e)}")

    def update_grad_norms(self):
        """更新当前阶段的梯度范数历史"""
        if self.current_stage < 0 or self.current_stage >= len(self.stages):
            return
        
        total_norm = 0.0
        param_count = 0
        
        stage_module = self._get_module_by_path(self.stages[self.current_stage])
        if stage_module is not None:
            for param in stage_module.parameters():
                if param.grad is not None and param.requires_grad:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
        
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
            self.grad_norm_history[self.current_stage].append(total_norm)
            self.logger.debug(f"阶段 {self.current_stage} 梯度范数: {total_norm:.6f}")

    def step_epoch(self, val_metric: float, weights_path: Optional[str] = None):
        """每个epoch结束时调用"""
        self.current_epoch += 1
        self.val_metrics.append(val_metric)
        
        if self.current_stage >= 0:
            self.stage_epochs[self.current_stage] += 1
            
            # 更新当前阶段的最佳指标
            if val_metric < self.stage_best_metrics[self.current_stage]:
                self.stage_best_metrics[self.current_stage] = val_metric
                
                # 保存当前阶段的最佳权重
                if weights_path and os.path.exists(weights_path):
                    self.save_stage_weights(weights_path, "best")
        
        self.logger.info(
            f"Epoch {self.current_epoch}: 验证指标={val_metric:.4f}, "
            f"阶段={self.current_stage}, 阶段轮数={self.stage_epochs[self.current_stage] if self.current_stage >= 0 else 0}"
        )
        
        # 检查是否需要切换阶段
        if self._should_switch_stage():
            self._switch_to_next_stage()

    def _should_switch_stage(self) -> bool:
        """判断是否应该切换到下一个阶段"""
        if self.current_stage < 0:
            return False
        
        # 检查最小轮数要求
        if self.stage_epochs[self.current_stage] < self.min_epochs_per_stage:
            return False
        
        # 检查是否达到patience
        if len(self.val_metrics) < self.patience:
            return False
        
        recent_metrics = self.val_metrics[-self.patience:]
        is_plateau = all(m >= recent_metrics[0] * 0.995 for m in recent_metrics[1:])
        
        if is_plateau:
            self.logger.info(f"阶段 {self.current_stage} 达到plateau，准备切换")
            return True
        
        return False

    def _switch_to_next_stage(self):
        """切换到下一个训练阶段"""
        if self.current_stage <= 0:
            self.logger.info("所有阶段已完成")
            self.current_stage = -1
            return
        
        prev_stage = self.current_stage
        self.current_stage -= 1
        self.stage_changed = True
        
        # 重新设置冻结策略
        self._freeze_all()
        
        # 解冻当前阶段及之前的所有阶段
        stages_to_unfreeze = self.stages[self.current_stage:]
        self._unfreeze(stages_to_unfreeze)
        
        self.logger.info(
            f"阶段切换: {prev_stage} -> {self.current_stage}\n"
            f"解冻模块: {stages_to_unfreeze}\n"
            f"剩余阶段: {self.current_stage + 1}"
        )
        
        # 尝试加载上个阶段的最佳权重
        if self.auto_load_best:
            self._try_load_previous_stage_weights()

    def is_stage_changed(self) -> bool:
        """检查阶段是否刚刚切换"""
        if self.stage_changed:
            self.stage_changed = False
            return True
        return False

    def get_training_status(self) -> dict:
        """获取当前训练状态"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'current_stage': self.current_stage,
            'total_stages': len(self.stages),
            'current_stage_name': self.stages[self.current_stage] if self.current_stage >= 0 else "完成",
            'stage_epochs': self.stage_epochs[self.current_stage] if self.current_stage >= 0 else 0,
            'stage_best_metric': self.stage_best_metrics[self.current_stage] if self.current_stage >= 0 else 0,
            'total_epochs': self.current_epoch,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'trainable_ratio': trainable_params / total_params if total_params > 0 else 0,
            'val_metrics_count': len(self.val_metrics),
            'recent_metrics': self.val_metrics[-5:] if len(self.val_metrics) >= 5 else self.val_metrics,
            'stage_progress': {
                'completed_stages': len(self.stages) - self.current_stage - 1 if self.current_stage >= 0 else len(self.stages),
                'remaining_stages': self.current_stage + 1 if self.current_stage >= 0 else 0,
                'progress_percentage': ((len(self.stages) - self.current_stage - 1) / len(self.stages) * 100) if self.current_stage >= 0 else 100
            }
        }

    def plot(self, save_path: Optional[str] = None):
        """绘制训练进度和梯度范数图"""
        if not self.val_metrics:
            self.logger.warning("没有验证指标数据，无法绘图")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 绘制验证指标曲线
        epochs = list(range(1, len(self.val_metrics) + 1))
        ax1.plot(epochs, self.val_metrics, 'b-', linewidth=2, label='验证指标')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('验证指标')
        ax1.set_title('训练进度 - 验证指标变化')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 标记阶段切换点
        stage_switch_epochs = []
        cumulative_epochs = 0
        for i, stage_epoch_count in enumerate(self.stage_epochs):
            if stage_epoch_count > 0:
                cumulative_epochs += stage_epoch_count
                if cumulative_epochs <= len(self.val_metrics):
                    stage_switch_epochs.append(cumulative_epochs)
                    ax1.axvline(x=cumulative_epochs, color='red', linestyle='--', alpha=0.7)
                    ax1.text(cumulative_epochs, max(self.val_metrics) * 0.9, 
                            f'阶段{len(self.stages)-i-1}', rotation=90, ha='right')
        
        # 绘制梯度范数历史
        for i, grad_norms in enumerate(self.grad_norm_history):
            if grad_norms:
                stage_name = f'阶段{len(self.stages)-i-1}'
                ax2.plot(grad_norms, label=stage_name, linewidth=2)
        
        ax2.set_xlabel('训练步数')
        ax2.set_ylabel('梯度范数')
        ax2.set_title('各阶段梯度范数变化')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_yscale('log')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'enhanced_freeze_training_progress.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"训练进度图已保存: {save_path}")

    def save_state(self, filepath: Optional[str] = None):
        """保存调度器状态"""
        if filepath is None:
            filepath = os.path.join(self.output_dir, 'enhanced_freeze_scheduler_state.yaml')
        
        state = {
            'current_stage': self.current_stage,
            'stages': self.stages,
            'val_metrics': self.val_metrics,
            'stage_epochs': self.stage_epochs,
            'stage_best_metrics': self.stage_best_metrics,
            'current_epoch': self.current_epoch,
            'patience': self.patience,
            'min_epochs_per_stage': self.min_epochs_per_stage
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(state, f, default_flow_style=False, allow_unicode=True)
        
        self.logger.info(f"调度器状态已保存: {filepath}")

    def load_state(self, filepath: str):
        """加载调度器状态"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = yaml.safe_load(f)
            
            self.current_stage = state['current_stage']
            self.val_metrics = state['val_metrics']
            self.stage_epochs = state['stage_epochs']
            self.stage_best_metrics = state['stage_best_metrics']
            self.current_epoch = state['current_epoch']
            self.patience = state.get('patience', self.patience)
            self.min_epochs_per_stage = state.get('min_epochs_per_stage', self.min_epochs_per_stage)
            
            self.logger.info(f"调度器状态已加载: {filepath}")
            
        except Exception as e:
            self.logger.error(f"加载调度器状态失败: {str(e)}")

    def get_resume_info(self) -> dict:
        """获取恢复训练所需的信息"""
        return {
            'current_stage': self.current_stage,
            'completed_epochs': self.current_epoch,
            'stage_epochs': self.stage_epochs,
            'best_metrics': self.stage_best_metrics,
            'can_resume': self.current_stage >= 0
        }


def _get_module_by_path(model, module_path: str) -> Optional[torch.nn.Module]:
    """安全地通过路径获取模块的辅助函数"""
    try:
        parts = module_path.split('.')
        current = model
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                return None
        return current
    except Exception:
        return None


def create_yolov8_freeze_scheduler(
    model, 
    model_size='n',
    output_dir="freeze_logs",
    stage_weights_dir="stage_weights"
) -> EnhancedFreezeScheduler:
    """
    为不同尺寸的YOLOv8模型创建冻结调度器

    :param model: YOLOv8模型实例
    :param model_size: 模型尺寸 ('n', 's', 'm', 'l', 'x')
    :param output_dir: 日志输出目录
    :param stage_weights_dir: 阶段权重保存目录
    :return: EnhancedFreezeScheduler实例
    """

    # 基础的分阶段策略 - 从深层到浅层逐步解冻
    if model_size in ['n', 's']:
        # 小模型策略
        freeze_stages = [
            'model.model.9',  # Head
            'model.model.8',  # Head input
            'model.model.7',  # Neck
            'model.model.6',  # Neck
            'model.model.5',  # Backbone deep
            'model.model.4',  # Backbone
            'model.model.3',  # Backbone
            'model.model.2',  # Backbone
            'model.model.1',  # Backbone shallow
            'model.model.0'  # Input layer
        ]
    else:
        # 大模型策略 - 更细致的分阶段
        freeze_stages = [
            'model.model.9',  # Detection Head
            'model.model.8',  # Head preparation
            'model.model.7',  # Neck - Top
            'model.model.6',  # Neck - Middle
            'model.model.5',  # Backbone - Deep
            'model.model.4',  # Backbone - Deep-Mid
            'model.model.3',  # Backbone - Mid
            'model.model.2',  # Backbone - Mid-Shallow
            'model.model.1',  # Backbone - Shallow
            'model.model.0'  # Input stem
        ]

    return EnhancedFreezeScheduler(
        model=model,
        freeze_stages=freeze_stages,
        patience=3,
        min_epochs_per_stage=15,
        auto_load_best=True,
        output_dir=output_dir,
        stage_weights_dir=stage_weights_dir
    )