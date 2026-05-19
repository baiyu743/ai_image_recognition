import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import random
import numpy as np
from PIL import Image
import torchvision.transforms as T
import yaml
import os

from ultralytics.models.yolo.classify.train import ClassificationTrainer
from ultralytics.utils import loss

class ClsDistillationTrainer(ClassificationTrainer):
    """
    一个用于图像分类的知识蒸馏训练器。
    - 支持旧样本回放 (Replay)
    - 支持一致性正则化 (Consistency)
    - 蒸馏内容为教师模型和学生模型输出的类别概率分布 (logits)。
    """
    def __init__(self, *args, **kwargs):
        # --- 1. 拦截并移除自定义参数 ---
        overrides = kwargs.get('overrides', {})
        self.teacher_model = overrides.pop('teacher_model', None)
        self.distill_cls_weight = overrides.pop('distill_cls_weight', 1.0)
        self.temperature = overrides.pop('temperature', 2.0)
        
        # Replay params
        self.old_data_yaml = overrides.pop('old_data_yaml', None)
        self.replay_ratio = overrides.pop('replay_ratio', 0.3)
        self.replay_distill_boost = overrides.pop('replay_distill_boost', 2.0) # 确保拦截此参数
        self.max_replay_samples = overrides.pop('max_replay_samples', 500)
        
        # Consistency params
        self.enable_consistency = overrides.pop('enable_consistency', False)
        self.consistency_weight = overrides.pop('consistency_weight', 1.0)
        
        # 移除其他任务可能传入的无效参数，避免冲突
        overrides.pop('distill_reg_weight', None)
        overrides.pop('distill_feat_weight', None)
        overrides.pop('distill_mask_weight', None)
        overrides.pop('distill_bg_weight', None)
        overrides.pop('class_weights', None)
        overrides.pop('pseudo_conf_threshold', None)
        overrides.pop('new_class_ids', None) # 确保拦截此参数

        if self.teacher_model is None:
            raise ValueError("ClsDistillationTrainer requires a 'teacher_model' argument.")

        # --- 2. 调用父类构造函数 ---
        super().__init__(*args, **kwargs)

        # --- 3. 初始化教师模型 ---
        self.teacher_model.to(self.device)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        print("✅ 分类任务蒸馏训练器初始化成功，教师模型已冻结。")
        print(f"   - 蒸馏权重 (Cls): {self.distill_cls_weight}, 温度: {self.temperature}")
        
        # --- 4. 初始化回放缓冲区 ---
        self.replay_buffer = []
        if self.old_data_yaml:
            self._load_replay_buffer()

        # --- 5. 定义强增强 ---
        self.strong_aug = T.Compose([
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        ])

        # 对于分类任务，KLDivLoss的reduction='batchmean'更常用
        self.distill_cls_loss = nn.KLDivLoss(reduction='batchmean')

        # 映射标记
        self.distill_indices_computed = False
        self.t_indices = None
        self.s_indices = None

    def _load_replay_buffer(self):
        print(f"\n📦 正在构建旧样本回放缓冲区 (Classify)...")
        try:
            old_data_path = Path(self.old_data_yaml)
            train_dir = None
            
            # 尝试解析YAML
            if old_data_path.is_file() and old_data_path.suffix in ['.yaml', '.yml']:
                with old_data_path.open('r', encoding='utf-8') as f:
                    cfg = yaml.safe_load(f)
                    # Classification yaml usually has 'path' or 'train' pointing to dir
                    base = Path(cfg.get('path', old_data_path.parent))
                    if 'train' in cfg:
                        train_dir = base / cfg['train']
            elif old_data_path.is_dir():
                # 假设直接提供了数据集根目录，且包含 'train' 子目录
                if (old_data_path / 'train').exists():
                    train_dir = old_data_path / 'train'
                else:
                    train_dir = old_data_path # 或者是直接的train目录
            
            if not train_dir or not train_dir.exists():
                print(f"   ❌ 无法定位旧数据的训练目录: {train_dir}")
                return

            print(f"   - 扫描目录: {train_dir}")
            
            # 获取教师模型的类别名称
            teacher_names = self.teacher_model.names # dict {0: 'name', ...}
            valid_class_names = set(teacher_names.values())
            
            valid_samples = []
            # 遍历目录
            for class_dir in train_dir.iterdir():
                if class_dir.is_dir() and class_dir.name in valid_class_names:
                    for img_file in class_dir.glob('*.*'):
                        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                            valid_samples.append({
                                'img_path': str(img_file),
                                'class_name': class_dir.name
                            })
            
            if len(valid_samples) > self.max_replay_samples:
                self.replay_buffer = random.sample(valid_samples, self.max_replay_samples)
            else:
                self.replay_buffer = valid_samples
                
            print(f"   ✅ 成功加载 {len(self.replay_buffer)} 个旧样本")

        except Exception as e:
            print(f"   ❌ 加载旧样本失败: {e}")

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        
        # Replay
        if self.replay_buffer and self.replay_ratio > 0:
            batch_size = batch['img'].shape[0]
            num_replay = int(batch_size * self.replay_ratio)
            if num_replay > 0:
                indices = random.sample(range(batch_size), num_replay)
                for idx in indices:
                    sample = random.choice(self.replay_buffer)
                    # Load Image
                    try:
                        img = Image.open(sample['img_path']).convert('RGB')
                        img = img.resize((self.args.imgsz, self.args.imgsz))
                        # Convert to tensor
                        img_t = T.ToTensor()(img).to(self.device)
                        batch['img'][idx] = img_t
                        
                        # Update Label
                        # Find class index in current model
                        # self.model.names is dict {id: name}
                        # We need name -> id
                        name_to_id = {v: k for k, v in self.model.names.items()}
                        if sample['class_name'] in name_to_id:
                            batch['cls'][idx] = name_to_id[sample['class_name']]
                    except Exception:
                        pass

        # Consistency
        if self.enable_consistency:
            batch['img_weak'] = batch['img'].clone()
            try:
                batch['img'] = self.strong_aug(batch['img'])
            except:
                pass
                
        return batch

    def get_loss(self, preds, batch):
        # --- 1. 计算标准GT损失 ---
        # ClassificationTrainer直接将损失函数实例保存在self.criterion
        loss_gt, loss_items = self.criterion(preds, batch)

        # --- 2. 准备学生和教师的输出 ---
        student_preds = preds
        
        # Teacher input
        teacher_img = batch.get('img_weak', batch['img']) if self.enable_consistency else batch['img']
        
        with torch.no_grad():
            teacher_preds = self.teacher_model(teacher_img)

        # --- 3. 计算分类蒸馏损失 (带类别映射) ---
        # 动态计算类别映射，确保即使类别顺序不同或有新增类别也能正确蒸馏
        if not self.distill_indices_computed:
            self.t_indices = []
            self.s_indices = []
            
            t_names = self.teacher_model.names
            s_names = self.model.names
            
            # 反转学生模型名称字典以便查找
            s_name_to_id = {v: k for k, v in s_names.items()}
            
            # 查找公共类别
            for t_id, t_name in t_names.items():
                if t_name in s_name_to_id:
                    self.t_indices.append(t_id)
                    self.s_indices.append(s_name_to_id[t_name])
            
            # 转为Tensor
            if self.t_indices:
                self.t_indices = torch.tensor(self.t_indices, device=self.device, dtype=torch.long)
                self.s_indices = torch.tensor(self.s_indices, device=self.device, dtype=torch.long)
            
            self.distill_indices_computed = True
            
            print(f"ℹ️  蒸馏类别映射已建立: 共有 {len(self.t_indices)}/{len(t_names)} 个教师类别被匹配。")
            if len(self.t_indices) < len(t_names):
                print("⚠️  警告: 学生模型缺少部分教师模型的类别，这些类别的知识将无法被蒸馏。")

        # 仅在有公共类别时计算蒸馏损失
        if self.t_indices is not None and len(self.t_indices) > 0:
            # 提取对应的logits子集
            s_logits_subset = student_preds[:, self.s_indices]
            t_logits_subset = teacher_preds[:, self.t_indices]
            
            # 使用 log_softmax 和 softmax 来计算KL散度
            loss_distill_cls = self.distill_cls_loss(
                F.log_softmax(s_logits_subset / self.temperature, dim=1),
                F.softmax(t_logits_subset / self.temperature, dim=1)
            ) * (self.temperature ** 2) # T^2 scaling
        else:
            loss_distill_cls = torch.tensor(0.0, device=self.device)

        # Consistency weight
        c_weight = self.consistency_weight if self.enable_consistency else 1.0

        # --- 4. 合并所有损失 ---
        total_loss = loss_gt + self.distill_cls_weight * loss_distill_cls * c_weight

        new_loss_items = torch.tensor([loss_distill_cls], device=self.device)
        # 分类任务的loss_items通常只有一个元素，直接拼接
        loss_items = torch.cat((loss_items, new_loss_items))
        
        return total_loss, loss_items