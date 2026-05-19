import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import loss
from ultralytics.utils.ops import xywh2xyxy, non_max_suppression
from pathlib import Path
import random
import numpy as np
from PIL import Image
import torchvision.transforms as T

class DistillationTrainer(DetectionTrainer):
    """
    一个集成了知识蒸馏、旧样本回放、伪标签生成和一致性正则化的YOLOv8检测训练器。
    """
    def __init__(self, *args, **kwargs):
        overrides = kwargs.get('overrides', {})
        self.teacher_model_arg = overrides.pop('teacher_model', None)
        self.distill_cls_weight = overrides.pop('distill_cls_weight', 1.0)
        self.distill_reg_weight = overrides.pop('distill_reg_weight', 2.0)
        self.distill_feat_weight = overrides.pop('distill_feat_weight', 5.0)
        self.temperature = overrides.pop('temperature', 2.0)
        self.class_weights = overrides.pop('class_weights', {})
        self.distill_bg_weight = overrides.pop('distill_bg_weight', 0.05)
        self.custom_names = overrides.pop('names', None)
        self.new_class_ids = overrides.pop('new_class_ids', [])
        
        self.old_data_yaml = overrides.pop('old_data_yaml', None)
        self.replay_ratio = overrides.pop('replay_ratio', 0.3)
        self.replay_distill_boost = overrides.pop('replay_distill_boost', 2.0)
        self.max_replay_samples = overrides.pop('max_replay_samples', 500)

        # 新增：伪标签和一致性参数
        self.pseudo_conf_threshold = overrides.pop('pseudo_conf_threshold', 0.7)
        self.enable_consistency = overrides.pop('enable_consistency', False)
        self.consistency_weight = overrides.pop('consistency_weight', 1.0)

        if self.teacher_model_arg is None:
            raise ValueError("DistillationTrainer requires a 'teacher_model' argument.")

        super().__init__(*args, **kwargs)

        self.teacher_model = self.teacher_model_arg
        self.distill_cls_loss = nn.KLDivLoss(reduction='none')
        self.feat_loss = nn.MSELoss()
        
        self.replay_buffer = []
        self.old_class_ids = []

        # 定义强增强变换 (仅光度变换，不改变几何坐标)
        self.strong_aug = T.Compose([
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        ])

    def _setup_train(self, world_size):
        super()._setup_train(world_size)

        if self.custom_names:
            print("\n✅ 在训练开始前，正在同步类别名称...")
            names_dict = {i: name for i, name in enumerate(self.custom_names)}
            self.data['names'] = names_dict
            self.model.names = names_dict
            if self.validator:
                self.validator.names = names_dict

        self.teacher_model.to(self.device)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        self.num_old_classes = self.teacher_model.nc
        student_nc = self.data.get('nc', '未知')
        self.old_class_ids = list(range(self.num_old_classes))
        
        print("\n✅ 知识蒸馏训练器已成功设置。")
        print(f"   - 教师/学生类别数: {self.num_old_classes} -> {student_nc}")
        print(f"   - 新类别ID: {self.new_class_ids}")
        
        if self.old_data_yaml:
            self._load_replay_buffer()
        else:
            print("   ⚠️  未提供旧数据集路径，旧样本回放功能已禁用")

    def _load_replay_buffer(self):
        print(f"\n📦 正在构建旧样本回放缓冲区...")
        print(f"   - 旧数据集配置: {self.old_data_yaml}")
        print(f"   - 回放比例: {self.replay_ratio:.1%}")
        print(f"   - 最大样本数: {self.max_replay_samples}")
        
        try:
            import yaml
            old_data_path = Path(self.old_data_yaml)
            if not old_data_path.exists():
                print(f"   ❌ 旧数据集配置文件不存在")
                return
            
            with old_data_path.open('r', encoding='utf-8') as f:
                old_data_cfg = yaml.safe_load(f)
            
            old_base_path = Path(old_data_cfg.get('path', old_data_path.parent))
            old_train_path = old_data_cfg.get('train', 'images/train')
            
            if Path(old_train_path).is_absolute():
                old_img_dir = Path(old_train_path)
            else:
                old_img_dir = (old_base_path / old_train_path).resolve()
            
            old_label_dir = old_img_dir.parent.parent / 'labels' / old_img_dir.name
            if not old_label_dir.exists():
                old_label_dir = Path(str(old_img_dir).replace('images', 'labels'))
            
            if not old_label_dir.exists():
                print(f"   ❌ 未找到旧数据集的标签目录")
                return
            
            image_files = list(old_img_dir.glob('*.jpg')) + list(old_img_dir.glob('*.png'))
            valid_samples = []
            
            for img_path in image_files:
                label_path = old_label_dir / (img_path.stem + '.txt')
                if label_path.exists():
                    valid_samples.append({
                        'img_path': str(img_path),
                        'label_path': str(label_path)
                    })
            
            if len(valid_samples) > self.max_replay_samples:
                self.replay_buffer = random.sample(valid_samples, self.max_replay_samples)
            else:
                self.replay_buffer = valid_samples
            
            print(f"   ✅ 成功加载 {len(self.replay_buffer)} 个旧样本到回放缓冲区")
            
        except Exception as e:
            print(f"   ❌ 加载旧样本失败: {e}")
            import traceback
            traceback.print_exc()

    def preprocess_batch(self, batch):
        """
        重写batch预处理，混入旧样本。
        核心修复：确保所有张量的维度正确匹配。
        """
        batch = super().preprocess_batch(batch)
        
        # 确保batch中的核心张量都在正确的设备上
        batch['cls'] = batch['cls'].to(self.device)
        batch['bboxes'] = batch['bboxes'].to(self.device)
        batch['batch_idx'] = batch['batch_idx'].to(self.device)
        
        if not self.replay_buffer or self.replay_ratio <= 0:
            batch['is_replay'] = torch.zeros(batch['img'].shape[0], dtype=torch.bool, device=self.device)
            return batch
        
        batch_size = batch['img'].shape[0]
        num_replay = int(batch_size * self.replay_ratio)
        
        if num_replay == 0:
            batch['is_replay'] = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            return batch
        
        replace_indices = random.sample(range(batch_size), min(num_replay, batch_size))
        is_replay = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        is_replay[replace_indices] = True
        
        for idx in replace_indices:
            old_sample = random.choice(self.replay_buffer)
            
            # 加载旧图片
            img = Image.open(old_sample['img_path']).convert('RGB')
            img_size = batch['img'].shape[2]
            img = img.resize((img_size, img_size))
            img = np.array(img).transpose(2, 0, 1)
            img = torch.from_numpy(img).float() / 255.0
            batch['img'][idx] = img.to(self.device)
            
            # 加载旧标签
            labels = []
            with open(old_sample['label_path'], 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        bbox = [float(x) for x in parts[1:5]]
                        labels.append([cls_id] + bbox)
            
            if labels:
                labels_tensor = torch.tensor(labels, dtype=torch.float32, device=self.device)
                
                # 计算mask
                mask = (batch['batch_idx'] == idx).squeeze()
                
                if mask.any():
                    # --- 关键修复：确保维度匹配 ---
                    keep_mask = ~mask
                    
                    # 处理 cls
                    if batch['cls'].dim() == 1:
                        new_cls = torch.cat([batch['cls'][keep_mask], labels_tensor[:, 0]], dim=0)
                    else:
                        new_cls = torch.cat([batch['cls'][keep_mask], labels_tensor[:, 0:1]], dim=0)
                    
                    new_bboxes = torch.cat([batch['bboxes'][keep_mask], labels_tensor[:, 1:5]], dim=0)
                    
                    # 处理 batch_idx
                    new_batch_idx_vals = torch.full((labels_tensor.shape[0],), idx, 
                                                    dtype=batch['batch_idx'].dtype, device=self.device)
                    
                    if batch['batch_idx'].dim() == 1:
                        new_batch_idx = torch.cat([batch['batch_idx'][keep_mask], new_batch_idx_vals], dim=0)
                    else:
                        new_batch_idx = torch.cat([batch['batch_idx'][keep_mask], new_batch_idx_vals.unsqueeze(1)], dim=0)
                    
                    batch['cls'] = new_cls
                    batch['bboxes'] = new_bboxes
                    batch['batch_idx'] = new_batch_idx
                else:
                    # 如果该索引原本没有标签，直接添加
                    if batch['cls'].dim() == 1:
                        batch['cls'] = torch.cat([batch['cls'], labels_tensor[:, 0]], dim=0)
                    else:
                        batch['cls'] = torch.cat([batch['cls'], labels_tensor[:, 0:1]], dim=0)
                    
                    batch['bboxes'] = torch.cat([batch['bboxes'], labels_tensor[:, 1:5]], dim=0)
                    
                    new_batch_idx_vals = torch.full((labels_tensor.shape[0],), idx,
                                                    dtype=batch['batch_idx'].dtype, device=self.device)
                    
                    if batch['batch_idx'].dim() == 1:
                        batch['batch_idx'] = torch.cat([batch['batch_idx'], new_batch_idx_vals], dim=0)
                    else:
                        batch['batch_idx'] = torch.cat([batch['batch_idx'], new_batch_idx_vals.unsqueeze(1)], dim=0)
        
        batch['is_replay'] = is_replay

        # --- 新增：一致性训练的强弱增强处理 ---
        if self.enable_consistency:
            # 保存弱增强视图供教师模型使用
            batch['img_weak'] = batch['img'].clone()
            
            # 对学生模型的输入应用强增强
            # 注意：batch['img'] 是 (B, 3, H, W)，值域 [0, 1]
            try:
                # 应用强增强
                batch['img'] = self.strong_aug(batch['img'])
            except Exception as e:
                # 偶尔可能会因为设备问题失败，捕获异常
                pass

        return batch

    def _add_pseudo_labels(self, batch, teacher_preds):
        """利用教师模型生成伪标签并合并到Batch中"""
        # teacher_preds: (B, 4+NC, Anchors) -> (B, Anchors, 4+NC)
        preds_for_nms = teacher_preds.permute(0, 2, 1)
        
        # 仅对旧类别生成伪标签
        pseudo_results = non_max_suppression(
            preds_for_nms,
            conf_thres=self.pseudo_conf_threshold,
            iou_thres=0.7,
            classes=self.old_class_ids,
            multi_label=True
        )
        
        new_cls_list = []
        new_bboxes_list = []
        new_batch_idx_list = []
        
        h, w = batch['img'].shape[2:]
        
        for i, det in enumerate(pseudo_results):
            if len(det) == 0: continue
            
            # det: (n, 6) [x1, y1, x2, y2, conf, cls]
            # 转换坐标 xyxy -> xywh (normalized)
            bboxes = det[:, :4].clone()
            xywh = torch.zeros_like(bboxes)
            xywh[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2 / w
            xywh[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2 / h
            xywh[:, 2] = (bboxes[:, 2] - bboxes[:, 0]) / w
            xywh[:, 3] = (bboxes[:, 3] - bboxes[:, 1]) / h
            
            cls = det[:, 5:6]
            
            new_cls_list.append(cls)
            new_bboxes_list.append(xywh)
            # batch_idx 需要与现有格式匹配
            new_batch_idx_list.append(torch.full((len(det), 1), i, device=self.device))
            
        if new_cls_list:
            p_cls = torch.cat(new_cls_list, dim=0)
            p_bboxes = torch.cat(new_bboxes_list, dim=0)
            p_bidx = torch.cat(new_batch_idx_list, dim=0)
            
            # 拼接到 batch
            if batch['cls'].dim() == 1:
                 batch['cls'] = torch.cat([batch['cls'], p_cls.squeeze(-1)], dim=0)
            else:
                 batch['cls'] = torch.cat([batch['cls'], p_cls], dim=0)
                 
            batch['bboxes'] = torch.cat([batch['bboxes'], p_bboxes], dim=0)
            
            if batch['batch_idx'].dim() == 1:
                batch['batch_idx'] = torch.cat([batch['batch_idx'], p_bidx.squeeze(-1)], dim=0)
            else:
                batch['batch_idx'] = torch.cat([batch['batch_idx'], p_bidx], dim=0)

    def get_loss(self, preds, batch):
        """计算总损失，包括标准检测损失、知识蒸馏损失、伪标签损失和一致性损失。"""
        student_nc = self.data['nc']

        # 1. 准备教师模型输入 (如果启用一致性，使用弱增强视图)
        teacher_img = batch.get('img_weak', batch['img']) if self.enable_consistency else batch['img']
        
        with torch.no_grad():
            teacher_output = self.teacher_model(teacher_img)
            teacher_preds, teacher_feats = teacher_output[0], teacher_output[1]

        # 2. 生成伪标签并合并到 Batch (如果启用且阈值有效)
        if self.pseudo_conf_threshold > 0 and self.pseudo_conf_threshold < 1.0:
            self._add_pseudo_labels(batch, teacher_preds)

        if not hasattr(self, 'iou_loss'):
            self.iou_loss = loss.v8DetectionLoss(self.model).iou_loss

        # 3. 计算标准损失 (此时 batch 可能已包含伪标签)
        loss_gt, loss_items = super().get_loss(preds, batch)
        
        student_preds, student_feats = preds[0], preds[1]
        
        # 特征蒸馏损失
        loss_distill_feat = 0.0
        if student_feats and teacher_feats:
            for feat_s, feat_t in zip(student_feats, teacher_feats):
                loss_distill_feat += self.feat_loss(feat_s, feat_t)

        loss_distill_cls = 0.0
        loss_distill_reg = 0.0
        
        target_scores, _, _, _, _, fg_mask_gt = self.criterion.assigner(
            student_preds, (batch['cls'].view(-1, 1), batch['bboxes'].view(-1, 4)), 
            batch['batch_idx'].view(-1, 1), self.model
        )
        
        new_class_mask_gt = torch.zeros_like(fg_mask_gt)
        if fg_mask_gt.any():
            gt_cls_in_pred_space = target_scores[fg_mask_gt].argmax(-1)
            is_new_class = torch.isin(gt_cls_in_pred_space, torch.tensor(self.new_class_ids, device=self.device))
            new_class_mask_gt[fg_mask_gt] = is_new_class

        _, anchor_points, stride_tensor = self.model.head.make_anchors(student_preds, self.model.head.stride, 0.5)
        is_replay = batch.get('is_replay', torch.zeros(batch['img'].shape[0], dtype=torch.bool, device=self.device))

        for i, (pred_s, pred_t_raw) in enumerate(zip(student_preds, teacher_preds)):
            with torch.no_grad():
                pred_t_aligned = torch.zeros_like(pred_s)
                pred_t_aligned[..., :4] = pred_t_raw[..., :4]
                pred_t_aligned[..., 4:4 + self.num_old_classes] = pred_t_raw[..., 4:]
                
                pred_t_cls = pred_t_aligned[..., 4:].view(-1, student_nc)
                teacher_probs = F.softmax(pred_t_cls / self.temperature, dim=1)
                teacher_max_probs, teacher_max_ids = torch.max(pred_t_cls.softmax(dim=1), dim=1)
                fg_mask_teacher = teacher_max_probs > self.args.conf
            
            distill_mask = ~new_class_mask_gt
            batch_indices = torch.arange(pred_s.shape[0], device=self.device).view(-1, 1, 1).expand_as(pred_s[..., 0])
            is_replay_expanded = is_replay[batch_indices.view(-1)].view_as(fg_mask_teacher)
            
            fg_distill_mask = fg_mask_teacher & distill_mask
            bg_distill_mask = ~fg_mask_teacher & distill_mask

            loss_fg = 0.0
            if fg_distill_mask.any():
                weights = torch.ones_like(teacher_max_ids[fg_distill_mask], dtype=torch.float32)
                replay_boost = torch.where(
                    is_replay_expanded[fg_distill_mask],
                    torch.tensor(self.replay_distill_boost, device=self.device),
                    torch.tensor(1.0, device=self.device)
                )
                
                if self.class_weights:
                    for class_id, weight in self.class_weights.items():
                        weights[teacher_max_ids[fg_distill_mask] == class_id] = weight
                
                weights *= replay_boost
                cls_s_fg = F.log_softmax(pred_s[..., 4:].view(-1, student_nc)[fg_distill_mask] / self.temperature, dim=1)
                kl_div_fg = self.distill_cls_loss(cls_s_fg, teacher_probs[fg_distill_mask]).sum(dim=1)
                loss_fg = (kl_div_fg * weights).mean()

            loss_bg = 0.0
            if bg_distill_mask.any():
                replay_boost_bg = torch.where(
                    is_replay_expanded[bg_distill_mask],
                    torch.tensor(self.replay_distill_boost * 0.5, device=self.device),
                    torch.tensor(1.0, device=self.device)
                )
                cls_s_bg = F.log_softmax(pred_s[..., 4:].view(-1, student_nc)[bg_distill_mask] / self.temperature, dim=1)
                kl_div_bg = self.distill_cls_loss(cls_s_bg, teacher_probs[bg_distill_mask]).sum(dim=1)
                loss_bg = (kl_div_bg * replay_boost_bg).mean()

            loss_distill_cls += (loss_fg + self.distill_bg_weight * loss_bg) * (self.temperature ** 2)

            if fg_distill_mask.any():
                box_s = self.decode_bboxes(pred_s[..., :4], anchor_points[i], stride_tensor[i])
                box_t = self.decode_bboxes(pred_t_aligned[..., :4], anchor_points[i], stride_tensor[i])
                replay_boost_reg = torch.where(
                    is_replay_expanded[fg_distill_mask],
                    torch.tensor(self.replay_distill_boost, device=self.device),
                    torch.tensor(1.0, device=self.device)
                )
                iou = self.iou_loss(box_s[fg_distill_mask], box_t[fg_distill_mask])
                loss_distill_reg += (iou * replay_boost_reg).mean()

        # 应用一致性权重 (如果有)
        consistency_factor = self.consistency_weight if self.enable_consistency else 1.0
        
        total_distill_loss = (self.distill_cls_weight * loss_distill_cls +
                              self.distill_reg_weight * loss_distill_reg +
                              self.distill_feat_weight * loss_distill_feat) * consistency_factor
        
        total_loss = loss_gt + total_distill_loss
        new_loss_items = torch.tensor([loss_distill_cls, loss_distill_reg, loss_distill_feat], device=self.device)
        loss_items = torch.cat((loss_items, new_loss_items))
        
        return total_loss, loss_items

    def decode_bboxes(self, pred_dist, anchor_points, stride):
        box_preds = self.model.head.dfl(pred_dist)
        box_preds = box_preds * stride
        return xywh2xyxy(torch.cat((anchor_points - box_preds[..., :2], anchor_points + box_preds[..., 2:]), -1))

