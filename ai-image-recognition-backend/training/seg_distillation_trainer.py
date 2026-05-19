import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.utils import loss
from ultralytics.utils.ops import xywh2xyxy
import torchvision.transforms as T

class SegDistillationTrainer(SegmentationTrainer):
    """
    一个用于图像分割的知识蒸馏训练器 (最终修正版)。
    - 解决了因教师/学生模型类别空间不一致导致的标签混淆问题。
    - 采用“解耦蒸馏”策略，避免新旧知识学习冲突。
    - 修正了初始化顺序问题。
    - 集成一致性正则化（强弱数据增强）。
    """
    def __init__(self, *args, **kwargs):
        # --- 1. 拦截并移除自定义参数 ---
        overrides = kwargs.get('overrides', {})
        self.teacher_model = overrides.pop('teacher_model', None)
        self.distill_cls_weight = overrides.pop('distill_cls_weight', 1.0)
        self.distill_reg_weight = overrides.pop('distill_reg_weight', 2.0)
        self.distill_feat_weight = overrides.pop('distill_feat_weight', 5.0)
        self.distill_mask_weight = overrides.pop('distill_mask_weight', 1.0)
        self.temperature = overrides.pop('temperature', 2.0)
        self.class_weights = overrides.pop('class_weights', {})
        self.distill_bg_weight = overrides.pop('distill_bg_weight', 0.05)
        self.new_class_ids = overrides.pop('new_class_ids', [])
        
        # 新增：一致性参数
        self.enable_consistency = overrides.pop('enable_consistency', False)
        self.consistency_weight = overrides.pop('consistency_weight', 1.0)

        # 新增：旧样本回放参数 (拦截并保存，避免传递给父类导致报错)
        self.old_data_yaml = overrides.pop('old_data_yaml', None)
        self.replay_ratio = overrides.pop('replay_ratio', 0.0)
        self.replay_distill_boost = overrides.pop('replay_distill_boost', 1.0)
        self.max_replay_samples = overrides.pop('max_replay_samples', 1000)

        # --- 核心修正：拦截 names 参数 ---
        self.custom_names = overrides.pop('names', None)

        if self.teacher_model is None:
            raise ValueError("SegDistillationTrainer requires a 'teacher_model' argument.")

        # --- 2. 调用父类构造函数 ---
        # self.nc 等属性在此之后才会被完全设置
        super().__init__(*args, **kwargs)

        # 定义强增强变换 (仅光度变换，不改变几何坐标)
        self.strong_aug = T.Compose([
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        ])

        # --- 核心修正：在父类初始化后，强制设置 names ---
        if self.custom_names:
            # self.data 是在 super().__init__ 中创建的
            if self.data is None:
                self.data = {}
            self.data['names'] = self.custom_names
            # 同时更新模型对象中的 names，确保一致性
            self.model.names = self.custom_names

        # --- 3. 继续进行原有的初始化 ---
        self.teacher_model.to(self.device)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        self.num_old_classes = self.teacher_model.nc
        
        # 设置一个标志位，用于延迟初始化
        self.distill_initialized = False

    def _initialize_distillation(self):
        """
        在第一次调用 get_loss 时执行的延迟初始化。
        此时 self.nc 和其他训练参数已准备就绪。
        """
        print("✅ 分割任务蒸馏训练器(最终修正版)初始化成功。")
        print(f"   - 教师/学生类别数: {self.num_old_classes} -> {self.nc}")
        if self.new_class_ids:
            print(f"   - 已识别新类别ID: {self.new_class_ids}")

        self.distill_cls_loss = nn.KLDivLoss(reduction='none') 
        self.feat_loss = nn.MSELoss()
        self.mask_loss = nn.BCEWithLogitsLoss()
        
        # 初始化完成，设置标志位为 True
        self.distill_initialized = True

    def preprocess_batch(self, batch):
        """
        重写batch预处理，支持一致性训练（强弱增强）。
        """
        batch = super().preprocess_batch(batch)
        
        # 一致性训练：强弱增强
        if self.enable_consistency:
            # 保存弱增强视图供教师模型使用
            batch['img_weak'] = batch['img'].clone()
            try:
                # 对学生模型的输入应用强增强
                batch['img'] = self.strong_aug(batch['img'])
            except Exception:
                pass
        return batch

    def get_loss(self, preds, batch):
        # --- 核心修正：执行延迟初始化 ---
        if not self.distill_initialized:
            self._initialize_distillation()

        # --- 1. 计算标准GT损失 ---
        if not hasattr(self, 'criterion'):
            self.criterion = loss.v8SegmentationLoss(self.model)
        loss_gt, loss_items = self.criterion(preds, batch)

        # --- 2. 准备学生和教师的输出 ---
        student_det_preds, student_seg_protos = preds[0], preds[1]
        student_feats = preds[2] if len(preds) > 2 else None

        # 准备教师模型输入 (如果启用一致性，使用弱增强视图)
        teacher_img = batch.get('img_weak', batch['img']) if self.enable_consistency else batch['img']

        with torch.no_grad():
            teacher_output = self.teacher_model(teacher_img)
            teacher_det_preds, teacher_seg_protos = teacher_output[0], teacher_output[1]
            teacher_feats = teacher_output[2] if len(teacher_output) > 2 else None

        # --- 3. 计算特征和掩码蒸馏损失 (全局) ---
        loss_distill_feat = sum(self.feat_loss(s, t) for s, t in zip(student_feats, teacher_feats)) if student_feats and teacher_feats else 0.0
        loss_distill_mask = self.mask_loss(student_seg_protos, teacher_seg_protos)

        # --- 4. 计算解耦的检测头蒸馏损失 (分类+回归) ---
        loss_distill_cls = 0.0
        loss_distill_reg = 0.0
        
        target_scores, _, _, _, _, fg_mask_gt = self.criterion.assigner(
            student_det_preds, (batch['cls'].view(-1, 1), batch['bboxes'].view(-1, 4)), 
            batch['batch_idx'].view(-1, 1), self.model
        )
        
        new_class_mask_gt = torch.zeros_like(fg_mask_gt)
        if fg_mask_gt.any():
            gt_cls_in_pred_space = target_scores[fg_mask_gt].argmax(-1)
            is_new_class = torch.isin(gt_cls_in_pred_space, torch.tensor(self.new_class_ids, device=self.device))
            new_class_mask_gt[fg_mask_gt] = is_new_class

        _, anchor_points, stride_tensor = self.model.head.make_anchors(student_det_preds, self.model.head.stride, 0.5)

        for i, (pred_s, pred_t_raw) in enumerate(zip(student_det_preds, teacher_det_preds)):
            cls_offset = 4 + self.nm 
            
            with torch.no_grad():
                pred_t_aligned = torch.zeros_like(pred_s)
                pred_t_aligned[..., :cls_offset] = pred_t_raw[..., :cls_offset]
                pred_t_aligned[..., cls_offset:cls_offset + self.num_old_classes] = pred_t_raw[..., cls_offset:]
                
                pred_t_cls = pred_t_aligned[..., cls_offset:].view(-1, self.nc)
                teacher_probs = F.softmax(pred_t_cls / self.temperature, dim=1)
                teacher_max_probs, teacher_max_ids = torch.max(pred_t_cls.softmax(dim=1), dim=1)
                fg_mask_teacher = teacher_max_probs > self.args.conf
            
            distill_mask = ~new_class_mask_gt
            fg_distill_mask = fg_mask_teacher & distill_mask
            bg_distill_mask = ~fg_mask_teacher & distill_mask

            loss_fg = 0.0
            if fg_distill_mask.any():
                weights = torch.ones_like(teacher_max_ids[fg_distill_mask], dtype=torch.float32)
                if self.class_weights:
                    for class_id, weight in self.class_weights.items():
                        weights[teacher_max_ids[fg_distill_mask] == class_id] = weight
                cls_s_fg = F.log_softmax(pred_s[..., cls_offset:].view(-1, self.nc)[fg_distill_mask] / self.temperature, dim=1)
                kl_div_fg = self.distill_cls_loss(cls_s_fg, teacher_probs[fg_distill_mask]).sum(dim=1)
                loss_fg = (kl_div_fg * weights).mean()

            loss_bg = 0.0
            if bg_distill_mask.any():
                cls_s_bg = F.log_softmax(pred_s[..., cls_offset:].view(-1, self.nc)[bg_distill_mask] / self.temperature, dim=1)
                loss_bg = self.distill_cls_loss(cls_s_bg, teacher_probs[bg_distill_mask]).sum(dim=1).mean()

            loss_distill_cls += (loss_fg + self.distill_bg_weight * loss_bg) * (self.temperature ** 2)

            if fg_distill_mask.any():
                box_s = self.decode_bboxes(pred_s[..., :4], anchor_points[i], stride_tensor[i])
                box_t = self.decode_bboxes(pred_t_aligned[..., :4], anchor_points[i], stride_tensor[i])
                iou = self.criterion.iou_loss(box_s[fg_distill_mask], box_t[fg_distill_mask])
                loss_distill_reg += iou.mean()

        # --- 5. 合并所有损失 ---
        # 应用一致性权重 (如果有)
        consistency_factor = self.consistency_weight if self.enable_consistency else 1.0

        total_distill_loss = (self.distill_cls_weight * loss_distill_cls +
                              self.distill_reg_weight * loss_distill_reg +
                              self.distill_feat_weight * loss_distill_feat +
                              self.distill_mask_weight * loss_distill_mask) * consistency_factor
        
        total_loss = loss_gt + total_distill_loss

        new_loss_items = torch.tensor([loss_distill_cls, loss_distill_reg, loss_distill_feat, loss_distill_mask], device=self.device)
        loss_items = torch.cat((loss_items, new_loss_items))
        
        return total_loss, loss_items

    def decode_bboxes(self, pred_dist, anchor_points, stride):
        if not hasattr(self.criterion, 'dfl'):
            self.criterion.dfl = loss.DFL(16)
        box_preds = self.criterion.dfl(pred_dist) * stride
        return xywh2xyxy(torch.cat((anchor_points - box_preds[..., :2], anchor_points + box_preds[..., 2:]), -1))