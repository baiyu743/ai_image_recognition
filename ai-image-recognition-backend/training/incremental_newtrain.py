import os
import yaml
import shutil
import tempfile
import random
from pathlib import Path
from ultralytics import YOLO

class IncrementalTrainer:
    """
    一个用于管理和执行YOLOv8增量训练的辅助类 (最终修复版 v6)。
    - 支持 Detect, Segment 和 Classify 任务。
    - 彻底重构了混合数据集的创建逻辑，从根本上分离了新旧数据的处理流程。
    """
    def __init__(self, existing_model_path, old_data_yaml, new_data_yaml, task):
        self.existing_model_path = Path(existing_model_path)
        self.old_data_yaml = Path(old_data_yaml)
        self.new_data_yaml = Path(new_data_yaml)
        self.task = task

        self.old_names = []
        self.new_names = []
        self.combined_names = []
        self.new_class_ids = []
        self.new_to_combined_map = {}

        self.temp_dir = Path(tempfile.mkdtemp())
        print(f"ℹ️  增量训练器已初始化，临时目录: {self.temp_dir}")

    def analyze_changes(self):
        """分析类别差异并创建ID映射。"""
        print("\n--- 正在分析数据集变化 ---")
        
        # --- 分类任务特殊处理 ---
        if self.task == 'classify':
            self._analyze_changes_classify()
            return self

        with self.old_data_yaml.open('r', encoding='utf-8') as f:
            old_data = yaml.safe_load(f)
        with self.new_data_yaml.open('r', encoding='utf-8') as f:
            new_data = yaml.safe_load(f)

        self.old_names = old_data.get('names', [])
        self.new_names = new_data.get('names', [])
        
        self.combined_names = list(self.old_names)
        added_classes = [name for name in self.new_names if name not in self.combined_names]
        self.combined_names.extend(added_classes)
        
        self.new_class_ids = [self.combined_names.index(name) for name in added_classes]
        for i, name in enumerate(self.new_names):
            self.new_to_combined_map[i] = self.combined_names.index(name)
        
        print(f"  - 总类别数: {len(self.combined_names)}")
        if added_classes:
            print(f"  - 新增类别: {added_classes}")
        print(f"  - 标签重映射规则 (新ID -> 合并后ID): {self.new_to_combined_map}")
        return self

    def _analyze_changes_classify(self):
        """分类任务的类别分析逻辑"""
        def get_cls_names(path_input):
            p = Path(path_input)
            train_dir = None
            # 情况1: 输入是 yaml 文件
            if p.is_file() and p.suffix in ['.yaml', '.yml']:
                with p.open('r', encoding='utf-8') as f:
                    d = yaml.safe_load(f)
                    root = Path(d.get('path', p.parent))
                    # 尝试寻找 train 目录
                    if 'train' in d:
                        train_dir = root / d['train']
            # 情况2: 输入是目录
            elif p.is_dir():
                if (p / 'train').exists():
                    train_dir = p / 'train'
                else:
                    train_dir = p # 假设直接指向了包含类别文件夹的目录
            
            if train_dir and train_dir.exists():
                return sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
            return []

        self.old_names = get_cls_names(self.old_data_yaml)
        self.new_names = get_cls_names(self.new_data_yaml)
        
        # 合并类别名称 (去重并排序)
        self.combined_names = sorted(list(set(self.old_names + self.new_names)))
        
        print(f"  - [分类] 旧数据集类别: {len(self.old_names)} 个")
        print(f"  - [分类] 新数据集类别: {len(self.new_names)} 个")
        print(f"  - [分类] 合并后总类别: {len(self.combined_names)} 个")
        print(f"  - 类别列表: {self.combined_names}")

    def _create_remapped_new_dataset_mirror(self):
        """为新数据集创建一个包含重映射标签的临时镜像。"""
        print("\n--- 正在为新数据集创建重映射镜像 ---")
        with self.new_data_yaml.open('r', encoding='utf-8') as f:
            new_data_config = yaml.safe_load(f)

        original_base_path = Path(new_data_config.get('path', self.new_data_yaml.parent))
        mirror_base_path = self.temp_dir / 'new_data_mirror'
        
        remapped_config = new_data_config.copy()
        remapped_config['path'] = str(mirror_base_path.resolve())

        for split in ['train', 'val', 'test']:
            if split not in new_data_config:
                continue

            original_img_dir = (original_base_path / str(new_data_config[split])).resolve()
            mirror_img_dir = (mirror_base_path / str(new_data_config[split])).resolve()
            mirror_label_dir = Path(str(mirror_img_dir).replace('images', 'labels'))
            mirror_img_dir.mkdir(parents=True, exist_ok=True)
            mirror_label_dir.mkdir(parents=True, exist_ok=True)
            original_label_dir = Path(str(original_img_dir).replace('images', 'labels'))

            image_files = list(original_img_dir.glob('*.jpg')) + list(original_img_dir.glob('*.png'))
            
            for img_path in image_files:
                symlink_path = mirror_img_dir / img_path.name
                if symlink_path.exists(): continue
                try:
                    os.symlink(img_path, symlink_path)
                except (OSError, AttributeError):
                    shutil.copy2(img_path, symlink_path)

                original_label_path = original_label_dir / (img_path.stem + '.txt')
                remapped_label_path = mirror_label_dir / (img_path.stem + '.txt')
                if original_label_path.exists():
                    with original_label_path.open('r') as fr, remapped_label_path.open('w') as fw:
                        for line in fr:
                            parts = line.strip().split()
                            try:
                                original_id = int(parts[0])
                                remapped_id = self.new_to_combined_map.get(original_id)
                                if remapped_id is not None:
                                    parts[0] = str(remapped_id)
                                    fw.write(' '.join(parts) + '\n')
                            except (ValueError, IndexError):
                                continue
        
        remapped_yaml_path = self.temp_dir / 'new_data_remapped.yaml'
        with remapped_yaml_path.open('w', encoding='utf-8') as f:
            yaml.dump(remapped_config, f, allow_unicode=True)
        
        print(f"✅ 新数据集的重映射镜像已创建完毕。")
        return remapped_yaml_path

    def _create_mixed_dataset(self, old_data_ratio=0.2):
        """
        使用原始旧数据集和重映射后的新数据集镜像来创建最终的混合数据集。
        """
        # --- 分类任务特殊处理 ---
        if self.task == 'classify':
            return self._create_mixed_dataset_classify(old_data_ratio)

        remapped_new_yaml = self._create_remapped_new_dataset_mirror()
        
        print(f"\n--- 正在创建最终混合数据集 (旧数据比例: {old_data_ratio}) ---")
        
        mixed_train_path = self.temp_dir / 'train.txt'
        mixed_val_path = self.temp_dir / 'val.txt'
        
        # --- 核心修正：分离新旧数据处理逻辑 ---

        # 1. 处理旧数据集
        with self.old_data_yaml.open('r', encoding='utf-8') as f:
            old_data = yaml.safe_load(f)
        old_base_path = Path(old_data.get('path', self.old_data_yaml.parent))
        if 'train' in old_data:
            old_img_dir = (old_base_path / str(old_data['train'])).resolve()
            old_image_files = sorted(list(old_img_dir.glob('*.jpg')) + list(old_img_dir.glob('*.png')))
            num_to_sample = int(len(old_image_files) * old_data_ratio)
            sampled_files = old_image_files[:num_to_sample]
            with mixed_train_path.open('w', encoding='utf-8') as f: # 'w' for overwrite
                for img_path in sampled_files:
                    f.write(str(img_path) + '\n')
            print(f"  - 已向 train.txt 添加 {len(sampled_files)} 张来自【旧数据集】的图片。")

        if 'val' in old_data:
            old_val_dir = (old_base_path / str(old_data['val'])).resolve()
            old_val_files = sorted(list(old_val_dir.glob('*.jpg')) + list(old_val_dir.glob('*.png')))
            with mixed_val_path.open('w', encoding='utf-8') as f: # 'w' for overwrite
                for img_path in old_val_files:
                    f.write(str(img_path) + '\n')
            print(f"  - 已向 val.txt 添加 {len(old_val_files)} 张来自【旧数据集】的图片。")

        # 2. 处理重映射后的新数据集
        with remapped_new_yaml.open('r', encoding='utf-8') as f:
            new_data_remapped = yaml.safe_load(f)
        new_base_path = Path(new_data_remapped['path']) # 必须有 'path'
        
        # 处理新训练集
        if 'train' in new_data_remapped:
            new_img_dir = (new_base_path / str(new_data_remapped['train'])).resolve()
            new_image_files = sorted(list(new_img_dir.glob('*.jpg')) + list(new_img_dir.glob('*.png')))
            with mixed_train_path.open('a', encoding='utf-8') as f: # 'a' for append
                for img_path in new_image_files:
                    f.write(str(img_path) + '\n')
            print(f"  - 已向 train.txt 添加 {len(new_image_files)} 张来自【新数据集镜像】的图片。")

        # 处理新验证集
        if 'val' in new_data_remapped:
            new_val_img_dir = (new_base_path / str(new_data_remapped['val'])).resolve()
            new_val_files = sorted(list(new_val_img_dir.glob('*.jpg')) + list(new_val_img_dir.glob('*.png')))
            mode = 'a' if mixed_val_path.exists() else 'w'
            with mixed_val_path.open(mode, encoding='utf-8') as f:
                for img_path in new_val_files:
                    f.write(str(img_path) + '\n')
            print(f"  - 已向 val.txt 添加 {len(new_val_files)} 张来自【新数据集镜像】的图片。")

        # --- 结束修正 ---

        mixed_yaml_path = self.temp_dir / 'mixed_data.yaml'
        mixed_data_config = {
            'train': str(mixed_train_path.resolve()),
            'val': str(mixed_val_path.resolve()),
            'nc': len(self.combined_names),
            'names': self.combined_names
        }
        with mixed_yaml_path.open('w', encoding='utf-8') as f:
            yaml.dump(mixed_data_config, f, allow_unicode=True)
        
        print(f"✅ 最终混合数据集配置文件已生成: {mixed_yaml_path}")
        return mixed_yaml_path

    def _create_mixed_dataset_classify(self, old_data_ratio):
        """为分类任务创建混合数据集 (通过符号链接合并文件夹)"""
        print(f"\n--- 正在创建混合分类数据集 (旧数据比例: {old_data_ratio}) ---")
        mixed_root = self.temp_dir / 'mixed_cls_data'
        mixed_root.mkdir(parents=True, exist_ok=True)
        
        for split in ['train', 'val', 'test']:
            target_split_dir = mixed_root / split
            target_split_dir.mkdir(exist_ok=True)
            
            # 内部函数：处理单个数据源
            def process_source(source_path, is_old_data=False):
                p = Path(source_path)
                src_dir = None
                if p.is_file() and p.suffix in ['.yaml', '.yml']:
                    with p.open('r', encoding='utf-8') as f:
                        d = yaml.safe_load(f)
                        root = Path(d.get('path', p.parent))
                        if split in d:
                            src_dir = root / d[split]
                        elif split == 'train' and 'train' not in d: # 兼容只有根目录的情况
                             src_dir = root
                elif p.is_dir():
                    src_dir = p / split if (p/split).exists() else (p if split == 'train' else None)
                
                if not src_dir or not src_dir.exists():
                    return

                # 遍历类别文件夹
                for class_dir in src_dir.iterdir():
                    if not class_dir.is_dir(): continue
                    
                    class_name = class_dir.name
                    target_class_dir = target_split_dir / class_name
                    target_class_dir.mkdir(exist_ok=True)
                    
                    images = list(class_dir.glob('*.*'))
                    images = [x for x in images if x.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']]
                    
                    if is_old_data and split == 'train':
                        # 旧数据采样
                        k = int(len(images) * old_data_ratio)
                        selected = random.sample(images, k) if k < len(images) else images
                    else:
                        selected = images
                    
                    for img in selected:
                        # 创建符号链接，文件名前加前缀避免冲突
                        prefix = 'old_' if is_old_data else 'new_'
                        dst = target_class_dir / f"{prefix}{img.name}"
                        try:
                            os.symlink(img, dst)
                        except (OSError, AttributeError):
                            shutil.copy2(img, dst)
            
            # 处理旧数据
            process_source(self.old_data_yaml, is_old_data=True)
            # 处理新数据
            process_source(self.new_data_yaml, is_old_data=False)
            
        print(f"✅ 混合分类数据集已创建: {mixed_root}")
        return mixed_root

    def _adapt_model(self):
        print("\n--- 正在调整模型以适应新类别 ---")
        model = YOLO(self.existing_model_path)
        
        # 检查是否需要重置头部
        # 注意：对于分类任务，即使类别数相同，如果类别名称顺序变了，理论上也需要调整，
        # 但这里主要关注类别数量变化或显式的新增类别。
        if len(self.combined_names) != len(self.old_names) or self.task == 'classify':
            if self.task == 'classify':
                self._reset_cls_head(model, len(self.combined_names))
            else:
                # 检测和分割任务可以使用内置的 reset_head
                try:
                    model.model.reset_head(nc=len(self.combined_names))
                except AttributeError:
                    print("⚠️ 警告: 该模型不支持 reset_head，尝试手动调整...")
            
            print(f"✅ 模型头部已重置，新的类别数: {len(self.combined_names)}")
        return model

    def _reset_cls_head(self, model, nc):
        """手动重置分类模型的头部 (Linear层)"""
        import torch
        import torch.nn as nn
        
        # 获取分类头层 (通常是最后一层 Classify)
        # model.model 是 ClassificationModel
        # model.model.model 是 Sequential
        c_layer = model.model.model[-1]
        
        # 检查是否是 Classify 模块且包含 linear 层
        if hasattr(c_layer, 'linear') and isinstance(c_layer.linear, nn.Linear):
            old_linear = c_layer.linear
            new_linear = nn.Linear(old_linear.in_features, nc)
            
            # 尝试保留旧权重
            print("   - 正在迁移旧类别权重...")
            with torch.no_grad():
                # 获取当前模型的类别名称映射
                current_names = model.names
                
                for old_idx, old_name in current_names.items():
                    if old_name in self.combined_names:
                        new_idx = self.combined_names.index(old_name)
                        if old_idx < old_linear.out_features and new_idx < nc:
                            new_linear.weight[new_idx] = old_linear.weight[old_idx]
                            new_linear.bias[new_idx] = old_linear.bias[old_idx]
            
            # 替换线性层
            c_layer.linear = new_linear
            # 移动到正确设备
            c_layer.linear.to(next(model.parameters()).device)
            
            # 更新模型属性
            model.model.nc = nc
            # 更新名称映射
            new_names_dict = {i: n for i, n in enumerate(self.combined_names)}
            model.model.names = new_names_dict
            # model.names 是一个 property，不能直接赋值，它会读取 model.model.names
            # 所以只需要更新 model.model.names 即可
            # model.names = new_names_dict 
        else:
            print("⚠️ 警告: 未找到标准的分类头线性层，跳过重置。")

    def _adjust_freeze(self, model, args_dict):
        """
        检查并调整冻结层数，防止冻结整个模型（特别是分类模型）。
        """
        if 'freeze' in args_dict:
            freeze_val = args_dict['freeze']
            # 检查 model.model.model 是否存在且有长度
            if hasattr(model.model, 'model') and hasattr(model.model.model, '__len__'):
                num_layers = len(model.model.model)
                # 如果冻结层数 >= 总层数，说明头部也被冻结了
                if freeze_val >= num_layers:
                    new_freeze = max(0, num_layers - 1)
                    print(f"⚠️ 警告: 请求冻结 {freeze_val} 层，但模型只有 {num_layers} 层。")
                    print(f"   自动调整冻结层数为 {new_freeze} 以确保头部可训练。")
                    args_dict['freeze'] = new_freeze

    def train(self, **kwargs):
        self.analyze_changes()
        model = self._adapt_model()
        data_yaml = self._create_mixed_dataset(kwargs.pop('old_data_ratio', 0.2))
        
        # 自动调整冻结参数
        self._adjust_freeze(model, kwargs)
        
        # Windows下减少workers以防止页面文件错误
        if os.name == 'nt' and 'workers' not in kwargs:
            kwargs['workers'] = 2
            print("ℹ️  Windows环境检测: 自动设置 workers=2 以防止内存错误")
        
        model.train(data=str(data_yaml), **kwargs)

    def train_two_stage(self, stage1_args, stage2_args, old_data_ratio, project, name):
        self.analyze_changes()
        model = self._adapt_model()
        data_yaml = self._create_mixed_dataset(old_data_ratio)
        
        # 自动调整第一阶段冻结参数
        self._adjust_freeze(model, stage1_args)
        
        # Windows下减少workers以防止页面文件错误
        if os.name == 'nt':
            if 'workers' not in stage1_args:
                stage1_args['workers'] = 2
                print("ℹ️  Windows环境检测: 自动设置 Stage 1 workers=2")
            if 'workers' not in stage2_args:
                stage2_args['workers'] = 2
                print("ℹ️  Windows环境检测: 自动设置 Stage 2 workers=2")
        
        print("\n" + "="*20 + " 🚀 开始第一阶段训练 " + "="*20)
        model.train(data=str(data_yaml), project=project, name=name, **stage1_args)
        print("\n" + "="*20 + " 🚀 开始第二阶段训练 " + "="*20)
        last_weights = Path(project) / name / 'weights' / 'last.pt'
        model_stage2 = YOLO(last_weights)
        model_stage2.train(data=str(data_yaml), project=project, name=name, **stage2_args)

    def __del__(self):
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"⚠️ 清理临时目录时出错: {e}")