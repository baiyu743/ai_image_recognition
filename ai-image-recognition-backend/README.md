# Vision model iterate hub - 后端服务

基于 FastAPI 和 YOLOv8 构建的高性能 AI 后端服务，支持目标检测、图像分割和分类任务的训练、推理与评估。

## 快速启动

**前提条件**:
- Python 3.9+
- CUDA（可选，仅在 Linux/Windows + NVIDIA GPU 场景需要）
- macOS（Apple Silicon/Intel）建议先安装 Homebrew，并安装构建依赖：`brew install cmake`

```bash
# 1. 进入后端目录
cd ai-image-recognition-backend

# 2. 安装依赖
# Windows:
pip install -r requirements.txt
# macOS/Linux:
pip3 install -r requirements.txt

# 3. 启动服务
# Windows:
uvicorn main:app --reload --host 0.0.0.0 --port 8000
# macOS/Linux 推荐使用 python3 -m 方式，避免出现 “command not found: uvicorn”
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

启动后访问 API 文档:
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 核心功能

- **多模式训练系统**:
  - **常规训练**: 标准 YOLOv8 训练流程
  - **增量训练**: 基于现有模型进行新类别或新数据的增量学习 (支持旧样本回放)
  - **知识蒸馏**: 支持教师-学生模型蒸馏，包含分类/回归/特征损失及一致性训练
  - **冻结策略**: 分阶段解冻训练，优化微调效果
- **智能辅助标注**: 提供自动化的 BBox、Polygon 和 Keypoint 标注接口
- **任务管理**: 异步训练任务调度，支持任务状态查询、日志实时获取和取消任务
- **项目化数据闭环**: 围绕 project_id 管理数据集、训练产物、评估结果，形成“创建项目 → 标注导入 → 训练 → 评估”的闭环

## 推荐使用流程（与前端对应）

1. 创建项目：`POST /api/projects`（生成 `projects/<project_id>/...` 目录结构）
2. 图像标注：前端上传图片并标注；完成后“一键导入到项目”
3. 数据集落盘：`POST /api/projects/{project_id}/dataset/from-annotations`（生成 YOLO 数据集与 `dataset.yaml`，写入 `meta.json`）
4. 模型训练：`POST /api/training/*`（训练结果写入 `projects/<project_id>/training/...`）
5. 评估优化：读取训练产物 `evaluation_data.json`，再发起评估 `POST /api/evaluation/...`，评估结果写入 `projects/<project_id>/evaluation/eval_*`

---

## 主要 API 接口

### 模型训练 (Training)
| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/training/regular` | 启动常规训练 |
| POST | `/api/training/incremental` | 启动增量训练 (支持新类别) |
| POST | `/api/training/distillation` | 启动知识蒸馏训练 |
| POST | `/api/training/freeze-strategy` | 启动冻结策略训练 |
| GET | `/api/training/tasks` | 获取所有训练任务列表 |
| GET | `/api/training/tasks/{id}/logs` | 获取实时训练日志 |
| GET | `/api/training/select-local-file` | 打开本地文件选择器 |
| POST | `/api/training/tasks/{id}/cancel` | 取消训练任务（前端“取消训练”按钮） |

### 图像标注 (Annotation)
| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/auto_annotate` | 上传图片并获取 AI 标注结果 |
| POST | `/api/projects/{project_id}/auto_annotate/batch` | 批量自动标注（项目级） |

### 项目与数据集 (Projects / Dataset)
| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/projects` | 创建项目 |
| GET | `/api/projects` | 项目列表 |
| GET | `/api/projects/{project_id}` | 获取项目详情 |
| POST | `/api/projects/{project_id}/dataset/from-annotations` | 将标注会话数据导入到项目数据集（生成 YOLO） |
| GET | `/api/projects/{project_id}/dataset/state` | 获取项目数据集图片列表（可选缩略图） |

### 评估优化 (Evaluation)
| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/api/projects/{project_id}/artifacts/evaluation-data` | 获取最新 `evaluation_data.json` |
| POST | `/api/projects/{project_id}/artifacts/evaluation-data/regenerate` | 重新生成 `evaluation_data.json`（修复历史产物） |
| GET | `/api/projects/{project_id}/evaluations` | 项目评估记录列表 |
| GET | `/api/projects/{project_id}/evaluations/{evaluation_id}` | 获取评估详情（metrics/report） |

## 数据与目录结构（项目化）

默认项目根目录是 `PROJECTS_ROOT`（未设置时默认为后端目录下的 `projects/`）：

```
projects/
└── <project_id>/
    ├── meta.json                 # 项目元信息（dataset_yaml_path 等）
    ├── dataset.yaml              # YOLO 数据集配置（train/val path、names）
    ├── data/
    │   ├── images/
    │   │   ├── train/
    │   │   └── val/
    │   └── labels/
    │       ├── train/
    │       └── val/
    ├── training/
    │   └── <run>/
    │       ├── weights/best.pt
    │       └── evaluation_data.json
    └── evaluation/
        └── eval_<evaluation_id>/
            ├── metrics.json
            └── analysis_report.md
```

---

## 项目结构（代码）

```
ai-image-recognition-backend/
├── main.py                 # FastAPI 入口
├── training/               # 训练核心模块
│   ├── api.py              # 训练相关 API 路由
│   ├── training_service.py # 任务调度服务
│   ├── incremental_newtrain.py # 增量训练逻辑
│   ├── distillation_trainer.py # 知识蒸馏逻辑
│   └── enhanced_training.py    # 常规/冻结训练逻辑
├── ai_models.py            # YOLO 模型推理封装
├── database.py             # SQLite 数据库连接
└── uploads/                # 临时文件存储
```

## 配置与环境变量

| 变量/配置 | 默认值 | 说明 |
|---|---|---|
| `PROJECTS_ROOT` | `projects` | 项目数据根目录（建议生产环境设置到持久化磁盘） |
| `DATABASE_URL` | 未设置时使用本地 sqlite | 数据库连接串（见 `database.py`） |
| `DASHSCOPE_API_KEY` | 空 | 增广/LLM 接口 Key（也支持 `.augmentation_api_key` 文件） |
| `OPENAI_API_KEY` | 空 | 评估报告 LLM（OpenAI 兼容） |
| `OPENAI_BASE_URL` | 空 | OpenAI 兼容 Base URL |
| `AUGMENTATION_*` | 见 `augmentation/config.py` | 增广模型、URL、预设等 |

增广 Key 的本地配置方式（任选一种）：
- 环境变量：`export DASHSCOPE_API_KEY=...`
- 文件：在后端根目录放 `.augmentation_api_key`（第一行写 key）
- `.env/.env.production/.env.local`：后端启动时会自动读取（不覆盖已设置环境变量）

---

## 常见问题排查

### 1) 评估结果全为 0 / `ground_truths` 为 0

典型原因：
- 验证集划分到了未标注图片，或 `labels/val` 下的 txt 为空
- 历史训练产物 `evaluation_data.json` 使用了错误的 `image_id`（如 `image0.jpg`），导致标签匹配失败

建议处理：
1. 回到前端“图像标注 → 导入到项目”，确保验证集图片有标注（`data/labels/val/*.txt` 非空）
2. 对已有训练产物执行一次重建（无需重训）：
   - `POST /api/projects/{project_id}/artifacts/evaluation-data/regenerate`

### 2) macOS 安装依赖失败（如 onnxruntime-gpu / onnx-simplifier）

- macOS 不支持 `onnxruntime-gpu`（无 CUDA），请使用 CPU 版本
- 部分包需要 `cmake`：`brew install cmake`

---

## 技术栈

- **Web Framework**: FastAPI
- **Deep Learning**: PyTorch, Ultralytics YOLOv8
- **Database**: SQLite, SQLAlchemy
- **Task Queue**: Python `threading` & `asyncio` (用于异步训练任务)

## 端口与运行

- 默认端口：`8000`
- 前端默认通过 `src/config/api.js` 访问 `http://localhost:8000`
