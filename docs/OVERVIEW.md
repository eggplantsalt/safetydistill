# VLSA-AEGIS 技术概览

> 本文档用于帮助新成员快速理解项目结构、运行流程与二次开发路径。缺失信息已用 [需补充] 标注。

## 第一部分：项目架构分析

### 1. 项目整体架构

**目录结构（tree）**
```
.
├─ LICENSE
├─ README.md
├─ requirements.txt
├─ main_demo_dummy.py
├─ main/
│  ├─ main_aegis.py
│  ├─ main_aegis_translational.py
│  ├─ requirements.txt
│  └─ utils.py
├─ scripts/
│  ├─ serve_policy.py
│  ├─ train.py
│  ├─ train_pytorch.py
│  ├─ train_test.py
│  └─ compute_norm_stats.py
├─ openpi/
│  ├─ README.md
│  ├─ pyproject.toml
│  ├─ docs/
│  ├─ examples/
│  ├─ packages/
│  ├─ scripts/
│  └─ src/
└─ safelibero/
   ├─ README.md
   ├─ requirements.txt
   ├─ setup.py
   ├─ benchmark_scripts/
   ├─ notebooks/
   ├─ scripts/
   └─ templates/
```

**核心模块与职责**
- AEGIS 评测入口：运行 SafeLIBERO 的评估循环、调用策略服务、叠加安全控制层。[main/main_aegis.py](main/main_aegis.py)
- AEGIS（平移版）入口：安全层仅对平移部分做控制（简化版）。[main/main_aegis_translational.py](main/main_aegis_translational.py)
- 安全与感知工具集：CBF 约束、点云处理、椭球拟合、障碍物识别。[main/utils.py](main/utils.py)
- SafeLIBERO：安全版基准任务与数据集工具。[safelibero/](safelibero/)
- openpi：VLA 模型训练/推理与策略服务。[openpi/](openpi/)
- 项目脚本：模型服务、训练、统计等脚本入口。[scripts/](scripts/)

**模块依赖与数据流（Mermaid）**
```mermaid
flowchart LR
  subgraph Eval[AEGIS Eval]
    A[main_aegis.py] --> B[LIBERO Env]
    A --> C[Policy Client]
    A --> D[Safety Layer (CBF/QP)]
    A --> E[Perception (GroundingDINO + Depth)]
  end

  subgraph Model[openpi Policy Server]
    F[serve_policy.py] --> G[openpi Policy]
  end

  subgraph Bench[SafeLIBERO]
    H[safelibero tasks] --> B
  end

  C -->|websocket actions| F
  E --> D
  D --> B
```

**各模块技术栈与关键依赖**
- AEGIS 评测：Python, MuJoCo, LIBERO, openpi-client, tyro, cvxpy, scipy, imageio
- 感知：GroundingDINO, OpenCV, Open3D
- 安全控制：CBF + QP（cvxpy/OSQP）
- 模型服务：openpi（JAX/PyTorch, transformers, torch）
- 基准：SafeLIBERO / LIBERO

### 2. 模块详细说明

**AEGIS 评测入口**
- 功能：在 SafeLIBERO 环境中执行任务，调用策略服务生成动作，并通过安全层进行实时约束。
- 主要接口：
  - `eval_libero(args)`：评测主循环（任务选择、episode 执行、视频保存）。[main/main_aegis.py](main/main_aegis.py)
  - `_get_libero_env(...)`：创建 OffScreenRenderEnv，支持深度渲染。[main/main_aegis.py](main/main_aegis.py)
  - `_quat2axisangle(...)`：四元数转轴角，兼容 robosuite 逻辑。[main/main_aegis.py](main/main_aegis.py)
- 关键参数（节选）：
  - `host`/`port`：策略服务地址与端口
  - `task_suite_name`：安全任务套件名（safelibero_spatial/object/goal/long）
  - `safety_level`：安全等级 I/II
  - `task_index`/`episode_index`：任务与 episode 子集
  - `replan_steps`：动作块重规划步数
  - `video_out_path`：视频输出目录

**AEGIS（平移版）**
- 功能：将安全约束主要施加到平移控制量，减少控制维度。
- 与完整版本差异：[main/main_aegis_translational.py](main/main_aegis_translational.py)

**安全与感知工具集**
- CBF 相关：`compute_h_coeffs_3d`、`compute_h_ij` 等用于构建安全约束。[main/utils.py](main/utils.py)
- 点云生成：`get_point_cloud`（深度 + GroundingDINO bbox）。[main/utils.py](main/utils.py)
- 点云过滤：`filtering_points`（场景范围 + 聚类）。[main/utils.py](main/utils.py)
- 椭球拟合：`fit_ellipse`（MVEE + 可视化）。[main/utils.py](main/utils.py)
- 障碍识别：`obstacle_detection`（ZhipuAI 视觉大模型）。[main/utils.py](main/utils.py)

**配置文件与参数说明**
- 全局依赖：[requirements.txt](requirements.txt)
- AEGIS 依赖：[main/requirements.txt](main/requirements.txt)
- openpi 依赖与版本：[openpi/pyproject.toml](openpi/pyproject.toml)
- ⚠️ 需要手动下载与配置：[README.md](README.md)
  - GroundingDINO 权重
  - 策略模型 checkpoint
  - `obstacle_detection` 的 `api_key`（ZhipuAI）

### 3. 代码组织逻辑

- 命名规范：入口脚本以 `main_*.py` 命名，功能类工具集中在 [main/utils.py](main/utils.py)
- 组织原则：
  - 评测逻辑与算法模块分离
  - openpi 作为独立子模块提供训练与策略服务
  - SafeLIBERO 作为数据与评测基准子模块
- 设计模式（隐式）：
  - 客户端/服务端：策略推理由策略服务统一提供（websocket）
  - 管道式处理：观测 -> 感知 -> 安全控制 -> 执行
- 数据流与控制流：
  - 观测图像与状态拼接后传入策略服务
  - 动作经 CBF/QP 修正后执行，记录成功率与安全性

**Checklist**
- [ ] 关键模块边界清晰（AEGIS / SafeLIBERO / openpi）
- [ ] 依赖与数据流关系已覆盖
- [ ] 关键配置项与外部依赖已标注

---

## 第二部分：快速上手指南

### 1. 环境准备

**系统要求**
- GPU：建议 8GB+（推理）/ 24GB+（训练）
- OS：当前仓库文档以 Linux 为主，Windows 需额外适配 [需补充]

**依赖安装（参考 README，按需调整）**
- AEGIS + SafeLIBERO 基础依赖（示例，Linux/bash）
```
# 创建与激活环境
uv venv --python 3.11 .aegis_venv
source .aegis_venv/bin/activate

# 安装依赖
uv pip sync requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
```

- AEGIS 运行时依赖（main 子环境）
```
# 创建并切换到 main 环境
cd main
uv venv --python 3.8 .venv
source .venv/bin/activate
uv pip sync requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
```

- Windows PowerShell 参考（路径按需调整）
```
# 仅示例，实际以本机 uv/conda 配置为准
uv venv --python 3.11 .aegis_venv
.\.aegis_venv\Scripts\Activate.ps1
```

💡 提示：openpi 依赖请参考 [openpi/README.md](openpi/README.md)，其默认要求 Python >= 3.11。

### 2. 项目启动流程

**启动策略服务（Terminal 1）**
```
# 启动策略服务（服务端）
uv run scripts/serve_policy.py --env LIBERO
```

**运行 AEGIS 评测（Terminal 2）**
```
# 设置 SafeLIBERO 路径（PowerShell 示例）
$env:PYTHONPATH = "$env:PYTHONPATH;$PWD\safelibero"

# 运行 AEGIS
python main/main_aegis.py --task-suite-name safelibero_spatial --safety-level I --task-index 0 --episode-index 0 1 2 --video-out-path data/libero/videos
```

**快速预览（无策略输出，仅场景回放）**
```
# 运行 dummy demo
python main_demo_dummy.py --task-suite-name safelibero_goal --safety-level I --task-index 0 --episode-index 0
```

⚠️ 警告：需要提前下载模型与权重并配置 `api_key`，否则安全层感知会失败。

### 3. 训练参数配置（openpi）

**常用训练入口（openpi）**
```
# 统计归一化参数
uv run scripts/compute_norm_stats.py --config-name pi05_libero

# 启动训练
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite
```

**关键参数说明（节选）**
- `config-name`：训练配置名（与 openpi config 对应）
- `exp-name`：实验目录名
- `overwrite`：覆盖已有实验结果

💡 提示：完整参数与数据配置请参照 [openpi/README.md](openpi/README.md) 与 openpi 的训练配置文件 [需补充]

**Checklist**
- [ ] 依赖环境安装步骤完整可执行
- [ ] 启动流程包含策略服务与评测端
- [ ] 训练流程与关键参数覆盖到位

---

## 第三部分：深入学习路线

### 1. 代码阅读顺序

**推荐路径**
1. 入口与评测流程：[main/main_aegis.py](main/main_aegis.py)
2. 安全与感知工具：[main/utils.py](main/utils.py)
3. SafeLIBERO 基准说明：[safelibero/README.md](safelibero/README.md)
4. openpi 策略与训练：[openpi/README.md](openpi/README.md)

**阶段目标**
- 阶段一：理解评测循环与动作生成流程
- 阶段二：理解安全层构造与感知链路
- 阶段三：掌握策略服务与训练配置

### 2. 核心概念理解

**关键术语**
- VLSA：Vision-Language-Safe Action
- AEGIS：Action Execution Guarded by Invariant Safety
- CBF：Control Barrier Function，用于安全约束
- SafeLIBERO：包含安全干扰的 LIBERO 任务子集

**核心逻辑**
- 观测输入 → 策略服务推理 → 动作块 → CBF/QP 安全约束 → 环境执行 → 记录成功率与安全率

**参考资料**
- 项目主页与论文：[README.md](README.md)
- openpi 与 LIBERO 官方文档：[openpi/README.md](openpi/README.md)、[safelibero/README.md](safelibero/README.md)

### 3. 二次开发指南

**可扩展点**
- 替换策略服务：修改 [scripts/serve_policy.py](scripts/serve_policy.py) 或接入自定义模型服务
- 自定义安全层：在 [main/utils.py](main/utils.py) 中新增 CBF 约束或目标函数
- 新任务集成：扩展 SafeLIBERO 任务或直接接入 LIBERO 新套件

**贡献规范与开发流程**
- 参考 openpi 与 SafeLIBERO 的贡献说明 [需补充]
- 建议使用 pre-commit/ruff 进行格式化与静态检查（openpi 内置）

**调试技巧/测试方法**
- 使用 dummy demo 快速验证环境与渲染链路
- 在安全层前后记录动作，用于比较行为差异
- 先固定 episode，再逐步扩展任务范围

**二开示例**
1. 增加新的障碍识别策略：在 [main/utils.py](main/utils.py) 中新增视觉模型或规则
2. 添加新的安全约束：在 `compute_h_coeffs_3d` 基础上扩展多障碍约束 [需补充]
3. 集成自定义策略服务：保持 websocket 协议，替换 openpi 服务端

**Checklist**
- [ ] 阅读路径覆盖关键入口与工具层
- [ ] 核心概念与安全流程已解释
- [ ] 二开扩展点与示例清晰
