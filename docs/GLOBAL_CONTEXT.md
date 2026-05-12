你可以把下面整段复制到 GLOBAL_CONTEXT.md。以后每次给 codeagent 发需求，都把这个文档作为附件或上下文一起发。
# KKT-SenseVLA 全局上下文文档

> 本文档用于在 codeagent 上下文有限的情况下，持续提供项目背景、目标、工程边界和当前开发计划。  
> 每次向 codeagent 提需求时，都请附带本文档，避免上下文断裂。

---

## 1. 项目背景

我们正在基于 VLSA-AEGIS / SafeLIBERO / OpenVLA-OFT 相关代码，开发一个新的 VLA safety defense 方法，暂定名：

**KKT-SenseVLA**

核心研究方向是：

> 不只是让 VLA 输出安全动作，而是让 VLA 学会安全优化器背后的安全判断结构：哪条约束正在变紧、约束压力多大、应该往哪个方向修正动作。

已有工作 VLSA/AEGIS 提供了一个外部安全层：

```text
VLA 输出原始动作 a0
        ↓
AEGIS / CBF-QP 安全层
        ↓
输出安全动作 a*
我们的方法不是简单复现 AEGIS，而是把 AEGIS 这类安全优化器作为 teacher system（教师系统），生成用于蒸馏训练的标签，然后训练 student VLA（学生模型）内化这种安全判断能力。

2. 当前总体目标
当前工程目标不是一次性实现完整论文方法，而是分阶段搭建基础设施。
最终希望实现：
VLSA/AEGIS teacher
        ↓
生成 KKT-SafeLIBERO 蒸馏标签
        ↓
OpenVLA-OFT / 其他 VLA student 读取标签训练
        ↓
student 学会预测安全压力
        ↓
通过 KKT correction layer 生成更安全动作
        ↓
在 SafeLIBERO 上评测
当前最重要的第一阶段目标是：
在 VLSA-AEGIS 仓库中搭建 KKT-SenseVLA 标签生成骨架，能够记录并导出 teacher rollout 中的原始动作、安全动作和未来需要的 KKT 信息字段。

3. 重要原则
3.1 代码实现可以和论文最终写法不完全一致
当前优先级是：
先把代码跑通
先能生成数据
先能训练/评测出有效结果
代码实现可以为了工程可行性做简化。只要最终实验能支撑我们想证明的核心结论，论文写法可以后续再抽象和美化。
3.2 不要追求一开始就完整实现所有数学细节
第一阶段允许很多字段为空、占位或 TODO。
例如：
dual_variables 可以先为空
active_set 可以先为空
constraint_gradients 可以先为空
qp_status 可以先用占位字符串
但接口和数据结构必须提前设计好，方便后续逐步接入真实逻辑。
3.3 优先保持 VLSA 原逻辑不被破坏
不要直接大改 main/main_aegis.py 或 main/main_aegis_translational.py 的主逻辑。
推荐方式：
新增 wrapper
新增 adapter
新增 kkt_sense/ 独立模块
在必要位置插入最小 hook
目标是：
原始 AEGIS 仍然可以正常运行
新增 KKT 标签导出功能不破坏原评测流程

4. 当前代码库理解
当前仓库路径：
E:\vlsa-aegis
codeagent 已生成过 docs/OVERVIEW.md，其中确认了仓库大致结构：
.
├─ main/
│  ├─ main_aegis.py
│  ├─ main_aegis_translational.py
│  └─ utils.py
├─ safelibero/
├─ openpi/
├─ scripts/
└─ requirements.txt
4.1 VLSA/AEGIS 相关部分
main/main_aegis.py 
AEGIS 主评测入口 
负责 SafeLIBERO 评测循环 
调用策略服务 
执行安全层修正 
保存视频/统计评测结果 
main/main_aegis_translational.py 
平移版 AEGIS 
安全修正主要作用在平移动作维度 
对我们第一版 KKT 标签更友好，因为我们也优先只处理平移控制 
main/utils.py 
安全与感知工具集合 
包含 CBF/QP、点云、障碍物识别、椭球拟合等逻辑 
后续可能从这里接入 constraint values、constraint gradients、QP 约束信息 
4.2 SafeLIBERO
safelibero/ 
安全版 LIBERO benchmark 
包含安全任务、障碍物场景、评测脚本 
后续用于生成 KKT-SafeLIBERO 数据和最终评测 
4.3 openpi
openpi/ 
VLSA 仓库内包含的策略模型相关代码 
当前不作为 KKT-SenseVLA 第一阶段重点 
后续可作为模型服务或附加实验参考 

5. VLSA 仓库在本项目中的角色
VLSA 仓库不是我们的最终学生训练仓库。
它当前主要承担三个角色：
1. SafeLIBERO 环境与 benchmark
2. AEGIS teacher system
3. KKT 标签生成的数据来源
也就是说：
VLSA/AEGIS = 老师 + 考场
OpenVLA-OFT = 学生
KKT-SenseVLA = 我们新增的蒸馏训练与安全动作生成框架

6. OpenVLA-OFT 在本项目中的角色
OpenVLA-OFT 是我们优先考虑的 student model。
原因：
1. OpenVLA-OFT 使用连续动作表示，更适合学习 a*、Δa*、λ* 等连续标签。
2. 我们需要修改 action head、增加 pressure head、增加 KKT correction layer。
3. 这些训练逻辑不适合直接塞进 VLSA 仓库。
因此后续训练阶段更可能在以下位置实现：
方案 A：新建独立 kkt-sense-vla 总仓库
方案 B：fork OpenVLA-OFT 仓库并加入 KKT 模块
方案 C：当前 VLSA 仓库只负责生成数据，OpenVLA-OFT 单独训练
当前第一阶段先不处理 OpenVLA-OFT 训练代码。

7. 两个代码库如何协同
不要让 OpenVLA-OFT 训练时直接调用 VLSA。
推荐通过离线数据文件连接：
VLSA/AEGIS 环境
        ↓
运行 SafeLIBERO rollout
        ↓
导出 KKT-SafeLIBERO 标签文件
        ↓
OpenVLA-OFT 训练代码读取标签文件
        ↓
训练学生模型
        ↓
训练好的 checkpoint 回到 SafeLIBERO 评测
也就是说：
VLSA 负责生产“练习册”
OpenVLA-OFT 负责读取“练习册”训练学生

8. 虚拟环境管理原则
建议至少使用两个环境。
8.1 kkt-vlsa 环境
用途：
运行 VLSA
运行 SafeLIBERO
运行 AEGIS teacher
生成 rollout
生成 KKT 标签
做 SafeLIBERO 评测
可能依赖：
MuJoCo
robosuite
LIBERO / SafeLIBERO
cvxpy
OSQP
OpenCV
Open3D
GroundingDINO
ZhipuAI API
h5py
numpy
scipy
8.2 kkt-openvla 环境
用途：
训练 OpenVLA-OFT student
加载大模型
LoRA / OFT fine-tuning
读取 KKT-SafeLIBERO 数据
可能依赖：
torch
transformers
accelerate
peft
deepspeed
datasets
OpenVLA-OFT dependencies
8.3 两个环境之间如何通信
通过硬盘数据文件通信，不要互相 import：
data/kkt_safelibero/
    task_xxx/
        episode_xxx.jsonl
        episode_xxx.hdf5

9. KKT-SafeLIBERO 标签数据应包含什么
第一版标签文件不要求全部字段都有真实值，但 schema 必须预留。
每一步 step record 至少包含：
task_suite_name
safety_level
task_index
episode_index
step_index
instruction

observation metadata

action_nominal      # AEGIS safety layer 前的原始 VLA 动作 a0
action_safe         # AEGIS safety layer 后的安全动作 a*
action_delta        # action_safe - action_nominal

constraint_values   # 可先为空
constraint_gradients # 可先为空
dual_variables      # 可先为空
active_set          # 可先为空
qp_status           # 可先为空或占位

collision_info
extra_debug
后续真实训练时，最核心字段是：
a0
a*
Δa*
dual pressure p* 或 λ*
constraint values c_i
constraint gradients ∇c_i

10. 当前 KKT-SenseVLA 方法的简化实现目标
不要一开始做完整 KKT 体系。
第一阶段只需要支持以下工程逻辑：
1. 捕获 AEGIS safety layer 前的原始动作 a0
2. 捕获 AEGIS safety layer 后的安全动作 a*
3. 计算 Δa* = a* - a0
4. 将一整条 episode 的 step records 保存为 JSONL
5. 为未来接入 λ*、active_set、constraint_values、constraint_gradients 预留接口
后续阶段再逐步实现：
1. 从 QP solver 中导出 dual variables
2. 计算 active set
3. 计算 constraint values
4. 计算 constraint gradients
5. 输出 HDF5 / RLDS 格式
6. 对接 OpenVLA-OFT training

11. 预测安全压力与 action 生成的当前设计
我们不希望只是：
安全 embedding
        ↓
cross-attention
        ↓
MLP 直接生成动作
这种方式太自由，容易退化成普通 attention fusion。
我们希望最终结构是：
VLA hidden state
        ↓
输出原始动作 a0
        ↓
预测每条安全约束压力 p_i
        ↓
几何模块提供每条约束方向 ∇c_i
        ↓
KKT correction layer
        ↓
a = a0 + Q^{-1} Σ p_i ∇c_i
cross-attention 可以存在，但它的推荐用途是：
用于从 constraint tokens 中判断每条约束的 pressure p_i
而不是直接自由生成最终动作。

12. 当前阶段不要做的事情
为了避免范围失控，当前阶段不要做：
1. 不要训练 OpenVLA-OFT
2. 不要重写 AEGIS 主算法
3. 不要追求完整 dual variables 导出
4. 不要实现完整 KKT correction layer
5. 不要改 openpi 训练逻辑
6. 不要构造完整全量数据集
7. 不要大规模重构原仓库
当前只做：
搭建 KKT 标签生成骨架
验证能保存 dummy labels
为接入 AEGIS rollout 做准备

13. 第一阶段建议新增目录
建议在 E:\vlsa-aegis 下新增：
kkt_sense/
    __init__.py
    README.md
    schema.py
    io_utils.py
    rollout_capture.py
    label_exporter.py
    constraints.py
    qp_interface.py
    scripts/
        generate_kkt_labels.py
各文件职责：
schema.py 
定义 KKT label 的数据结构 
io_utils.py 
保存/读取 JSONL 或其他格式标签 
rollout_capture.py 
将单步 rollout 信息打包成 step record 
label_exporter.py 
导出一整个 episode 标签 
constraints.py 
预留 constraint values / gradients 接口 
qp_interface.py 
预留从 QP solver 提取 dual variables / active set 的接口 
scripts/generate_kkt_labels.py 
命令行脚本，第一版可以只生成 dummy JSONL，验证流程 

14. 第一阶段完成标准
第一阶段完成后，应该能够运行类似命令：
python kkt_sense/scripts/generate_kkt_labels.py \
  --task-suite-name safelibero_spatial \
  --safety-level I \
  --task-index 0 \
  --episode-index 0 \
  --output-dir data/kkt_safelibero_debug
并生成一个可读的 JSONL 文件，例如：
data/kkt_safelibero_debug/
    safelibero_spatial_level_I_task_0_episode_0.jsonl
里面至少包含 dummy 或真实的字段：
{
  "task_suite_name": "safelibero_spatial",
  "safety_level": "I",
  "task_index": 0,
  "episode_index": 0,
  "step_index": 0,
  "action_nominal": [...],
  "action_safe": [...],
  "action_delta": [...],
  "dual_variables": null,
  "active_set": null,
  "constraint_values": null,
  "constraint_gradients": null,
  "qp_status": "placeholder"
}

15. 当前待解决问题
15.1 需要找到 AEGIS 中 a0 和 a* 的位置
后续需要 codeagent 阅读：
main/main_aegis.py
main/main_aegis_translational.py
main/utils.py
确认：
原始 VLA action 在哪里生成
安全层修正后的 action 在哪里生成
两者是否都能在 step 前被捕获
15.2 需要确认 QP solver 是否能导出 dual variables
后续需要确认：
VLSA 当前 QP 用的是 cvxpy/OSQP 还是其他 solver
是否能从 solver_result 里拿到 dual variables
如果不能，是否需要我们自己实现 lightweight QP labeler
15.3 需要确认 constraint values 和 gradients 的来源
后续需要确认：
main/utils.py 是否已有 CBF 约束函数
是否能直接计算 h_i 或 c_i
是否能计算 ∇c_i
是否需要先实现简化版 obstacle/workspace/action_bound 约束
15.4 需要确认数据格式
第一版使用 JSONL，方便调试。
后续若训练 OpenVLA-OFT，可能需要转换为：
HDF5
RLDS
parquet
npz

16. 对 codeagent 的工作方式要求
请 codeagent 遵守：
1. 每次只做一个阶段的改造，不要跨太多目标。
2. 尽量新增文件，不要大改原文件。
3. 如果必须修改原文件，请明确说明修改位置和原因。
4. 代码要可运行，至少提供最小测试命令。
5. 每次完成后输出：
   - 新增/修改文件列表
   - 每个文件作用
   - 运行命令
   - 当前 TODO
   - 可能风险
6. 如果遇到不确定，不要瞎猜，先标注 [需确认]。
7. 不要在本地进行依赖复杂的真实实验，因为本地没有完整 Python/GPU/MuJoCo 环境。
8. 可以做静态检查、阅读文件、创建目录、写代码，但不要假设本地环境可以运行完整 SafeLIBERO。

17. 当前下一步任务
下一步不是训练模型，而是：
在 VLSA 仓库中创建 kkt_sense/ 标签生成骨架。
目标是让仓库具备：
保存 KKT-SafeLIBERO step records 的能力
完成后，再进入下一步：
将 main/main_aegis_translational.py 或 main/main_aegis.py 中的原始动作 a0 和安全动作 a* 接入 kkt_sense 标签导出模块。

18. 一句话总结
本项目当前不是要马上训练模型，而是先把 VLSA/AEGIS 变成一个能生成 KKT 蒸馏标签的 teacher data generator。
VLSA 负责老师和环境，OpenVLA-OFT 负责学生训练，二者通过离线 KKT-SafeLIBERO 标签数据连接。
