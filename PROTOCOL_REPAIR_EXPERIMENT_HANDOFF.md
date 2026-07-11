# CircuitGCL 协议修复与后续实验交接

## 0. 文档定位

本文档用于给后续 agent 提供统一、可执行的 CircuitGCL 工作入口。它整合并修正以下信息：

- `AGENT_HANDOFF_ISSUES.md` 中已经确认的代码与实验协议问题；
- 对该审阅文档中过时、不完整或表述过强部分的修正；
- 当前 P1/P2/P4 已完成实验的最新状态；
- online GNN encoder 与 downstream GNN backbone 复用主线的后续实验计划；
- label rebalancing 与 GCL/reuse 交互实验的正式任务清单。

本文档不覆盖原审阅文档。原文档保留为 2026-07-11 18:48 CST 的历史审阅快照；后续 agent 应以本文档的最新状态和执行顺序为准。

审阅与计划快照：

- 日期：2026-07-11 CST
- 分支：`test`
- 当前代码提交：`af061316193b054dcdd3bc481919bdf77c0bc15d`
- 核心协议修复提交：`ab1f252`
- 远端工作分支：`lixc929/test`
- Python 环境：`/home/lixc/.conda/envs/RCG/bin/python`
- 当前成熟任务：edge regression
- 当前核心目标：将 GCL online GNN encoder 与 downstream GNN backbone 合并为一个可部署共享 backbone，并研究 label rebalancing 单独使用和联合使用时的效果。

## 1. 后续 agent 必须遵守的约束

1. 只在用户 fork 的 `test` 分支工作；不得向老师仓库的主分支或其他分支推送。
2. 不使用 GPU0。启动任务前检查 GPU 状态，不中断、不抢占、不终止其他用户进程。
3. 所有正式实验使用 RCG 环境，并记录实际 Python、CUDA、GPU、Git commit 和完整命令。
4. 不覆盖已有实验目录、checkpoint 或 cache；每个运行必须使用独立 artifact 目录。
5. 不把 `status=running`、空 `test_results` 或中间 best 值写入正式结果。
6. `logs/`、checkpoint、embedding 和数据集被 Git 忽略；Git clean 不代表实验产物不存在。
7. `sp8192w` 暂定为最终盲测集。在模型、loss 和超参数锁定前，不处理它、不查看其标签分布、不运行模型。
8. 避免继续增加零散 Markdown。实验过程更新现有 `EXPERIMENT_LOG.md`；高层状态更新 `TEACHER_TASKS_README.md`；本文档仅维护协议修复与任务清单。

## 2. 当前实验的最新状态

`logs/s6_p1_p2_protocol_20260711` 已完成统一汇总：

- artifact 数：11
- completed：11
- running：0
- anomalous：0
- GPU4 上已无本项目训练进程
- 队列正常结束时间：2026-07-11 19:54 CST

汇总入口：

- `logs/s6_p1_p2_protocol_20260711/experiment_summary.json`
- `logs/s6_p1_p2_protocol_20260711/experiment_runs.tsv`

### 2.1 P1/P2 seed0 结果

| 结构 | Val MSE | digtime | timing_ctrl | array | 三迁移集平均 |
| --- | ---: | ---: | ---: | ---: | ---: |
| static | 0.007845 | 0.015749 | 0.009772 | 0.010257 | 0.011926 |
| joint, lambda=0 | 0.009417 | 0.013731 | 0.011223 | 0.012385 | 0.012446 |
| LoRA r8, lambda=0, backbone LR=1e-4 | 0.009170 | 0.013827 | 0.011120 | 0.011877 | 0.012274 |
| LoRA r8, lambda=0.05, backbone LR=1e-4 | 0.009115 | 0.014073 | 0.010324 | 0.011804 | 0.012067 |
| LoRA r8, lambda=0.05, backbone LR=1e-5 | 0.009549 | 0.014218 | 0.011308 | 0.011530 | 0.012352 |
| LoRA r8, lambda=0.05, backbone LR=1e-6 | 0.009650 | 0.013941 | 0.011679 | 0.010804 | 0.012141 |
| LoRA r8, lambda=0, backbone LR=1e-5 | 0.009612 | 0.014226 | 0.011739 | 0.011001 | 0.012322 |

在当前旧协议下：

- static 的 source validation 最好；
- 复用方案普遍改善 digtime，但没有在三个迁移集和 source validation 上全面超过 static；
- positive-lambda LoRA 中，backbone LR `1e-4` 的 source validation 和迁移平均最好；
- 所有 best epoch 位于 77-79，说明 80 epoch 并未明显过长。

### 2.2 static seeds 0-4

固定 SGRL checkpoint 下的 downstream 结果：

| 指标 | mean +/- population std |
| --- | ---: |
| Val MSE | 0.00789372 +/- 0.00003349 |
| digtime | 0.01623054 +/- 0.00101114 |
| timing_ctrl | 0.01027478 +/- 0.00084640 |
| array | 0.01026561 +/- 0.00044917 |
| 三迁移集平均 | 0.01225698 +/- 0.00056905 |

重要限定：当前缓存键不包含 seed，因此这五个 seed 共享同一份 SGRL 预训练 checkpoint。它们只能证明 downstream 训练的稳定性，不能称为端到端方法的五 seed 稳定性。

### 2.3 当前结果的证据等级

当前 S5/S6/P1/P2 结果应统一标记为：

```text
legacy-transductive-v1
relation-balanced
fixed-pretraining-checkpoint
downstream-only seeds
development benchmark
```

这些结果仍可用于理解复用结构、优化方向和表示漂移，但不能支持论文所宣称的 strict unseen-circuit zero-shot 结论。

## 3. 对 `AGENT_HANDOFF_ISSUES.md` 的核验与修正

### 3.1 已确认且必须处理的问题

| 问题 | 核验结论 | 对当前主线的影响 |
| --- | --- | --- |
| SGRL 在全部电路拼接图上训练 | 确认 | 目标拓扑参与参数拟合，不是 strict inductive |
| node feature normalization 使用全部电路 | 确认 | 测试域 covariate statistics 参与训练 |
| LDS 在 loader 创建后才写权重 | 确认 | loader 中第二列仍是 class ID，LDS 实际训练错误 |
| GAI 使用完整 SSRAM labels | 确认 | GMM 包含 source validation labels |
| BNI/LDS 使用全部电路 labels | 确认 | 存在 transfer-label leakage |
| 关系类型等量采样 | 确认 | 当前测试指标对应 relation-balanced 子集，不是自然分布 |
| static/SGRL/processed cache key 不完整 | 确认 | multi-seed 与配置复现语义不完整 |
| validation/test 邻居采样可能随机 | 确认风险 | 噪声大小尚未量化，需固定并审计 |
| node/classification root supervision 与 loss 协议问题 | 确认 | 不阻塞 edge regression，但旧分类结果不可作为正式证据 |
| 非默认 edge-aware SGRL encoder 使用错误 edge input | 确认 | ClusterGCN 主线未触发，其他公开模型路径不可靠 |
| LoRA 尚无真实 deployment-only artifact | 确认 | 当前 29,378 参数只是核算和 merge API 证据 |

### 3.2 修订：原始 SGRL target 更新存在文字与上游算法歧义

论文 `papers/CircuitGCL_Transferable_Parasitic_Estimation_arXiv2507.06535.pdf` 明确说明：

- GCL 只在同一个训练设计上预训练；
- 所有测试设计严格排除在训练和验证之外；
- target encoder 不接受梯度更新，只通过 online encoder 的 EMA 更新。

当前原始 SGRL 代码同时：

1. 在 `CustomOnline.update_target_encoder()` 中做 EMA；
2. 为同一个 target encoder 建立独立 optimizer；
3. 在 `train_target_encoder()` 中对 target encoder 反向传播。

SGRL NeurIPS 2024 论文和官方实现明确采用 dual 机制：target 通过 scattering loss 优化，online 对 target stop-gradient，每轮末再做 EMA。CircuitGCL 论文的 alignment 段落支持 EMA-only 文字直译，但其 RSM 段落仍定义了 target scattering loss。因此当前 dual 路径不能直接定性为 bug，也不能在 matched 消融前预设 EMA-only 为正式赢家。

后续使用两个中性模式名：

```text
sgrl_dual_rsm_ema
circuitgcl_text_ema_only
```

EMA-only 模式必须从 online 同步初始化 target，并与 S6 downstream EMA target 使用不同开关。

### 3.3 原文档已经过时或需要降级的表述

1. “P1/P2/P4 仍在运行”已经过时：现在 11 个 artifact 全部完成。
2. “9 个运行都在 GPU4，启动来源异常”不是未知异常：这些任务是按用户要求从等待中的 GPU3 lane 迁移，并通过独立 tmux 在 GPU4 并行启动。
3. P4 static seeds 在 P3 前运行是为了利用空闲 GPU；由于 static 不依赖 P3 选型，这不会污染 P3，但其 seed 仅为 downstream-only。
4. “不得进入 P3”属于正式研究流程建议，不是代码事实。探索性 P3 可以运行，但在协议修复前运行的结果之后必须重跑，因而不建议继续消耗资源。
5. 邻居采样噪声机制存在，但其 MSE 方差尚未实际测量；正式表述应为“已确认风险，待量化”。
6. relation-balanced 不是天然错误，它可能是论文预处理协议的一部分；错误在于把它称为自然分布性能，或不保留 relation type 和采样指纹。
7. 原文档列出的各电路 raw-to-final 保留比例来自历史日志，代码中的平衡机制已确认，但具体比例仍应由新 structured audit 重新生成。

## 4. 正式协议决策

除非用户或研究负责人明确改变，后续主线采用以下决策：

### 4.1 数据可见性

- 正式主线：`strict_inductive`
- 兼容对照：`transductive_legacy`
- strict 模式下，只有 SSRAM 参与 SGRL 和 downstream 参数拟合。
- 测试电路的图结构和节点特征只在冻结模型的最终推理阶段进入 encoder。

### 4.2 模型选择边界

- 训练集：SSRAM source-train edges
- 选择集：SSRAM source-validation edges
- 开发迁移诊断：digtime、timing_ctrl、array
- 最终盲测：sp8192w
- digtime、timing_ctrl、array 已长期参与人工分析，不能再称为最终盲测集。
- 正式口径为 `strict-inductive training + development-informed design + sp8192w blind confirmation`。
- source validation 是主要选择指标；digtime、timing_ctrl、array 使用预注册 veto/Pareto 规则参与开发选型，不能再声称完全未参与选择。

### 4.3 采样分布

- 第一阶段继续使用 `relation_balanced`，保持与历史和论文处理方式可比。
- 最终锁定模型后，增加 `natural_distribution` 评估。
- 两种分布必须使用不同 cache key，并分别报告 overall 和 per-relation 指标。

### 4.4 随机种子语义

- `downstream-only seed`：固定同一 SGRL checkpoint，只改变 split/downstream 初始化。
- `end-to-end seed`：每个 seed 都产生独立 source split、SGRL checkpoint、downstream 初始化和固定 eval sampler。
- 正式 P4/P5 使用 paired end-to-end seeds。
- 同一 seed 内不同方法允许共享同一个 source-only SGRL checkpoint，以减少方差并保证公平比较。

## 5. 实现任务清单

### WP0：冻结 legacy 结果

- [ ] 在 `EXPERIMENT_LOG.md` 把当前 11 个结果标记为 `legacy-transductive-v1`。
- [ ] 记录统一 summary 路径、commit、协议、缓存语义和运行完成状态。
- [ ] 不删除、不覆盖当前 artifact。
- [ ] 不把旧协议和新 strict 协议结果放进同一均值或排名表。

### WP1：source/test 数据边界

- [ ] 增加 `--protocol strict_inductive|transductive_legacy`。
- [ ] strict 模式的 SGRL train graph 只包含 `dataset[0]`。
- [ ] 将“预训练图构造”和“冻结 encoder 推理目标图”拆分为不同函数。
- [ ] 增加断言：strict 训练期间不得访问 `dataset[1:]` 的图、特征或标签。
- [ ] 在 artifact 中记录 source graph names 和 transfer graph names。

验收测试：

- [ ] 使用访问 transfer graph 就抛异常的伪数据集，strict SGRL smoke test 必须通过。
- [ ] transductive legacy 模式仍能复现原有 all-circuit graph 构造。

当前实现审计（`af06131`）：source-only SGRL 的参数拟合边界已经成立，但最强版本的“训练期间完全不访问 transfer 数据”尚未成立。当前程序仍会在训练前加载并 collate 全部图，在 downstream 训练前构造 transfer loaders、读取 transfer labels，并在 source-only SGRL 结束后立即用冻结 encoder 为全部图生成 embedding。以上操作不让 transfer 数据参与梯度或统计量拟合，因此可以称为 parameter-fit strict inductive；在完成 lazy/deferred transfer construction 前，不能声称训练阶段从未访问 transfer graph/features/labels。

### WP2：source-fit normalization

- [ ] 将 normalization 拆为 `fit_normalizer(source_graph)` 和 `transform(graph, state)`。
- [ ] 对 node_attr 只使用 SSRAM 全图节点特征拟合最大值。
- [ ] validation/test 只能加载和应用已保存 state。
- [ ] normalization state 写入 artifact，并记录 SHA256。
- [ ] 不再原地依赖全部拼接 `_data` 计算统计量。

验收测试：

- [ ] 修改 transfer graph 为极端 node_attr 后，source normalization state 不变。
- [ ] 保存再加载 normalization state，输出完全一致。

### WP3：split 与 label-prior

- [ ] `dataset_sampling()` 返回或接收持久化的 train/val indices。
- [ ] split indices 写入 artifact，包含 seed、数量和 hash。
- [ ] GAI GMM 只读取 source-train 连续标签。
- [ ] BNI/LDS/class counts 只读取 source-train labels。
- [ ] source validation 与 transfer labels 不得参与任何 prior 拟合。
- [ ] GMM 文件保持 run-local 原子写入。

验收测试：

- [ ] 访问 source-val 或 transfer label 就抛异常的数据集上，GAI/BNI/LDS prior 构造仍能完成。
- [ ] GMM 输入数量与 source-train edge 数量一致。

### WP4：LDS 权重链路

- [ ] 不在 loader 创建后回写 `dataset._data.edge_label`。
- [ ] 在 split 确定后直接构造 train sample weights。
- [ ] 明确 loader 的 label schema，建议为连续标签、离散标签、sample weight 分字段保存。
- [ ] validation/test 不需要用 train LDS weights 修改真实标签。
- [ ] 修正 `lds_sigma` 默认值或帮助文本，使其与离散类别轴一致。

验收测试：

- [ ] 从 train loader 取 batch，weight 与预计算结果逐项一致。
- [ ] weight 不是整数 class ID。
- [ ] class 0 样本不会因时序错误获得固定零权重。

### WP5：可审计的 SGRL target-update 消融

- [ ] 增加 `--sgrl_pretrain_target_update sgrl_dual_rsm_ema|circuitgcl_text_ema_only`。
- [ ] dual 模式保留 target scattering optimizer 和 epoch EMA。
- [ ] EMA-only 模式从 online 同步初始化，target 参数不进入任何 optimizer。
- [ ] target forward 使用 stop-gradient。
- [ ] target 只在每个 epoch 按 EMA 更新。
- [ ] 预训练 target 开关与 S6 downstream target 开关明确分离。

验收测试：

- [ ] 一个 online optimization step 后，target 不产生梯度更新。
- [ ] EMA 后 target 参数满足预期公式。
- [ ] 两种模式使用不同 cache key，且不预设实验赢家。

### WP6：确定性 evaluation

- [ ] 为 val/test 定义独立 `eval_seed`。
- [ ] 每次评估使用相同 root order 和相同 neighbor samples。
- [ ] 评估过程不得改变后续训练 RNG 状态。
- [ ] artifact 记录 sampler 配置、eval seed 和 fingerprint。
- [ ] 若 PyG generator 不能可靠固定采样，则使用 RNG-preserving context 或预物化固定 eval batches。
- [ ] 将 `pretraining_seed`、`downstream_seed`、`split_seed`、`train_sampler_seed`、`eval_seed` 分开，不再让一个全局 RNG 状态串联所有阶段。
- [ ] 在 SGRL 训练或 cache load 完成后、构建 downstream model/loader 前，显式重置 downstream RNG。
- [ ] 固定 SGRL frozen-embedding inference 的 sampling seed/view；不同 SGRL 训练条件不得因为此前 RNG 消耗量不同而使用不同的 embedding inference neighborhoods。

验收测试：

- [ ] 同一 checkpoint 连续评估至少 5 次，MSE 完全一致或差异低于预设数值容差。
- [ ] 不同方法在同一 paired seed 下使用相同 evaluation neighborhoods。
- [ ] 同一 embedding、normalization state 和 split 在 cache miss 与 cache hit 两条路径下，downstream 初始权重、首批数据和训练轨迹一致。

### WP7：缓存和 artifact 指纹

- [ ] static embedding cache key 包含完整 scientific config、seed、protocol 和 data fingerprint。
- [ ] SGRL checkpoint cache key包含 target update、seed、采样和优化参数。
- [ ] processed graph key包含 sampling distribution、sample seed、raw hash、to_undirected 和 neg ratio。
- [ ] 避免 `neg_edge_ratio=0.04` 与 `0.0` 的一位小数命名冲突。
- [ ] run config 记录所有实际加载 cache 的绝对路径和 SHA256。
- [ ] 并行运行不得竞争写同一 checkpoint/cache。
- [ ] checkpoint/embedding 使用锁或 run-local 临时文件后原子发布；runner 依赖顺序不能替代实现层并发保护。
- [ ] static artifact 同时记录生成 embedding 的 SGRL online checkpoint 路径和 SHA256，而不只记录 embedding。

### WP8：relation-aware 与 natural evaluation

- [ ] processed data 保留 target relation type。
- [ ] 保存 relation-balanced 采样 indices 和 fingerprint。
- [ ] 记录 raw、capacitance-filtered、per-relation、final counts。
- [ ] 增加 natural-distribution 处理模式。
- [ ] structured metrics 增加 per-relation MSE/MAE/bias。
- [ ] 在模型锁定前不处理 sp8192w。

### WP9：真实 LoRA deployment export

- [ ] 从 isolated best checkpoint 恢复模型。
- [ ] merge task LoRA。
- [ ] 移除 EMA target、predictor、optimizer 和其他 training-only state。
- [ ] 保存 deployment-only config 和 state dict。
- [ ] 新进程只依赖 deployment artifact 完成 inference。
- [ ] 固定 batch 上比较训练 checkpoint 与部署模型输出。
- [ ] 报告实际 deployment 参数量、文件大小、峰值显存和推理时间。

## 6. 实验执行计划

所有正式长实验必须在 WP1-WP7 的最小测试通过后启动。代码修改后先跑 CPU/unit tests 和 1 epoch GPU smoke，不直接提交大规模任务。

### E0：协议差异审计

目的：量化旧结果中 all-circuit topology、全域 normalization 和 target dual update 的影响。

所有单元必须在同一个修复后 harness、相同 split、seed 和 eval views 下重跑。先固定 `sgrl_dual_rsm_ema` 做 topology x normalization 2x2，再隔离 target update：

| ID | Structure | Protocol | GCL train graph | Normalization | Target update | Loss | Tuning seeds | Epoch |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: |
| E0-A | static | transductive | all circuits | all circuits | dual RSM+EMA | MSE | 0-2 | 80/160 rule |
| E0-B | static | transductive | all circuits | source only | dual RSM+EMA | MSE | 0-2 | 80/160 rule |
| E0-C | static | transductive | source only | all circuits | dual RSM+EMA | MSE | 0-2 | 80/160 rule |
| E0-D | static | strict inductive | source only | source only | dual RSM+EMA | MSE | 0-2 | 80/160 rule |
| E0-E | static | strict inductive | source only | source only | text EMA-only | MSE | 0-2 | 80/160 rule |
| E0-F | no-GCL | transductive | N/A | all circuits | N/A | MSE | 0-2 | 80/160 rule |
| E0-G | no-GCL | strict inductive | N/A | source only | N/A | MSE | 0-2 | 80/160 rule |

说明：

- 历史 static 结果只作复现锚点，不能替代新 harness 的 E0-A。
- E0-A/B/C/D 识别 topology、normalization 和二者交互。
- E0-D/E 隔离 target update；matched 结果出来前不预设赢家。
- E0-F/G 单独识别 no-GCL normalization 影响。

#### 2026-07-11 21:37 seed0 现场审计：运行中历史快照

以下内容保留当时的运行中证据；七个任务之后已经全部完成，最终状态和结果以紧随其后的“完成后复核”为准。`logs/protocol_e0_20260711` 当时有 A-G 七个 seed0 运行，全部来自 commit `af06131`，位于 GPU4，对应 tmux pane 均存活。截至 2026-07-11 21:37 CST：

```text
completed = 0
running_active = 7
failed = 0
stale = 0
test_results = {}
```

不得把其中间 best epoch/MSE 写入正式结果。七个运行先自然完成，不中断、不覆盖；但即使完成，也只能标记为 `E0-seed0-calibration-v1`，原因如下：

1. 七份 `normalization_state.pt` 的 SHA256 完全相同：`df87fe04822420c6ada72e88a887fb9a840ee1903e42515ce8e397198d380749`。SSRAM 已提供当前 max-normalizer 的逐维全局最大值，因此 source/all 拟合域虽然不同，实际变换数值相同。A/B、C/D、F/G 的 MSE 差不得解释为 normalization effect；当前可审计的机制结论是该 leakage 在这批数据和此 normalizer 下的实测数值影响为零。
2. A、C、E 是 SGRL cache miss，会先训练 SGRL并生成 embedding；B、D 是 cache hit。`main.py` 只在 SGRL 前设置一次随机种子，进入 downstream 前没有重置 RNG。因此 A/B、C/D、D/E，以及不同图规模的 A/C，downstream RNG 起点不匹配，当前不是纯 normalization/topology/target-update 因子对照。
3. A/B 实际共享同一 all-graph embedding hash，C/D 共享同一 source-dual embedding hash，所有 split hash 也相同；启动顺序已避免本轮 cache 并发写。但 cache 文件仍无锁、非原子发布，不能依赖人工启动顺序作为正式并发保证。
4. 当前五项 `tests.test_protocol_boundaries` CPU 单测通过，但它们没有覆盖真实 LinkNeighborLoader 连续五次评估、cache hit/miss downstream 等价性、target EMA 数值公式、完整 forbidden-transfer training smoke 或 artifact 最低字段。

因此当前 seed0 结束后禁止直接启动 seeds 1-2 或 E1。先完成 WP6/WP7 的 RNG/cache 修复和缺失验收，再从固定 cache、独立 downstream seed 的干净状态重跑 formal E0。

#### 2026-07-11 完成后复核：`E0-seed0-calibration-v1`

七个任务均在约 21:46 CST 正常完成。最终审计结果为：

```text
completed = 7
failed = 0
OOM / traceback / NaN = 0
final transfer evaluation = 7/7 在重新加载各自 best checkpoint 后执行
our GPU4 training processes = 全部正常退出
```

产物目录：`logs/protocol_e0_20260711`。下表报告 raw MSE，越低越好；“迁移均值”是 `digtime`、`timing_ctrl` 和 `array_128_32_8t` 的算术平均。

| ID | 配置 | Best epoch | Source Val | digtime | timing_ctrl | array | 迁移均值 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | static, all graph, all norm, dual | 76 | 0.007846 | 0.018416 | 0.011085 | 0.009738 | 0.013080 |
| B | static, all graph, source norm, dual | 76 | 0.007883 | 0.016833 | 0.010045 | 0.010059 | 0.012312 |
| C | static, source graph, all norm, dual | 77 | 0.007887 | 0.016569 | 0.010114 | 0.010818 | 0.012500 |
| D | static, source graph, source norm, dual | 77 | 0.007880 | 0.015616 | 0.010174 | 0.009998 | 0.011929 |
| E | static, source graph, source norm, EMA-only | 79 | 0.007853 | 0.015033 | 0.010582 | 0.009892 | 0.011836 |
| F | no-GCL, all norm | 79 | 0.007853 | 0.014833 | 0.010474 | 0.010354 | 0.011887 |
| G | no-GCL, source norm | 77 | 0.007854 | 0.014516 | 0.010228 | 0.010048 | **0.011597** |

完成后能够直接成立的事实：

1. 这一个 calibration seed 中，G 的迁移均值最低；static GCL 没有显示出相对 no-GCL 的稳定优势，A 在 `digtime` 上尤其差。但 calibration 不进入正式均值或方法排名。
2. E 的迁移均值比 D 低约 0.79%，不能据此选择 EMA-only，因为 D 是 cache hit、E 是 cache miss，target update 与 downstream RNG 路径同时变化。
3. B/D 都是 cache hit、使用相同 source normalization，B 比 D 的迁移均值高约 3.21%；这只能作为 source-only topology 可能更好的指示性证据，单 seed 和未锁定的 CUDA/training determinism 不足以形成结论。
4. A/B 共享同一个 all-graph embedding，C/D 共享同一个 source-dual embedding；七份 normalization state 和 split 分别完全相同。A/B 仍相差约 6.23%，C/D 相差约 4.79%，F/G 在无 cache 差异时也相差约 2.50%。这些差异不能归因 normalization，并说明 cache RNG 不是唯一噪声源，训练级 CUDA/采样确定性也必须审计。
5. 七个 best epoch 全部位于 76-79，A-E 与 F-G 两个 matched group 均触发最后 10% 刷新规则。calibration 已足以在 formal run 前统一预注册 160 epochs；无需先正式跑一遍 80 epochs 再重复到 160。

#### 对现场分析的修正与边界

另一个 agent 对 cache/RNG、normalization、artifact 和 runner 缺口的诊断总体正确，后续沿用这些证据，但采用以下更精确的表述：

1. 当前已建立 source-only gradient/statistics 边界：SGRL 参数、normalizer 和 rebalancing prior 不使用 transfer labels 拟合。端到端 target-independent training 尚未建立，因为 cache miss 会在 downstream 前对全部迁移图执行随机邻居 embedding inference，并且没有随后重置 downstream RNG；目标图可能通过 RNG 消耗间接改变 downstream 轨迹。
2. `parameter-fit strict inductive` 与强时间边界 `no-access` 是两种不同协议。E0-E5 至少要证明 transfer 数据不参与参数拟合、模型选择或训练 RNG；E6 blind 才要求目标图和标签在最终一次性评估前完全不可访问。推理时读取未见图的特征/拓扑本身可以属于 inductive GNN inference，不应自动定性为泄漏。
3. formal E0 的硬阻塞项是：独立阶段 seed、cache/pretraining 后重置 downstream RNG、固定 SGRL embedding inference/eval view、真实 loader 重复评估、cache hit/miss 等价测试，以及实际 embedding/checkpoint hash。cache locking、完整环境/数据指纹、lazy transfer loading 和 processed-key 修复仍须完成，但若 runner 保证单写者和依赖顺序，可放在 confirmation/blind 阶段前闭环，不必全部阻塞早期 E1 探索。
4. WP5 中“没有真实 online step”应理解为“缺少覆盖真实 online step 的 EMA 公式和 cadence 测试”；当前代码本身存在 online training step。
5. `tests.test_protocol_boundaries` 的 5 项测试通过；完整 `tests/` suite 同时复核为 22/22 通过。前者仍不等于完整 protocol regression suite。

通过条件：

- [ ] E0-E5 strict artifact 证明 transfer 数据不参与参数拟合、模型选择或训练 RNG。
- [ ] E6 blind artifact 证明最终揭盲前未访问 transfer graph/labels。
- [ ] 重复 eval 结果稳定。
- [ ] 所有 cache 和 split 都可通过 artifact 恢复。
- [ ] 能解释 legacy 与 strict 的主要性能差异来源。
- [ ] cache hit/miss 不改变 downstream 初始化、训练采样或评估视图。
- [ ] normalization state 相同的 cell 不依据随机 MSE 差异宣称 normalization effect。
- [ ] 明确区分 parameter-fit strict inductive 与强时间边界 no-access，并按最终采用的定义提供测试证据。

统一延长规则：若任一主要 matched 候选在最后 10% epoch 内刷新 best，
则同组全部候选从 80 epoch 统一延长到 160 epoch；不得只延长表现较好的方法。
当前 `scripts/run_protocol_e0.sh` 只硬编码 80 epochs，没有 160 参数、组级触发器或独立 run tag。calibration 的 A-G 全部满足 `best_epoch >= 72`，因此 formal E0 应统一从头运行 160 epochs，并使用独立 formal log root；不得从 calibration 的 best checkpoint 接续。

### E1：严格协议最小复用基线

统一设置：strict inductive、source-fit normalization、E0 选定或并列保留的 target 模式、fixed evaluation、relation-balanced、MSE。调参使用 seeds 0-2；确认阶段使用全新 seeds 10-14。

| ID | 结构 | 作用 |
| --- | --- | --- |
| E1-A | no-GCL | 无对比学习下限 |
| E1-B | static | 原始 online encoder + downstream GNN 双 GNN 基线 |
| E1-C | init_reuse | 单 backbone 初始化复用基线 |
| E1-D | joint_shared, lambda=0 | 共享 backbone，仅监督训练 |
| E1-E | LoRA r8, lambda=0 | 共享 backbone轻量适配，不加联合 GCL |
| E1-F | LoRA r8, lambda=0.05 | 共享 backbone + 联合 GCL |

继续条件：

- [ ] 复用结构相对 static 的 source Val MSE 恶化不超过约 5%，或形成明确的 accuracy/compactness Pareto 优势。
- [ ] 部署参数量相对原双 GNN 推理路径减少至少约 35%。
- [ ] 预注册开发迁移 veto：三迁移集平均相对 static 恶化超过 10%，或任一迁移集恶化超过 25%，视为系统性退化。
- [ ] 表示漂移、base weight drift 和 LoRA delta 可解释。

若 LoRA 仍比 static source Val MSE 差 10% 以上，应暂停 P3，优先重新检查共享路径、stats fusion、supervised/GCL 梯度冲突和预训练接口，不直接扩大 rank sweep。

### E2：backbone learning-rate 复核

只在 E1-F 仍有竞争力时运行：

```text
LoRA r8, lambda=0.05
backbone_lr = 1e-4 / 1e-5 / 1e-6
LoRA/head lr = 1e-4
seed = 0
```

只依据 source validation 选择唯一 backbone LR。旧协议 P2 结果不能直接沿用为严格协议最优值。

### E3：P3 LoRA rank/layer

固定 E2 最优 LR 和 positive lambda：

```text
rank = 4 / 8 / 16
layer = first / last
seed = 0
```

共 6 个配置，可并行运行。seed0 初筛后仅保留 source-val Pareto 前列配置补 seeds1-2；筛选规则必须在运行前写入 selection manifest。选择顺序：

1. source Val MSE；
2. deployment 参数量；
3. 表示漂移和训练稳定性；
4. 应用已预注册的开发迁移 veto，不临时改变阈值。

### E4：P4 paired end-to-end seeds

锁定结构后使用全新 confirmation seeds 10-14，E4 后不得返回 E2/E3 重选：

| 方法 | Seeds |
| --- | --- |
| static | 10-14 |
| init_reuse | 10-14 |
| best LoRA, lambda=0 | 10-14 |
| best LoRA, lambda>0 | 10-14 |

执行依赖：

1. 每个 seed 先生成独立 source-only SGRL checkpoint。
2. 同一 seed 的四种 downstream 方法共享该 seed checkpoint。
3. 同一 seed 使用相同 source split 和 eval sampler fingerprint。
4. 不同 seed 的 cache 路径完全隔离。

正式报告 paired difference、mean/std 和每个 seed 原始值，不只报告各组独立均值。

### E5：P5 GCL/reuse 与 label rebalancing 因子实验

目标是回答“label rebalancing 和对比学习分别有效，但合起来可能变差”的教师任务。

完整结构：

| Backbone/GCL | MSE | GAI | BMC |
| --- | --- | --- | --- |
| no-GCL | neither | rebalancing only | rebalancing only |
| static | GCL only | original GCL + rebalancing | original GCL + rebalancing |
| shared, lambda=0 | reuse only | reuse + GAI | reuse + BMC |
| shared, fixed positive lambda | reuse + joint GCL | reuse + joint GCL + GAI | reuse + joint GCL + BMC |

执行方式：

1. 先跑所有尚缺的 seed0 单元，确认 loss 数值和训练稳定性。
2. 再为完整 4x3 因子矩阵补 confirmation seeds 10-14；配置完全相同的 E4 MSE 运行可以复用。
3. GAI 的 GMM、BMC/GAI 的 trainable sigma 和 checkpoint criterion state 必须进入 artifact。
4. 比较 interaction effect，而不只是分别挑每行最小值。
5. 预注册 GAI/BMC 在 positive-lambda 与 lambda=0 下相对 MSE 改变量之差作为 primary interaction。
6. 报告 overall、fixed source-train label bins、bias；relation type 可用后再报告 per-relation。

### E6：最终盲测与自然分布

只有以下项目全部锁定后才能开始：

- backbone 结构；
- LoRA rank/layer；
- lambda 和 learning rate；
- loss；
- checkpoint selection；
- relation-balanced/natural evaluator；
- deployment export。

盲测步骤：

1. 记录 sp8192w raw file SHA256，不预先读取 labels 做设计选择。
2. 一次性生成 relation-balanced 和 natural-distribution 两套 processed artifact。
3. 在任何结果展示前，一次性批量完成 `method x confirmation seed x distribution`，不是挑单个最佳 seed。
4. 不根据 sp8192w 结果重新调参。
5. 若需要与论文公开大图比较，再评估 ultra8t/sandwich，并明确它们不是本项目真正盲测集。

## 7. 并行运行策略

1. 先完成代码与测试，不边修改共享代码边跑正式实验。
2. 预训练 checkpoint 是 downstream 任务的依赖；必须先完成并校验，再并行启动下游矩阵。
3. 每个实验使用唯一 log/artifact/cache 路径，避免并发覆盖。
4. 当前模型单任务显存通常约 0.6GB，但并行度不能只按显存决定；还要检查 GPU utilization、CPU 线程、I/O 和其他用户进程。
5. GPU4 可在资源允许时运行多个独立下游任务；建议逐步增加并发并保留显存缓冲。
6. 每个任务限制 BLAS/OMP 线程，避免大量小模型并发压垮 CPU。
7. 使用 tmux 或可靠队列保存任务；断线后通过 artifact status、PID 和日志更新时间核验，不凭 tmux 名称推断结果。
8. 不因 GPU 利用率高而终止任何其他用户进程。

### E0 预算快照

| 项目 | Seed0 首轮 | 完整 tuning seeds 0-2 |
| --- | ---: | ---: |
| Static topology x normalization | 4 runs | 12 runs |
| Strict target-update contrast | 1 additional run | 3 additional runs |
| no-GCL normalization contrast | 2 runs | 6 runs |
| 合计 downstream cells | 7 | 21 |

按旧 80-epoch运行估算，单 cell 约 25-40 分钟，聚合 GPU 时间约 9-14
小时；GPU4 上 5-7 路并发时 seed0 首轮预计约 1-1.5 小时墙钟时间。
Static A/B 共享 all-graph SGRL cache，C/D 共享 source-only dual cache，必须先让
A/C 完成预训练和 cache 写入，再启动 B/D，禁止并发竞争同一 cache。EMA-only 和
no-GCL 单元相互独立。若触发统一 160-epoch 延长规则，预算近似翻倍。

当前 7 个 seed0 任务已经启动并作为 calibration sunk cost 运行。由于 cache-dependent downstream RNG 混杂，它们不能替代修复后的 formal seed0；预算重算时应把“已消耗 calibration GPU-hours”和“仍需 formal E0 GPU-hours”分开列出。

## 8. 正式 artifact 最低字段

每个新协议运行至少记录：

```text
git_commit
git_dirty / source_tree_hash
full_command
python_path
host / OS / torch / PyG / CUDA / cuDNN / driver / GPU
protocol
source_graph_names
transfer_graph_names
sampling_distribution
raw_data_hashes
processed_data_hashes
split_seed
pretraining_seed
downstream_seed
train_sampler_seed
relation_sample_seed
split_indices_hash
normalization_state_hash
sgrl_target_update
sgrl_target_initialization / update_cadence
sgrl_checkpoint_path/hash
embedding_path/hash
eval_seed
eval_sampler_fingerprint
loss and criterion state
best_epoch
best raw source-val MSE
final-only transfer metrics
deployment parameter count
status and ended_at
```

summary 只允许读取 structured metrics，不从 console 正则解析正式结果。

## 9. 结果表述边界

协议修复前可以说：

- S6 joint-shared + mergeable LoRA 是可运行的单-backbone 方向；
- 当前旧协议下 positive-lambda LoRA 有部分数据集收益，但未稳定超过 static；
- static 的 downstream-only seed 波动较小；
- 低 backbone LR 能减少 base drift，但不一定提高 source validation；
- 当前关系平衡开发集上，reuse 对 digtime 较有帮助。

协议修复前不能说：

- 已实现论文所述 strict zero-shot 复现；
- 新方案已稳定超过 static；
- static 五 seed 是完整端到端稳定性；
- 当前 test MSE 是自然边分布性能；
- LDS 已正确集成；
- GAI/BNI/LDS 的 prior 完全无泄漏；
- 已生成可独立部署的 29,378 参数 artifact；
- README 的 42.48% 改善已在当前分支复现。

## 10. 后续 agent 的立即任务

不要直接启动 P3。按以下顺序推进：

1. [x] 只读核验 Git、五项现有协议单测和 11 个 legacy completed artifacts。
2. [x] 在 `EXPERIMENT_LOG.md` 标明 legacy 结果的数据边界，禁止与 strict 结果混合。
3. [x] 让 7 个 `E0-seed0-calibration-v1` 任务自然完成；已核验 final status、best epoch、final-only transfer 和 artifact。
4. [x] 将七份相同 normalization state 写成机制结论；不得从 A/B、C/D、F/G 的随机 MSE 差异推断 normalization effect。
5. [ ] 修复 SGRL cache hit/miss 导致的 downstream RNG 起点差异，拆分并记录 pretraining/downstream/train-sampler/eval seeds。
6. [ ] 补真实 fixed-eval 重复测试、cache-hit/miss 等价测试、固定 embedding inference view，以及覆盖真实 online step 的 EMA 数值/cadence 测试。
7. [ ] formal E0 前记录实际 embedding/checkpoint hash；confirmation/blind 前补 raw/processed/code/eval-view 指纹、原子 cache 发布、并发锁和 strong-boundary forbidden-transfer smoke。
8. [ ] 将 formal E0 runner 固定为 160 epochs、独立 formal run tag，并增加 cache 单写者/依赖检查。
9. [ ] 在修复后重新运行 formal E0 seeds 0-2；当前 calibration 不进入 formal group mean。
10. [ ] formal E0 通过后才运行 E1；只有达到继续条件后才运行 E2/E3。
11. [ ] 锁定架构后运行 E4/E5，最后完成 WP8/WP9 和 E6 blind evaluation。

## 11. 一句话结论

现有实验已经证明“把 online GNN 与 downstream GNN 合成单一共享 backbone”在工程上可行，但完成的 E0 seed0 仍只是 calibration：normalization state 在本数据上完全相同，cache hit/miss 和训练级非确定性又污染了 matched 对照。先修复独立阶段 RNG、固定 inference/eval view 和必要 cache provenance，再统一以 160 epochs 重跑 formal E0 seeds 0-2；强 no-access、完整环境指纹和原子 cache 等增强项最迟在 confirmation/blind 阶段前闭环。只有在此基础上，E1-E5 的精度、紧凑性和 rebalancing interaction 结论才具有正式研究价值。
