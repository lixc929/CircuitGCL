# CircuitGCL 问题清单与后续 Agent 交接

## 0. 文档用途与审阅快照

本文档用于把当前 CircuitGCL 改进工作的真实状态、已确认问题、证据位置和建议处理顺序交接给后续 agent。它不是新的实验结论，也不是要求一次性修改所有问题。

- 初次审阅时间：2026-07-11 18:48 CST
- 本次更新：2026-07-11 20:49 CST；已复核 `PROTOCOL_REPAIR_EXPERIMENT_HANDOFF.md`、完成后的 P1/P2/P4 artifacts、SGRL target-update 语义及 E0-E6 设计
- 分支：`test`
- 审阅提交：`71bb8829c8c6ba8db3738c4133baac10c0b16fa5`
- 初次审阅前状态：工作树 clean，`test` 与 `lixc929/test` 同步
- 本次更新前状态：仅 `AGENT_HANDOFF_ISSUES.md` 和 `PROTOCOL_REPAIR_EXPERIMENT_HANDOFF.md` 为 untracked；未发现代码改动
- 当前成熟主线：edge regression
- 当前总判断：S6/LoRA 方向可行，但尚未稳定超过 `static + MSE`；严格迁移协议和数据边界仍未闭环

初次审阅和本次复核都只读取代码、文档、论文及已有 artifact，没有运行会写缓存的测试。2026-07-11 20:49 的唯一写操作是更新本文档，未修改项目代码、实验 artifact 或 `PROTOCOL_REPAIR_EXPERIMENT_HANDOFF.md`。

## 1. 后续 Agent 首先必须遵守的约束

1. P1/P2/P4 的 11 个 structured artifacts 已全部完成；正式引用前仍必须核对 completed/status/test_results/checkpoint/config，而不是从 console 文本取数。
2. 不要直接进入 P3、P5、HCA 或 SRL 实现。先处理本文的 P0 数据/协议问题，并修订 E0-E6 的选择与消融规则，否则后续实验可能建立在错误评估边界上。
3. 不要把 SGRL 当前的 dual target update 简化成后续误加的 legacy bug，也不要未经消融就把 `ema_only` 设成唯一正式默认值；详见 11.3 和 17.1。
4. 修改前先重新执行只读的 `git status --short --branch`；`logs/` 等被忽略目录中的运行状态不会出现在 Git 状态里。
5. 所有正式比较必须区分：
   - strict inductive：目标电路的拓扑、特征、标签都不参与训练期拟合；
   - transductive/domain adaptation：目标电路拓扑和无标签特征可见，但目标标签不可见。
6. 当前 digtime、timing_ctrl、array 已经长期参与架构分析，应视为开发 benchmark，不应继续称为最终盲测集。
7. `sp8192w` 在模型、loss、超参数、checkpoint 和分析脚本冻结前不得被用于设计选择；技术管线可以先在非盲大图上验证。

## 2. 当前研究推进状态

| 阶段 | 状态 | 目前能成立的结论 |
| --- | --- | --- |
| S0-S3 | 完成 | 已澄清 online encoder reuse，并证明保留原 downstream GNN 的在线特征复用可恢复到 static/no-GCL 附近 |
| S4 | 完成 | `init_reuse` 实现单一紧凑 downstream GNN，但未全面胜过 static |
| S5 | 当前轮结束 | partial sharing 可运行；三 seed 下验证和三个迁移均值仍落后 static，digtime seed 敏感 |
| S5.5 | 完成诊断 | 解冻后的 shared representation 快速漂移是主要问题，不是 gate 饱和 |
| S6 | 已实现、探索性验证 | joint-shared + mergeable LoRA 有小幅分数据集收益，尚未稳定胜过 static |
| P0（旧运行协议修复项） | 完成 | raw-MSE 选模、best reload 后 final-only transfer、隔离 artifact 已实现；不代表本文标为 P0 优先级的数据边界问题已解决 |
| P1/P2 | 已完成旧协议审计运行 | 11 个 artifacts 均 completed；positive-lambda LoRA 有局部收益，但没有任何复用方案同时胜过 static 的 source validation 和全部三个迁移集 |
| P4 static | 已完成 downstream-only 五 seed | 共享同一 static embedding/pretraining realization，只能说明 downstream 波动，不能代表端到端预训练稳定性 |
| P3/P5 | 未完成 | rank/layer、严格协议下的独立调参/确认 seeds、MSE/GAI/BMC 因子比较仍被协议修复阻塞 |
| HCA/SRL | 仅有阅读和设计 | 尚未实现，不应写成当前功能 |
| node/classification | 旧路径 | 存在 split、指标和 loss 协议错误，不能作为当前正式证据 |

主要状态文档：

- `TEACHER_TASKS_README.md:127-152`
- `TEACHER_TASKS_README.md:398-491`
- `EXPERIMENT_LOG.md:1103-1388`
- `EXPERIMENT_LOG.md:1390-1672`

## 3. P1/P2/P4 完成状态与证据边界

`logs/s6_p1_p2_protocol_20260711/experiment_summary.json` 已核实：

```text
num_artifacts = 11
num_completed = 11
num_running = 0
num_anomalous = 0
```

这里的 `anomalous=0` 仅表示 structured artifact 没有缺 metrics、状态不一致、缺 checkpoint 或重复配置等汇总器定义的异常，不代表科学协议、GPU 争用或启动来源没有问题。

11 个 run 均有 completed `run_config.json`/`metrics.json`、三个迁移集结果及 `best_model.pt`。正式复用时仍应检查：

```text
run_config.json.status == completed
metrics.json.status == completed
metrics.json.test_results 非空
best_model.pt 存在
run_config.json.ended_at 存在
```

已核实的主要结果：

- 所有复用方案在 digtime 上都优于 seed0 static。
- 没有任何复用方案同时胜过 static 的 source validation 和三个迁移集。
- positive-lambda LoRA 中，默认统一 `lr=1e-4` 的 validation 与三个迁移集均值最好；这里是全局统一学习率，不应误写成显式独立 `backbone_lr=1e-4`。
- static 五 seed 的三个迁移集平均为 `0.012256976341 ± 0.000569053913`（原汇总使用 population std）；正式统计应改报 sample SD、每个 paired difference 和置信区间。
- 所有 best epoch 位于 77-79。它只能支持“80 epoch 没有明显过长”，同时强烈提示 80 epoch 可能偏短；不能据此认定已经充分收敛。

运行来源的本地证据只能精确支持：历史 9 个 GPU4 artifacts 中，`p1_joint_lambda0` 来自最终 GPU4 runner，4 个任务由原 GPU3 lane 改到 GPU4，另 4 个 P4 static seeds 是额外并发任务。现有 artifact 没有记录 launcher parent 或 tmux session，因此“按用户要求”“全部由独立 tmux 迁移”等说法只能标为操作者说明，不能写成完全可独立审计的事实。`queue_status.tsv` 中只有最终 GPU4 lane 有明确 `queue_finished`；应表述为“最终 GPU4 串行 lane 于 19:54 正常结束”。

详细数值和原始证据入口：

- `PROTOCOL_REPAIR_EXPERIMENT_HANDOFF.md:39-83`
- `logs/s6_p1_p2_protocol_20260711/experiment_runs.tsv`
- `logs/s6_p1_p2_protocol_20260711/experiment_summary.json`
- `logs/s6_p1_p2_protocol_20260711/queue_status.tsv`

## 4. P0：严格迁移协议边界不成立

### 4.1 SGRL 预训练读取全部目标电路拓扑

证据链：

- `main.py:385-424`：一次加载 `ssram+digtime+timing_ctrl+array_128_32_8t`，在 downstream source/test split 前调用 SGRL。
- `sram_dataset.py:420-449`：`adaption_for_sgrl()` 遍历 `dataset.names` 的所有电路并拼成一个 Batch。
- `sgrl_train.py:143-179`：在这个全电路 Batch 上训练 SGRL。

SGRL 当前不读取寄生标签，因此不是监督标签泄漏；但它看到了 digtime、timing_ctrl、array 的节点类型和拓扑。这属于无标签目标域可见的 transductive/domain-adaptation protocol，不是 strict unseen-circuit inductive transfer。

影响：

- 当前结果不能直接支撑“目标电路在训练期完全未见”的表述。
- static embedding、online feature、partial shared 和 joint shared 都继承该边界。
- 当前代码行为与 README 中 SSRAM 为 Train/Val、其他电路为 Test 的直观口径不一致，见 `readme.md:121-134`。

后续必须先做协议选择：

- 若保留 transductive：文档、表格和论文表述必须明确目标拓扑/特征可见。
- 若目标是 inductive：SGRL 只能基于 source graph 拟合；目标电路只能在最终推理时进入 encoder。

验收标准：

- 运行 config 明确记录 `protocol=inductive|transductive`。
- source-only 模式下，测试应断言 SGRL 数据构造没有访问 `dataset[1:]`。
- source-only 和 all-circuit 版本必须作为两个不同实验配置，不能共享 checkpoint/cache key。

### 4.2 节点统计归一化使用全部目标电路

证据：

- `sram_dataset.py:75-99`：所有电路先 collate 到 `_data`。
- `sram_dataset.py:101-120`：net/device 的最大值在整份 `_data` 上统计。
- `downstream_train.py:1197-1209`：所有模式在 source/test sampling 前调用 `dataset.norm_nfeat()`。

因此即使 `--sgrl 0`，no-GCL 基线也使用了目标域 covariate 统计量。内部横向比较可能仍相对公平，但不能称为 strict inductive。

验收标准：

- normalization state 只在 source-train 或至少 source graph 上 fit。
- 把 normalization state 保存进 artifact，并在 val/test 只做 transform。
- 增加测试：改变 transfer 图的极端 node_attr 不应改变 source normalization state。

### 4.3 目标电路已被反复用于架构开发

S2-S6 的下一步选择多次依据 digtime/timing_ctrl/array 指标决定，例如：

- `TEACHER_TASKS_README.md:424-451`
- `TEACHER_TASKS_README.md:462-479`
- `EXPERIMENT_LOG.md:1465-1484`

即使今后 P2/P3 只用 source validation 选超参，这三个电路也已经是开发 benchmark。当前计划将 `sp8192w` 保留为最终盲测电路；ultra8t/sandwich 若继续用于管线验证或补充比较，必须明确属于非盲辅助电路，且是否运行应在看到 sp8192w 结果前决定。

## 5. P0：LDS 当前实际训练链路错误

### 5.1 权重计算发生在 loader 创建之后

执行顺序：

1. `downstream_train.py:1206-1209` 先调用 `dataset_sampling()`。
2. `sampling.py:101-109,117-125` 通过高级索引生成 train/val 的 `edge_label` 并传给 loader。
3. `downstream_train.py:1230-1236` 才计算 LDS 权重并改写 `dataset._data.edge_label[:,1]`。

loader 已经持有切片后的标签张量，后续改写 `_data` 不会追溯更新这些 tensors。于是 `compute_loss()` 在 `downstream_train.py:188-195` 实际把原始 class ID `0..4` 当作 sample weight：class 0 权重为 0，class 4 权重为 4。

现有审计只证明 WeightedMSE 的形状广播从 `[N,N]` 修成 `[N,1]`，没有证明 LDS 权重进入了 loader：

- `scripts/audit_rebalancing_original_vs_clean.py:134-151`
- `tests/test_rebalancing_protocol.py:12-40` 只测试 GMM source-only

另外，`main.py:274-278` 的默认 `--lds_sigma=0.02` 与离散类别轴 `0..4` 不匹配，几乎不会产生跨类别平滑，帮助文本却建议 1 或 2。

验收标准：

- 在 split 已确定后，直接为 train/val 样本构造正确 LDS weight，或让 loader 显式携带第三列/独立 weight tensor。
- 端到端测试从 loader 取一个 batch，断言 weight 与预计算结果一致，且不是整数 class ID。
- class 0 样本不能因实现时序意外获得 0 权重。
- LDS 分布只能由 source-train labels 拟合。

## 6. P0：GAI/BNI/LDS 的 label-prior 边界仍不正确

### 6.1 GAI 已 source-only，但仍包含 source validation 标签

积极改进：`balanced_mse.py:76-101` 的 GMM 只访问 `dataset[0]` 的连续标签，并在 run-local artifact 中原子保存。

剩余问题：

- source train/val indices 只存在于 `sampling.py:94-102` 的局部变量。
- `train_gmm()` 仍拟合完整 SSRAM，包含 20% source validation 标签。

这会让 validation 不再完全独立于训练期 label-prior 拟合。

### 6.2 BNI/LDS 直接读取全部 transfer 标签

`downstream_train.py:1223-1235` 从 `dataset._data.edge_label[:,1]` 构造 BNI/LDS 分布；该 `_data` 包含全部四个电路。这是目标标签泄漏。

P5 当前计划只比较 MSE/GAI/BMC，所以 BNI/LDS 不阻塞 P5 的最小矩阵，但这两个公开 CLI 选项不能视为协议正确。

验收标准：

- split indices 必须被持久化并传给所有 label-prior 方法。
- GMM、LDS、BNI、class counts 只读取 source-train labels。
- 单测提供一个访问 transfer label 就抛异常的数据集，并额外验证 source validation 也没有被访问。

## 7. P0：当前测试集是关系类型平衡子集，不是自然分布

数据预处理会对每个电路的三类耦合关系分别降采样到该电路最少关系类型的数量：

- `sram_dataset.py:332-344`
- `utils.py:105-153`

历史日志显示，从电容范围过滤后的目标边到最终保存边，保留比例约为：

| 电路 | 过滤后目标边 | 最终保存边 | 约保留比例 |
| --- | ---: | ---: | ---: |
| SSRAM | 624,268 | 175,413 | 28.10% |
| digtime | 33,606 | 2,814 | 8.37% |
| timing_ctrl | 33,581 | 3,333 | 9.93% |
| array | 358,684 | 73,569 | 20.51% |

最终数量都精确是 3 的倍数，对应三类关系等量拼接。随后 `sram_dataset.py:347-372` 只保存正边及电容标签，并丢弃目标关系类型，后续无法直接做 per-relation error audit。

影响：

- 当前 test MSE 和 label-distribution audit 描述的是“指定电容范围内、关系类型等量的随机子集”。
- 它适合固定协议内横向比较，但不能等同于真实部署自然分布。
- 小测试电路只保留约 8%-10%，一次随机预处理子集可能影响结果。

验收标准：

- 明确命名并同时报告 `natural-distribution` 和 `relation-balanced` 两种评估。
- 保存 target relation type、采样 indices、raw/filtered/per-relation/final counts。
- per-relation 和 overall 指标都进入 structured metrics。
- processed cache key 必须包含采样协议和随机种子/indices fingerprint。

## 8. P0：node/classification 路径不能作为正式结果

### 8.1 node train/val supervision 泄漏

- `sampling.py:30-45`：虽然用 net mask 统计类别，但仍把所有节点作为 train/val roots。
- PyG NeighborLoader 只保证前 `batch.batch_size` 个节点是 seed roots。
- 模型却对采样子图中的全部 net 节点或全部节点计算监督，没有截取 roots：
  - `model.py:403-413`
  - `model.py:891-900`
  - `model.py:1120-1129`
  - `model.py:1318-1327`

训练 batch 的邻域节点可能包含 validation roots，且其标签会参与训练 loss。

### 8.2 classification loss 和指标协议不一致

- `downstream_train.py:78-100`：训练 F1 随机只保留约 10% 多数类，指标非标准且每次不同。
- `downstream_train.py:224`：validation Logger 没有传 `max_label`，训练与验证指标口径不同。
- `downstream_train.py:951-958`：Balanced Softmax 使用 mini-batch 类别计数；缺失类可导致 `log(0)`。
- `downstream_train.py:170-177`：validation `compute_loss()` 对 bsmCE 使用普通 CE，不是训练目标。
- `downstream_train.py:990-1011`：每次 validation F1 提升就读取 test；没有 best checkpoint reload 或 structured metrics；打印的 “Best epoch” 实际是当前 epoch。

验收标准：

- node loss 仅作用于 seed roots，并且 train/val root sets 持久化且不交叉。
- Balanced Softmax 使用 source-train 全局 class counts，并对零计数有明确策略。
- 训练/验证/测试使用相同、标准且确定的指标定义。
- classification 与 regression 一样使用 isolated best checkpoint 和 final-only test。

在这些条件满足前，README 中的 node classification 结果不能用于证明当前分支改进。

## 9. P1：评估仍包含未量化的邻居采样噪声

`sampling.py:105-158` 的 val/test LinkNeighborLoader 虽然 `shuffle=False`，但有限邻居采样仍是随机的。当前影响包括：

- 同一 checkpoint 重复评估，MSE 可能变化。
- checkpoint 当时的 `best_val_mse` 与训练结束后 `val:best` 复评值可能不同。
- 各方法训练过程消耗 RNG 的方式不同，最终 test loader 看到的邻域样本也可能不同。
- P2 的 `2e-5` tie threshold 可能小于评估采样噪声，但仓库尚未量化该噪声。

验收标准：

- 对相同 checkpoint 重复评估多次，报告 sampling variance。
- 正式比较使用固定 eval batches、固定 generator，或完整邻域评估。
- structured metrics 记录 eval seed/sampler fingerprint。
- paired-seed 比较应使用相同 evaluation neighborhoods。

## 10. P1：缓存键和 multi-seed 语义不完整

### 10.1 static embedding 缓存

`main.py:403-415` 的缓存键只编码 dataset、model、layers、dim、activation，缺少：

- seed
- dropout
- batch size / neighbor count
- learning rates / momentum / weight decay
- pretraining epochs
- data hash / processed sample indices
- Git commit / implementation version

### 10.2 SGRL checkpoint 缓存

`sgrl_train.py:193-201` 的 checkpoint 名虽包含 dropout，仍缺少 seed、采样和多数优化配置；只要文件存在就无条件复用。

因此当前 multi-seed 主要表示“同一个 pretrained checkpoint 上的 downstream seed 波动”，不是端到端预训练 seed 稳定性。该设计可以用于隔离 downstream variance，但必须明确报告，不能写成完整方法的 multi-seed 稳定性。

### 10.3 processed graph cache

`sram_dataset.py:382-417` 的 processed key 只包含 name、部分 sample rate 和一位小数的 `neg_edge_ratio`，缺少 raw hash、`to_undirected`、随机 seed 等；`neg_edge_ratio=0.04` 甚至可能与 `0.0` 命名冲突。

验收标准：

- 使用完整 scientific config + data fingerprint 生成内容寻址缓存键。
- artifact 记录加载的 processed graph、SGRL checkpoint、embedding 的绝对路径和 SHA256。
- split indices、采样 indices、normalization state 都可恢复。
- 明确区分 `downstream-only seeds` 与 `end-to-end seeds`。

## 11. P1：SGRL 声明支持的部分路径尚未闭环

### 11.1 edge-aware encoder 输入错误

`sgrl_models.py:93-98` 把 `edge_type_embed` 应用于 `batch.x`，而不是 `batch.edge_type`。默认 ClusterGCN 不使用 `ze`，所以当前 canonical 实验未触发；GINE/ResGatedGCN 则可能得到错误长度或错误语义的 edge features。

`OnlineFeatureGraphHead._make_sgrl_batch()` 也没有传递 edge type，见 `model.py:488-500`。

### 11.2 不可达或未使用逻辑

- `sgrl_models.py:174-182`：`num_hop` 路径位于提前 `return` 后，不可达。
- `sgrl_train.py:143-145,210-212`：计算并传入 normalized adjacency，但当前训练函数不使用它。
- `sgrl_models.py:31`：`use_stats=False` 硬编码，当前 SGRL 实际只用节点类型和拓扑。

### 11.3 target encoder 算法定义需要澄清

当前 SGRL target encoder 同时接受 RSM/scattering loss 的独立 optimizer 更新和 EMA：

- `sgrl_models.py:154-157`
- `sgrl_models.py:159-165,185-196`
- `sgrl_train.py:27-52,158-164,204-227`

复核后的结论不是“当前代码明确错误”，而是论文文字和上游方法存在需要显式审计的歧义：

1. CircuitGCL 论文第 5 页 §III-B 的文字确实说 alignment 时只有 online encoder 接收梯度、target 固定，并在每个 epoch 后 EMA。这支持把 EMA-only 作为一种论文文字直译。
2. 同一 CircuitGCL 论文第 4 页 §III-A Eq. (5) 又对 target representation 定义 `L_scattering`，并说明通过最小化该损失执行 RSM。
3. 原始 SGRL NeurIPS 论文 Fig. 3 和官方实现都采用 dual 机制：online 对 target stop-gradient；target 通过 RSM/scattering loss 更新；随后再 EMA。
4. `git blame` 显示本地 dual 实现来自 CircuitGCL 作者仓库的初始提交，不是本轮改进引入的错误。

外部一手证据：

- [SGRL NeurIPS 2024 论文](https://proceedings.neurips.cc/paper_files/paper/2024/file/d0ffb35aaa7faa894afe5060c694d674-Paper-Conference.pdf)
- [SGRL 官方仓库](https://github.com/hedongxiao-tju/SGRL)
- [官方训练实现](https://github.com/hedongxiao-tju/SGRL/blob/main/train.py)

因此 `PROTOCOL_REPAIR_EXPERIMENT_HANDOFF.md:117-131` 把当前实现定性为明确的 paper-faithful mismatch、把 `ema_only` 设成唯一正式默认值，证据不足。建议使用中性且可审计的模式名：

```text
sgrl_dual_rsm_ema          # SGRL 官方/作者代码基线
circuitgcl_text_ema_only   # CircuitGCL §III-B 文字直译消融
```

如果实现 `circuitgcl_text_ema_only`：

- target 必须从 online 完整同步初始化；当前两个 encoder 是分别随机初始化，直接使用 `tau=0.99` 的 epoch EMA 会长期保留随机 teacher 偏差。
- 必须明确显式 scattering loss 是删除、迁移到其他参数，还是重新定义；不能只删除 target optimizer 后仍无条件称为完整 SGRL。
- 预训练开关应命名为 `sgrl_pretrain_target_update`。S6 downstream 的 EMA target 是另一条链路，当前每 batch 更新，必须另设 `joint_target_update`/`joint_target_cadence`，不能共用一个含糊开关。
- E0 中可以比较两种模式，但在得到 matched 结果前，E1 不应预设 EMA-only 为正式默认。

验收标准：

- 对每个 CLI 声明支持的 `cl_model` 增加 forward smoke test。
- 明确 SGRL 的实际输入、loss、target 初始化、更新顺序和与 S6 joint GCL 的差异。
- 对两种 target 模式分别测试 optimizer 参数集合、stop-gradient、EMA cadence、cache key 和 checkpoint 恢复。
- 删除或实现不可达逻辑前，先确认是否要保持上游复现兼容性。

## 12. P1：LoRA 部署目前只有理论核算，没有真实部署 artifact

代码提供：

- `model.py:875-888`：deployment parameter count 和 LoRA merge 方法。
- `tests/test_joint_shared.py:135-177`：零初始化与 merge 等价性的单元测试。

但生产训练路径没有调用 `merge_task_lora_for_deployment()`。`run_artifacts.py:83-101` 保存的 best checkpoint 仍包含：

- 未合并 LoRA
- training-only EMA target backbone
- predictor
- optimizer state

因此“部署仍为 29,378 参数”目前是参数核算和 API 可行性验证，不是已经导出并加载验证过的精简模型。

验收标准：

- 独立 export 命令恢复 best checkpoint、merge LoRA、移除 target/predictor/optimizer。
- 保存 deployment-only config/state。
- 在固定 batch 上比较训练 checkpoint 与部署 artifact 输出，误差满足预设容差。
- 新进程能够只依赖部署 artifact 完成 inference。

## 13. P1：正式结果证据和远端复现仍不完整

积极部分：

- `run_artifacts.py` 已保存 args、command、git commit、model/optimizer/criterion state。
- `scripts/summarize_experiments.py` 能识别 duplicate、running、completed，并按 config/seed 选择 canonical artifact。
- P0 后 regression 使用 raw MSE 和 final-only transfer。

剩余问题：

- `.gitignore:14-27` 忽略全部 logs、checkpoint、embedding、pkl 和 PDF。
- 远端 clone 只有 Markdown 数字，没有原始 artifact、hash 或可恢复 checkpoint。
- S5 和部分旧 S6 运行早于隔离 artifact，历史共享 checkpoint 已可能被覆盖。
- 旧 GAI/BMC 结果的 `noise_sigma` 实际未进入 optimizer；只能视为 fixed-sigma 开发结果，见 `EXPERIMENT_LOG.md:1657-1672`。
- 主 `readme.md:217-248` 的最高 42.48% MSE 改善是上游论文口径，不是当前 S5/S6 分支重新验证出的结果。

验收标准：

- 为正式表格发布机器可读 summary、artifact manifest 和 hashes。
- 把“上游论文结果”“本地开发结果”“修正协议后的正式结果”分成三层。
- 禁止从 console 正则解析正式结果；只读取 structured metrics。

## 14. P2：其他已确认但不阻塞当前 edge-regression 主线的问题

### 14.1 negative sampling 接口未闭环

- `utils.py:29-55` 的 global 分支返回 3 个值，`sram_dataset.py:332-334` 只接收 2 个，`neg_ratio>1` 会解包失败。
- structured negative sampling 不约束负边终点节点类型，却直接继承正边关系类型。
- `sram_dataset.py:347-350` 对 edge task 最终只保留正边，负边没有进入任务标签。

当前 canonical regression 使用 `neg_edge_ratio=0`，所以主实验不受影响；但公开参数的非零路径不能视为可用功能。

### 14.2 CLI 类型解析不可靠

- `main.py:24`：`--net_only type=bool`，命令行字符串 `False` 仍通常解析为 True。
- `main.py:269`：`--class_boundaries type=list`，从 CLI 传字符串不会可靠解析为浮点列表。

### 14.3 static-online consistency audit 覆盖有限

- `scripts/audit_static_online_consistency.py:116-122` 只取每图最前 2,048 条边，不是随机或关系/标签分层抽样。
- 边按关系类型拼接，因此可能只覆盖前一两类关系。
- `scripts/audit_static_online_consistency.py:150-152` 通过前 `2*edge_count` 个节点推断 endpoints，依赖 PyG 排列约定，没有用显式映射交叉验证。

该审计足以排除当前 ClusterGCN 配置的 gross endpoint mismatch，但不能证明所有关系、标签 bins、模型和中间层都一致。

## 15. 已经有价值且应保留的改进

后续修复时不要破坏以下成果：

1. GAI 只使用连续标签且 run-local 原子保存 GMM：`balanced_mse.py:76-101`。
2. BMC `[N,N]` logits + long targets：`balanced_mse.py:133-152`。
3. WeightedMSE 显式对齐 `[N,1]`：`balanced_mse.py:181-191`。
4. raw validation MSE：`downstream_train.py:103-113,249-255`。
5. isolated run config/checkpoint/metrics：`run_artifacts.py:37-147`。
6. best checkpoint reload 后 final-only transfer：`downstream_train.py:836-892`。
7. criterion parameter 进入 optimizer，criterion state 随 checkpoint 保存：`downstream_train.py:1095-1185`、`run_artifacts.py:96-116`。
8. partial-shared 解冻时保留 Adam state：`downstream_train.py:294-336`。
9. RNG-neutral representation audit：`downstream_train.py:359-652`。
10. joint-shared、EMA target、独立 backbone LR、mergeable LoRA：`model.py:660-924`。
11. fixed ten-bin MSE/MAE/bias：`downstream_train.py:124-155`。
12. duplicate/artifact summarizer：`scripts/summarize_experiments.py`。

## 16. 当前结果哪些可以说，哪些不能说

### 可以说

- 原 public regression rebalancing 路径存在具体 GAI/BMC/LDS 形状或标签使用问题；当前分支已修复其中 GAI 连续标签、BMC 公式和 WeightedMSE 广播。
- S5 partial sharing 实现可运行，但三 seed 下没有匹配 static 的平均准确率和稳定性。
- static cached embedding 与相同 checkpoint 的 online endpoint representation 在当前审计配置下近乎一致。
- S5 解冻后发生明显 shared representation drift，gate 没有饱和。
- S6 LoRA 是可行的单-backbone adaptation 方向；已完成的 80-epoch 结果仍属于 `legacy-transductive-v1`，不能替代严格协议结论。
- timing_ctrl 在当前“关系平衡后的处理数据”上具有最大的标签分布偏移。

### 不能说

- 不能说当前新方案已稳定超过 `static + MSE`。
- 不能把已完成的 80-epoch legacy 结果与未来 strict artifacts 直接合表或据此预选 strict 协议赢家。
- 不能把 best epoch 77-79 解读成 80 epoch 已充分收敛。
- 不能把 `ema_only` 或 dual 中任一模式在未经 matched 消融前称为唯一正确实现。
- 不能说当前实验是 strict unseen-circuit inductive transfer。
- 不能把当前 test MSE 当成自然原始边分布性能。
- 不能说 LDS 集成已经正确。
- 不能把旧 GAI/BMC fixed-sigma 结果当作最终 paper-faithful 比较。
- 不能把当前 node/classification 结果当作无泄漏的正式结果。
- 不能说已经生成了 29,378 参数的可部署 artifact。
- 不能把 README 的 42.48% 改善当成本分支的本地复现结论。

## 17. 对 `PROTOCOL_REPAIR_EXPERIMENT_HANDOFF.md` 的复核与修订要求

总体判断：主方向、协议修复优先级和 artifact 意识是合理的，但当前版本更像实验路线图，还不能原样作为正式预注册执行。以下问题会直接改变归因或结论，必须先修订。

### 17.1 E0 target-update 消融应保留，但不预设赢家

- 按 11.3 使用 `sgrl_dual_rsm_ema` 与 `circuitgcl_text_ema_only` 两个中性模式名。
- dual 是上游作者实现和复现基线，不应仅降格为 legacy audit。
- EMA-only 是有价值的论文文字直译消融，但必须同步初始化 target，并明确 scattering loss 的去向。
- 在 matched E0 结果出来前，删除 E1 中“统一使用 EMA-only”的预设。

### 17.2 E0 当前无法分别归因 topology 与 normalization

`E0-A2 -> E0-C` 同时改变 all/source GCL graph、all/source normalization，以及旧/新 harness 的 split/cache/evaluator。至少应在同一个修复后 harness 中补 static 2×2：

```text
GCL graph:      all circuits / source only
normalization: all circuits / source only
target update: 固定 sgrl_dual_rsm_ema
```

随后再用 `source/source dual` 对 `source/source ema_only` 隔离 target update。历史 E0-A2 只能作为复现锚点；若要做因果比较，应在新 harness 下重跑 legacy cell，并匹配 raw data、split、pretraining/downstream seed 和 eval neighborhoods。

### 17.3 开发迁移集的选择规则必须统一

`PROTOCOL_REPAIR_EXPERIMENT_HANDOFF.md:161` 说 digtime/timing_ctrl/array 只作解释、不参与选择，但 `:341` 又用“开发迁移集不系统性崩溃”作为 E1 继续条件，这已经属于 selection。

必须明确二选一：

1. 真正 source-only selection：训练/调参阶段只输出 source-val；先写入不可变 selection manifest，再统一解封开发迁移结果，并删除 transfer-collapse gate。
2. 承认三个电路参与方法工程：正式口径写成 `strict-inductive training + development-informed design + sp8192w blind confirmation`，并预注册量化的 transfer veto/Pareto 规则。

鉴于这三个电路已经长期用于架构诊断，第二种表述更诚实。`strict_inductive` 仍可描述训练数据边界，但不能抹去历史人工设计信息。

### 17.4 80 epoch 上限存在右截断

历史 best epoch 全在 77-79，部分曲线在最后若干 epoch 仍改善。必须先说明研究目标是固定 80-epoch compute budget，还是近似收敛后的精度。

若目标是后者，应预注册统一 extension rule，例如：任一主要候选在最后 10% epoch 刷新 best，则同阶段所有候选统一从 80 延长到 160。不能只延长看起来有希望的方法。

### 17.5 E1-E4 需要分离 tuning 与 confirmation seeds

当前 E1-E3 仅用 seed0 选择，E4 又纳入 seed0，会产生 winner's curse。建议：

- tuning 使用独立 seeds，例如 0-2；预算有限时可预注册 successive halving。
- confirmation 使用全新 seeds，例如 10-14，E4 后不再返回 E2/E3 重选。
- 分别记录 split、pretrain、downstream、train sampler、eval 和 relation sample seeds。
- relation-balanced transfer indices/views 在所有方法和训练 seeds 间固定，避免测试子集变化混入方差。
- `lambda=0` 与 `lambda>0` 必须使用相同 checkpoint 初始化、rank、layer、base/head LR 和 LoRA scaling，只改变 lambda，才能归因联合 GCL。
- `lambda=0.05` 必须明确是预注册固定假设还是 strict 协议下待选择超参。

五个 paired seeds 适合稳定性描述，但对小效应的统计功效有限。正式报告应使用 sample SD，并给出每个 seed、paired difference/ratio 和 95% CI；“恶化不超过 5%”若作为 non-inferiority 条件，应以 paired relative degradation 的 CI 上界判断，而不是单点估计。

### 17.6 E5 当前是 method × loss，不是干净的 GCL/reuse × rebalancing

现有 `no-GCL / static / best shared` 三行同时改变 GCL、模型结构、参数量和训练路径。若研究问题是“联合 GCL 是否与 rebalancing 冲突”，建议使用：

```text
no-GCL
static
shared, lambda=0
shared, lambda=lambda_fixed_positive
```

每行匹配 `MSE / GAI / BMC`。预注册 primary interaction，例如：

```text
[(GAI - MSE) under shared lambda>0]
-
[(GAI - MSE) under shared lambda=0]
```

BMC 同理。label bins 必须由 source-train 边界定义并固定应用，不能按各 transfer/blind 电路重新计算分位点。若预算不允许 4×3，结论必须降为“backbone strategy × loss interaction”，不能声称已经分离 GCL 与 reuse。

### 17.7 E6 应定义为一次冻结后的批量揭盲

不能挑 source-val 最好的单个 seed，也不应先看 static 再决定是否评估 shared。开始前冻结：

- commit 和 dirty patch/source hash；
- 全部 confirmation checkpoint 路径与 SHA256；
- model/loss/config；
- relation-balanced/natural evaluator；
- analysis script、表格字段和技术失败重跑规则。

随后在任何结果展示前，用一个非交互批处理完成：

```text
method × confirmation seed × relation-balanced/natural
```

“各评估一次”应解释为一次冻结后的揭盲事件，而不是每个方法只选一个 checkpoint。只允许相同 checkpoint/config 的基础设施重试；任何模型、预处理或分析逻辑变化都构成新分析。ultra8t/sandwich 应提前决定总是跑或不跑，不能根据 sp8192w 结果临时决定。

### 17.8 `sp8192w` 在揭盲前有必须处理的管线风险

- `sram_dataset.py:392-397` 由 dataset name 直接生成 `<name>.pt`，现有 raw 文件名是 `sp8192w.pt`。
- `sram_dataset.py:194-196` 的特殊 power-net 分支只识别 `sram_sp_8192w`。
- `sram_dataset.py:459-463` 的 large-dataset sampling 只识别 ultra8t/sandwich。

因此使用 `sp8192w` 虽能匹配 raw 文件，却可能漏掉特殊 power-net 和 large-graph 配置；使用 `sram_sp_8192w` 又可能找不到 raw 文件。应把 raw name、canonical circuit ID、power-net IDs 和 size class 配置化，并在非盲 large circuits 或 synthetic fixture 上先验证完整 evaluator/processor。该修复不要求现在读取 sp8192w 标签或生成其 processed artifact。

还应预注册 blind technical failure 规则，避免第一次处理失败后根据已看到的结果修改科学配置。

### 17.9 artifact 与统计字段还需补全

除 `PROTOCOL_REPAIR_EXPERIMENT_HANDOFF.md:451-477` 已列字段外，至少增加：

- `git_dirty`、patch/source tree hash、untracked scientific file hashes；
- hostname、OS、PyTorch/PyG/CUDA/cuDNN/driver 和 GPU 型号；
- deterministic flags、worker count/worker seeds；
- train sampler seed、relation sample seed、eval view hashes；
- target 初始化方式、target update cadence；
- source code/config hash、selection manifest hash。

部署参数量减少 35% 的条件也要定义场景：cold-start unseen graph 应计入 SGRL + downstream，cached-embedding steady state 则是另一口径；两者都应报告参数、时间、存储和峰值显存。

### 17.10 修订后的推荐执行顺序

1. 修订 target-update 命名、开发集选择规则、seed 角色和盲测规则。
2. 完成 WP1-WP7、对应 regression tests 和 1 epoch strict smoke。
3. 在统一 harness 下运行 E0 static 2×2，并对关键 contrasts 补 paired seeds；再做 target-update 消融。
4. E1-E3 使用独立 tuning seeds 和预注册的筛选/延长规则。
5. 写入不可变 selection manifest。
6. E4 使用全新 confirmation seeds，只确认、不返工选型。
7. E5 使用 matched-config 4×3，或明确降级为 method × loss interaction。
8. 完成 deployment export，并在非盲图上验证 natural/balanced evaluator 和 blind pipeline。
9. 冻结 commit、patch、checkpoints、配置、分析脚本及失败规则后，一次性批量揭盲 sp8192w。
10. 最后再处理 node/classification、HCA/SRL、negative sampling 等扩展线。

开始完整矩阵前应补一张预算表：每阶段新 pretrains、downstream runs、预计 GPU-hours、磁盘、并发度、停止/延长规则和可复用 artifacts。按当前表面设计 E0-E5 约有 85 个 downstream cells，显存约 0.6GB/任务不足以证明 CPU、I/O 和总时长可承受。

## 18. 推荐后续 Agent 的首个交付物

不要直接提交大规模重构。第一个交付物建议是一个小型“协议审计 PR/commit”，内容仅包括：

1. 明确 source/test boundary 的配置和断言；
2. split indices、normalization state、eval views 持久化；
3. train-only rebalancing statistics 与 LDS weight propagation 修复；
4. `sgrl_dual_rsm_ema` / `circuitgcl_text_ema_only` 两种预训练模式及清晰初始化、cache key 和单测，但不预设正式赢家；
5. target 预训练开关与 S6 downstream EMA target 开关分离；
6. dataset alias/power-net/size-class 配置化，并仅用非盲图验证 blind pipeline；
7. 对应最小回归测试和 1 epoch strict smoke；
8. 不改变 S6 共享 backbone/LoRA 的数学定义。

完成后先跑小型 source-only smoke/matched baseline，确认数据量、参数量、loss、artifact 和 evaluation sampling 全部符合预期，再决定是否继续 P3。

## 19. 快速文件地图

| 文件 | 作用 | 当前关注点 |
| --- | --- | --- |
| `sram_dataset.py` | 原始图处理、归一化、SGRL 图适配 | 全电路 normalization、全电路 SGRL、关系平衡下采样、processed cache |
| `sampling.py` | train/val/test loaders | split 持久化、eval sampling、node roots |
| `utils.py` | 正负边和关系平衡采样 | natural vs balanced、negative path |
| `sgrl_train.py` | SGRL 预训练和 checkpoint | source-only 边界、cache key、target update |
| `sgrl_models.py` | SGRL encoder | edge type bug、不可达 hop path、实际输入定义 |
| `model.py` | static/reuse/partial/joint/LoRA 模型 | node root supervision、deployment export |
| `downstream_train.py` | loss、训练、评估、artifact 调用 | LDS 时序、train-only priors、eval noise、classification 旧协议 |
| `balanced_mse.py` | GAI/BMC/BNI/LDS/BSCE | source-train prior、LDS、zero-count handling |
| `run_artifacts.py` | 隔离运行产物 | 增加 data/cache/split hashes |
| `scripts/summarize_experiments.py` | structured 汇总 | 正式结果入口 |
| `scripts/audit_static_online_consistency.py` | static-online 审计 | 分层抽样与显式 endpoint 校验 |
| `scripts/run_s6_p1_p2.sh` | staged runner | singleton/重复启动/GPU provenance |
| `PROTOCOL_REPAIR_EXPERIMENT_HANDOFF.md` | 协议修复与 E0-E6 路线图 | target-update 定性、E0 混杂、selection seeds、E5 因子、E6 揭盲规则 |
| `EXPERIMENT_LOG.md` | 运行证据与结论 | 当前最权威实验记录，但旧结果证据等级不同 |
| `TEACHER_TASKS_README.md` | 任务路线与阶段状态 | 高层状态入口 |
| `LABEL_REBALANCING_ANALYSIS.md` | 论文阅读与候选设计 | HCA/SRL 仍是计划，不是实现 |

## 20. 一句话交接结论

当前项目最值得继续的是 S6 的单 backbone + LoRA 方向和已经建立的 artifact/审计体系；`PROTOCOL_REPAIR_EXPERIMENT_HANDOFF.md` 的主线合理，但还必须先解决 strict-vs-transductive 边界、SGRL target-update 歧义、source-train-only statistics、LDS 权重链路、E0 因果混杂、独立 tuning/confirmation seeds、E5 可识别性、自然分布评估和一次性 blind reveal。否则即使后续数字更好，也不足以支撑可信的跨电路泛化或算法归因结论。
