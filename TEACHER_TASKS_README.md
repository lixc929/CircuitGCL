# CircuitGCL Optimization Tasks

This note summarizes the current understanding of the optimization tasks assigned by the advisor, based on the project structure, README, code behavior, reproduced baseline runs, and follow-up discussion.

## 1. Current Project Behavior

CircuitGCL currently has two relatively separate stages:

1. Self-supervised graph contrastive learning, implemented in `sgrl_train.py` and `sgrl_models.py`.
2. Downstream capacitance prediction, implemented in `downstream_train.py` and `model.py`.

When `--sgrl 1` is enabled, the current flow is:

```text
Training graph
  -> SGRL online/target training
  -> generate static node embeddings
  -> dataset.set_cl_embeds(...)
  -> downstream GNN consumes these embeddings as extra node features
```

This means the GCL module is currently used mainly as an offline embedding generator. The downstream GNN still has its own encoder/backbone.

Relevant code path:

- `main.py`: calls `sgrl_train(...)`, saves/loads `embeddings/*.pkl`.
- `sgrl_train.py`: trains the SGRL model and returns node embeddings from the online encoder.
- `sram_dataset.py`: `set_cl_embeds(...)` replaces `self._data.x` with learned embeddings.
- `model.py`: when `args.sgrl` is enabled, applies `cl_linear(batch.x)` and concatenates it with node type embeddings.

## 2. Advisor's First Task: Reuse Online Encoder and Downstream GNN

The current best interpretation is:

> The advisor likely pointed to the Online Encoder and the Downstream GNN, not the Target Encoder.

So the task is probably not to merge online and target encoders into one training branch. The target encoder is part of the BYOL-style/GCL training mechanism and can remain as a training-time auxiliary branch.

The more likely task is:

```text
Current:
GCL Online Encoder -> static embeddings -> separate Downstream GNN -> Head

Target:
GCL Online Encoder / learned backbone -> reused as Downstream Encoder -> Head
```

In other words, avoid using GCL only as a static embedding generator. Instead, reuse the GCL encoder/backbone directly in downstream training.

Possible implementation directions:

1. **Online feature reuse**
   - Pretrain the online encoder with GCL.
   - Use it online to produce `H_online` for downstream batches.
   - Keep the original downstream GNN and prediction head.

2. **Online feature finetuning**
   - Pretrain the online encoder.
   - Feed its online features into the downstream GNN.
   - Allow supervised gradients to finetune the online encoder with a conservative learning rate.

3. **Pretrain then initialize**
   - Pretrain the online encoder with GCL.
   - Copy compatible weights into downstream GNN layers where architectures match.
   - Finetune downstream with supervised capacitance loss.

4. **Partial or joint shared encoder**
   - Use one encoder backbone for both GCL loss and downstream prediction.
   - During training, combine contrastive loss and supervised loss with a tunable weight.

The first version should be conservative: implement frozen online feature reuse while keeping the original downstream `GraphHead`. This directly tests the advisor's online-encoder/downstream-GNN reuse idea without immediately deleting downstream model capacity.

## 3. What Not To Misinterpret

Do not treat the main task as simply merging the online and target encoders.

The online/target split is useful during contrastive learning:

- Online encoder: main learnable encoder.
- Predictor: maps online representation toward target representation.
- Target encoder: stable training target, often updated by EMA or separately constrained.

For downstream prediction, only one encoder/backbone should be used. That encoder should preferably come from or share weights with the online encoder.

## 4. Shared Backbone Roadmap

This is the concrete roadmap for gradually merging the GCL online GNN encoder and the downstream GNN backbone into one shared backbone.

### 4.1 Final Target

The final model should not merge the online encoder and target encoder. The target encoder remains a training-time teacher/stabilizer for the GCL objective.

The final target is:

```text
Training:
graph
  -> shared online GNN backbone f_theta
      -> predictor q_theta -> GCL alignment loss against target encoder f_phi
      -> downstream head -> supervised capacitance loss

target encoder f_phi:
  -> EMA / stop-gradient teacher for GCL only

Inference:
graph
  -> shared online GNN backbone f_theta
  -> downstream head
  -> prediction
```

So "reuse" means:

```text
GCL online GNN encoder + downstream GNN backbone
  -> gradually become one shared GNN backbone
```

It does not mean:

- Merging online and target encoders.
- Removing the downstream GNN immediately.
- Using only static GCL embeddings forever.
- Expecting label rebalancing to fix a bad reuse architecture.

### 4.2 Step-by-Step Plan

Use the current `test` branch for all implementation and experiment commits. Do not push to the teacher upstream repository.

| Step | Status | Goal | Implementation | Main comparison |
| --- | --- | --- | --- | --- |
| S0. Definition and evidence | Done | Lock down what "reuse" means and collect paper support. | Use CircuitGCL, BYOL, BGRL, GraphCL, GNN pretraining references in `papers/README.md`. | N/A |
| S1. Code-path diagnosis | Done | Explain why the first `SgrlBackboneHead` attempt is too aggressive. | Document that it replaces the downstream `GraphHead` instead of feeding online features into it. | Static GCL vs replacement-style reuse |
| S2. Online feature reuse | Done | Use the pretrained online encoder online, but keep the original downstream GNN. | Added `--sgrl_mode online_feature`; frozen/eval online encoder produces `H_online` per batch, then `GraphHead` still runs downstream message passing and prediction. | `static + MSE` vs `online_feature_frozen + MSE` |
| S3. Online feature finetuning | Done | Check whether supervised gradients should update the online encoder. | Added `--sgrl_mode online_feature_finetune`; use separate optimizer groups with lower online-encoder LR. | `online_feature_frozen` vs `online_feature_finetune` |
| S4. Parameter initialization reuse | Done | Reuse GCL online encoder weights to initialize compatible downstream layers. | Added `init_reuse`; copied only matching tensors, logged skipped keys, and recorded parameter counts. | compact `no-GCL/static/init_reuse + MSE`, all GPU verified |
| S5. Partial shared backbone | Done for current round | Share early/lower GNN layers while keeping task-specific later layers/head. | Added `partial_shared`, `SharedGNNBackbone`, and `PartialSharedGraphHead`; full `k1 + gate` is the reliable shared-backbone path, while `scalar_gate/vector_gate + freeze3` reduce parameters but lose transfer. | `init_reuse` vs `partial_shared_k1/k2/gate/freeze/slim_gate` |
| S6. Joint shared backbone | Pending | Train one online backbone with both GCL and supervised losses. | Optimize `L = L_supervised + lambda_gcl * L_gcl`; target encoder remains EMA/stop-gradient. | `partial_shared` vs `joint_shared` |
| S7. Label rebalancing integration | Current sweep done | Test whether rebalancing helps after reuse is architecturally correct. | Ran GAI/BMC on `gate+freeze3`, `vector_gate+freeze3`, and `gate+freeze2`; BMC helps the stable full-gate backbone most, not the over-slim vector gate. | best reuse + `MSE/GAI/BMC` |

### 4.3 First Implementation Target

The S2 implementation target was:

```text
Input sampled downstream graph
  -> pretrained GCL online encoder, eval/frozen
  -> H_online
  -> original downstream GraphHead feature path
  -> original downstream GNN layers
  -> original edge/node prediction head
```

This differs from the previous replacement-style reuse attempt:

```text
Input graph
  -> SGRL encoder
  -> edge MLP
```

The previous attempt is still useful as a negative control, but it should not be treated as the main shared-backbone method.

### 4.4 Experiment Gate

Each step must pass a small development gate before expanding experiments:

1. Run on the lightweight development matrix:

```text
dataset: ssram+digtime+timing_ctrl+array_128_32_8t
task: edge regression
loss: MSE first
epochs: 20
gpu: non-zero GPU, usually GPU 3
```

2. Compare against:

```text
no GCL + MSE
static GCL + MSE
previous replacement-style reuse + MSE
```

3. Only expand to `GAI` and `BMC` if the reuse method is at least close to static GCL with MSE.

4. Record every run in `EXPERIMENT_LOG.md`, including command, checkpoint/log path, best epoch, validation MSE, and per-testset MSE.

### 4.5 Success Criteria

The shared-backbone line is worth continuing if:

- S2 frozen online feature reuse is close to or better than static GCL on MSE.
- Finetuning does not destabilize validation/test transfer.
- Partial/joint sharing improves at least one transfer testset without large regression on the others.
- Rebalancing gains appear after the reuse path itself is stable.

If S2 is much worse than static GCL, debug feature alignment first:

- Whether the online encoder sees the same features/edge types as SGRL pretraining.
- Whether dropout/eval mode matches static embedding extraction.
- Whether sampled downstream subgraphs match the neighborhood assumptions of GCL embeddings.
- Whether `H_online` should be detached, normalized, projected, or fused with node statistics before downstream GNN.

## 5. Advisor's Second Task: Separate GCL and Label Rebalancing

The advisor also noted that:

> Label rebalancing and contrastive learning may work well separately, but the combined version is not necessarily better.

So the work should be split into clear ablations.

### 5.1 Label Rebalancing Only

Disable SGRL/GCL and compare supervised losses:

```text
--sgrl 0 --regress_loss mse
--sgrl 0 --regress_loss gai
--sgrl 0 --regress_loss bmc
--sgrl 0 --regress_loss bni
--sgrl 0 --regress_loss lds
```

Goal:

- Identify which label rebalancing method is actually stable and useful.
- Compare per-testset behavior, especially on `timing_ctrl`, `array_128_32_8t`, `ultra8t`, and `sandwich`.

### 5.2 GCL Only

Use ordinary supervised loss after GCL:

```text
--sgrl 1 --regress_loss mse
```

Compare at least three GCL usage styles:

1. Current static embedding concatenation.
2. Online encoder initialization for downstream GNN.
3. Shared/finetuned online encoder as downstream backbone.

Goal:

- Check whether GCL itself improves transferability.
- Determine whether static embedding concatenation is weaker than backbone reuse.

### 5.3 GCL + Label Rebalancing

After the separate effects are clear, test combinations:

```text
--sgrl 1 --regress_loss gai
--sgrl 1 --regress_loss bmc
--sgrl 1 --regress_loss bni
--sgrl 1 --regress_loss lds
```

Goal:

- Understand why the combination may degrade.
- Check whether the issue is caused by conflicting objectives, unstable tail weighting, overfitting rare labels, or embedding/backbone mismatch.

## 6. Current Local Baseline Results

The following older 20-epoch runs were completed locally before explicit CUDA
diagnostics were added to the logs. Treat them as historical references unless
rerun with `CUDA status: available=True` and `Using GPU: ...` in the log:

```text
Dataset:
ssram+digtime+timing_ctrl+array_128_32_8t

Task:
edge regression

Batch size:
128
```

### 6.1 MSE Baseline

Command type:

```bash
python main.py \
  --dataset ssram+digtime+timing_ctrl+array_128_32_8t \
  --task regression \
  --task_level edge \
  --regress_loss mse \
  --batch_size 128 \
  --epochs 20 \
  --num_workers 0 \
  --gpu 3
```

Best result:

```text
Best epoch: 17
Val MSE: 0.0078
Test MSE: digtime 0.0165, timing_ctrl 0.0089, array_128_32_8t 0.0102
```

Log:

```text
logs/20260625_160900_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossmse_batch128.txt
```

Checkpoint:

```text
downstream_model/model_17-mse.pth
```

### 6.2 GAI Baseline

Command type:

```bash
OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16 \
python main.py \
  --dataset ssram+digtime+timing_ctrl+array_128_32_8t \
  --task regression \
  --task_level edge \
  --regress_loss gai \
  --batch_size 128 \
  --epochs 20 \
  --num_workers 0 \
  --gpu 3
```

Best result:

```text
Best epoch: 17
Val MSE: 0.0078
Test MSE: digtime 0.0151, timing_ctrl 0.0087, array_128_32_8t 0.0107
```

Log:

```text
logs/20260625_190032_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossgai_batch128.txt
```

Checkpoint:

```text
downstream_model/model_17-gai.pth
```

## 7. Immediate Next Steps

Recommended order:

1. Completed S2 online feature reuse: frozen/eval GCL online encoder feeding the original downstream `GraphHead`.
2. Completed the S2 MSE comparison:

```text
No GCL + MSE
Static GCL embedding + MSE
Previous replacement-style reuse + MSE
Online feature reuse + MSE
```

3. Completed S3 online encoder finetuning with a conservative learning rate:

```text
Online feature reuse, frozen online encoder + MSE
Online feature reuse, finetuned online encoder + MSE
```

4. Completed S4 parameter initialization reuse:

```text
compact no-GCL + MSE, GPU verified
compact static + MSE, GPU verified
init_reuse + MSE, GPU verified
```

5. S5 partial shared backbone is now implemented as the next architecture:
   `--sgrl_mode partial_shared` reuses the SGRL online node/edge encoder and
   the first `k` online GNN layers, then keeps a downstream-specific tail GNN
   and prediction head. GPU3 MSE results so far:

```text
partial_shared_k1 + add: Val 0.0104; digtime/timing_ctrl/array 0.0180/0.0123/0.0130
partial_shared_k2 + add: Val 0.0108; digtime/timing_ctrl/array 0.0294/0.0152/0.0187
partial_shared_k1 + gate: Val 0.0102; digtime/timing_ctrl/array 0.0140/0.0119/0.0126
partial_shared_k1 + residual_gate: Val 0.0105; digtime/timing_ctrl/array 0.0147/0.0114/0.0120
partial_shared_k1 + gate + backbone_lr=1e-5: Val 0.0107; digtime/timing_ctrl/array 0.0150/0.0123/0.0127
partial_shared_k1 + gate + freeze1: Val 0.0098; digtime/timing_ctrl/array 0.0141/0.0117/0.0121
partial_shared_k1 + gate + freeze2: Val 0.0098; digtime/timing_ctrl/array 0.0139/0.0117/0.0120
partial_shared_k1 + gate + freeze3: Val 0.0098; digtime/timing_ctrl/array 0.0140/0.0117/0.0119
partial_shared_k1 + scalar_gate + freeze3: Val 0.0102; digtime/timing_ctrl/array 0.0172/0.0117/0.0130
partial_shared_k1 + vector_gate + freeze3: Val 0.0101; digtime/timing_ctrl/array 0.0189/0.0118/0.0134
partial_shared_k1 + gate + freeze5: Val 0.0099; digtime/timing_ctrl/array 0.0139/0.0118/0.0120
partial_shared_k1 + gate + freeze3 + backbone_lr=3e-5: Val 0.0099; digtime/timing_ctrl/array 0.0141/0.0116/0.0121
partial_shared_k1 + gate + freeze3 + GAI: Val 0.0098; digtime/timing_ctrl/array 0.0140/0.0117/0.0120
partial_shared_k1 + gate + freeze3 + BMC: Val 0.0098; digtime/timing_ctrl/array 0.0142/0.0116/0.0118
partial_shared_k1 + vector_gate + freeze3 + GAI: Val 0.0101; digtime/timing_ctrl/array 0.0192/0.0119/0.0133
partial_shared_k1 + vector_gate + freeze3 + BMC: Val 0.0101; digtime/timing_ctrl/array 0.0181/0.0117/0.0130
partial_shared_k1 + gate + freeze2 + GAI: Val 0.0098; digtime/timing_ctrl/array 0.0140/0.0117/0.0119
partial_shared_k1 + gate + freeze2 + BMC: Val 0.0098; digtime/timing_ctrl/array 0.0137/0.0114/0.0120
```

   k2 makes the architecture more structurally shared but badly hurts
   cross-dataset transfer. Learned gate fusion repairs most of the k1/add
   transfer loss. The strongest single-seed row was `freeze2 + BMC` at
   `0.0137/0.0114/0.0120` on digtime/timing_ctrl/array, but the completed
   seeds `0/1/2` audit shows that this advantage is not stable. Across three
   seeds, original `static + MSE` has the best validation and transfer means;
   all tested `k1 + gate + freeze` rows are worse and are especially sensitive
   on digtime. Slim gates also remain weaker, particularly on digtime. The
   current conclusion is therefore narrower: full `gate` is still the best
   reuse implementation tested, but it is not yet a replacement for static
   GCL on accuracy/stability. Keep `vector_gate` only as a compact ablation and
   avoid adding new branches until the multi-seed result is discussed. Full
   per-seed metrics are in `EXPERIMENT_LOG.md`.
6. Only after the compact reuse architecture is stable, compare:

```text
Best reuse + MSE
Best reuse + GAI
Best reuse + BMC
```

   This comparison has been run for the current S5 candidates. The multi-seed
   audit shows that BMC only makes small, dataset-dependent changes inside the
   full-gate family and does not close the gap to static GCL. It also does not
   fix the weaker `vector_gate` digtime transfer.

7. If GCL + rebalancing is worse than either alone, test:

```text
freeze encoder vs finetune encoder
smaller rebalancing strength / smoother bins
GCL pretrain epochs
warmup with MSE, then switch to GAI/BMC
loss weighting for joint GCL + supervised training
```

## 8. Working Hypothesis

The likely research story is:

> The current CircuitGCL implementation underuses GCL because the contrastive encoder is only converted into static embeddings. Reusing the online encoder as the downstream backbone may improve transferability. Meanwhile, label rebalancing and GCL need to be evaluated separately because rebalancing emphasizes rare/tail labels while GCL may emphasize structural invariance, and their objectives can conflict when naively combined.

This hypothesis should guide the next round of implementation and experiments.
