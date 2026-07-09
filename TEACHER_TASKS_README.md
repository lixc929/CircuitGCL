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

1. **Pretrain then initialize**
   - Pretrain the online encoder with GCL.
   - Copy compatible weights into the downstream GNN encoder.
   - Finetune downstream with supervised capacitance loss.

2. **Pretrain then freeze**
   - Pretrain the online encoder.
   - Use it as a frozen feature extractor.
   - Train only the downstream prediction head.

3. **Pretrain then partially finetune**
   - Freeze lower GNN layers first.
   - Finetune later layers and the prediction head.

4. **Joint/shared encoder**
   - Use one encoder backbone for both GCL loss and downstream prediction.
   - During training, combine contrastive loss and supervised loss with a tunable weight.

The first version should be conservative: implement pretrain -> initialize -> finetune, because it is easiest to compare against the current code and has the lowest engineering risk.

## 3. What Not To Misinterpret

Do not treat the main task as simply merging the online and target encoders.

The online/target split is useful during contrastive learning:

- Online encoder: main learnable encoder.
- Predictor: maps online representation toward target representation.
- Target encoder: stable training target, often updated by EMA or separately constrained.

For downstream prediction, only one encoder/backbone should be used. That encoder should preferably come from or share weights with the online encoder.

## 4. Advisor's Second Task: Separate GCL and Label Rebalancing

The advisor also noted that:

> Label rebalancing and contrastive learning may work well separately, but the combined version is not necessarily better.

So the work should be split into clear ablations.

### 4.1 Label Rebalancing Only

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

### 4.2 GCL Only

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

### 4.3 GCL + Label Rebalancing

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

## 5. Current Local Baseline Results

The following 20-epoch runs have already been completed locally on GPU 3:

```text
Dataset:
ssram+digtime+timing_ctrl+array_128_32_8t

Task:
edge regression

Batch size:
128
```

### 5.1 MSE Baseline

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

### 5.2 GAI Baseline

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

## 6. Immediate Next Steps

Recommended order:

1. Run a clean ablation table for `--sgrl 0` with different rebalancing losses.
2. Run current `--sgrl 1 --regress_loss mse` to measure the existing static-embedding GCL effect.
3. Implement online-encoder-to-downstream-backbone reuse.
4. Compare:

```text
No GCL + MSE
No GCL + GAI
Static GCL embedding + MSE
Static GCL embedding + GAI
Backbone reuse GCL + MSE
Backbone reuse GCL + GAI
```

5. If GCL + rebalancing is worse than either alone, test:

```text
freeze encoder vs finetune encoder
smaller rebalancing strength / smoother bins
GCL pretrain epochs
warmup with MSE, then switch to GAI/BMC
loss weighting for joint GCL + supervised training
```

## 7. Working Hypothesis

The likely research story is:

> The current CircuitGCL implementation underuses GCL because the contrastive encoder is only converted into static embeddings. Reusing the online encoder as the downstream backbone may improve transferability. Meanwhile, label rebalancing and GCL need to be evaluated separately because rebalancing emphasizes rare/tail labels while GCL may emphasize structural invariance, and their objectives can conflict when naively combined.

This hypothesis should guide the next round of implementation and experiments.
