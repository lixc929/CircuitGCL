# Experiment Log

This is the single running log for local CircuitGCL experiments in this fork.
Keep new experiment records here instead of creating many separate Markdown files.

## Document Map

- `TEACHER_TASKS_README.md`: advisor task interpretation and high-level roadmap.
- `LABEL_REBALANCING_ANALYSIS.md`: paper reading notes and label-rebalancing design analysis.
- `EXPERIMENT_LOG.md`: commands, logs, metrics, and conclusions from actual runs.

## Device Validity Note

Formal experiment logs should now include both lines below:

```text
CUDA status: available=True, device_count=..., requested_gpu=...
Using GPU: ...
```

Runs made before this diagnostic was added are kept as development references,
but should not be used as final GPU evidence unless they are rerun or otherwise
verified. In particular, logs with `PID = 2` were sandbox-style runs and should
be treated as CPU/unverified. The GPU-verified runs below were executed with the
RCG environment at `/home/lixc/.conda/envs/RCG/bin/python` and GPU 3.

## 2026-07-09: SGRL Backbone Reuse Pilot

### Goal

Test whether reusing the SGRL online encoder as the downstream backbone is better
than the original static embedding path.

### Git State

```text
branch: test
base: 1fcfe5c Add SGRL backbone reuse path
remote: lixc929/test
```

### Shared Settings

```text
dataset: ssram+digtime+timing_ctrl+array_128_32_8t
task: edge regression
loss: mse
gpu: 4
seed: 42
```

### Pilot Runs

The first pilot intentionally uses lighter SGRL sampling than the README default,
because the default SGRL loader is very slow on the full SRAM graph.

```text
cl_gnn_layers: 2
cl_hid_dim: 64
cl_batch_size: 32768
cl_num_neighbors: 8
num_hops: 2
num_neighbors: 8
downstream epochs: 20
batch_size: 512
```

| Run | Mode | Status | Log | Best Val MSE | Test MSEs |
| --- | --- | --- | --- | --- | --- |
| reuse-nogcl-pilot | `--sgrl 0` | done | `logs/reuse_pilot_20260709/20260709_164310_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossmse_batch512.txt` | 0.0096 | digtime 0.0141; timing_ctrl 0.0099; array_128_32_8t 0.0113 |
| reuse-static-pilot | `--sgrl 1 --sgrl_mode static` | done | `logs/reuse_pilot_20260709/20260709_161848_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossmse_batch512.txt` | 0.0096 | digtime 0.0136; timing_ctrl 0.0103; array_128_32_8t 0.0110 |
| reuse-init-pilot | `--sgrl 1 --sgrl_mode init` | done | `logs/reuse_pilot_20260709/20260709_163412_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossmse_batch512.txt` | 0.0132 | digtime 0.0150; timing_ctrl 0.0226; array_128_32_8t 0.0117 |

### Best Checkpoints

```text
reuse-nogcl-pilot: best epoch 18, val loss 0.00962609
reuse-static-pilot: best epoch 15, val loss 0.00963094
reuse-init-pilot: best epoch 19, val loss 0.01321202
```

### Interpretation Notes

- Static mode tests the original CircuitGCL use of SGRL as cached node embeddings.
- Init mode tests the advisor-style reuse of the SGRL online encoder as the downstream backbone.
- In this pilot, static SGRL and no-GCL are close on validation MSE. Static SGRL is better on digtime and array_128_32_8t, while no-GCL is better on timing_ctrl.
- Direct online-backbone reuse underperforms both static SGRL and no-GCL, especially on timing_ctrl.
- The first suspect is implementation mismatch rather than proof that reuse is bad: `SgrlBackboneHead` currently reuses the online encoder but does not yet include the downstream circuit-statistics adapter used by the original edge model.
- Next reuse experiment should add a hybrid head: online encoder backbone plus downstream circuit-statistics adapter, then compare `init` and `freeze` under the same settings.

## 2026-07-09: SGRL Hybrid Reuse Adapter Pilot

### Goal

Test the next reuse variant: reuse the SGRL online encoder as the downstream
backbone, but restore the downstream circuit-statistics adapter before the edge
head.

### Code Delta

Added two downstream reuse switches:

```text
--sgrl_reuse_stats 1
--sgrl_reuse_stats_fusion concat
```

When `--sgrl_mode init` or `--sgrl_mode freeze` is active, the reused SGRL node
features are concatenated with node-type/statistics features before the edge
head. This keeps the teacher-suggested online/downstream reuse path, while
avoiding the previous pilot's pure-GCL-embedding bottleneck.

### Shared Settings

```text
dataset: ssram+digtime+timing_ctrl+array_128_32_8t
task: edge regression
loss: mse
gpu: 4
seed: 42
cl_epochs: 5
cl_gnn_layers: 2
cl_hid_dim: 64
cl_batch_size: 32768
cl_num_neighbors: 8
num_hops: 2
num_neighbors: 8
downstream epochs: 20
batch_size: 512
```

### Pilot Runs

| Run | Mode | Status | Log | Best Epoch | Best Val MSE | Test MSEs |
| --- | --- | --- | --- | --- | --- | --- |
| reuse-hybrid-init-pilot | `--sgrl 1 --sgrl_mode init --sgrl_reuse_stats 1 --sgrl_reuse_stats_fusion concat` | done | `logs/reuse_hybrid_pilot_20260709/20260709_170216_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossmse_batch512.txt` | 18 | 0.0106 | digtime 0.0246; timing_ctrl 0.0168; array_128_32_8t 0.0213 |
| reuse-hybrid-freeze-pilot | `--sgrl 1 --sgrl_mode freeze --sgrl_reuse_stats 1 --sgrl_reuse_stats_fusion concat` | done | `logs/reuse_hybrid_pilot_20260709/20260709_172140_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossmse_batch512.txt` | 18 | 0.0105 | digtime 0.0252; timing_ctrl 0.0180; array_128_32_8t 0.0291 |

### Comparison

| Variant | Best Val MSE | digtime MSE | timing_ctrl MSE | array_128_32_8t MSE |
| --- | --- | --- | --- | --- |
| no-GCL pilot | 0.0096 | 0.0141 | 0.0099 | 0.0113 |
| static SGRL embedding pilot | 0.0096 | 0.0136 | 0.0103 | 0.0110 |
| direct online init reuse pilot | 0.0132 | 0.0150 | 0.0226 | 0.0117 |
| hybrid online init reuse pilot | 0.0106 | 0.0246 | 0.0168 | 0.0213 |
| hybrid online freeze reuse pilot | 0.0105 | 0.0252 | 0.0180 | 0.0291 |

### Interpretation Notes

- Adding the circuit-statistics adapter fixes a real part of the reuse mismatch:
  validation MSE improves from direct reuse `0.0132` to about `0.0105-0.0106`.
- The hybrid reuse variants still do not beat the no-GCL/static baselines on
  validation, and their cross-dataset test MSEs are much worse.
- `freeze` is slightly better than `init` on validation in this pilot, but worse
  on all three test datasets. This suggests the reused SGRL backbone is not yet
  aligned with downstream generalization.
- Current conclusion: online/downstream reuse is a plausible teacher-task
  direction, but the simple concat adapter is not enough. Keep static/no-GCL as
  baselines, and treat the next reuse work as alignment/regularization rather
  than just wiring reuse into the head.
- Reasonable next reuse candidates: gated/residual fusion instead of raw concat,
  lower learning rate on reused encoder layers, or reuse only the early encoder
  initialization while preserving more of the original downstream `GraphHead`.

## 2026-07-09: SGRL Gated Reuse Fusion Pilot

### Goal

Test whether a learned gate is better than raw concat when reusing the SGRL
online encoder as the downstream backbone.

### Code Delta

Added two fusion choices for the reused SGRL backbone:

```text
--sgrl_reuse_stats_fusion gate
--sgrl_reuse_stats_fusion residual_gate
```

`gate` learns an elementwise mix between reused SGRL features and
circuit-statistics features. `residual_gate` keeps reused SGRL features as the
base and adds a gated statistics residual.

### Shared Settings

```text
dataset: ssram+digtime+timing_ctrl+array_128_32_8t
task: edge regression
loss: mse
seed: 42
cl_epochs: 5
cl_gnn_layers: 2
cl_hid_dim: 64
cl_batch_size: 32768
cl_num_neighbors: 8
num_hops: 2
num_neighbors: 8
downstream epochs: 20
batch_size: 512
```

GPU0 was avoided. The full `gate` pilots used GPU3 for `init` and GPU1 for
`freeze`. `residual_gate` was only smoke-tested on GPU1.

### Pilot Runs

| Run | Mode | Status | Log | Best Epoch | Best Val MSE | Test MSEs |
| --- | --- | --- | --- | --- | --- | --- |
| reuse-gate-init-pilot | `--sgrl 1 --sgrl_mode init --sgrl_reuse_stats 1 --sgrl_reuse_stats_fusion gate` | done | `logs/reuse_gate_pilot_20260709/20260709_174655_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossmse_batch512.txt` | 18 | 0.0114 | digtime 0.0177; timing_ctrl 0.0138; array_128_32_8t 0.0184 |
| reuse-gate-freeze-pilot | `--sgrl 1 --sgrl_mode freeze --sgrl_reuse_stats 1 --sgrl_reuse_stats_fusion gate` | done | `logs/reuse_gate_pilot_20260709/20260709_180109_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossmse_batch512.txt` | 19 | 0.0108 | digtime 0.0276; timing_ctrl 0.0151; array_128_32_8t 0.0207 |
| reuse-residual-gate-smoke | `--sgrl 1 --sgrl_mode init --sgrl_reuse_stats 1 --sgrl_reuse_stats_fusion residual_gate --epochs 1` | smoke only | `logs/reuse_residual_gate_smoke_20260709/20260709_181121_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossmse_batch512.txt` | 0 | 0.0212 | digtime 0.0279; timing_ctrl 0.0408; array_128_32_8t 0.0237 |

### Comparison

| Variant | Best Val MSE | digtime MSE | timing_ctrl MSE | array_128_32_8t MSE |
| --- | --- | --- | --- | --- |
| no-GCL pilot | 0.0096 | 0.0141 | 0.0099 | 0.0113 |
| static SGRL embedding pilot | 0.0096 | 0.0136 | 0.0103 | 0.0110 |
| concat online init reuse pilot | 0.0106 | 0.0246 | 0.0168 | 0.0213 |
| concat online freeze reuse pilot | 0.0105 | 0.0252 | 0.0180 | 0.0291 |
| gate online init reuse pilot | 0.0114 | 0.0177 | 0.0138 | 0.0184 |
| gate online freeze reuse pilot | 0.0108 | 0.0276 | 0.0151 | 0.0207 |

### Interpretation Notes

- `gate` improves the `init` test MSEs compared with raw concat, especially on
  digtime (`0.0246 -> 0.0177`) and timing_ctrl (`0.0168 -> 0.0138`), but its
  validation MSE is worse (`0.0106 -> 0.0114`).
- `gate + freeze` reaches a reasonable validation MSE (`0.0108`), but its
  cross-dataset tests are still much worse than no-GCL/static baselines.
- None of the current reuse variants beats no-GCL or static SGRL embedding.
  Fusion quality helps, but it is not the main bottleneck.
- `residual_gate` passed a one-epoch runtime smoke test only, so it should not
  be used as an accuracy conclusion yet.
- Current next direction: keep this gated fusion available, but shift reuse
  experiments toward alignment/regularization, such as a lower learning rate
  for reused encoder layers or partial/early-layer reuse. In parallel, start a
  separate label-rebalancing track, because the advisor's feedback suggests it
  may be useful independently even if combined reuse plus rebalancing is not
  immediately additive.

## 2026-07-09: Rebalancing Original-vs-Clean Audit

### Goal

Verify whether the suspected label-rebalancing implementation issues are real
in the public repository code, without making a broad claim that the paper or
advisor's internal code is wrong.

### Setup

Two implementations were compared:

```text
original: origin/master at f43f83c, temporary worktree /tmp/circuitgcl_origin_master_audit
clean: current test branch at b5ef96b plus local audit script
```

The original worktree used a symlink to the existing `datasets/` directory. The
original logs were copied into `logs/rebalancing_impl_audit_original_20260709/`
for local traceability. Logs remain ignored by git.

### Formula/Shape Audit

Command:

```bash
/home/lixc/.conda/envs/RCG/bin/python scripts/audit_rebalancing_original_vs_clean.py \
  --json_out logs/rebalancing_audit_20260709.json
```

Key findings:

| Item | Original behavior | Clean behavior | Evidence |
| --- | --- | --- | --- |
| GAI labels | `edge_label` shape is `[175413, 2]`; flattening gives 350826 values, 43.43% greater than 1.0 | uses only continuous labels, all in `[0.003854, 0.990668]` | `logs/rebalancing_audit_20260709.json` |
| GAI GMM | component means include exact class-id modes `1.0, 2.0, 3.0, 4.0` | component means stay in normalized label range `0.148347 ... 0.804206` | `logs/rebalancing_audit_20260709.json` |
| BMC logits | legacy logits `[4, 1] -> [4]`, target shape `[4]`, target dtype `float32` | clean logits `[4, 4]`, target dtype `int64` | `logs/rebalancing_audit_20260709.json` |
| LDS loss shape | legacy broadcast shape `[4, 4]`, toy loss `0.309275` | clean shape `[4, 1]`, toy loss `0.003775` | `logs/rebalancing_audit_20260709.json` |

Interpretation: the GAI and LDS issues are concrete under the public
`origin/master` code and current normalized labels. BMC does not crash in the
current PyTorch environment, but it uses a non-standard one-dimensional
cross-entropy path rather than Balanced MSE's `[N, N]` batch comparison.

### Training-Level 1-Epoch Check

Shared command settings:

```text
dataset: ssram+digtime+timing_ctrl+array_128_32_8t
task: edge regression
sgrl: 0
epochs: 1
batch_size: 512
num_hops: 2
num_neighbors: 8
num_workers: 0
gpu: 2
seed: 42
```

| Impl | Loss | Log | Val MSE | digtime MSE | timing_ctrl MSE | array_128_32_8t MSE |
| --- | --- | --- | --- | --- | --- | --- |
| original | mse | `logs/rebalancing_impl_audit_original_20260709/20260709_183240_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossmse_batch512.txt` | 0.0140 | 0.0168 | 0.0173 | 0.0142 |
| clean | mse | `logs/rebalancing_impl_audit_clean_20260709/20260709_183543_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossmse_batch512.txt` | 0.0140 | 0.0168 | 0.0173 | 0.0142 |
| original | gai | `logs/rebalancing_impl_audit_original_20260709/20260709_183323_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossgai_batch512.txt` | 0.0143 | 0.0171 | 0.0175 | 0.0144 |
| clean | gai | `logs/rebalancing_impl_audit_clean_20260709/20260709_183628_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossgai_batch512.txt` | 0.0138 | 0.0166 | 0.0169 | 0.0138 |
| original | bmc | `logs/rebalancing_impl_audit_original_20260709/20260709_183413_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossbmc_batch512.txt` | 0.0157 | 0.0191 | 0.0181 | 0.0151 |
| clean | bmc | `logs/rebalancing_impl_audit_clean_20260709/20260709_183719_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossbmc_batch512.txt` | 0.0128 | 0.0164 | 0.0171 | 0.0131 |
| original | lds | `logs/rebalancing_impl_audit_original_20260709/20260709_183456_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_losslds_batch512.txt` | 0.0244 | 0.0248 | 0.0255 | 0.0243 |
| clean | lds | `logs/rebalancing_impl_audit_clean_20260709/20260709_183809_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_losslds_batch512.txt` | 0.0169 | 0.0212 | 0.0169 | 0.0178 |

### Interpretation Notes

- The identical original/clean MSE run validates that the temporary worktree and
  current branch are comparable under the same seed and training settings.
- The GAI issue is real in `origin/master`: the original GMM sees class IDs as
  label values. Clean GAI improves all 1-epoch validation/test MSEs in this
  audit, but this is still too short to claim final performance.
- The BMC issue is real as a formula mismatch. Clean BMC changes both loss scale
  and behavior: original val loss is `2037.76995797`, clean val loss is
  `0.01278353`, and clean BMC is best in this 1-epoch audit.
- The LDS broadcasting issue is real. Clean LDS improves over original LDS, but
  it remains worse than MSE/GAI/BMC in this short run.
- Current conclusion: do not say the advisor's project is "wrong"; say the
  public `origin/master` rebalancing code has implementation issues that affect
  the current environment/data path. The current `test` branch contains the
  clean regression fixes, and the next serious experiment should run 20-epoch
  clean no-GCL rebalancing ablations.

## 2026-07-09: Reuse x Rebalancing Development Matrix

### Goal

Run a compact 3 x 3 matrix to separate three questions:

1. original static SGRL embedding path vs no-GCL downstream;
2. whether fixed `GAI` / `BMC` help over `MSE`;
3. whether label rebalancing can rescue the current online-encoder reuse path.

This is a development anchor, not a full README/default reproduction. It uses
the lightweight settings from the reuse pilots so the matrix is comparable with
earlier local results and can reuse the cached SGRL artifacts.

### Command Driver

```bash
/home/lixc/.conda/envs/RCG/bin/python -u scripts/run_reuse_rebalance_matrix.py \
  --gpu 3 \
  --python /home/lixc/.conda/envs/RCG/bin/python \
  --quiet
```

Summary files:

```text
logs/reuse_rebalance_matrix_20260709/summary.json
logs/reuse_rebalance_matrix_20260709/summary.md
```

### Shared Settings

```text
dataset: ssram+digtime+timing_ctrl+array_128_32_8t
task: edge regression
seed: 42
gpu: 3
epochs: 20
batch_size: 512
num_workers: 0
cl_epochs: 5
cl_gnn_layers: 2
cl_hid_dim: 64
cl_batch_size: 32768
cl_num_neighbors: 8
num_hops: 2
num_neighbors: 8
```

Device note: this matrix was run before explicit CUDA diagnostics were added to
`main.py`. It remains useful as a compact development comparison, but final
claims should be based on GPU-verified reruns that print `CUDA status` and
`Using GPU`.

Modes:

```text
static: --sgrl 1 --sgrl_mode static
nogcl: --sgrl 0
reuse_gate_init: --sgrl 1 --sgrl_mode init --sgrl_reuse_stats 1 --sgrl_reuse_stats_fusion gate
```

### Results

| Mode | Loss | Best epoch | Val MSE | digtime MSE | timing_ctrl MSE | array MSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| static | mse | 17 | 0.0097 | 0.0151 | 0.0101 | 0.0111 |
| static | gai | 19 | 0.0096 | 0.0147 | 0.0102 | 0.0111 |
| static | bmc | 16 | 0.0095 | 0.0146 | 0.0099 | 0.0112 |
| nogcl | mse | 18 | 0.0096 | 0.0141 | 0.0099 | 0.0113 |
| nogcl | gai | 15 | 0.0097 | 0.0140 | 0.0106 | 0.0117 |
| nogcl | bmc | 19 | 0.0095 | 0.0136 | 0.0101 | 0.0110 |
| reuse_gate_init | mse | 18 | 0.0114 | 0.0177 | 0.0138 | 0.0183 |
| reuse_gate_init | gai | 18 | 0.0114 | 0.0175 | 0.0139 | 0.0184 |
| reuse_gate_init | bmc | 19 | 0.0112 | 0.0190 | 0.0142 | 0.0201 |

### Interpretation Notes

- Fixed BMC is the best validation run for both `static` and `nogcl` in this
  development matrix, but the gains are modest and not uniform across every
  test dataset.
- `nogcl + bmc` has the best cross-test profile in this matrix: digtime
  `0.0136`, timing_ctrl `0.0101`, array `0.0110`.
- Static SGRL is competitive but not clearly dominant over no-GCL under these
  lightweight settings. This means downstream architecture and label loss need
  to be compared carefully before claiming GCL benefit.
- The current online-reuse path is still much worse than both static and no-GCL.
  `GAI` does not improve it, and `BMC` slightly improves validation but hurts
  cross-dataset tests. This supports separating the reuse problem from the
  rebalancing problem for the next iteration.
- Current next step: keep `nogcl/static + bmc` as the clean rebalancing anchor,
  then improve reuse by changing the reuse architecture or optimization rather
  than expecting label rebalancing alone to fix it.

## 2026-07-09: S2 Online Feature Reuse

### Goal

Implement and test the conservative shared-backbone step:

```text
pretrained frozen SGRL online encoder
  -> online H_online per downstream batch
  -> original downstream GraphHead
  -> original downstream GNN layers and edge head
```

This differs from the previous replacement-style reuse path, which replaced the
downstream GNN with `SgrlBackboneHead`.

### Code Path

- `--sgrl_mode online_feature` added in `main.py`.
- `OnlineFeatureGraphHead` added in `model.py`.
- `GraphHead.forward(..., cl_x=None)` now accepts external CL features while
  preserving the static embedding path.
- `downstream_train.py` builds `OnlineFeatureGraphHead` when
  `args.use_sgrl_online_features` is enabled.

### Command

```bash
OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16 \
/home/lixc/.conda/envs/RCG/bin/python main.py \
  --dataset ssram+digtime+timing_ctrl+array_128_32_8t \
  --task regression \
  --task_level edge \
  --regress_loss mse \
  --batch_size 512 \
  --epochs 20 \
  --num_workers 0 \
  --gpu 3 \
  --sgrl 1 \
  --sgrl_mode online_feature \
  --cl_epochs 5 \
  --cl_gnn_layers 2 \
  --cl_hid_dim 64 \
  --cl_batch_size 32768 \
  --cl_num_neighbors 8 \
  --num_hops 2 \
  --num_neighbors 8 \
  --log_dir logs/online_feature_dev_20260709
```

Log:

```text
logs/online_feature_dev_20260709/20260709_213202_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossmse_batch512.txt
```

### Result

| Mode | Loss | Best epoch | Val MSE | digtime MSE | timing_ctrl MSE | array MSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| static | mse | 17 | 0.0097 | 0.0151 | 0.0101 | 0.0111 |
| nogcl | mse | 18 | 0.0096 | 0.0141 | 0.0099 | 0.0113 |
| reuse_gate_init | mse | 18 | 0.0114 | 0.0177 | 0.0138 | 0.0183 |
| online_feature_frozen | mse | 17 | 0.0096 | 0.0142 | 0.0101 | 0.0112 |

### Interpretation Notes

- S2 passes the development gate: it is close to both `static + MSE` and
  `nogcl + MSE`, and much better than the replacement-style `reuse_gate_init`.
- This supports the current interpretation that reuse should keep the
  downstream `GraphHead` first, rather than immediately replacing it with the
  SGRL encoder and an MLP head.
- The result does not yet prove that GCL online reuse is better than no-GCL, but
  it fixes the architecture-level failure from the first reuse attempt.
- Next step: S3 online feature finetuning, with a conservative learning rate or
  a separate optimizer group for the online encoder.

## 2026-07-09: S3 Online Feature Finetuning

### Goal

Test whether the S2 online-feature path benefits from supervised finetuning of
the pretrained SGRL online encoder. This is still not the compact final model:
it keeps both the online encoder and downstream `GraphHead`, but lets downstream
gradients update the online encoder with a much smaller learning rate.

### Code Path

- Added `--sgrl_mode online_feature_finetune`.
- Added `--sgrl_online_lr` for the downstream finetuning learning rate of the
  online encoder.
- `OnlineFeatureGraphHead.load_online_encoder_state(..., freeze=False)` is used
  in finetune mode.
- `downstream_train.py` now builds two optimizer groups:
  online encoder at `--sgrl_online_lr`, downstream parameters at `--lr`.

### Command

```bash
OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16 \
/home/lixc/.conda/envs/RCG/bin/python main.py \
  --dataset ssram+digtime+timing_ctrl+array_128_32_8t \
  --task regression \
  --task_level edge \
  --regress_loss mse \
  --batch_size 512 \
  --epochs 20 \
  --num_workers 0 \
  --gpu 3 \
  --sgrl 1 \
  --sgrl_mode online_feature_finetune \
  --sgrl_online_lr 1e-6 \
  --cl_epochs 5 \
  --cl_gnn_layers 2 \
  --cl_hid_dim 64 \
  --cl_batch_size 32768 \
  --cl_num_neighbors 8 \
  --num_hops 2 \
  --num_neighbors 8 \
  --log_dir logs/online_feature_finetune_dev_20260709
```

Log:

```text
logs/online_feature_finetune_dev_20260709/20260709_214440_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossmse_batch512.txt
```

### Result

| Mode | Loss | Best epoch | Val MSE | digtime MSE | timing_ctrl MSE | array MSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| static | mse | 17 | 0.0097 | 0.0151 | 0.0101 | 0.0111 |
| nogcl | mse | 18 | 0.0096 | 0.0141 | 0.0099 | 0.0113 |
| online_feature_frozen | mse | 17 | 0.0096 | 0.0142 | 0.0101 | 0.0112 |
| online_feature_finetune | mse | 17 | 0.0096 | 0.0148 | 0.0100 | 0.0109 |

Extra note: the raw validation loss kept decreasing after the script's selected
best epoch, reaching `0.00958864` at epoch 19. The training script selects best
checkpoints by the rounded `mse` field, so epoch 18/19 were treated as ties and
did not trigger fresh test evaluation.

### Interpretation Notes

- S3 is stable and reaches the same validation MSE range as S2.
- Finetuning slightly improves the array test split and timing-control is about
  the same, but digtime gets worse than frozen online-feature reuse.
- This does not yet justify spending more time tuning S3. It should remain as a
  candidate setting, while the main reuse work moves to S4/S5 where we can
  actually reduce duplicated GNN capacity.
- Next step: add a compactness-oriented reuse stage, starting with parameter
  counting and initialization/partial-sharing experiments.

## 2026-07-09: S4 Parameter Initialization Reuse

### Goal

Move from feature reuse to a compactness-oriented reuse path:

```text
pretrain SGRL online encoder
  -> copy compatible online-encoder weights into the original downstream GraphHead
  -> train one downstream GraphHead for inference
```

This is different from S2/S3: S2/S3 still carry an online encoder plus the
downstream `GraphHead` during downstream training, while S4 uses the online
encoder only for initialization and keeps a single downstream backbone.

### Code Path

- Added `--sgrl_mode init_reuse`.
- Added `GraphHead.load_sgrl_encoder_init(...)` to copy compatible tensors and
  print copied/skipped summaries.
- Added downstream parameter-count logging.
- Added CUDA diagnostics to every run log.

### Commands

GPU-verified `init_reuse + MSE`:

```bash
/home/lixc/.conda/envs/RCG/bin/python main.py \
  --dataset ssram+digtime+timing_ctrl+array_128_32_8t \
  --task regression \
  --task_level edge \
  --regress_loss mse \
  --batch_size 512 \
  --epochs 20 \
  --num_workers 0 \
  --gpu 3 \
  --sgrl 1 \
  --sgrl_mode init_reuse \
  --model clustergcn \
  --hid_dim 64 \
  --num_gnn_layers 2 \
  --cl_model clustergcn \
  --cl_epochs 5 \
  --cl_gnn_layers 2 \
  --cl_hid_dim 64 \
  --cl_batch_size 32768 \
  --cl_num_neighbors 8 \
  --num_hops 2 \
  --num_neighbors 8 \
  --log_dir logs/init_reuse_dev_gpu_20260709
```

GPU-verified paired compact `no-GCL + MSE` baseline:

```bash
/home/lixc/.conda/envs/RCG/bin/python main.py \
  --dataset ssram+digtime+timing_ctrl+array_128_32_8t \
  --task regression \
  --task_level edge \
  --regress_loss mse \
  --batch_size 512 \
  --epochs 20 \
  --num_workers 0 \
  --gpu 3 \
  --sgrl 0 \
  --model clustergcn \
  --hid_dim 64 \
  --num_gnn_layers 2 \
  --num_hops 2 \
  --num_neighbors 8 \
  --log_dir logs/compact_clustergcn_baseline_gpu_20260709
```

GPU-verified paired compact `static + MSE` baseline:

```bash
/home/lixc/.conda/envs/RCG/bin/python main.py \
  --dataset ssram+digtime+timing_ctrl+array_128_32_8t \
  --task regression \
  --task_level edge \
  --regress_loss mse \
  --batch_size 512 \
  --epochs 20 \
  --num_workers 0 \
  --gpu 3 \
  --sgrl 1 \
  --sgrl_mode static \
  --model clustergcn \
  --hid_dim 63 \
  --num_gnn_layers 2 \
  --cl_model clustergcn \
  --cl_epochs 5 \
  --cl_gnn_layers 2 \
  --cl_hid_dim 64 \
  --cl_batch_size 32768 \
  --cl_num_neighbors 8 \
  --num_hops 2 \
  --num_neighbors 8 \
  --log_dir logs/static_mse_compact_gpu_20260709
```

Static compact uses `--hid_dim 63` because the static embedding path combines
node type, circuit statistics, and CL features, so `GraphHead` requires the
hidden dimension to be divisible by 3.

### Result

All successful logs start with `CUDA status: available=True` and `Using GPU: 3`.

| Mode | Loss | Params | Best epoch | Val MSE | digtime MSE | timing_ctrl MSE | array MSE | Log |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| compact no-GCL | mse | 27,170 | 16 | 0.0101 | 0.0145 | 0.0124 | 0.0113 | `logs/compact_clustergcn_baseline_gpu_20260709/20260709_221835_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossmse_batch512.txt` |
| compact static | mse | 27,134 | 17 | 0.0100 | 0.0137 | 0.0117 | 0.0115 | `logs/static_mse_compact_gpu_20260709/20260709_223724_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossmse_batch512.txt` |
| init_reuse | mse | 27,170 | 19 | 0.0097 | 0.0141 | 0.0112 | 0.0121 | `logs/init_reuse_dev_gpu_20260709/20260709_220924_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossmse_batch512.txt` |

The `init_reuse` log reports:

```text
SGRL GraphHead init reuse summary: copied_tensors=12, copied_values=17025, skipped=6.
```

### Interpretation Notes

- S4 is the first reuse variant in this line that actually makes the downstream
  inference model compact: after initialization, inference uses one `GraphHead`
  with the same parameter count as the paired no-GCL baseline.
- Compared with the GPU-verified compact no-GCL baseline, `init_reuse` improves
  validation MSE (`0.0101 -> 0.0097`) and improves digtime/timing_ctrl, while
  array is slightly worse (`0.0113 -> 0.0121`).
- Compared with the GPU-verified compact static baseline, `init_reuse` has
  better validation and timing-control MSE, while compact static is better on
  digtime and array. This means S4 is promising but not a clean win over static
  GCL yet.
- This is a better reuse direction than the earlier replacement-style
  `reuse_gate_init` path, which was both less accurate and less faithful to the
  original downstream model.
- The `static + MSE` rerun had two failed starts before the successful log:
  `--hid_dim 64` violated the static-path divisibility check, and the first
  `--hid_dim 63` attempt hit CUDA OOM while GPU3 was nearly full. No external
  processes were killed or interrupted; the successful run started after GPU3
  memory was naturally released.
- Next direction: move to S5 partial shared backbone. S5 should try to keep the
  compactness benefit of `init_reuse` while preserving the stronger digtime and
  array transfer behavior of static GCL.

## 2026-07-09: S5 Partial Shared Backbone Design

### Goal

Move from one-time initialization reuse to a real shared-backbone architecture:

```text
pretrain SGRL online encoder
  -> reuse node/edge encoder + lower online GNN layers as shared backbone
  -> keep a downstream-specific tail GNN and prediction head
  -> train supervised downstream task with MSE first
```

This is the first S5 step toward merging the GCL online GNN encoder and the
downstream GNN backbone into one smaller structure. It is deliberately not yet
joint contrastive/supervised training; the immediate question is whether a
partly shared backbone is a better reuse unit than S4's one-time parameter
initialization.

### Architecture

```text
batch graph
  -> SharedGNNBackbone
       node/edge type embedding from SGRL online encoder
       first k SGRL online GNN layers
  -> optional circuit-statistics fusion
  -> downstream tail GNN layers
  -> downstream edge/node prediction head
```

Implemented mode:

```text
--sgrl_mode partial_shared
--shared_gnn_layers 1
```

Code path:

- `main.py`
  - Added `partial_shared` to `--sgrl_mode`.
  - Added `--shared_gnn_layers`.
  - Added `--partial_shared_stats_fusion`.
- `model.py`
  - Added `SharedGNNBackbone`.
  - Added `PartialSharedGraphHead`.
  - Shared lower layers are loaded from the SGRL online encoder checkpoint.
  - Downstream tail/head remains task-specific and trainable.
- `downstream_train.py`
  - Added `PartialSharedGraphHead` to `build_downstream_model`.

### First Experiment Plan

First run only MSE. Rebalancing losses should wait until the shared-backbone
architecture is validated.

| Mode | Loss | Purpose |
| --- | --- | --- |
| compact no-GCL | mse | Lower compact baseline |
| compact static | mse | Original static GCL baseline |
| init_reuse | mse | S4 one-time initialization reuse |
| partial_shared_k1 | mse | S5 conservative shared lower layer |
| partial_shared_k2 | mse | S5 more aggressive sharing |

The first S5 command should be:

```bash
/home/lixc/.conda/envs/RCG/bin/python main.py \
  --dataset ssram+digtime+timing_ctrl+array_128_32_8t \
  --task regression \
  --task_level edge \
  --regress_loss mse \
  --batch_size 512 \
  --epochs 20 \
  --num_workers 0 \
  --gpu 3 \
  --sgrl 1 \
  --sgrl_mode partial_shared \
  --shared_gnn_layers 1 \
  --partial_shared_stats_fusion add \
  --model clustergcn \
  --hid_dim 64 \
  --num_gnn_layers 2 \
  --cl_model clustergcn \
  --cl_epochs 5 \
  --cl_gnn_layers 2 \
  --cl_hid_dim 64 \
  --cl_batch_size 32768 \
  --cl_num_neighbors 8 \
  --num_hops 2 \
  --num_neighbors 8 \
  --log_dir logs/partial_shared_k1_mse_gpu_20260709
```

Expected decision rule:

- If `partial_shared_k1` matches or improves on `init_reuse`, S5 is a stronger
  reuse direction than S4.
- If `partial_shared_k2` degrades, downstream still needs a task-specific tail.
- If both degrade, the shared lower-layer design should be revised before
  adding GAI/BMC.

### First GPU Result

Smoke run:

```bash
/home/lixc/.conda/envs/RCG/bin/python main.py \
  --dataset ssram+digtime+timing_ctrl+array_128_32_8t \
  --task regression \
  --task_level edge \
  --regress_loss mse \
  --batch_size 512 \
  --epochs 1 \
  --num_workers 0 \
  --gpu 3 \
  --sgrl 1 \
  --sgrl_mode partial_shared \
  --shared_gnn_layers 1 \
  --partial_shared_stats_fusion add \
  --model clustergcn \
  --hid_dim 64 \
  --num_gnn_layers 2 \
  --cl_model clustergcn \
  --cl_epochs 5 \
  --cl_gnn_layers 2 \
  --cl_hid_dim 64 \
  --cl_batch_size 32768 \
  --cl_num_neighbors 8 \
  --num_hops 2 \
  --num_neighbors 8 \
  --log_dir logs/partial_shared_k1_smoke_gpu_20260709
```

The smoke run completed one epoch on GPU3 and verified that checkpoint loading,
shared-layer copying, forward/backward, validation, and three test loaders all
work.

Formal 20-epoch run:

```bash
/home/lixc/.conda/envs/RCG/bin/python main.py \
  --dataset ssram+digtime+timing_ctrl+array_128_32_8t \
  --task regression \
  --task_level edge \
  --regress_loss mse \
  --batch_size 512 \
  --epochs 20 \
  --num_workers 0 \
  --gpu 3 \
  --sgrl 1 \
  --sgrl_mode partial_shared \
  --shared_gnn_layers 1 \
  --partial_shared_stats_fusion add \
  --model clustergcn \
  --hid_dim 64 \
  --num_gnn_layers 2 \
  --cl_model clustergcn \
  --cl_epochs 5 \
  --cl_gnn_layers 2 \
  --cl_hid_dim 64 \
  --cl_batch_size 32768 \
  --cl_num_neighbors 8 \
  --num_hops 2 \
  --num_neighbors 8 \
  --log_dir logs/partial_shared_k1_mse_gpu_20260709
```

Successful formal log:

```text
logs/partial_shared_k1_mse_gpu_20260709/20260709_230332_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossmse_batch512.txt
```

The log reports:

```text
Partial shared backbone load summary: shared_layers=1, copied_tensors=10, copied_values=9409, skipped=1.
Model parameters: total=29,762, trainable=29,762, frozen=0.
```

The more aggressive S5-k2 run shared both downstream GNN layers with the SGRL
online GNN and therefore left no downstream-specific tail GNN:

```bash
/home/lixc/.conda/envs/RCG/bin/python main.py \
  --dataset ssram+digtime+timing_ctrl+array_128_32_8t \
  --task regression --task_level edge --regress_loss mse \
  --batch_size 512 --epochs 20 --num_workers 0 --gpu 3 \
  --sgrl 1 --sgrl_mode partial_shared \
  --shared_gnn_layers 2 \
  --partial_shared_stats_fusion add \
  --model clustergcn --hid_dim 64 --num_gnn_layers 2 \
  --cl_model clustergcn --cl_epochs 5 --cl_gnn_layers 2 \
  --cl_hid_dim 64 --cl_batch_size 32768 --cl_num_neighbors 8 \
  --num_hops 2 --num_neighbors 8 \
  --log_dir logs/partial_shared_k2_mse_gpu_20260709
```

The k2 log reports:

```text
Partial shared backbone load summary: shared_layers=2, copied_tensors=13, copied_values=17665, skipped=1.
Model parameters: total=29,762, trainable=29,762, frozen=0.
```

| Mode | Loss | Params | Best epoch | Val MSE | digtime MSE | timing_ctrl MSE | array MSE | Log |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| compact no-GCL | mse | 27,170 | 16 | 0.0101 | 0.0145 | 0.0124 | 0.0113 | `logs/compact_clustergcn_baseline_gpu_20260709/20260709_221835_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossmse_batch512.txt` |
| compact static | mse | 27,134 | 17 | 0.0100 | 0.0137 | 0.0117 | 0.0115 | `logs/static_mse_compact_gpu_20260709/20260709_223724_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossmse_batch512.txt` |
| init_reuse | mse | 27,170 | 19 | 0.0097 | 0.0141 | 0.0112 | 0.0121 | `logs/init_reuse_dev_gpu_20260709/20260709_220924_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossmse_batch512.txt` |
| partial_shared_k1_add | mse | 29,762 | 19 | 0.0104 | 0.0180 | 0.0123 | 0.0130 | `logs/partial_shared_k1_mse_gpu_20260709/20260709_230332_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossmse_batch512.txt` |
| partial_shared_k2_add | mse | 29,762 | 18 | 0.0108 | 0.0294 | 0.0152 | 0.0187 | `logs/partial_shared_k2_mse_gpu_20260709/20260709_233131_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossmse_batch512.txt` |
| partial_shared_k1_gate | mse | 38,018 | 19 | 0.0102 | 0.0140 | 0.0119 | 0.0126 | `logs/partial_shared_k1_gate_mse_gpu_20260709/20260709_234332_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossmse_batch512.txt` |
| partial_shared_k1_residual_gate | mse | 38,018 | 19 | 0.0105 | 0.0147 | 0.0114 | 0.0120 | `logs/partial_shared_k1_residual_gate_mse_gpu_20260709/20260709_234916_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossmse_batch512.txt` |
| partial_shared_k1_gate_backbone_lr1e-5 | mse | 38,018 | 19 | 0.0107 | 0.0150 | 0.0123 | 0.0127 | `logs/partial_shared_k1_gate_backbone_lr1e-5_mse_gpu_20260709/20260709_235738_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossmse_batch512.txt` |
| partial_shared_k1_gate_freeze1 | mse | 38,018 total / 28,738 warmup trainable | 18 | 0.0098 | 0.0141 | 0.0117 | 0.0121 | `logs/partial_shared_k1_gate_freeze1_mse_gpu_20260710/20260710_002548_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossmse_batch512.txt` |
| partial_shared_k1_gate_freeze2 | mse | 38,018 total / 28,738 warmup trainable | 18 | 0.0098 | 0.0139 | 0.0117 | 0.0120 | `logs/partial_shared_k1_gate_freeze2_mse_gpu_20260710/20260710_003140_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossmse_batch512.txt` |
| partial_shared_k1_gate_freeze3 | mse | 38,018 total / 28,738 warmup trainable | 19 | 0.0098 | 0.0140 | 0.0117 | 0.0119 | `logs/partial_shared_k1_gate_freeze3_mse_gpu_20260709/20260710_000538_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossmse_batch512.txt` |
| partial_shared_k1_scalar_gate_freeze3 | mse | 29,763 total / 20,483 warmup trainable | 18 | 0.0102 | 0.0172 | 0.0117 | 0.0130 | `logs/partial_shared_k1_scalar_gate_freeze3_mse_gpu_20260710/20260710_005658_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossmse_batch512.txt` |
| partial_shared_k1_vector_gate_freeze3 | mse | 29,826 total / 20,546 warmup trainable | 19 | 0.0101 | 0.0189 | 0.0118 | 0.0134 | `logs/partial_shared_k1_vector_gate_freeze3_mse_gpu3_20260710/20260710_010211_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossmse_batch512.txt` |
| partial_shared_k1_gate_freeze5 | mse | 38,018 total / 28,738 warmup trainable | 19 | 0.0099 | 0.0139 | 0.0118 | 0.0120 | `logs/partial_shared_k1_gate_freeze5_mse_gpu_20260710/20260710_003708_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossmse_batch512.txt` |
| partial_shared_k1_gate_freeze3_backbone_lr3e-5 | mse | 38,018 total / 28,738 warmup trainable | 19 | 0.0099 | 0.0141 | 0.0116 | 0.0121 | `logs/partial_shared_k1_gate_freeze3_backbone_lr3e-5_mse_gpu_20260710/20260710_004222_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossmse_batch512.txt` |

### Interpretation Notes

- The S5-k1 implementation is functional: it loads the SGRL online lower layer,
  trains normally, and steadily reduces validation MSE from 0.0133 to 0.0104.
- S5-k2 is also functional, but it is not a good reuse candidate in this form:
  source validation MSE keeps improving to 0.0108, while cross-dataset transfer
  becomes much worse than k1 and all compact baselines, especially digtime
  (0.0294) and array (0.0187).
- This confirms that source validation alone is misleading for S5. The stronger
  sharing can overfit the training/source distribution even when val improves.
- The parameter count is also larger than the compact baselines because this
  implementation keeps stats adapters and edge/head modules. Even k2 does not
  yet provide the simpler final architecture numerically, although it is more
  structurally shared.
- Next S5 direction should not push sharing deeper with plain `add` fusion.
  Better next tests:
  - `partial_shared_k1` with `gate` or `residual_gate` stats fusion.
  - `partial_shared_k1` with a smaller supervised LR or frozen shared layer
    warmup, to reduce overwriting of the pretrained GCL representation.
  - clean parameter accounting so unused tail modules are not instantiated when
    `tail_layers == 0`.

### S5 Follow-up: Fusion and Warmup

Follow-up S5 experiments kept `shared_gnn_layers=1` and changed only the
statistics fusion or shared-backbone optimization schedule.

Code support added:

```text
--partial_shared_backbone_lr <float>
--partial_shared_freeze_epochs <int>
--partial_shared_stats_fusion scalar_gate
--partial_shared_stats_fusion vector_gate
```

`--partial_shared_backbone_lr` creates a separate optimizer group for
`shared_backbone.*`. `--partial_shared_freeze_epochs N` freezes the shared
backbone for the first `N` downstream epochs, keeps its batchnorm/dropout path in
eval mode during the freeze window, then unfreezes it and rebuilds the optimizer.

Key observations:

- `partial_shared_k1_gate` fixes most of the transfer collapse from
  `partial_shared_k1_add`: digtime improves from `0.0180` to `0.0140`, timing
  from `0.0123` to `0.0119`, and array from `0.0130` to `0.0126`. The cost is a
  larger parameter count because the learned gate adds `8,256` parameters.
- `partial_shared_k1_residual_gate` is not better on validation, but it improves
  timing_ctrl and array relative to plain gate. This suggests that keeping a
  statistics residual can help some transfer targets even when source val is
  weaker.
- `partial_shared_k1_gate_backbone_lr1e-5` is stable but too conservative. Val
  reaches only `0.0107`, and digtime remains `0.0150`; lowering the whole shared
  backbone update rate is less promising than a short freeze warmup.
- `partial_shared_k1_gate_freeze3` remains the best S5 variant after the warmup
  sweep. Its exact reported Val MSE is `0.00979278`, with test MSE
  `0.01404435/0.01165146/0.01194173` for
  digtime/timing_ctrl/array. It matches or slightly beats the compact baselines
  on source validation while keeping transfer much healthier than k2/add. Its
  remaining weakness is digtime, still slightly worse than compact static
  (`0.0137`).
- The warmup sweep shows that `freeze1` and `freeze2` are close to `freeze3`
  (`0.00983751` and `0.00984017` reported Val MSE), but do not beat it overall.
  `freeze2` has the best digtime among the freeze sweep (`0.01392501`), while
  `freeze3` keeps the best validation and array transfer.
- `freeze5` catches up late (`0.00988753`) but is slower and still behind
  `freeze3`; longer freezing seems to delay downstream adaptation.
- `freeze3 + backbone_lr=3e-5` also catches up late (`0.00989967`), but it is
  slower than plain `freeze3`. This suggests that the useful ingredient is the
  short freeze warmup, not an overly small shared-backbone learning rate.
- For `freeze1` and `freeze2`, the final epoch printed slightly lower val losses
  without emitting matching test results. The table uses each log's final
  reported `Best epoch` and `Test results` for reproducible comparison.
- The slim gate variants replace the 8,256-parameter learned gate MLP with
  either one scalar gate (`scalar_gate`) or one hidden-dimension gate vector
  (`vector_gate`). This reduces the `k1 + gate + freeze3` model from `38,018`
  total parameters to `29,763` or `29,826`, roughly a 21.6% reduction.
- `partial_shared_k1_vector_gate_freeze3` reaches a slightly better validation
  MSE than `scalar_gate` (`0.01012813` vs reported-best `0.01020981`), but both
  lose the strong transfer behavior of the full learned gate. The clearest
  regression is digtime: `0.0172` for scalar and `0.0189` for vector versus
  `0.0140` for full `gate + freeze3`.
- The scalar run printed a final epoch validation loss of `0.01015140` without
  new matching test results; the table keeps the log's reported best epoch and
  associated tests. The first vector attempt on GPU1 failed before training
  because the current PyTorch build does not support the Blackwell `sm_120`
  architecture; the recorded vector result is the GPU3 rerun.

Current S5 conclusion after the warmup/slim-gate sweep:

- Do not continue deeper sharing (`k2`) with the current head.
- Keep full `k1 + gate` as the main shared-backbone architecture. `freeze2`
  and `freeze3` are both stable; `freeze3` was the first accuracy candidate,
  while `freeze2` later became the best BMC row in the rebalancing sweep.
- Keep `k1 + vector_gate + freeze3` and `k1 + scalar_gate + freeze3` as compact
  ablation baselines: they prove the architecture can be slimmed, but the
  current slim gates are not accurate enough to replace the full learned gate.
- If compactness becomes the primary target, the next design should try a
  middle ground such as grouped/low-rank gate or a tiny bottleneck adapter,
  instead of jumping directly from 8,256 gate parameters down to 1 or 64.

### S5 Rebalancing Sweep: Gate Accuracy vs Vector Slim

Goal: finish the already-planned label-rebalancing comparison on the current
best shared-backbone candidates, without adding new architecture branches.

Common setup:

```text
dataset: ssram+digtime+timing_ctrl+array_128_32_8t
epochs: 20
batch_size: 512
sgrl_mode: partial_shared
shared_gnn_layers: 1
cl_epochs: 5
GPUs: 3/4
```

The first vector-GAI attempt hit a native GMM/OpenBLAS crash (`code 139`). The
successful GAI reruns capped BLAS threads with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`; they still used
GPU execution.

| Mode | Loss | Params | Freeze | Best epoch | Val MSE/loss | digtime MSE | timing_ctrl MSE | array MSE | Log |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| partial_shared_k1_gate_freeze3 | mse | 38,018 total / 28,738 warmup trainable | 3 | 19 | 0.0098 / 0.00979278 | 0.0140 | 0.0117 | 0.0119 | `logs/partial_shared_k1_gate_freeze3_mse_gpu_20260709/20260710_000538_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossmse_batch512.txt` |
| partial_shared_k1_gate_freeze3 | gai | 38,018 total / 28,738 warmup trainable | 3 | 19 | 0.0098 / 0.00977143 | 0.0140 | 0.0117 | 0.0120 | `logs/partial_shared_k1_gate_freeze3_gai_gpu4_20260710/20260710_011656_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossgai_batch512.txt` |
| partial_shared_k1_gate_freeze3 | bmc | 38,018 total / 28,738 warmup trainable | 3 | 19 | 0.0098 / 0.00980692 | 0.0142 | 0.0116 | 0.0118 | `logs/partial_shared_k1_gate_freeze3_bmc_gpu3_20260710/20260710_011656_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossbmc_batch512.txt` |
| partial_shared_k1_vector_gate_freeze3 | mse | 29,826 total / 20,546 warmup trainable | 3 | 19 | 0.0101 / 0.01012813 | 0.0189 | 0.0118 | 0.0134 | `logs/partial_shared_k1_vector_gate_freeze3_mse_gpu3_20260710/20260710_010211_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossmse_batch512.txt` |
| partial_shared_k1_vector_gate_freeze3 | gai | 29,826 total / 20,546 warmup trainable | 3 | 19 | 0.0101 / 0.01013507 | 0.0192 | 0.0119 | 0.0133 | `logs/partial_shared_k1_vector_gate_freeze3_gai_gpu3_20260710_retry_gpu_threads1/20260710_012324_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossgai_batch512.txt` |
| partial_shared_k1_vector_gate_freeze3 | bmc | 29,826 total / 20,546 warmup trainable | 3 | 18 | 0.0101 / 0.01012138 | 0.0181 | 0.0117 | 0.0130 | `logs/partial_shared_k1_vector_gate_freeze3_bmc_gpu4_20260710/20260710_011934_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossbmc_batch512.txt` |
| partial_shared_k1_gate_freeze2 | gai | 38,018 total / 28,738 warmup trainable | 2 | 18 | 0.0098 / 0.00982826 | 0.0140 | 0.0117 | 0.0119 | `logs/partial_shared_k1_gate_freeze2_gai_gpu4_20260710_threads1/20260710_012939_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossgai_batch512.txt` |
| partial_shared_k1_gate_freeze2 | bmc | 38,018 total / 28,738 warmup trainable | 2 | 19 | 0.0098 / 0.00982118 | 0.0137 | 0.0114 | 0.0120 | `logs/partial_shared_k1_gate_freeze2_bmc_gpu3_20260710/20260710_012939_edge_regression_ssram+digtime+timing_ctrl+array_128_32_8t_lossbmc_batch512.txt` |

Single-seed rebalancing observations (superseded where noted by the multi-seed
audit below):

- Full `gate` is the best-performing reuse structure in this sweep. GAI/BMC
  do not materially improve source validation over MSE, but BMC improves
  timing_ctrl and the `freeze2 + BMC` row gives the best transfer result for
  this seed: `0.0137/0.0114/0.0120`.
- `gate + freeze3` is internally consistent for this seed: MSE, GAI, and BMC stay
  around `0.0098` validation and `0.014x/0.011x/0.012x` transfer.
- `vector_gate` is compact but not a primary candidate yet. BMC slightly
  repairs array/timing_ctrl versus vector-MSE, but digtime remains much worse
  (`0.0181`-`0.0192`) than the full-gate rows (`0.0137`-`0.0142`).
- These single-seed results motivated the full-gate multi-seed audit below;
  they should not be used alone to claim a stable improvement. Rebalancing also
  should not be used to rescue an over-slimmed reuse architecture.

### S5 Multi-Seed Stability Audit

Goal: check whether the strongest single-seed S5 rows remain competitive across
seeds `0/1/2`, using matched original-static controls. No new architecture was
introduced in this audit.

Common setup:

```text
dataset: ssram+digtime+timing_ctrl+array_128_32_8t
epochs: 20
batch_size: 512
lr: 1e-4
cl_epochs: 5
seeds: 0, 1, 2
environment: /home/lixc/.conda/envs/RCG
GPUs: 3/4
log root: logs/multiseed_s5_gpu_20260710
```

Every row below completed with `Done!`. `Val MSE/loss` and test metrics come
from the final reported `Best epoch` block, so each test result remains tied to
the checkpoint that produced it.

| Mode | Loss | Seed | Best epoch | Val MSE/loss | digtime MSE | timing_ctrl MSE | array MSE | Run directory |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| static | mse | 0 | 18 | 0.0097 / 0.00970833 | 0.0139 | 0.0119 | 0.0114 | `logs/multiseed_s5_gpu_20260710/static_mse_seed0` |
| static | mse | 1 | 17 | 0.0099 / 0.00994769 | 0.0148 | 0.0116 | 0.0113 | `logs/multiseed_s5_gpu_20260710/static_mse_seed1` |
| static | mse | 2 | 19 | 0.0099 / 0.00994125 | 0.0142 | 0.0121 | 0.0111 | `logs/multiseed_s5_gpu_20260710/static_mse_seed2` |
| static | bmc | 0 | 15 | 0.0098 / 0.00979413 | 0.0142 | 0.0117 | 0.0112 | `logs/multiseed_s5_gpu_20260710/static_bmc_seed0` |
| static | bmc | 1 | 16 | 0.0099 / 0.00991911 | 0.0146 | 0.0119 | 0.0113 | `logs/multiseed_s5_gpu_20260710/static_bmc_seed1` |
| static | bmc | 2 | 17 | 0.0099 / 0.00994669 | 0.0143 | 0.0126 | 0.0114 | `logs/multiseed_s5_gpu_20260710/static_bmc_seed2` |
| partial_shared_k1_gate_freeze2 | bmc | 0 | 18 | 0.0098 / 0.00984451 | 0.0140 | 0.0119 | 0.0133 | `logs/multiseed_s5_gpu_20260710/partial_shared_k1_gate_freeze2_bmc_seed0` |
| partial_shared_k1_gate_freeze2 | bmc | 1 | 18 | 0.0101 / 0.01009922 | 0.0146 | 0.0127 | 0.0129 | `logs/multiseed_s5_gpu_20260710/partial_shared_k1_gate_freeze2_bmc_seed1` |
| partial_shared_k1_gate_freeze2 | bmc | 2 | 18 | 0.0102 / 0.01024479 | 0.0163 | 0.0126 | 0.0119 | `logs/multiseed_s5_gpu_20260710/partial_shared_k1_gate_freeze2_bmc_seed2` |
| partial_shared_k1_gate_freeze3 | mse | 0 | 17 | 0.0099 / 0.00990024 | 0.0142 | 0.0122 | 0.0133 | `logs/multiseed_s5_gpu_20260710/partial_shared_k1_gate_freeze3_mse_seed0` |
| partial_shared_k1_gate_freeze3 | mse | 1 | 19 | 0.0100 / 0.01001535 | 0.0144 | 0.0127 | 0.0123 | `logs/multiseed_s5_gpu_20260710/partial_shared_k1_gate_freeze3_mse_seed1` |
| partial_shared_k1_gate_freeze3 | mse | 2 | 17 | 0.0103 / 0.01027607 | 0.0165 | 0.0124 | 0.0121 | `logs/multiseed_s5_gpu_20260710/partial_shared_k1_gate_freeze3_mse_seed2` |
| partial_shared_k1_gate_freeze3 | bmc | 0 | 18 | 0.0099 / 0.00992583 | 0.0137 | 0.0126 | 0.0139 | `logs/multiseed_s5_gpu_20260710/partial_shared_k1_gate_freeze3_bmc_seed0` |
| partial_shared_k1_gate_freeze3 | bmc | 1 | 19 | 0.0100 / 0.01002019 | 0.0141 | 0.0123 | 0.0129 | `logs/multiseed_s5_gpu_20260710/partial_shared_k1_gate_freeze3_bmc_seed1` |
| partial_shared_k1_gate_freeze3 | bmc | 2 | 17 | 0.0103 / 0.01029715 | 0.0166 | 0.0121 | 0.0121 | `logs/multiseed_s5_gpu_20260710/partial_shared_k1_gate_freeze3_bmc_seed2` |

The summary uses population standard deviation (`ddof=0`) over the three seeds.
Validation MSE is limited to the four decimal places emitted by the training
logger; the per-seed table also keeps the higher-precision validation loss.

| Mode | Loss | Val MSE mean +/- std | digtime mean +/- std | timing_ctrl mean +/- std | array mean +/- std |
| --- | --- | ---: | ---: | ---: | ---: |
| static | mse | 0.009833 +/- 0.000094 | 0.014300 +/- 0.000374 | 0.011867 +/- 0.000205 | 0.011267 +/- 0.000125 |
| static | bmc | 0.009867 +/- 0.000047 | 0.014367 +/- 0.000170 | 0.012067 +/- 0.000386 | 0.011300 +/- 0.000082 |
| partial_shared_k1_gate_freeze2 | bmc | 0.010033 +/- 0.000170 | 0.014967 +/- 0.000974 | 0.012400 +/- 0.000356 | 0.012700 +/- 0.000589 |
| partial_shared_k1_gate_freeze3 | mse | 0.010067 +/- 0.000170 | 0.015033 +/- 0.001040 | 0.012433 +/- 0.000205 | 0.012567 +/- 0.000525 |
| partial_shared_k1_gate_freeze3 | bmc | 0.010067 +/- 0.000170 | 0.014800 +/- 0.001283 | 0.012333 +/- 0.000205 | 0.012967 +/- 0.000736 |

Stability conclusions:

- The earlier single-seed `freeze2 + BMC` best row does not hold across seeds.
  Its seed-2 digtime MSE rises to `0.0163`, and its three-seed transfer means
  are all worse than both static controls.
- Original `static + MSE` has the best three-seed mean on validation and all
  three transfer datasets. `static + BMC` is close and has lower variance on
  validation, digtime, and array, but does not improve the means.
- Within the shared-backbone rows, BMC slightly improves freeze3 digtime and
  timing_ctrl means relative to MSE, while MSE remains better on array. BMC is
  therefore not a general fix for the current reuse gap.
- All three partial-shared configurations are more seed-sensitive on digtime
  than the static controls. The large seed-2 degradation is the main warning
  signal and should be treated as an architecture/optimization stability issue,
  not hidden by selecting seed 0.
- S5 still demonstrates real structural reuse of the online lower GNN layer,
  but the current `k1 + gate + freeze` implementation has not yet matched the
  original static model in multi-seed accuracy or stability. Do not claim the
  earlier single-seed result as the final reuse improvement.

### S5.5 Optimizer, Eval-Mode, and Frozen-Backbone Audit

Goal: separate three possible causes of S5 instability before adding another
architecture or loss branch:

1. downstream Adam state was reset when the shared backbone was unfrozen;
2. gradient freezing and dropout/batchnorm eval mode were coupled;
3. cached static embeddings and online embeddings might see incompatible
   downstream sampling contexts.

Code changes:

- Unfreezing now adds the newly trainable backbone parameters to the existing
  Adam optimizer instead of rebuilding it. Existing gate/tail/head optimizer
  moments and step counts are preserved.
- Added `--partial_shared_backbone_eval_policy`:
  - `frozen_only` preserves the previous module-mode behavior;
  - `always` keeps backbone dropout/batchnorm in eval mode even after gradients
    are enabled.
- Added `scripts/audit_static_online_consistency.py` to compare cached static
  embeddings with outputs from the same online checkpoint inside downstream
  disjoint `LinkNeighborLoader` batches.
- Added three focused unit tests for optimizer-state preservation and eval-mode
  behavior. All tests pass with the RCG environment.

#### Static-vs-Online Representation Consistency

Audit setup:

```text
checkpoint: best_online_ssram+digtime+timing_ctrl+array_128_32_8t_clustergcn_layer2_dim64_tanh_dr0.3_small.pkl
cached embeddings: embeddings_ssram+digtime+timing_ctrl+array_128_32_8t_clustergcn_layer2_dim64_tanh.pkl
downstream sampling: 2 hops, 8 neighbors, disjoint=True
sample: first 2,048 target edges per circuit
device: GPU 3
result: logs/static_online_consistency_20260710/audit_seed0_gpu3.json
```

| Circuit | Root cosine mean | Root cosine p05 | Root relative L2 mean | Root MSE | All-node cosine mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| ssram | 0.9999989 | 0.9999925 | 0.000565 | 7.05e-7 | 0.969866 |
| digtime | 0.9999995 | 0.9999925 | 0.000242 | 3.02e-7 | 0.972392 |
| timing_ctrl | 0.9999983 | 0.9999925 | 0.000855 | 1.07e-6 | 0.968608 |
| array | 0.9999991 | 0.9999925 | 0.000433 | 5.40e-7 | 0.970323 |

The target-edge root embeddings are effectively identical across the cached
static and online disjoint-batch paths. The lower all-node cosine is expected:
non-root neighbors in layer-wise sampling do not all receive a complete two-hop
receptive field, while the target roots do. The downstream edge head consumes
the roots. This rules out a gross checkpoint/cache/sampler mismatch at the
prediction endpoints, although it does not by itself validate the current k1
intermediate-layer fusion.

#### Freeze-All MSE Multi-Seed Control

This control kept the pretrained k1 shared backbone frozen and in eval mode for
all 20 epochs. Only the statistics gate, downstream tail, and prediction head
were trained (`28,738` trainable of `38,018` total parameters).

```text
sgrl_mode: partial_shared
shared_gnn_layers: 1
partial_shared_stats_fusion: gate
partial_shared_freeze_epochs: 20
partial_shared_backbone_eval_policy: always
loss: MSE
seeds: 0, 1, 2
log root: logs/freeze_all_mse_multiseed_gpu_20260710
```

| Seed | Best epoch | Val MSE/loss | digtime MSE | timing_ctrl MSE | array MSE | Run directory |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 18 | 0.0102 / 0.01020449 | 0.0152 | 0.0131 | 0.0153 | `logs/freeze_all_mse_multiseed_gpu_20260710/partial_shared_k1_gate_freeze_all_mse_seed0` |
| 1 | 18 | 0.0102 / 0.01018344 | 0.0152 | 0.0131 | 0.0137 | `logs/freeze_all_mse_multiseed_gpu_20260710/partial_shared_k1_gate_freeze_all_mse_seed1` |
| 2 | 19 | 0.0104 / 0.01035412 | 0.0149 | 0.0126 | 0.0141 | `logs/freeze_all_mse_multiseed_gpu_20260710/partial_shared_k1_gate_freeze_all_mse_seed2` |

Population mean and standard deviation over the three seeds:

| Mode | Val MSE mean +/- std | digtime mean +/- std | timing_ctrl mean +/- std | array mean +/- std |
| --- | ---: | ---: | ---: | ---: |
| static + MSE | 0.009833 +/- 0.000094 | 0.014300 +/- 0.000374 | 0.011867 +/- 0.000205 | 0.011267 +/- 0.000125 |
| k1 + gate + freeze3 + MSE | 0.010067 +/- 0.000170 | 0.015033 +/- 0.001040 | 0.012433 +/- 0.000205 | 0.012567 +/- 0.000525 |
| k1 + gate + freeze_all + MSE | 0.010267 +/- 0.000094 | 0.015100 +/- 0.000141 | 0.012933 +/- 0.000236 | 0.014367 +/- 0.000680 |

Conclusions:

- Freeze-all greatly reduces digtime variance relative to freeze3, so updating
  the pretrained layer is one source of seed sensitivity.
- Permanent freezing does not improve mean accuracy. It is worse than freeze3
  on validation, timing_ctrl, and especially array. The shared layer needs some
  supervised adaptation after a short warmup.
- The root consistency audit makes a basic cached-vs-online sampler mismatch
  unlikely as the primary cause. The remaining high-value variables are the
  optimizer reset (now fixed), the dropout/eval transition, and the destructive
  full-gate fusion of intermediate GCL features with circuit statistics.
- This motivated a corrected freeze3 MSE rerun with `frozen_only` versus
  `always` eval policy. The completed comparison is recorded below.

#### Corrected Freeze3 Optimizer and Eval-Policy Sweep

This sweep used `partial_shared_k1 + gate + freeze3 + MSE` for seeds 0, 1,
and 2. At epoch 3, all six logs contain the new
`Added partial-shared backbone to the existing optimizer` message and none
contains the old optimizer-rebuild path. Thus the gate/tail/head Adam state is
preserved in every run.

```text
log root: logs/freeze3_optimizer_eval_multiseed_gpu_20260710
partial_shared_backbone_eval_policy: frozen_only / always
seeds: 0, 1, 2
epochs: 20
loss: MSE
```

| Eval policy | Seed | Best epoch | Val MSE/loss | digtime MSE | timing_ctrl MSE | array MSE | Run directory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| frozen_only | 0 | 18 | 0.0101 / 0.01009078 | 0.0157 | 0.0128 | 0.0125 | `logs/freeze3_optimizer_eval_multiseed_gpu_20260710/frozen_only_seed0` |
| frozen_only | 1 | 19 | 0.0102 / 0.01024406 | 0.0161 | 0.0130 | 0.0119 | `logs/freeze3_optimizer_eval_multiseed_gpu_20260710/frozen_only_seed1` |
| frozen_only | 2 | 18 | 0.0104 / 0.01041682 | 0.0163 | 0.0126 | 0.0120 | `logs/freeze3_optimizer_eval_multiseed_gpu_20260710/frozen_only_seed2` |
| always | 0 | 18 | 0.0100 / 0.01004581 | 0.0140 | 0.0128 | 0.0139 | `logs/freeze3_optimizer_eval_multiseed_gpu_20260710/always_seed0` |
| always | 1 | 15 | 0.0101 / 0.01012399 | 0.0189 | 0.0126 | 0.0129 | `logs/freeze3_optimizer_eval_multiseed_gpu_20260710/always_seed1` |
| always | 2 | 18 | 0.0103 / 0.01031581 | 0.0148 | 0.0124 | 0.0132 | `logs/freeze3_optimizer_eval_multiseed_gpu_20260710/always_seed2` |

Population mean and standard deviation over the three seeds:

| Mode | Val MSE mean +/- std | digtime mean +/- std | timing_ctrl mean +/- std | array mean +/- std |
| --- | ---: | ---: | ---: | ---: |
| static + MSE | 0.009833 +/- 0.000094 | 0.014300 +/- 0.000374 | 0.011867 +/- 0.000205 | 0.011267 +/- 0.000125 |
| old freeze3 + MSE (optimizer reset) | 0.010067 +/- 0.000170 | 0.015033 +/- 0.001040 | 0.012433 +/- 0.000205 | 0.012567 +/- 0.000525 |
| corrected freeze3 + MSE, frozen_only | 0.010233 +/- 0.000125 | 0.016033 +/- 0.000249 | 0.012800 +/- 0.000163 | 0.012133 +/- 0.000262 |
| corrected freeze3 + MSE, always | 0.010133 +/- 0.000125 | 0.015900 +/- 0.002146 | 0.012600 +/- 0.000163 | 0.013333 +/- 0.000419 |
| freeze_all + MSE | 0.010267 +/- 0.000094 | 0.015100 +/- 0.000141 | 0.012933 +/- 0.000236 | 0.014367 +/- 0.000680 |

Conclusions:

- Preserving Adam state fixes an implementation confound, but it does not
  recover the static baseline. Neither corrected policy improves the overall
  mean relative to the old freeze3 result.
- `frozen_only` is the more stable corrected policy. It has much lower digtime
  variance than `always` and better array mean, although `always` is slightly
  better on validation and timing_ctrl means.
- Keeping the shared backbone in eval mode after unfreezing is not a general
  solution: `always` produces a large digtime seed-1 failure and the worst
  digtime variance in this comparison.
- The original `static + MSE` remains best on validation and all three transfer
  means. The remaining limitation is therefore in partial-sharing/fusion and
  supervised adaptation, not just optimizer reset or train/eval coupling.

### S5.6 Isolated Artifacts and Representation-Drift Audit

#### Run Artifact Isolation

Parallel runs no longer write to the shared
`downstream_model/model_<epoch>-<loss>.pth` path. Each timestamped log now owns
a sibling artifact directory containing:

```text
run_config.json   exact args, command, Git commit, host, PID, timestamps/status
best_model.pt     model state, optimizer state, best epoch, args, and metrics
metrics.json      machine-readable best validation and named test results
representation_audit.jsonl  optional fixed-batch audit records
```

Writes use temporary files plus `os.replace`, and successful or failed
downstream exits finalize `run_config.json`. This removes checkpoint collisions
between parallel seeds and makes the trained model tied to each reported row
recoverable.

#### RNG-Neutral Freeze3 Audit

The audit captures one fixed source-validation batch and one fixed batch from
each transfer circuit. Python, NumPy, CPU Torch, and all CUDA RNG states are
restored immediately after batch capture, so enabling the audit does not alter
the subsequent sampling trajectory. A frozen copy of the initialized shared
backbone is the reference.

```text
mode: partial_shared_k1 + gate + freeze3 + frozen_only + MSE
seed: 0
GPU: 3
Git commit: 2db823ea3259360574ffa9ad2f943ba03de9b6da
log root: logs/freeze3_representation_audit_seed0_rng_neutral_gpu3_20260710
artifact: 20260710_234609_870963_edge_regression_..._artifacts
```

The run reproduces the corrected seed-0 row: best epoch 18, validation
MSE/loss `0.0101 / 0.01008070`, and test MSE
`0.0157 / 0.0129 / 0.0124` for digtime/timing_ctrl/array.

Source-validation representation trajectory:

| Epoch | State | Root cosine | Root relative L2 | Weight relative L2 | Gate mean | Shared norm |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| -1 | initialized | 1.0000 | 0.0000 | 0.0000 | 0.4946 | 5.8288 |
| 2 | last frozen epoch | 1.0000 | 0.0000 | 0.0000 | 0.4849 | 5.8288 |
| 3 | first unfrozen epoch | 0.9562 | 0.2893 | 0.0148 | 0.4557 | 5.3747 |
| 4 | second unfrozen epoch | 0.9038 | 0.4116 | 0.0225 | 0.4458 | 4.9926 |
| 10 | adapted | 0.7716 | 0.6266 | 0.0392 | 0.4351 | 3.9163 |
| 18 | best checkpoint epoch | 0.5923 | 0.8087 | 0.0637 | 0.4374 | 3.0268 |
| 19 | final epoch | 0.5540 | 0.8332 | 0.0659 | 0.4378 | 2.9328 |

At best epoch 18, root cosine is `0.5923 / 0.5793 / 0.5920 / 0.5955`
for source_val/digtime/timing_ctrl/array. Gate values do not collapse: mean is
about `0.43-0.44`, no values exceed `0.9`, and at most `0.24%` fall below
`0.1` at epoch 18. The shared-feature norm falls from about `5.8` to `3.0`
while statistics norms remain about `4.0-4.5`.

Conclusion: the dominant observed failure is rapid shared-representation drift
after unfreezing, not gate saturation. A small parameter displacement
(`6.37%` relative L2 at the best epoch) changes root representation direction
substantially. This supports adding an explicit GCL/teacher constraint to the
single deployment backbone instead of continuing freeze or eval-policy sweeps.

### S6 Joint Shared Backbone: Implementation and Smoke Test

Implemented `--sgrl_mode joint_shared` with this training/deployment split:

```text
node type embedding + alpha * statistics embedding (alpha initialized to 0)
  -> complete two-layer shared GNN initialized from the online encoder
  -> supervised edge head

training only:
  shared representation -> predictor -> EMA target alignment loss

L_total = L_supervised + joint_gcl_lambda * L_gcl
```

The zero-initialized signed statistics scale preserves the pretrained GCL path
exactly at initialization. The EMA target receives no gradients and is updated
after optimizer steps. Target and predictor are omitted from deployment
parameter accounting.

One-epoch GPU smoke test:

```text
mode: joint_shared
joint_shared_gnn_layers: 2
joint_gcl_lambda: 0.01
seed: 0
Git commit: 67851dada10bfe73b534732290bda3a38f2787d3
log root: logs/joint_shared_lambda001_smoke_seed0_gpu3_20260711
```

| Check | Result |
| --- | --- |
| Complete online layer load | 13 tensors / 17,665 values copied |
| Training parameters | 58,628 total / 37,699 trainable |
| Deployment parameters | 29,378 |
| Epoch-0 GCL loss | 0.41444825 |
| Learned online stats scale | -0.006521 |
| EMA target stats scale | -0.003833 |
| Artifact status | completed; checkpoint/config/metrics present |

The smoke result (Val MSE `0.0146`) is not an accuracy comparison after only
one epoch. It verifies full-data loading, supervised plus GCL backward,
predictor optimization, EMA update, compact deployment accounting, and
isolated checkpoint recovery. The next controlled experiments are the same
architecture with `joint_gcl_lambda = 0, 0.01, 0.05, 0.1` on seed 0; only the
best candidate should advance to seeds 1/2. Rebalancing remains paused.

#### S6 Seed-0 GCL-Weight Sweep

All four 20-epoch runs used the same `joint_shared` architecture, pretrained
checkpoint, MSE loss, data split, and seed. Only `joint_gcl_lambda` changed.

```text
Git commit: ba5cdc9c91048575d6e41e890c609db267edda4e
log root: logs/s6_lambda_seed0_gpu_parallel_20260711
valid run directories: lambda_0, lambda_001_gpu3_retry, lambda_005, lambda_01
GPU: 3/4
```

| Lambda | Best epoch | Val MSE/loss | digtime MSE | timing_ctrl MSE | array MSE | Online stats scale | Final GCL loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 19 | 0.0119 / 0.01186482 | 0.0138 | 0.0182 | 0.0117 | -0.2653 | N/A |
| 0.01 | 19 | 0.0116 / 0.01158730 | 0.0135 | 0.0176 | 0.0115 | -0.2711 | 0.02005927 |
| 0.05 | 19 | 0.0115 / 0.01154649 | 0.0136 | 0.0167 | 0.0119 | -0.2770 | 0.01899753 |
| 0.1 | 19 | 0.0116 / 0.01156188 | 0.0137 | 0.0170 | 0.0121 | -0.2746 | 0.01847927 |

Reference rows:

| Mode | Val MSE | digtime MSE | timing_ctrl MSE | array MSE |
| --- | ---: | ---: | ---: | ---: |
| static + MSE, three-seed mean | 0.009833 | 0.014300 | 0.011867 | 0.011267 |
| corrected freeze3 + MSE, frozen_only mean | 0.010233 | 0.016033 | 0.012800 | 0.012133 |
| joint_shared, lambda=0.05 seed0 | 0.0115 | 0.0136 | 0.0167 | 0.0119 |

Conclusions:

- Joint GCL regularization helps relative to the identical `lambda=0`
  architecture. `lambda=0.05` improves validation, digtime, and timing_ctrl;
  `lambda=0.01` gives the best digtime and array values. The alignment branch
  is therefore useful rather than inert.
- `lambda=0.05` is the best overall seed-0 tradeoff, but it does not satisfy the
  gate for a multi-seed promotion. Its validation is worse than both static and
  corrected partial sharing, and timing_ctrl remains much worse (`0.0167`
  versus static `0.011867`).
- The fully shared two-layer backbone improves digtime and keeps array close,
  but removing the downstream message-passing tail severely weakens
  timing_ctrl adaptation. Increasing lambda cannot repair that structural
  limitation; `0.1` is already slightly worse than `0.05`.
- All best checkpoints occur at epoch 19, so the current 20-epoch budget may
  not be fully converged. Longer training alone is not the next priority,
  because the transfer imbalance is large and systematic.
- Do not run seeds 1/2 or add GAI/BMC yet. The next design should preserve one
  deployment backbone while adding lightweight task-specific adaptation
  inside it, rather than restoring a second full downstream GNN.

Run-selection note: `logs/s6_lambda_seed0_parallel_20260711` contains an
accidental CPU-only launch and is marked failed. The first GPU4 `lambda_001`
run under the valid root was intentionally interrupted at epoch 9 and marked
failed when it was migrated to GPU3. Only the four completed directories named
above are used in the table.

#### S6 40-Epoch LoRA Multi-Seed Audit

The next design keeps the same single two-layer deployment backbone and adds a
rank-8 LoRA update only to its final GNN layer. The adapter contains 2,048
train-time values and is merged into the backbone after training, so deployment
still has 29,378 parameters and no second GNN. All runs below use MSE, 40
epochs, and seeds 0/1/2.

```text
base seed-0 root: logs/s6_convergence40_seed0_gpu_parallel_20260711
LoRA seed-0 root: logs/s6_lora_r8_convergence40_seed0_gpu_parallel_20260711
seed-1/2 root: logs/s6_overnight_multiseed_20260711
base Git commit: 8ea5a18 (seed 0), ec1994d (seeds 1/2)
LoRA Git commit: ec1994d
GPU: 3/4
```

| Mode | Seed | Best epoch | Val loss | digtime MSE | timing_ctrl MSE | array MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| joint_shared, lambda=0 | 0 | 37 | 0.00994910 | 0.01376977 | 0.01252905 | 0.01176881 |
| joint_shared, lambda=0 | 1 | 36 | 0.01021504 | 0.01374099 | 0.01225465 | 0.01112517 |
| joint_shared, lambda=0 | 2 | 39 | 0.01048058 | 0.01490439 | 0.01245758 | 0.01166681 |
| LoRA r8, lambda=0 | 0 | 39 | 0.00999989 | 0.01369054 | 0.01269085 | 0.01127510 |
| LoRA r8, lambda=0 | 1 | 39 | 0.01024845 | 0.01367797 | 0.01262781 | 0.01138243 |
| LoRA r8, lambda=0 | 2 | 39 | 0.01040588 | 0.01440152 | 0.01258184 | 0.01118199 |
| LoRA r8, lambda=0.05 | 0 | 38 | 0.00990960 | 0.01407262 | 0.01265228 | 0.01163638 |
| LoRA r8, lambda=0.05 | 1 | 36 | 0.01004999 | 0.01420917 | 0.01168040 | 0.01127877 |
| LoRA r8, lambda=0.05 | 2 | 39 | 0.01044632 | 0.01434549 | 0.01262315 | 0.01084139 |

Population mean and standard deviation over three seeds:

| Mode | Val loss mean +/- std | digtime mean +/- std | timing_ctrl mean +/- std | array mean +/- std |
| --- | ---: | ---: | ---: | ---: |
| joint_shared, lambda=0 | 0.010215 +/- 0.000217 | 0.014138 +/- 0.000542 | 0.012414 +/- 0.000116 | 0.011520 +/- 0.000282 |
| LoRA r8, lambda=0 | 0.010218 +/- 0.000167 | 0.013923 +/- 0.000338 | 0.012634 +/- 0.000045 | 0.011280 +/- 0.000082 |
| LoRA r8, lambda=0.05 | 0.010135 +/- 0.000227 | 0.014209 +/- 0.000111 | 0.012319 +/- 0.000451 | 0.011252 +/- 0.000325 |
| static + MSE reference | 0.009833 +/- 0.000094 | 0.014300 +/- 0.000374 | 0.011867 +/- 0.000205 | 0.011267 +/- 0.000125 |

Conclusions:

- The mergeable LoRA adapter solves the structural adaptation problem without
  increasing deployment size. With `lambda=0`, it matches base validation,
  improves digtime and array means, but slightly worsens timing_ctrl.
- `LoRA r8 + lambda=0.05` has the best S6 validation mean and the best average
  of the three transfer MSEs. Relative to base `lambda=0`, it improves
  validation, timing_ctrl, and array, while digtime is slightly worse.
- The gain is small and dataset-dependent. Compared with static + MSE, the best
  S6 row is still worse on validation and timing_ctrl, approximately tied on
  array, and slightly better on digtime. Three seeds are not enough to claim a
  statistically stable accuracy improvement over static.
- Most best checkpoints occur at epochs 38/39, so 40 epochs is a materially
  fairer comparison than the earlier 20-epoch sweep, but convergence remains
  close to the budget boundary.

Run-selection note: the overnight queue orchestration accidentally launched
`LoRA r8 + lambda=0.05` seeds 1 and 2 three times each. The table uses the
earliest completed artifact for each seed. Their
validation-loss ranges are only `0.01003465-0.01004999` and
`0.01044632-0.01046414`, respectively, so the conclusion does not depend on
which duplicate is selected. The non-LoRA `lambda=0.05` seed-0 process under
`logs/s6_convergence40_seed0_gpu_parallel_20260711/lambda_005` stopped during
epoch 37 and retains `running` artifact metadata; it is incomplete and excluded.

### P0 Corrected Selection Protocol and P1/P2 Queue

The S6 audit exposed a protocol issue: regression metrics were displayed after
rounding MSE to four decimals, and the same rounded value was also used to
select the best checkpoint. Four of the nine canonical 40-epoch S6 runs had a
later epoch with a lower raw validation MSE that was not saved because both
epochs occupied the same four-decimal bin. The largest observed raw-MSE gap was
`5.922e-5`, which is comparable to the small differences among S6 candidates.

P0 changes the protocol as follows:

- `Logger` retains `mse_raw` while preserving rounded metrics for display.
- Checkpoints and early stopping use raw validation MSE plus an explicit
  `early_stopping_min_delta`.
- Transfer test sets are not evaluated whenever validation improves. Training
  first finishes, reloads `best_model.pt`, and evaluates each transfer set once.
- Final transfer metrics are written to both `metrics.json` and the checkpoint.

P2 adds `--joint_backbone_lr`. When set, pretrained
`shared_backbone.gnn.*` parameters use this learning rate, while LoRA,
statistics adapters, predictor, and supervised head retain `--lr`. EMA-target
parameters remain frozen.

The staged seed-0 queue is versioned in `scripts/run_s6_p1_p2.sh`. Every run
uses at most 80 epochs, raw-MSE early stopping with patience 12 and minimum
delta `1e-6`, and the corrected final-only transfer protocol.

| Stage | Candidate | Purpose |
| --- | --- | --- |
| P1 | static + MSE | Matched 80-epoch reference |
| P1 | joint_shared, lambda=0 | No-adapter shared reference |
| P1 | LoRA r8, lambda=0 | Isolate task adapter |
| P1 | LoRA r8, lambda=0.05 | Isolate joint GCL contribution |
| P2 | LoRA r8, lambda=0.05, backbone LR=1e-5/1e-6 | Limit pretrained-backbone drift |
| P2 | LoRA r8, lambda=0, backbone LR=1e-5 | Paired low-LR GCL ablation |

GPU3 and GPU4 have independent serial lanes. A lane waits until its GPU has at
least 8 GB free and utilization is at most 80%, so no external process is
stopped and two CircuitGCL runs are never placed on the same GPU concurrently.
P3 rank/layer screening starts only after this queue identifies the best P2
optimization setting; P4 multi-seed and P5 rebalancing remain gated on P3.

### Pre-Registered P2-P5 Selection and Reporting Rules

These rules were fixed before observing any corrected P1/P2 result:

1. P2 selects `joint_backbone_lr` using seed-0 source-validation raw MSE only.
   Transfer-circuit metrics and representations are not selection criteria.
2. If two P2 candidates differ by at most `2e-5` raw validation MSE, prefer the
   candidate with lower source-validation base-representation drift at the
   restored best checkpoint.
3. P3 applies the selected P2 optimization setting to LoRA rank `4/8/16` and
   GNN layer `0/1`. It again selects on source-validation raw MSE only. Within
   `2e-5`, prefer the lower rank because it uses fewer train-time adapter values;
   merged deployment size is unchanged.
4. P4 compares `static`, `init_reuse`, best-LoRA `lambda=0`, and best-LoRA
   positive lambda on paired seeds `0-4`. Report population mean/std and paired
   candidate-minus-static differences. No method is dropped because of one
   transfer circuit after the matrix begins.
5. P5 keeps the P4 architecture fixed and compares MSE/GAI/BMC. Primary
   selection remains source-validation raw MSE. Overall and ten fixed label-bin
   MSE/MAE/bias are reported for validation and every transfer circuit so tail
   improvements cannot be hidden by aggregate MSE.

Protocol tooling:

- `scripts/summarize_experiments.py` reads isolated JSON artifacts, selects one
  canonical completed artifact per scientific configuration and seed, detects
  duplicate/missing/inconsistent runs, and computes grouped and paired stats.
- `--joint_shared_audit 1` records source-validation base/task drift every five
  epochs. Transfer representations are recorded only after restoring the best
  checkpoint.
- `scripts/audit_label_distribution.py` records fixed-bin circuit label
  distributions, source-to-target Jensen-Shannon shift, source-tail coverage,
  and optional binned prediction errors without requiring a GPU.

#### CPU Label-Distribution and Artifact Audit

The pre-P5 CPU audit used the same normalized edge labels as downstream
training. Outputs are under `logs/label_distribution_audit_20260711`.

| Circuit | Labels | Mean | Std | Q10 | Median | Q90 | 10-bin JS from SSRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ssram | 175,413 | 0.5716 | 0.1472 | 0.3677 | 0.6086 | 0.7234 | 0.0000 |
| digtime | 2,814 | 0.5644 | 0.1448 | 0.3301 | 0.5892 | 0.7236 | 0.0186 |
| timing_ctrl | 3,333 | 0.6374 | 0.1777 | 0.3612 | 0.6388 | 0.8575 | 0.0787 |
| array_128_32_8t | 73,569 | 0.5691 | 0.1412 | 0.3897 | 0.6063 | 0.7250 | 0.0093 |

Relative to SSRAM's Q10/Q90, `12.19%/10.80%` of digtime labels fall below/above
the thresholds, `10.11%/31.59%` of timing_ctrl labels do so, and
`9.30%/14.40%` of array labels do so. Timing control therefore has the largest
and most asymmetric label shift, concentrated in the high-label region. This
supports reporting fixed-bin errors in P5 and helps explain why timing_ctrl has
been the most sensitive transfer circuit. It does not by itself prove that
reweighting will improve timing_ctrl, because covariate/graph shift remains.

The generic artifact audit found 14 selected S5/S6 artifacts: 13 completed and
one stale `running` record from the interrupted non-LoRA lambda-0.05 run. Six
artifacts were marked anomalous because LoRA lambda-0.05 seeds 1/2 were each
launched three times. The summarizer keeps one canonical completed artifact per
scientific configuration and seed, so duplicates do not enter group means.

#### Rebalancing Protocol Corrections Before P5

Three additional issues were corrected before running P5:

- GAI's GMM now reads source circuit labels only. The old function fitted only
  on source labels but still inspected transfer labels to print KL diagnostics.
  Distribution comparison is now an explicit offline audit instead of a
  training-time test-label access.
- Each GAI run writes its GMM inside its isolated artifact directory using an
  atomic replacement, eliminating the shared `pkl/gmm/gmm.pkl` race between
  parallel seeds.
- GAI/BMC/BNI `noise_sigma` parameters now enter the task-learning-rate optimizer
  group and their criterion state is saved/restored with the best checkpoint.
  Earlier local GAI/BMC results used a fixed initial sigma despite the parameter
  being declared trainable; they remain measurements of that fixed-sigma
  implementation but are not the final paper-faithful P5 comparison.

### Legacy P1/P2/P4 Completion and Protocol Boundary

The corrected 80-epoch P1/P2 queue and four additional static downstream seeds
completed on 2026-07-11. Structured outputs are under
`logs/s6_p1_p2_protocol_20260711`; the summarizer reports 11 completed runs, no
running runs, and no artifact-schema anomalies.

These runs are frozen as `legacy-transductive-v1`, not as strict zero-shot
evidence. SGRL fitted a concatenation of source and transfer graph topology,
node-feature normalization fitted all loaded circuits, evaluation used the
relation-balanced processed edge subsets, and static seeds 0-4 shared one
pretraining/embedding realization. The five static runs therefore measure
downstream-only variation.

The completed seed-0 results are recorded in
`logs/s6_p1_p2_protocol_20260711/experiment_runs.tsv`. Static has the best
source-validation MSE (`0.00784496`). The best positive-lambda LoRA row uses
the default unified learning rate `1e-4` and reaches validation MSE
`0.00911473`; it improves digtime relative to seed-0 static but does not beat
static simultaneously on validation and all transfer circuits. This supports
continued single-backbone research, not a stable superiority claim.

Formal follow-up is gated on a unified protocol harness with independently
controlled SGRL graph scope and normalization scope, source-train-only label
priors, persisted splits and normalization state, fixed evaluation views, and
matched `sgrl_dual_rsm_ema` versus `circuitgcl_text_ema_only` target-update
experiments. Legacy and strict results must not be pooled in one group mean.

### Strict Seed-0 Seven-Method Selection

The provenance-repaired 160-epoch seed-0 group completed on 2026-07-12 under
`logs/strict_seed0_seven_20260712_v2`. The summarizer reports seven completed
artifacts, no running artifacts, and no anomalies. All runs use source-only
SGRL fitting and normalization, relation seed `20260711`, fixed embedding and
evaluation views, and commit `fe542d9`.

| Method | Val MSE | digtime | timing_ctrl | array | Transfer mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| static dual | 0.007810 | 0.016487 | 0.010408 | 0.010042 | 0.012312 |
| static EMA-only | 0.007813 | 0.017125 | 0.010185 | 0.009633 | 0.012314 |
| no-GCL | 0.007781 | 0.015456 | 0.011312 | 0.009716 | 0.012161 |
| init_reuse | **0.007770** | 0.016106 | 0.010707 | **0.009221** | **0.012011** |
| joint lambda=0 | 0.008091 | **0.014517** | 0.011066 | 0.012115 | 0.012566 |
| LoRA r8 lambda=0 | 0.008130 | 0.014855 | 0.011083 | 0.013578 | 0.013172 |
| LoRA r8 lambda=0.05 | 0.008143 | 0.014833 | 0.011231 | 0.012388 | 0.012817 |

Relative to static dual, `init_reuse` improves validation by `0.51%` and the
three-circuit transfer mean by `2.44%`. Static dual and EMA-only are effectively
tied; dual remains the mainline because it matches the author implementation
and initializes all reuse candidates. LoRA lambda=0 is excluded from the next
stage because array degrades by `35.21%`, exceeding the preregistered `25%`
single-circuit veto. Tuning seeds 1-2 retain no-GCL, static dual, init_reuse,
joint lambda=0, and LoRA r8 lambda=0.05. Distillation is deferred because the
retained shared candidates remain within the `5%` source-validation gate.

### 后续改进方法尝试

目标优先级固定为：先争取使用完全共享、零额外部署参数的两层 GNN
达到 static dual 的效果；只有纯共享方案不能通过门槛时，才采用可合并
LoRA。以下实验不得接触最终 `sp8192w` 盲测集，并继续使用 strict
source-only provenance、固定 evaluation view 和相同的 160-epoch 协议。

| 顺序 | 方法 | 要回答的问题 | 部署额外参数 | 启动条件 |
| --- | --- | --- | ---: | --- |
| R0 | 完成当前 seeds 1-2 队列并汇总 | 当前候选是否稳定 | 0 | 当前运行项 |
| R1 | `joint rank0 lambda=0` 对 `lambda=0.05` | 持续 GCL 是否改善完全共享模型 | 0 | R0 完成；先 seed0，过门槛再跑 seeds 1-2 |
| R2 | supervised/GCL 梯度余弦审计 | 性能损失是否来自共享层梯度冲突 | 0 | 与 R1 seed0 同时记录，不改变优化行为 |
| R3 | 从强到弱的 lambda 调度 | 防止早期 GCL 表示漂移并允许后期任务适配 | 0 | R1 固定 lambda 有帮助但仍落后 static |
| R4 | GCL/downstream 交替优化 | 避免同一步中两个目标直接抵消 | 0 | R2 冲突明显或 R3 无法通过门槛 |
| R5 | PCGrad | 投影共享参数上的冲突梯度 | 0 | R2 显示负梯度余弦占比较高 |
| R6 | 预训练权重锚定（L2-SP） | 限制共享表示过快偏离 GCL 初始化 | 0 | 仅作低成本补充，不再扩展 freeze/低学习率 sweep |
| R7 | static-dual teacher distillation | 将两套 GNN 的预测知识压缩进一套共享 GNN | 0 | R1-R5 最佳纯共享方案仍未通过门槛，只做一次 |
| R8 | EMA/SWA 参数平均 | 减少联合训练末期波动 | 0 | 仅在最佳 epoch 波动明显时补充 |
| R9 | mergeable LoRA | 用低秩任务增量缓解容量/任务冲突 | 合并后 0 | 纯共享方案均失败时的最终折中 |

R1 是下一项必须补齐的严格同构实验：两行使用完全相同的 rank-0
共享架构，只改变 `joint_gcl_lambda`。当前运行中的 `joint rank0 lambda=0`
与 `LoRA rank8 lambda=0.05` 不能单独分离 GCL 和 LoRA 的贡献。

R2 至少记录共享 GNN 梯度余弦的均值、中位数、负值比例和分位数。
只有审计支持梯度冲突假设时才实现 PCGrad，避免无证据地增加训练复杂度。
R3 首选单一预注册调度，例如在 160 epoch 内将 lambda 从 `0.05`
衰减到 `0.005`，不进行无界超参数搜索。

R7 的教师只在训练期间存在。学生损失优先使用
`L_label + lambda * L_GCL + beta * L_teacher_output`；输出蒸馏不需要新增
projection head。最终导出必须删除 teacher、EMA target、GCL predictor 和
优化器状态，只保留一套两层共享 GNN 与 downstream head。

所有候选继续使用既定门槛：相对 static dual 的 source validation 退化
不超过 `5%`，三迁移集均值退化不超过 `10%`，任一单电路退化不超过
`25%`。每轮只改变一个主要因素；未通过门槛的方法不扩展到 rebalancing
矩阵。锁定架构后再依次执行 MSE/GAI/BMC、deployment export、合并前后
预测一致性验证，以及一次性的 `sp8192w` 盲测。

### Strict Seeds 0-2 Selection and Rank-0 Positive-Lambda Control

The seeds 1-2 selection queue completed on 2026-07-12 under
`logs/strict_selection_seeds12_20260712` with ten completed runs, no running
runs, and no anomalous artifacts. Combined with the matching seed-0 runs in
`logs/strict_seed0_seven_20260712_v2`, the three-seed population statistics
are:

| Method | Val MSE | digtime | timing_ctrl | array | Transfer mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| static dual | 0.007714 +/- 0.000068 | 0.015027 +/- 0.001035 | 0.010874 +/- 0.000422 | 0.010086 +/- 0.000213 | 0.011996 +/- 0.000262 |
| no-GCL | 0.007713 +/- 0.000060 | 0.015206 +/- 0.000179 | 0.011248 +/- 0.000144 | 0.009922 +/- 0.000184 | 0.012125 +/- 0.000095 |
| init reuse | 0.007714 +/- 0.000065 | 0.014964 +/- 0.000808 | 0.011276 +/- 0.000403 | 0.010009 +/- 0.000581 | 0.012083 +/- 0.000073 |
| joint rank0, lambda=0 | 0.007978 +/- 0.000081 | 0.014801 +/- 0.000743 | 0.010936 +/- 0.000101 | 0.012388 +/- 0.000193 | 0.012708 +/- 0.000246 |
| LoRA r8, lambda=0.05 | 0.008040 +/- 0.000074 | 0.014527 +/- 0.000224 | 0.011076 +/- 0.000176 | 0.013060 +/- 0.001017 | 0.012888 +/- 0.000330 |

Relative to static dual, `init_reuse` is effectively neutral on source
validation (`-0.01%`) and costs `0.73%` on transfer mean. It remains the
strongest compact engineering fallback. Exact `joint rank0, lambda=0` costs
`3.42%` on validation and `5.94%` on transfer mean; digtime improves `1.50%`,
timing_ctrl costs `0.57%`, and array costs `22.82%`. It passes the aggregate
selection gates but leaves little array margin. LoRA r8 lambda=0.05 costs
`4.22%` on validation and `7.44%` on transfer mean, while array costs `29.48%`.
It therefore fails the preregistered single-circuit `25%` veto and is not the
primary reuse candidate.

The missing strict same-architecture GCL control completed under
`logs/strict_joint_rank0_lambda005_seed0_20260712`:

| Rank-0 seed0 | Val MSE | digtime | timing_ctrl | array | Transfer mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| lambda=0 | 0.008091 | 0.014517 | 0.011066 | 0.012115 | 0.012566 |
| lambda=0.05 | 0.008036 | 0.015224 | 0.010998 | 0.011696 | 0.012640 |

Adding GCL loss improves source validation by `0.68%`, timing_ctrl by `0.61%`,
and array by `3.46%` relative to the identical lambda-zero architecture, but
digtime worsens by `4.87%` and transfer mean worsens by `0.59%`. Relative to
seed-0 static dual, the positive-lambda row costs `2.90%` on validation and
`2.66%` on transfer mean; its worst single-circuit degradation is array at
`16.47%`. It therefore passes the promotion gates, but one seed is not evidence
of a stable GCL benefit. The next controlled step is rank-0 lambda=0.05 on
seeds 1-2 before lambda scheduling, PCGrad, distillation, or rebalancing.

The compared commits differ only in experiment scripts and this log; model,
loss, dataset, and training implementation files are unchanged across
`fe542d9`, `ddc5eaa`, and `9680fda`.

### Strict Rank-0 Positive-Lambda Three-Seed Completion and Gradient Audit

The missing rank-0 `lambda=0.05` seeds 1-2 completed normally on 2026-07-12
under `logs/strict_joint_rank0_lambda005_seeds12_audit_20260712`. Both cells
ran all 160 epochs, restored their source-validation-selected best checkpoints,
evaluated the three development transfer circuits once, and finished with
return code zero. Together with the preregistered seed-0 control, the raw rows
are:

| Seed | Best epoch | Val MSE | digtime | timing_ctrl | array | Transfer mean | Commit |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 156 | 0.008035913 | 0.015224475 | 0.010998346 | 0.011695725 | 0.012639515 | `9680fda` |
| 1 | 157 | 0.007928901 | 0.014354646 | 0.010798935 | 0.011398138 | 0.012183907 | `2d0ea91` |
| 2 | 155 | 0.007928575 | 0.013772232 | 0.010914260 | 0.011053859 | 0.011913450 | `2d0ea91` |

The final same-architecture three-seed comparison uses population standard
deviation and the exact raw MSE stored in each `metrics.json`:

| Method | Val MSE | digtime | timing_ctrl | array | Transfer mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| static dual | 0.007714418 +/- 0.000067849 | 0.015026925 +/- 0.001034566 | 0.010873532 +/- 0.000422270 | 0.010086297 +/- 0.000212843 | 0.011995585 +/- 0.000261850 |
| joint rank0, lambda=0 | 0.007978297 +/- 0.000080919 | 0.014801401 +/- 0.000742506 | 0.010935558 +/- 0.000100515 | 0.012387784 +/- 0.000193157 | 0.012708248 +/- 0.000245843 |
| joint rank0, lambda=0.05 | 0.007964463 +/- 0.000050523 | 0.014450451 +/- 0.000596734 | 0.010903847 +/- 0.000081742 | 0.011382574 +/- 0.000262272 | 0.012245624 +/- 0.000299610 |

Relative to static dual, rank-0 `lambda=0.05` changes validation by `+3.241%`,
digtime by `-3.836%`, timing_ctrl by `+0.279%`, array by `+12.852%`, and the
three-circuit transfer mean by `+2.084%`. It therefore passes the preregistered
`5%` source-validation, `10%` transfer-mean, and `25%` any-circuit gates. Each
individual seed also passes all three gates. Relative to the identical
rank-0 `lambda=0` architecture, positive lambda improves the mean validation,
digtime, timing_ctrl, array, and transfer mean by `0.173%`, `2.371%`, `0.290%`,
`8.115%`, and `3.640%`, respectively. The improvement is not pointwise
universal: seed 0 transfer mean is `0.586%` worse and seed 2 validation is
`0.304%` worse than their lambda-zero pairs. The population result nevertheless
supports a stable positive-GCL contribution rather than the seed-0-only mixed
signal.

The diagnostic gradient audit did not alter the normal backward pass. It
sampled batch zero every ten epochs from epoch 0 through 150 for seeds 1-2,
giving 32 records. The pooled cosine statistics are:

| Shared parameter group | Mean | Median | Q10 | Q25 | Q75 | Q90 | Negative fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| global | 0.173236 | 0.118057 | -0.079026 | -0.005059 | 0.289289 | 0.627376 | 25.000% |
| embeddings | 0.220134 | 0.182139 | -0.327768 | 0.015305 | 0.461272 | 0.649180 | 21.875% |
| layer 0 | 0.182120 | 0.113826 | -0.081090 | 0.002471 | 0.309506 | 0.637739 | 25.000% |
| layer 1 | 0.041681 | 0.048010 | -0.061532 | -0.028455 | 0.100847 | 0.149597 | 34.375% |

The conflict is seed-dependent. Seed 1 has global mean `0.268212` and `12.5%`
negative global records; seed 2 has global mean `0.078259` and `37.5%`
negative records. Layer 1 is the weakest group, reaching `50%` negative records
for seed 2. The global mean and median remain positive, so this is not a
catastrophic optimization failure, but a `25%` pooled global and `34.375%`
layer-1 negative fraction is too large to describe as no conflict. Following
the preregistered R3/R5 ordering, the next controlled experiment is one seed-0
lambda schedule from `0.05` to `0.005`; PCGrad is not expanded in parallel.

#### Canonical Teacher-Report Artifact

To prevent manual transcription from multiple log directories,
`scripts/summarize_strict_reuse.py` is the canonical strict reuse report
builder. With no positional arguments it reads exactly these four whitelisted
roots:

- `logs/strict_seed0_seven_20260712_v2`
- `logs/strict_selection_seeds12_20260712`
- `logs/strict_joint_rank0_lambda005_seed0_20260712`
- `logs/strict_joint_rank0_lambda005_seeds12_audit_20260712`

It requires the exact 20-run method/seed matrix, validates every formal strict
argument and restored raw validation MSE, verifies checkpoint existence,
checks source-only graph scope, recomputes paired statistics and gates, checks
the rank-0 lambda-zero/positive-lambda provenance pairing, and recomputes the
gradient statistics from raw JSONL. The current audit has 20/20 completed
artifacts, zero anomalies, and 68 unique path/SHA256 provenance references with
zero missing files or hash mismatches. `SP8192W` is absent from every formal
configuration, runtime graph list, and reported metric; no formal run artifact
records access to it.

Regenerate the centralized machine-readable report with:

```bash
/home/lixc/.conda/envs/RCG/bin/python scripts/summarize_strict_reuse.py
```

The only two generated report files are:

- `logs/strict_reuse_report_20260712/strict_reuse_summary.json`
- `logs/strict_reuse_report_20260712/strict_reuse_summary.tsv`

The current SHA256 values are
`50ff9c82cd9f3d8ad0b949c74f72cf7f9ccb3981aa0f1b70c884f764b6d6540f`
and `51aa7dc229f4598255d5508a43c66b43971646230d5f9c5846105a08b0893470`,
respectively. These generated files are ignored by Git and can always be
reconstructed from the immutable raw artifacts; this experiment log remains
the single human-readable project record rather than introducing another
standalone Markdown report.

### Pre-Registered R3 Linear-Lambda Seed-0 Control

The completed rank-0 `lambda=0.05` gradient audit has positive pooled mean and
median cosine but `25.0%` negative global records and `34.375%` negative
layer-1 records. R3 therefore tests one lower-complexity conflict mitigation
before PCGrad: linearly decay the GCL weight while leaving the shared
architecture, optimizer, data, checkpoint selection, and evaluation protocol
unchanged. No PCGrad, alternating optimization, or distillation experiment is
launched in parallel.

For epoch `e` in the fixed 160-epoch run, the only new factor is:

```text
lambda(e) = 0.05 + (0.005 - 0.05) * e / 159
lambda(0) = 0.05
lambda(159) = 0.005
```

`args.joint_gcl_lambda` remains the initial value `0.05`; training uses a local
effective value and never recursively mutates the argument. Run config,
checkpoint metrics, epoch logs, and gradient-audit JSONL record the schedule
and effective value. The formal seed-0 root and method are:

```text
logs/strict_joint_rank0_lambda005_to0005_linear_seed0_audit_20260712
joint_rank0_lambda005_to0005_linear_seed0
```

The strict runner is
`scripts/run_strict_joint_lambda_schedule_seed0.py`. It is write-free unless
`--execute` is supplied, rejects physical GPU0/GPU1, maps one selected A100 to
logical GPU0, and refuses an existing log root. The initial conservative plan
required an empty device, but the operator subsequently authorized co-location
with existing GPU work when memory is available. The executable gate therefore
records utilization and process count but gates only on at least 6.5 GB free
memory; it never stops or modifies another process. It also locks the queued
commit and tracked-worktree state so code cannot change while waiting for a GPU,
then rechecks that state before invoking the post-run validator.

A successful queue completion additionally requires
`scripts/summarize_joint_lambda_schedule_seed0.py` to validate the exact
artifact count, strict runtime configuration, source-only provenance, paired
seed-0 identities, scheduled audit values, and reproduced raw metrics. A
validation mismatch propagates a nonzero summary and overall queue return code.

Selection is preregistered before observing the result:

1. The schedule must pass the existing `5%/10%/25%` gates relative to strict
   seed-0 static dual.
2. Candidate choice between scheduled and fixed `lambda=0.05` uses source
   validation raw MSE only. Transfer metrics are gates/reporting, not a tuning
   signal.
3. The schedule advances to seeds 1-2 only if seed-0 validation improves by
   more than `2e-5`. Within `2e-5`, retain fixed lambda because it is simpler
   and already has three-seed evidence.
4. Failure to advance does not automatically trigger PCGrad. Fixed
   `lambda=0.05` already passes every strict gate; any further conflict method
   requires a separate evidence review.

The paired controls remain
`logs/strict_joint_rank0_lambda005_seed0_20260712` and
`logs/strict_seed0_seven_20260712_v2`. All stage seeds and fixed evaluation
views remain unchanged, and the blind circuit is absent from the runner and
manifest.

### R3 Linear-Lambda Seed-0 Result

The formal run at commit `4375c9f` completed all 160 epochs on physical GPU4.
The method, dedicated validator, and overall queue all returned `rc=0`. The
best checkpoint was epoch 156, where the effective GCL weight was
`0.0058490566`; epoch 159 used the exact registered endpoint `0.005`.

| Method | Val raw MSE | digtime | timing_ctrl | array | Transfer mean |
|---|---:|---:|---:|---:|---:|
| static dual seed0 | 0.007809748 | 0.016486799 | 0.010408374 | 0.010042039 | 0.012312404 |
| fixed lambda=0.05 seed0 | 0.008035913 | 0.015224475 | 0.010998346 | 0.011695725 | 0.012639515 |
| linear 0.05 -> 0.005 seed0 | 0.008049868 | 0.015027398 | 0.010868106 | 0.011445360 | 0.012446955 |

Relative to static, the linear schedule is `+3.0746%` on source validation,
`+1.0928%` on transfer mean, `-8.8519%` on digtime, `+4.4169%` on
timing_ctrl, and `+13.9745%` on array. It passes every `5%/10%/25%` gate.
Relative to fixed `lambda=0.05`, it improves transfer mean by `1.5235%` and
all three individual transfer circuits by `1.1842%` to `2.1407%`, but worsens
the registered primary source-validation metric by `1.395494e-5` (`0.1737%`).
The registered expansion rule required an improvement greater than `2e-5`, so
`advances_to_seeds12=false`. Transfer improvements cannot be used after the
fact to reverse that decision. Fixed `lambda=0.05` remains the incumbent.

The schedule also did not remove the motivating conflict. Across 16 fixed
audit views, global cosine mean/median were `0.02065/0.00148` with `50.0%`
negative records. Embeddings and layer 0 were negative in `50.0%/43.75%` of
records. Layer 1 had mean/median `-0.02590/-0.02883` and `68.75%` negative
records; all final eight layer-1 records were negative. This supplies evidence
for exactly one sequential PCGrad seed-0 control, not for expanding the failed
schedule or launching multiple optimization variants.

The immutable result sources are:

```text
logs/strict_joint_rank0_lambda005_to0005_linear_seed0_audit_20260712/
  schedule_seed0_summary.json
  schedule_seed0_summary.tsv
```

### Pre-Registered R5 PCGrad Seed-0 Control

R5 keeps the incumbent rank-0 architecture, constant `lambda=0.05`, optimizer,
data, stage seeds, audits, checkpoint selection, and evaluations unchanged.
The only factor is deterministic two-task PCGrad on trainable parameters under
`shared_backbone.gnn`. Let

```text
u = grad(L_supervised)
v = grad(0.05 * L_GCL)
d = dot(u, v)
```

When `d < 0` and both global norms are nonzero, R5 uses both original task
gradients to compute

```text
u_projected = u - d / ||v||^2 * v
v_projected = v - d / ||u||^2 * u
g_shared = u_projected + v_projected
```

Otherwise `g_shared = u + v`. With exactly two tasks there is no randomized
task order and therefore no new RNG consumption. Head/criterion gradients stay
supervised-only, predictor gradients remain `0.05 * grad(L_GCL)`, parameters
outside the GNN scope retain the standard combined-loss gradient, and the EMA
target is updated once after the optimizer step as before. PCGrad is restricted
to rank 0 and fails fast if a selected parameter belongs to only one task.

The formal root and method are:

```text
logs/strict_joint_rank0_lambda005_pcgrad_seed0_audit_20260713
joint_rank0_lambda005_pcgrad_seed0
```

The runner accepts physical GPU3/GPU4 only, maps the selected device to logical
GPU0, ignores utilization/process count, and waits only for at least 6.5 GB of
free memory. It records a per-epoch PCGrad JSONL audit covering every training
batch, including raw cosine, conflict/projection fraction, and combined-gradient
norm changes. The dedicated validator requires exactly 160 epoch records,
finite values, at least one applied projection, matching file hashes, canonical
strict controls, and paired source-only provenance. No blind circuit appears.

Selection is fixed before observing PCGrad performance:

1. PCGrad must pass the existing `5%/10%/25%` gates relative to static seed0.
2. Expansion versus fixed `lambda=0.05` uses source-validation raw MSE only.
3. PCGrad advances to seeds 1-2 only for an improvement greater than `2e-5`.
4. Within `2e-5`, or if worse, retain fixed `lambda=0.05`; transfer metrics are
   gates/reporting only and cannot override the primary selection metric.
5. No linear schedule, distillation, or other reuse method runs in parallel.
   If PCGrad does not advance, the evidence supports locking the already-passing
   fixed positive-lambda shared architecture before rebalancing integration.

### R5 PCGrad Seed-0 Result

The formal PCGrad control completed at commit `95cb948` on physical GPU4. The
training process, dedicated validator, and overall queue all returned `rc=0`.
The source-validation-selected checkpoint was epoch 159.

| Method | Val raw MSE | digtime | timing_ctrl | array | Transfer mean |
|---|---:|---:|---:|---:|---:|
| static dual seed0 | 0.007809748 | 0.016486799 | 0.010408374 | 0.010042039 | 0.012312404 |
| fixed lambda=0.05 seed0 | 0.008035913 | 0.015224475 | 0.010998346 | 0.011695725 | 0.012639515 |
| PCGrad lambda=0.05 seed0 | 0.008055134 | 0.013704132 | 0.010625650 | 0.010672306 | 0.011667363 |

Relative to static dual, PCGrad changes source validation by `+3.1420%`,
transfer mean by `-5.2390%`, digtime by `-16.8782%`, timing_ctrl by `+2.0875%`,
and array by `+6.2763%`; all `5%/10%/25%` gates pass. Relative to fixed
`lambda=0.05`, PCGrad improves transfer mean by `7.6914%` and all three
transfer circuits by `3.3887%` to `9.9862%`, but worsens the preregistered
primary source-validation metric by `1.922064e-5` (`0.2392%`). It therefore
does not satisfy the required source improvement greater than `2e-5` and does
not advance to seeds 1-2. The observed transfer improvement cannot override
the registered primary metric. Fixed `lambda=0.05` remains the incumbent.

Across all 44,000 training batches, PCGrad projected 20,480 conflicts
(`46.5455%`). Raw global cosine had mean/median `0.029427/0.015871`, with
Q10/Q25/Q75/Q90 of `-0.230510/-0.106989/0.156354/0.317414`. The mean ratio of
projected to standard combined-gradient norm was `0.997096`. Thus conflict is
real and frequent, but removing its negative component shifts the
source/transfer tradeoff rather than improving the registered source metric.

The immutable summaries are:

```text
logs/strict_joint_rank0_lambda005_pcgrad_seed0_audit_20260713/
  pcgrad_seed0_summary.json
  pcgrad_seed0_summary.tsv
```

### Pre-Registered Compact GraphSAGE Shared Seed-0 Screen

The completed compact ClusterGCN rank-0 `lambda=0.05` model passes every
strict gate but remains `3.241%` worse than static dual on mean source
validation. The paper used GraphSAGE for its independent downstream GNN. A
single compact GraphSAGE screen therefore tests whether the remaining source
gap reflects the shared ClusterGCN operator rather than sharing itself. This
is an architecture screen, not a relaxation of the strict protocol or a
paper-scale reproduction.

At two layers and hidden dimension 64, PyG `SAGEConv` and `ClusterGCNConv`
each contain 8,256 trainable values per layer. The two-layer shared GNN and
the full deployment model therefore retain the same parameter count. The
paper-scale five-layer, hidden-144 GraphSAGE configuration is not tested here
because it would confound operator choice with a large capacity increase.

The only two candidate arms are:

```text
joint_sage_rank0_lambda0_seed0
joint_sage_rank0_lambda005_seed0
```

under:

```text
logs/strict_joint_sage_rank0_seed0_screen_20260714
```

Both arms use a two-layer hidden-64 GraphSAGE online/shared backbone and the
same two-layer head. Every other strict factor remains fixed: SSRAM-only SGRL
and normalization, five pretraining epochs, 160 downstream epochs without
early stopping, MSE, batch 512, learning rate `1e-4`, two hops with eight
neighbors, source/transfer sampling rates `1.0/0.1`, stage seed 0, and fixed
relation/embedding/evaluation seeds `20260711`. Representation audit remains
enabled every ten epochs. Gradient audit, PCGrad, LoRA, lambda schedules, and
distillation are excluded. The blind circuit is absent from commands,
manifests, runtime graph lists, and metrics.

The GraphSAGE SGRL checkpoint does not exist before this screen. To prevent a
checkpoint/metadata writer race, execution is fail-closed and sequential:

1. `lambda=0` is the only source-only GraphSAGE checkpoint producer.
2. Only after that run and its checkpoint provenance complete successfully may
   `lambda=0.05` validate and reuse the same checkpoint.
3. Any failure stops the queue; no second writer or automatic retry is allowed.

The dedicated validator must confirm exactly two completed artifacts, exact
strict arguments and stage seeds, source-only graph provenance, finite raw
metrics, restored-best validation equality, identical SAGE checkpoint/hash,
split, normalization, train/evaluation sampler provenance across the two arms,
absence of embedding-cache use, and the complete 21-record representation
audit. It also compares immutable processed-cache references with the canonical
seed-0 controls. SAGE and Cluster checkpoints or initial-model fingerprints are
intentionally not required to match because the operator is the experimental
factor.

Selection is fixed before observing either GraphSAGE result:

1. Each eligible arm must pass the existing gates relative to canonical
   seed-0 static dual: source degradation at most `5%`, transfer-mean
   degradation at most `10%`, and every transfer-circuit degradation at most
   `25%`.
2. The exact upper bounds are source Val `0.008200235828`, transfer mean
   `0.013543644268`, digtime `0.020608499181`, timing_ctrl `0.013010466937`,
   and array `0.012552548433`.
3. Advancement additionally requires source Val to improve over incumbent
   ClusterGCN `lambda=0.05` by more than `2e-5`, i.e. GraphSAGE Val must be
   strictly below `0.008015913110`.
4. Transfer metrics are gates and reporting only; they cannot select the
   architecture or reverse a source-based decision.
5. An arm qualifies only if it satisfies both the safety gates and the source
   improvement in item 3. If both GraphSAGE arms qualify, choose the lower raw
   source Val. If their difference is at most `2e-5`, prefer `lambda=0.05`
   because it retains the continuous GCL objective at identical deployment
   cost.
6. If neither qualifies, retain ClusterGCN `lambda=0.05` and stop GraphSAGE.
   If at least one qualifies, do not automatically launch more seeds: first
   register a matched static-GraphSAGE control and the exact multi-seed plan.

### Compact GraphSAGE Shared Seed-0 Result

The formal two-arm screen completed at commit `62894d9`. The sequential queue
ran `lambda=0` on physical GPU3, completed the source-only GraphSAGE checkpoint
handoff, and then ran `lambda=0.05` on physical GPU4. Both 160-epoch training
processes, the dedicated summary validator, and the overall queue returned
`rc=0`. Both source-validation-selected checkpoints were from epoch 156.

| Method | Val raw MSE | digtime | timing_ctrl | array | Transfer mean |
|---|---:|---:|---:|---:|---:|
| static dual seed0 | 0.007809748 | 0.016486799 | 0.010408374 | 0.010042039 | 0.012312404 |
| ClusterGCN lambda=0 seed0 | 0.008090950 | 0.014517009 | 0.011066118 | 0.012114635 | 0.012565921 |
| ClusterGCN lambda=0.05 seed0 | 0.008035913 | 0.015224475 | 0.010998346 | 0.011695725 | 0.012639515 |
| GraphSAGE lambda=0 seed0 | 0.008095136 | 0.014649815 | 0.011226653 | 0.012096140 | 0.012657536 |
| GraphSAGE lambda=0.05 seed0 | 0.008076897 | 0.014475943 | 0.011024745 | 0.010522621 | 0.012007770 |

Relative to static dual, GraphSAGE `lambda=0` changes source validation by
`+3.6542%`, transfer mean by `+2.8031%`, digtime by `-11.1422%`, timing_ctrl
by `+7.8617%`, and array by `+20.4550%`. GraphSAGE `lambda=0.05` changes the
same metrics by `+3.4207%`, `-2.4742%`, `-12.1968%`, `+5.9219%`, and
`+4.7857%`, respectively. Both arms therefore pass every registered
`5%/10%/25%` safety gate.

Within the matched GraphSAGE pair, `lambda=0.05` improves source validation by
`0.2253%`, transfer mean by `5.1334%`, digtime by `1.1869%`, timing_ctrl by
`1.7985%`, and array by `13.0084%` relative to `lambda=0`. Continuous GCL is
therefore beneficial inside the compact GraphSAGE architecture at identical
deployment cost.

GraphSAGE `lambda=0.05` also improves transfer mean by `4.9982%`, digtime by
`4.9166%`, and array by `10.0302%` relative to the seed-0 ClusterGCN
`lambda=0.05` incumbent; timing_ctrl is `0.2400%` worse. However, its source
validation is `0.5100%` worse: `0.008076896891` versus `0.008035913110`.
The preregistered expansion threshold was strictly below `0.008015913110`, so
the best GraphSAGE arm misses that threshold by `6.098378e-5`. Transfer results
are gates and reporting rather than an architecture-selection signal and
cannot reverse this source-based decision after observation.

The validator found both arms gate-eligible but no advancement-eligible
candidate, with selection reason `no_candidate_met_source_improvement` and
`advances_to_seeds12=false`. The paired checkpoint, split, normalization,
processed-cache, sampler, and evaluation provenance checks all pass. Each arm
contains the required 21 representation-audit records, and the blind circuit
is absent from commands, runtime graphs, and metrics.

The compact GraphSAGE and ClusterGCN deployment models each contain one
two-layer hidden-64 GNN and 29,378 trainable parameters, so parameter count
does not decide this comparison. GraphSAGE shows useful seed-0 transfer
behavior, especially on array, but does not satisfy the pre-observation source
criterion and is not expanded to seeds 1-2. The strict shared-backbone
incumbent is therefore locked as rank-0 ClusterGCN with constant
`lambda=0.05`, for which three-seed evidence already exists. The next reuse
milestone is single-GNN deployment export and export-equivalence validation,
not another architecture or optimizer sweep.

The immutable GraphSAGE screen summaries are:

```text
logs/strict_joint_sage_rank0_seed0_screen_20260714/
  sage_seed0_summary.json
  sage_seed0_summary.tsv
```
