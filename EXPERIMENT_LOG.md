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
- The next controlled experiment should rerun short freeze2/freeze3 MSE with
  the corrected optimizer and compare `frozen_only` against `always` eval
  policy. Do not add BMC/GAI or a new architecture until that comparison shows
  whether optimizer-state preservation and dropout control recover stability.
