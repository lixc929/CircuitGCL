# Experiment Log

This is the single running log for local CircuitGCL experiments in this fork.
Keep new experiment records here instead of creating many separate Markdown files.

## Document Map

- `TEACHER_TASKS_README.md`: advisor task interpretation and high-level roadmap.
- `LABEL_REBALANCING_ANALYSIS.md`: paper reading notes and label-rebalancing design analysis.
- `EXPERIMENT_LOG.md`: commands, logs, metrics, and conclusions from actual runs.

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
