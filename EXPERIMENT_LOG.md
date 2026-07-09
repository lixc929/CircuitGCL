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
