# Label Rebalancing and GCL Improvement Analysis

This note summarizes the reading of three local materials:

- `iccad25-ppt-final.pdf`: CircuitGCL ICCAD presentation.
- `CVPR2024.HCA.pdf`: Deep Imbalanced Regression via Hierarchical Classification Adjustment.
- `CVPR2025.SRL.pdf`: Improve Representation for Imbalanced Regression through Geometric Constraints.

The goal is to connect these ideas to the current CircuitGCL implementation and propose concrete improvement directions.

## 1. What the ICCAD Presentation Says

The ICCAD presentation frames CircuitGCL around two core ideas:

1. Graph Contrastive Learning
   - Learns topology-invariant and transferable node representations.
   - Uses online and target encoders.
   - Uses representation scattering mechanism.

2. Label Rebalancing
   - Addresses skewed parasitic capacitance distributions.
   - For regression, the presentation describes Balanced MSE / BMSE-style statistical conversion.
   - For classification, it uses Balanced Softmax Cross-Entropy.

The presentation also lists separate model components:

- SGRL encoders: 4-layer ClusterGCN, hidden dim 256, tanh, dropout 0.3, lr 1e-6.
- Downstream GNN: 5-layer GraphSAGE, hidden dim 144, PReLU, dropout 0.3, lr 1e-4.
- Task heads: 2-layer MLP.

This supports the earlier interpretation that the current system has two separate encoders:

```text
SGRL online encoder -> offline embedding
Downstream GNN      -> supervised capacitance prediction
```

The likely advisor direction is to reuse the online encoder as the downstream backbone, instead of treating SGRL as only a static embedding generator.

## 2. Current CircuitGCL Architecture

### 2.1 Data Flow

Current implementation:

```text
raw netlist/SPF/CDL
  -> heterogeneous graph
  -> homogeneous PyG graph
  -> subgraph/link sampling
  -> optional SGRL embedding
  -> downstream GNN + task head
  -> regression/classification loss
```

### 2.2 Current SGRL Usage

When `--sgrl 1`:

```text
main.py
  -> sgrl_train(...)
  -> returns node embeddings from online encoder
  -> saves embeddings/*.pkl
  -> dataset.set_cl_embeds(...)
  -> model.py uses cl_linear(batch.x)
  -> concatenates CL embedding with node type/stat features
```

So the current SGRL output is an extra feature, not the actual downstream backbone.

### 2.3 Current Label Normalization

For edge regression, `sram_dataset.py` normalizes coupling capacitance:

```text
y_norm = log10(y_raw * 1e21) / 6
y_norm is clipped to [0, 1]
y_class = bucketize(y_norm, [0.2, 0.4, 0.6, 0.8])
edge_label = [y_norm, y_class]
```

For node tasks, a similar rule is used:

```text
y_norm = log10(y_raw * 1e20) / 6
```

This is important because every rebalancing method should operate on either:

- the continuous normalized label `y_norm`; or
- a carefully designed discretization of `y_norm`.

### 2.4 Observed Label Distribution

Using the existing processed edge datasets and the current normalization rule:

```text
array_128_32_8t:
  fixed bins [0, .2, .4, .6, .8, 1] -> [3.60%, 8.03%, 37.00%, 51.28%, 0.10%]

digtime:
  fixed bins -> [2.42%, 10.73%, 39.45%, 46.52%, 0.89%]

ssram:
  fixed bins -> [3.11%, 10.02%, 34.07%, 50.42%, 2.38%]

timing_ctrl:
  fixed bins -> [2.46%, 8.82%, 24.00%, 39.27%, 25.44%]
```

Two observations:

1. The current fixed bins are not balanced.
2. Distribution shifts are strong across datasets; for example, `timing_ctrl` has a much larger high-capacitance bin than `ssram`.

This matters because label rebalancing based only on the training distribution may overfit `ssram` and fail on shifted test designs.

## 3. Label Rebalancing Implementation Audit

Status after the 2026-07-09 audit:

- The public `origin/master` implementation has concrete issues in regression
  rebalancing when interpreted against the normalized two-column labels.
- The current `test` branch already fixes the GAI, BMC, and LDS regression
  issues through commit `1fcfe5c`.
- The classification `BalancedSoftmax` mini-batch-count issue still remains a
  candidate fix because we have not changed that path yet.
- This should be reported as an implementation audit, not as a claim that the
  paper results are invalid. The paper/internal experiment code may differ from
  the public repository state.

### 3.1 GAI Used the Wrong Label Tensor in `origin/master`

`train_gmm(dataset)` currently uses:

```python
train_labels = dataset[graph_idx].edge_label
gmm.fit(train_labels.reshape(-1, 1))
```

After `dataset.norm_nfeat(...)`, `edge_label` has two columns:

```text
[continuous_label, discrete_class]
```

So the original public implementation fits the GMM on both continuous labels
and discrete class IDs flattened together. This is likely incorrect. GAI should
fit only:

```python
dataset[0].edge_label[:, 0]
```

The current `test` branch now keeps only the first column before fitting GMM.

### 3.2 LDS WeightedMSE Had a Broadcasting Risk in `origin/master`

`WeightedMSE.forward(inputs, targets, weights)` expects aligned shapes. In `compute_loss`, LDS currently passes:

```python
pred: [N, 1]
true[:, 0].squeeze(): [N]
true[:, 1].squeeze(): [N]
```

`(inputs - targets)` can broadcast to `[N, N]`, which is wrong. It should use:

```python
target = true[:, 0].view(-1, 1)
weight = true[:, 1].view(-1, 1)
```

The current `test` branch reshapes target and weight tensors to `[N, 1]`.

### 3.3 BMC Implementation Looked Incorrect in `origin/master`

`bmc_loss` currently creates:

```python
logits = -0.5 * (pred - target).pow(2) / noise_var
loss = F.cross_entropy(logits.view(-1), torch.arange(pred.shape[0], dtype=torch.float32, ...))
```

This does not match the standard Balanced MSE formulation, where logits should compare each prediction against all targets in the batch, producing an `[N, N]` matrix, and target indices should be integer class indices `0..N-1`.

The expected shape is closer to:

```python
logits = -0.5 * (pred - target.T).pow(2) / noise_var  # [N, N]
labels = torch.arange(N, device=device).long()
loss = F.cross_entropy(logits, labels)
```

The current `test` branch now uses `[N, N]` logits and integer target indices.

### 3.4 Balanced Softmax Uses Mini-batch Counts

For node classification, `class_train` computes `sample_per_class` from the current mini-batch:

```python
sample_per_class.append(torch.sum(true_class == i).item())
```

Balanced Softmax should preferably use global training-set class counts, not noisy per-batch counts. Per-batch zero-count classes can also create unstable `log(0)` behavior.

## 4. Lessons from CVPR 2024 HCA

Paper:

```text
Deep Imbalanced Regression via Hierarchical Classification Adjustment
```

### 4.1 Core Idea

HCA reformulates imbalanced regression as hierarchical classification.

It builds multiple classifiers over the same continuous label range:

```text
coarse classifier: balanced, enough samples per class, high range accuracy
fine classifier: high resolution, low quantization error, but less balanced
```

Then it uses coarse classifiers to adjust the finest classifier.

Main insight:

```text
Coarse classifiers locate the correct range more reliably.
Fine classifiers reduce quantization error inside that range.
```

### 4.2 Why It Helps Imbalanced Regression

Single-bin quantization has a tradeoff:

- coarse bins: balanced and easy to classify, but large quantization error;
- fine bins: smaller quantization error, but tail bins are low-shot and unstable.

HCA combines both.

For CircuitGCL, this is highly relevant because parasitic capacitance labels are:

- continuous;
- highly skewed;
- often evaluated by regression metrics;
- already discretized into classes for classification.

### 4.3 How To Adapt HCA to CircuitGCL

For edge regression, add a hierarchical classification head alongside the regression head:

```text
shared GNN encoder
  -> regression head: predict y_norm
  -> HCA heads:
       h=1: 2 bins by train quantiles
       h=2: 4 bins by train quantiles
       h=3: 8 bins by train quantiles
       h=4: 16 bins or 32 bins by train quantiles
```

Training objective:

```text
L = L_reg + lambda_hca * sum_h CE(head_h, bin_h)
```

Inference options:

1. Use HCA as an auxiliary loss only.
2. Use HCA to calibrate the regression prediction range.
3. Replace direct regression with expected bin value from a distilled HCA classifier.

Recommended first version:

```text
Auxiliary HCA loss + normal regression output
```

This is simpler and less disruptive.

### 4.4 Important Detail: Quantile Bins

HCA should not use the current fixed boundaries `[0.2, 0.4, 0.6, 0.8]` for all heads.

Instead, use training-set quantiles:

```text
h=1: median split
h=2: quartile split
h=3: 8-quantile split
h=4: 16-quantile split
```

This directly balances sample counts per class on the training set.

### 4.5 Circuit-Specific Concern

Training only on `ssram` and testing on other designs means quantile boundaries from `ssram` may not align with test distributions.

So HCA should be evaluated in two modes:

1. Train-quantile bins only, which is realistic and deployable.
2. Fixed physical/log bins, which may preserve cross-design semantics better.

The first may improve training balance; the second may improve transfer consistency.

## 5. Lessons from CVPR 2025 SRL

Paper:

```text
Improve Representation for Imbalanced Regression through Geometric Constraints
```

### 5.1 Core Idea

SRL argues that imbalanced regression is not only a loss-weighting problem. It is also a representation geometry problem.

For regression, representations should be:

```text
continuous
ordered
smooth
uniformly spread over feature space
```

The paper introduces two geometric losses:

1. Enveloping loss
   - encourages the label-ordered representation trace to occupy the hypersphere.

2. Homogeneity loss
   - encourages adjacent label-bin centroids to be smoothly and evenly spaced.

It also uses a centroid contrastive loss:

```text
sample representation -> close to its own label-bin centroid
sample representation -> far from other label-bin centroids
```

### 5.2 Why It Is Relevant to CircuitGCL

CircuitGCL already uses graph contrastive learning, but its current GCL objective is topology/representation-scattering based. It is not explicitly aware of continuous capacitance labels.

SRL suggests a missing ingredient:

```text
Make supervised regression representations label-aware and geometrically balanced.
```

This directly targets the weakness where rare capacitance ranges are poorly represented in feature space.

### 5.3 How To Adapt SRL to CircuitGCL

Add an optional supervised representation regularizer during downstream regression.

For each training batch:

1. Get edge representation before the final head:

```text
edge_repr = concat(src_emb, dst_emb)
```

2. Normalize:

```text
z = normalize(project(edge_repr))
```

3. Assign each sample to a label bin.

Recommended bins:

```text
quantile bins over training y_norm, e.g. 32 or 64 bins
```

4. Maintain a memory/surrogate centroid table:

```text
centroid[k] = EMA of normalized representations in bin k
```

5. Add losses:

```text
L_contrast: sample-to-centroid contrastive loss
L_homo: adjacent centroids should change smoothly/evenly
L_env: centroids should use the hypersphere broadly
```

Total loss:

```text
L = L_reg + lambda_con * L_contrast + lambda_homo * L_homo + lambda_env * L_env
```

Recommended first version:

```text
Implement only L_contrast + L_homo first.
Add L_env later if needed.
```

Reason:

- `L_env` requires sampled hypersphere points and is more sensitive.
- `L_homo` is easier and directly matches ordered capacitance labels.

### 5.4 Relationship with Existing GCL

SRL should not be confused with the existing unsupervised SGRL.

Current SGRL:

```text
graph structure self-supervision
online/target alignment
representation scattering
```

SRL-style regularizer:

```text
label-aware supervised representation geometry
edge representation ordered by capacitance labels
```

The two can be tested separately and then combined.

## 6. Proposed Improvement Roadmap

### Phase 0: Fix and Audit Existing Rebalancing Bugs

Before adding new methods:

1. Fix GAI to fit GMM only on continuous labels.
2. Fix LDS broadcasting.
3. Fix BMC batch logits.
4. Change Balanced Softmax to use global class counts.

Regression items 1-3 are already fixed on the current `test` branch and were
audited against `origin/master` on 2026-07-09. Item 4 remains for the
classification path. This phase is necessary because conclusions about
rebalancing from the public `origin/master` implementation may be unreliable.

### Phase 1: Clean Baselines

Run:

```text
No SGRL + MSE
No SGRL + fixed GAI
No SGRL + fixed BMC
No SGRL + fixed LDS
```

Track:

```text
overall MSE / MAE / R2
per-bin MSE / MAE
few/medium/many-shot metrics
prediction distribution vs label distribution
```

Per-bin metrics are essential; overall MSE alone can hide tail failure.

### Phase 2: HCA Auxiliary Head

Implement:

```text
--regress_aux hca
--hca_bins 2,4,8,16
--hca_lambda 0.1
--hca_binning quantile
```

Loss:

```text
L = L_reg + lambda_hca * sum_h CE_h
```

Compare:

```text
MSE
fixed GAI
MSE + HCA
fixed GAI + HCA
```

### Phase 3: SRL-Style Representation Regularization

Implement:

```text
--regress_aux srl
--srl_bins 32
--srl_lambda_con 0.01
--srl_lambda_homo 0.01
--srl_memory_momentum 0.9
```

Start with:

```text
L = L_reg + lambda_con * L_con + lambda_homo * L_homo
```

Add enveloping loss later only after the simpler version works.

### Phase 4: Reuse Online Encoder as Downstream Backbone

Implement the advisor's architecture direction:

```text
SGRL pretrain online encoder
  -> initialize downstream encoder
  -> finetune with MSE / GAI / HCA / SRL
```

Compare:

```text
static SGRL embedding
online encoder initialization
online encoder frozen
online encoder finetuned
```

### Phase 5: Combination Study

After each component is stable:

```text
MSE
GAI
HCA
SRL
GCL-backbone
GCL-backbone + GAI
GCL-backbone + HCA
GCL-backbone + SRL
GCL-backbone + GAI + SRL
```

The goal is not to stack everything blindly. The goal is to understand which component helps which failure mode.

## 7. Recommended Priority

Most important first:

```text
1. Fix current rebalancing implementation bugs.
2. Add per-bin/few-shot metrics.
3. Implement HCA auxiliary heads.
4. Implement simple SRL contrastive + homogeneity regularizer.
5. Implement online-encoder backbone reuse.
```

Why HCA before SRL:

- HCA is easier to implement.
- It directly targets label rebalancing.
- It fits existing classification code and class labels.
- It is easier to explain to the advisor.

Why SRL after HCA:

- SRL is more novel relative to current CircuitGCL.
- It may better explain why GCL and label rebalancing do not combine well: current GCL is structure-aware but label-agnostic, while SRL makes the representation label-aware.

## 8. Main Hypothesis

The current CircuitGCL label rebalancing is mostly loss-level and may have implementation issues. The next improvement should move beyond scalar loss reweighting:

```text
HCA: rebalance the label space through hierarchical quantile classification.
SRL: rebalance the representation space through label-aware geometric constraints.
GCL backbone reuse: remove the mismatch between offline GCL embeddings and the downstream encoder.
```

Together, these form a coherent research direction:

```text
from loss-level balancing
to label-space balancing
to representation-space balancing
to shared-backbone transfer learning
```
