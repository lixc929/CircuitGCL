# Paper Index

This directory keeps local reference PDFs for the CircuitGCL improvement work.
PDF files are intentionally ignored by Git via `*.pdf`; this index is tracked.

## CircuitGCL / Advisor Materials

| Local file | Source | Notes |
| --- | --- | --- |
| `CircuitGCL_Transferable_Parasitic_Estimation_arXiv2507.06535.pdf` | https://arxiv.org/abs/2507.06535 | Formal CircuitGCL paper. |
| `ICCAD2025_CircuitGCL_presentation.pdf` | Local advisor-provided slides | ICCAD presentation/slides copied from the repo root. |

## Self-Supervised / Encoder Reuse References

| Local file | Source | Notes |
| --- | --- | --- |
| `BYOL_Bootstrap_Your_Own_Latent_arXiv2006.07733.pdf` | https://arxiv.org/abs/2006.07733 | Core online/target self-supervised learning reference; downstream uses the learned online representation. |
| `BGRL_Large_Scale_Representation_Learning_on_Graphs_arXiv2102.06514.pdf` | https://arxiv.org/abs/2102.06514 | Graph adaptation of BYOL-style online/target learning; useful for online encoder reuse in GNNs. |
| `GraphCL_Graph_Contrastive_Learning_with_Augmentations_arXiv2010.13902.pdf` | https://arxiv.org/abs/2010.13902 | Graph contrastive pretraining baseline covering transfer and semi-supervised settings. |
| `GNN_Pretraining_Strategies_arXiv1905.12265.pdf` | https://arxiv.org/abs/1905.12265 | General GNN pretraining and downstream fine-tuning reference; includes negative-transfer cautions. |
| `SimSiam_Simple_Siamese_Representation_Learning_arXiv2011.10566.pdf` | https://arxiv.org/abs/2011.10566 | Non-contrastive Siamese reference showing the role of stop-gradient and predictor. |
| `MoCo_Momentum_Contrast_arXiv1911.05722.pdf` | https://arxiv.org/abs/1911.05722 | Momentum/target encoder reference for transferable self-supervised representations. |

## Imbalanced Regression References

| Local file | Source | Notes |
| --- | --- | --- |
| `Balanced_MSE_Imbalanced_Visual_Regression_arXiv2203.16427.pdf` | https://arxiv.org/abs/2203.16427 | Source for Balanced MSE, including BMC-style batch contrastive implementation. |
| `Delving_Into_Deep_Imbalanced_Regression_LDS_arXiv2102.09554.pdf` | https://arxiv.org/abs/2102.09554 | Source for label distribution smoothing (LDS). |
| `CVPR2024_HCA_Deep_Imbalanced_Regression.pdf` | Existing local CVF PDF | Advisor-recommended HCA paper. |
| `CVPR2024_HCA_arXiv2310.17154.pdf` | https://arxiv.org/abs/2310.17154 | ArXiv copy of HCA. |
| `CVPR2025_SRL_Geometric_Constraints_Imbalanced_Regression.pdf` | Existing local PDF | Advisor-recommended SRL paper. |
| `CVPR2025_SRL_arXiv2503.00876.pdf` | https://arxiv.org/abs/2503.00876 | ArXiv copy of SRL. |

## Imbalanced Classification References

| Local file | Source | Notes |
| --- | --- | --- |
| `Balanced_Meta_Softmax_arXiv2007.10740.pdf` | https://arxiv.org/abs/2007.10740 | Source for Balanced Softmax / BSCE. |
