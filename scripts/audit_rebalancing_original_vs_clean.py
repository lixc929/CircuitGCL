import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from balanced_mse import WeightedMSE, bmc_loss
from sram_dataset import performat_SramDataset


NET = 0
DEV = 1


def legacy_bmc_loss(pred, target, noise_var):
    logits = -0.5 * (pred - target).pow(2) / noise_var
    labels = torch.arange(pred.shape[0], dtype=torch.float32, device=pred.device)
    loss = F.cross_entropy(logits.view(-1), labels)
    loss = loss * (2 * noise_var).detach()
    return loss


def sample_1d(values, max_samples, seed):
    values = values.detach().cpu().view(-1)
    if max_samples <= 0 or values.numel() <= max_samples:
        return values
    generator = torch.Generator().manual_seed(seed)
    index = torch.randperm(values.numel(), generator=generator)[:max_samples]
    return values[index]


def fit_gmm(values, max_samples, seed):
    sampled = sample_1d(values, max_samples=max_samples, seed=seed)
    array = sampled.numpy().reshape(-1, 1)
    gmm = GaussianMixture(
        n_components=8,
        random_state=0,
        max_iter=200,
        reg_covar=1e-6,
    ).fit(array)
    order = np.argsort(gmm.means_.reshape(-1))
    return {
        "num_fit_samples": int(sampled.numel()),
        "means": [round(float(x), 6) for x in gmm.means_.reshape(-1)[order]],
        "weights": [round(float(x), 6) for x in gmm.weights_.reshape(-1)[order]],
        "variances": [round(float(x), 8) for x in gmm.covariances_.reshape(-1)[order]],
    }


def summarize_values(values):
    values = values.detach().cpu().view(-1)
    return {
        "num_values": int(values.numel()),
        "min": round(float(values.min()), 6),
        "max": round(float(values.max()), 6),
        "mean": round(float(values.mean()), 6),
        "std": round(float(values.std(unbiased=False)), 6),
        "fraction_gt_1": round(float((values > 1.0).float().mean()), 6),
    }


def audit_gai(args):
    dataset = performat_SramDataset(
        name=args.dataset,
        dataset_dir=args.dataset_dir,
        neg_edge_ratio=0.0,
        to_undirected=True,
        small_dataset_sample_rates=args.small_dataset_sample_rates,
        large_dataset_sample_rates=args.large_dataset_sample_rates,
        task_level="edge",
        net_only=True,
        class_boundaries=[0.2, 0.4, 0.6, 0.8],
    )
    dataset.norm_nfeat([NET, DEV])

    train_labels = dataset[0].edge_label.detach().cpu()
    legacy_labels = train_labels.reshape(-1)
    clean_labels = train_labels[:, 0]
    discrete_labels = train_labels[:, 1]

    return {
        "edge_label_shape": list(train_labels.shape),
        "continuous_label_summary": summarize_values(clean_labels),
        "discrete_label_summary": summarize_values(discrete_labels),
        "legacy_flattened_label_summary": summarize_values(legacy_labels),
        "clean_gmm": fit_gmm(clean_labels, args.max_gmm_samples, args.seed),
        "legacy_gmm": fit_gmm(legacy_labels, args.max_gmm_samples, args.seed),
    }


def audit_bmc():
    pred = torch.tensor([[0.05], [0.25], [0.65], [0.85]], dtype=torch.float32)
    target = torch.tensor([[0.06], [0.30], [0.60], [0.80]], dtype=torch.float32)
    noise_var = torch.tensor(0.001, dtype=torch.float32) ** 2

    legacy_logits = -0.5 * (pred - target).pow(2) / noise_var
    legacy_labels = torch.arange(pred.shape[0], dtype=torch.float32, device=pred.device)
    clean_logits = -0.5 * (pred.view(-1, 1) - target.view(1, -1)).pow(2) / noise_var

    result = {
        "legacy_logits_shape_before_flatten": list(legacy_logits.shape),
        "legacy_logits_shape_after_flatten": list(legacy_logits.view(-1).shape),
        "legacy_label_shape": list(legacy_labels.shape),
        "legacy_label_dtype": str(legacy_labels.dtype),
        "clean_logits_shape": list(clean_logits.shape),
        "clean_label_shape": [pred.shape[0]],
        "clean_label_dtype": "torch.int64",
    }
    try:
        result["legacy_status"] = "ok"
        result["legacy_loss"] = float(legacy_bmc_loss(pred, target, noise_var).item())
    except Exception as exc:
        result["legacy_status"] = "error"
        result["legacy_error"] = f"{type(exc).__name__}: {exc}"

    try:
        result["clean_status"] = "ok"
        result["clean_loss"] = float(bmc_loss(pred, target, noise_var).item())
    except Exception as exc:
        result["clean_status"] = "error"
        result["clean_error"] = f"{type(exc).__name__}: {exc}"

    return result


def audit_lds():
    pred = torch.tensor([[0.05], [0.25], [0.65], [0.85]], dtype=torch.float32)
    target_legacy = torch.tensor([0.06, 0.30, 0.60, 0.80], dtype=torch.float32)
    weights_legacy = torch.tensor([1.0, 1.5, 2.0, 2.5], dtype=torch.float32)
    target_clean = target_legacy.view(-1, 1)
    weights_clean = weights_legacy.view(-1, 1)

    legacy_loss_tensor = (pred - target_legacy) ** 2
    legacy_weighted = legacy_loss_tensor * weights_legacy
    clean_loss = WeightedMSE()(pred, target_clean, weights_clean)

    return {
        "legacy_broadcast_loss_shape": list(legacy_loss_tensor.shape),
        "legacy_loss": float(legacy_weighted.mean().item()),
        "clean_loss_shape": list(((pred - target_clean) ** 2).shape),
        "clean_loss": float(clean_loss.item()),
        "absolute_difference": float(abs(legacy_weighted.mean().item() - clean_loss.item())),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Audit original vs clean rebalancing behavior on CircuitGCL labels."
    )
    parser.add_argument(
        "--dataset",
        default="ssram+digtime+timing_ctrl+array_128_32_8t",
    )
    parser.add_argument("--dataset_dir", default="./datasets/")
    parser.add_argument("--small_dataset_sample_rates", type=float, default=1.0)
    parser.add_argument("--large_dataset_sample_rates", type=float, default=0.1)
    parser.add_argument("--max_gmm_samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--json_out",
        default="logs/rebalancing_original_vs_clean_audit.json",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    results = {
        "dataset": args.dataset,
        "max_gmm_samples": args.max_gmm_samples,
        "gai": audit_gai(args),
        "bmc": audit_bmc(),
        "lds": audit_lds(),
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))
    print(f"Saved audit JSON to {out_path}")


if __name__ == "__main__":
    main()
