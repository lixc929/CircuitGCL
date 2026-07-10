#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import LinkNeighborLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sgrl_models import CustomConv
from sram_dataset import performat_SramDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Compare cached static SGRL embeddings with online-encoder outputs '
            'computed inside downstream disjoint LinkNeighborLoader batches.'
        )
    )
    parser.add_argument(
        '--dataset',
        default='ssram+digtime+timing_ctrl+array_128_32_8t',
    )
    parser.add_argument('--dataset_dir', default='./datasets/')
    parser.add_argument('--small_dataset_sample_rates', type=float, default=1.0)
    parser.add_argument('--large_dataset_sample_rates', type=float, default=0.1)
    parser.add_argument('--num_hops', type=int, default=2)
    parser.add_argument('--num_neighbors', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--max_edges_per_circuit', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--cl_model', default='clustergcn')
    parser.add_argument('--cl_hid_dim', type=int, default=64)
    parser.add_argument('--cl_gnn_layers', type=int, default=2)
    parser.add_argument('--cl_act_fn', default='tanh')
    parser.add_argument('--cl_dropout', type=float, default=0.3)
    parser.add_argument(
        '--checkpoint',
        default=(
            'pkl/pkl_online/best_online_'
            'ssram+digtime+timing_ctrl+array_128_32_8t_'
            'clustergcn_layer2_dim64_tanh_dr0.3_small.pkl'
        ),
    )
    parser.add_argument(
        '--embeddings',
        default=(
            'embeddings/embeddings_'
            'ssram+digtime+timing_ctrl+array_128_32_8t_'
            'clustergcn_layer2_dim64_tanh.pkl'
        ),
    )
    parser.add_argument('--output_json', default=None)
    return parser.parse_args()


class MetricAccumulator:
    def __init__(self):
        self.online = []
        self.cached = []

    def update(self, online, cached):
        self.online.append(online.detach().float().cpu())
        self.cached.append(cached.detach().float().cpu())

    def summarize(self):
        online = torch.cat(self.online, dim=0)
        cached = torch.cat(self.cached, dim=0)
        cosine = F.cosine_similarity(online, cached, dim=1)
        delta = online - cached
        relative_l2 = delta.norm(dim=1) / cached.norm(dim=1).clamp_min(1e-12)
        return {
            'count': int(online.size(0)),
            'cosine_mean': float(cosine.mean()),
            'cosine_std': float(cosine.std(unbiased=False)),
            'cosine_p05': float(torch.quantile(cosine, 0.05)),
            'mse': float(delta.square().mean()),
            'relative_l2_mean': float(relative_l2.mean()),
            'online_norm_mean': float(online.norm(dim=1).mean()),
            'cached_norm_mean': float(cached.norm(dim=1).mean()),
        }


def build_online_encoder(args, device):
    model_args = SimpleNamespace(
        cl_hid_dim=args.cl_hid_dim,
        cl_act_fn=args.cl_act_fn,
        cl_gnn_layers=args.cl_gnn_layers,
        cl_model=args.cl_model,
        use_bn=0,
        cl_dropout=args.cl_dropout,
    )
    encoder = CustomConv(model_args)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    prefix = 'online_encoder.'
    encoder_state = {
        key[len(prefix):]: value
        for key, value in checkpoint.items()
        if key.startswith(prefix)
    }
    encoder.load_state_dict(encoder_state, strict=True)
    encoder.eval()
    return encoder.to(device)


def make_loader(graph, args):
    edge_count = min(
        args.max_edges_per_circuit,
        graph.edge_label_index.size(1),
    )
    edge_label_index = graph.edge_label_index[:, :edge_count]
    edge_label = graph.edge_label[:edge_count]
    return LinkNeighborLoader(
        graph,
        num_neighbors=[args.num_neighbors] * args.num_hops,
        edge_label_index=edge_label_index,
        edge_label=edge_label,
        subgraph_type='bidirectional',
        disjoint=True,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )


def audit_circuit(encoder, graph, cached, node_offset, args, device):
    all_nodes = MetricAccumulator()
    root_nodes = MetricAccumulator()
    batches = 0
    edges = 0

    with torch.no_grad():
        for batch in make_loader(graph, args):
            batch = batch.to(device)
            online = encoder.encode(batch)
            global_ids = batch.n_id.detach().cpu() + node_offset
            cached_batch = cached[global_ids].to(device)
            all_nodes.update(online, cached_batch)

            edge_count = int(batch.edge_label.size(0))
            root_count = min(2 * edge_count, online.size(0))
            root_nodes.update(online[:root_count], cached_batch[:root_count])
            batches += 1
            edges += edge_count

    return {
        'sampled_edges': edges,
        'batches': batches,
        'all_sampled_nodes': all_nodes.summarize(),
        'root_endpoint_nodes': root_nodes.summarize(),
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.gpu >= 0:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is not available.')
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    dataset = performat_SramDataset(
        name=args.dataset,
        dataset_dir=args.dataset_dir,
        neg_edge_ratio=0.0,
        to_undirected=True,
        small_dataset_sample_rates=args.small_dataset_sample_rates,
        large_dataset_sample_rates=args.large_dataset_sample_rates,
        task_level='edge',
        net_only=True,
        class_boundaries=[0.2, 0.4, 0.6, 0.8],
    )
    cached = torch.load(args.embeddings, map_location='cpu').float()
    expected_nodes = sum(dataset[i].num_nodes for i in range(len(dataset)))
    if cached.shape != (expected_nodes, args.cl_hid_dim):
        raise ValueError(
            f'Cached embedding shape {tuple(cached.shape)} does not match '
            f'expected {(expected_nodes, args.cl_hid_dim)}.'
        )

    encoder = build_online_encoder(args, device)
    results = {
        'config': {
            **vars(args),
            'device': str(device),
            'cached_shape': list(cached.shape),
        },
        'circuits': {},
    }
    node_offset = 0
    for index, circuit_name in enumerate(dataset.names):
        graph = dataset[index]
        results['circuits'][circuit_name] = audit_circuit(
            encoder,
            graph,
            cached,
            node_offset,
            args,
            device,
        )
        node_offset += graph.num_nodes

    print(json.dumps(results, indent=2, sort_keys=True))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(results, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )


if __name__ == '__main__':
    main()
