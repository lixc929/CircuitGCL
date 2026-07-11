import torch
import networkx as nx
from torch_geometric.utils import to_dense_adj, to_networkx
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader,NeighborLoader
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import pickle
from collections import Counter
from torch_geometric.data import HeteroData
NET = 0
DEV = 1
PIN = 2

def build_split_indices(args, dataset):
    """Build deterministic source train/validation indices."""
    train_graph = dataset[0]
    split_seed = getattr(args, 'split_seed', args.seed)
    if args.task_level == 'node':
        indices = np.arange(train_graph.y.size(0))
    elif args.task_level == 'edge':
        indices = np.arange(train_graph.edge_label.size(0))
    else:
        raise ValueError(f"Invalid task level: {args.task_level}")
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        shuffle=True,
        random_state=split_seed,
    )
    return {
        'train': torch.tensor(train_indices, dtype=torch.long),
        'val': torch.tensor(val_indices, dtype=torch.long),
        'seed': int(split_seed),
    }


def _append_regression_weights(labels, class_weights):
    if class_weights is None:
        return labels
    classes = labels[:, 1].long()
    weights = class_weights.to(labels.device)[classes].to(labels.dtype)
    return torch.cat((labels, weights.view(-1, 1)), dim=1)


def dataset_sampling(args, dataset, split_indices=None, class_weights=None):
    """ 
    Sampling subgraphs for each graph in dataset
    Args:
        args (argparse.Namespace): The arguments
        dataset (torch_geometric.data.InMemoryDataset): The dataset
    Return:
        train_loader, val_loader, test_loaders
    """

    ## default training data come from the first dataset
    graph_idx = 0
    train_graph = dataset[graph_idx]
    if split_indices is None:
        split_indices = build_split_indices(args, dataset)
    if args.task_level == 'node':
        train_node_ind = split_indices['train']
        val_node_ind = split_indices['val']
        class_labels = train_graph.y[train_node_ind, 1]
    
        train_loader = NeighborLoader(
            train_graph,
            num_neighbors=args.num_hops * [args.num_neighbors],
            input_nodes=train_node_ind,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = NeighborLoader(
            train_graph,
            num_neighbors=args.num_hops * [args.num_neighbors],
            input_nodes=val_node_ind,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
       
        test_loaders = {}
        for i in range(1, len(dataset)):
            test_graph = dataset[i]
            graph_name = test_graph.name

            all_nodes = np.arange(test_graph.num_nodes)
            all_test_nodes = torch.arange(test_graph.num_nodes)
            
            if graph_name == 'sandwich' or graph_name == 'ultra8t':
                # random sample all test nodes
                sampled_size = max(1, int(test_graph.num_nodes *args.large_dataset_sample_rates))
                perm = torch.randperm(test_graph.num_nodes)
                test_input_nodes = all_test_nodes[perm[:sampled_size]]
            else:
                sampled_size = max(1, int(test_graph.num_nodes *args.small_dataset_sample_rates))
                perm = torch.randperm(test_graph.num_nodes)
                test_input_nodes = all_test_nodes[perm[:sampled_size]]

            test_loaders[graph_name] = NeighborLoader(
                test_graph,
                num_neighbors=args.num_hops * [args.num_neighbors],
                input_nodes=test_input_nodes,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
    
    elif args.task_level == 'edge':
        train_ind = split_indices['train']
        val_ind = split_indices['val']
        class_labels = train_graph.edge_label[train_ind, 1]

        train_edge_label_index = train_graph.edge_label_index[:, train_ind]
        train_edge_label = _append_regression_weights(
            train_graph.edge_label[train_ind], class_weights
        )

        ## Create the dataloaders for training dataset
        train_loader = LinkNeighborLoader(
            train_graph,
            num_neighbors=args.num_hops * [args.num_neighbors],
            edge_label_index=train_edge_label_index,
            edge_label=train_edge_label,
            subgraph_type='bidirectional',
            disjoint=True,
            batch_size=args.batch_size,
            shuffle=False, 
            num_workers=args.num_workers,
        )

        val_edge_label_index = train_graph.edge_label_index[:, val_ind]
        val_edge_label = _append_regression_weights(
            train_graph.edge_label[val_ind], class_weights
        )

        ## Create the dataloaders for validation dataset
        val_loader = LinkNeighborLoader(
            train_graph,
            num_neighbors=args.num_hops * [args.num_neighbors],
            edge_label_index=val_edge_label_index,
            edge_label=val_edge_label,
            subgraph_type='bidirectional',
            disjoint=True,
            batch_size=args.batch_size,
            shuffle=False, 
            num_workers=args.num_workers,
        )

        test_loaders = {}

        ## The remaining datasets are all used for testing
        for i in range(graph_idx+1, len(dataset.names)):
            test_graph = dataset[i]
            graph_name = test_graph.name
           
           # edge sampling is completed in get_balanced_edges
            test_input_edge_labels = test_graph.edge_label
            test_edge_label_index = test_graph.edge_label_index

            test_edge_label = _append_regression_weights(
                test_input_edge_labels, class_weights
            )

            ## Create the dataloaders for each test dataset
            test_loaders[graph_name] = \
                LinkNeighborLoader(
                    test_graph,
                    num_neighbors=args.num_hops * [args.num_neighbors],
                    edge_label_index=test_edge_label_index,
                    edge_label=test_edge_label,
                    subgraph_type='bidirectional',
                    disjoint=True,
                    batch_size=args.batch_size,
            shuffle=False, 
            num_workers=args.num_workers,
        )
            
    else:
        raise ValueError(f"Invalid task level: {args.task_level}")
    
    if isinstance(class_labels, torch.Tensor):
        class_labels = class_labels.cpu().numpy().tolist()
    #print("class_labels", class_labels)
    label_counts = Counter(class_labels)
    print("label_counts", label_counts)
    max_label = max(label_counts, key=label_counts.get)
    print(f"The most common label in the training set is: {max_label}, with {label_counts[max_label]} samples")

    return (train_loader, val_loader, test_loaders, max_label, split_indices)

