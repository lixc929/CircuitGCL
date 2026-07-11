from sklearn.multiclass import OneVsRestClassifier
from sgrl_models import CustomConv, CustomOnline, Target
import hashlib
import json
import torch
import time
from tqdm import tqdm
import sys
import os

from sram_dataset import adaption_for_sgrl
from torch_geometric.loader import NeighborLoader, ShaDowKHopSampler, LinkNeighborLoader
from rng_utils import SamplerFingerprint, fixed_rng, seed_all


def state_dict_to_cpu(state_dict):
    return {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in state_dict.items()
    }


def sgrl_cache_fingerprint(args, train_graph_names):
    fields = {
        'train_graph_names': list(train_graph_names),
        'graph_scope': args.sgrl_graph_scope,
        'pretraining_seed': getattr(args, 'pretraining_seed', args.seed),
        'target_update': args.sgrl_pretrain_target_update,
        'model': args.cl_model,
        'layers': args.cl_gnn_layers,
        'hidden_dim': args.cl_hid_dim,
        'activation': args.cl_act_fn,
        'dropout': args.cl_dropout,
        'batch_size': args.cl_batch_size,
        'num_neighbors': args.cl_num_neighbors,
        'epochs': args.cl_epochs,
        'online_lr': args.e1_lr,
        'target_lr': args.e2_lr,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
    }
    serialized = json.dumps(fields, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()[:20]


def embedding_cache_fingerprint(args, sgrl_cache_key):
    """Identify a frozen-embedding view independently of its checkpoint."""
    fields = {
        'sgrl_cache_key': sgrl_cache_key,
        'embedding_inference_seed': getattr(
            args, 'embedding_inference_seed', args.seed
        ),
    }
    serialized = json.dumps(fields, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()[:20]


def make_sgrl_loader(args, graph, num_layers, shuffle):
    return NeighborLoader(
        graph,
        num_neighbors=[args.cl_num_neighbors] * num_layers,
        batch_size=args.cl_batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )


def build_sgrl_training_state(args, device):
    """Construct SGRL models and non-overlapping optimizer parameter sets."""
    online_conv = CustomConv(args).to(device)
    target_conv = CustomConv(args).to(device)
    target_update = args.sgrl_pretrain_target_update
    if target_update == 'circuitgcl_text_ema_only':
        target_conv.load_state_dict(online_conv.state_dict())

    online_model = CustomOnline(
        online_conv,
        target_conv,
        args.cl_hid_dim,
        args.num_hops,
        args.momentum,
    ).to(device)
    target_model = Target(target_conv).to(device)
    online_parameters = list(online_model.online_encoder.parameters()) + list(
        online_model.predictor.parameters()
    )
    online_optimizer = torch.optim.Adam(
        online_parameters,
        lr=args.e1_lr,
        weight_decay=args.weight_decay,
    )
    target_optimizer = None
    if target_update == 'sgrl_dual_rsm_ema':
        target_optimizer = torch.optim.Adam(
            target_model.parameters(),
            lr=args.e2_lr,
            weight_decay=args.weight_decay,
        )
    return online_model, target_model, online_optimizer, target_optimizer

def train_online_encoder(online: CustomOnline, optimizer, loader, graph_adj, device):
    tot_loss = 0.0
    num_loss = 0
    online.train()

    for batch in tqdm(iterable=loader, desc='Online batches'):
        optimizer.zero_grad()
        batch = batch.to(device)
        h,h_pred,h_target = online(batch)
        loss = online.get_loss(h_pred, h_target.detach())
        loss.backward()
        tot_loss += loss.item()
        num_loss += 1
        optimizer.step()

    online.update_target_encoder()
    return tot_loss / num_loss

def train_target_encoder(target: Target, optimizer, loader, device):
    tot_loss = 0.0
    num_loss = 0
    target.train()

    for batch in tqdm(iterable=loader, desc='Target batches', leave=False):
        optimizer.zero_grad()
        batch = batch.to(device)
        h_target = target(batch)
        loss = target.get_loss(h_target)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
        num_loss += 1

    return tot_loss / num_loss

def adj_norm(data):
    nb_nodes = data.x.size(0)
    self_loop_for_adj = torch.Tensor([i for i in range(nb_nodes)]).unsqueeze(0)
    self_loop_for_adj = torch.concat([self_loop_for_adj, self_loop_for_adj], dim=0)
    
    slsp_adj = torch.concat([data.edge_index.cpu(), self_loop_for_adj], dim=1)
    slsp_adj = torch.sparse_coo_tensor(slsp_adj.long(), torch.ones(slsp_adj.size()[1]),
                                        torch.Size([nb_nodes, nb_nodes]))
    adj_t = slsp_adj
    deg = torch.sparse.sum(adj_t, dim=1).to_dense()
    deg_inv_sqrt = deg.pow(-0.5)
    adj_t = adj_t * deg_inv_sqrt.view(1, -1)
    adj_t = adj_t * deg_inv_sqrt.view(-1, 1)
    return adj_t

def get_all_contrastive_embed(
        online_model, pkl_path, 
        train_graph, loader, 
        hidden_dim, num_hop, device,
        inference_seed=0, return_sampler_fingerprint=False,
    ):
    """
    Get all node embeddings from the online encoder of SGRL model.
    Args:
        online_model (CustomOnline): The online encoder of SGRL model
        pkl_path (str): The path to the state_dict file of the online encoder
        train_graph (torch_geometric.data.Batch): The training graph
        loader (torch_geometric.loader.NeighborLoader): The loader for the dataset
        hidden_dim (int): The hidden dimension of the model
        num_hop (int): The number of hops
        device (torch.device): The device
    Returns:
        torch.Tensor: The embeddings of all nodes in the graph
    """
    with fixed_rng(inference_seed):
        print(f"Loading model from {pkl_path}")
        online_model.load_state_dict(torch.load(
            pkl_path, map_location=device, weights_only=True
        ))
        online_model.eval()

        # Initialize all CL embeddings
        embeds = torch.zeros(
            (train_graph.num_nodes, hidden_dim), requires_grad=False
        )
        sampler_fingerprint = SamplerFingerprint()

        for batch in tqdm(
                iterable=loader, desc='Getting CL embeddings', leave=False):
            sampler_fingerprint.update(batch)
            # slsp_adj = graph_adj.index_select(0, batch.n_id)
            # batch.slsp_adj = slsp_adj.index_select(1, batch.n_id)
            # assert batch.input_id.size(0) == batch.batch_size, \
            #   f"input_id size {batch.input_id.size(0)} != batch_size {batch.batch_size}"
            batch = batch.to(device)

            ## We only record the embeddings of the sampled (specified by batch.input_id)
            ## nodes in this batch, even though the model return all neighbors' embeddings in the batch.
            ## The first batch_size embeddings are the embeddings of the sampled nodes.
            ## See docs about pyg loader.
            embeds[batch.input_id] = online_model.embed(
                batch, num_hop
            ).detach().cpu()[:batch.input_id.size(0)]

        ## Assign the embeddings to the batched big training graph
        # train_graph.cl_embed = embeds
        online_model = online_model.cpu()

    # ## We slice node embeds in the large `train_graph` 
    # ## and map them back to the corresponding each dataset
    # cl_embeds_for_dataset = [
    #     embeds[ train_graph.ptr[i] : train_graph.ptr[i+1] ] 
    #     for i in range(train_graph.num_graphs)
    # ]

    # return cl_embeds_for_dataset
    if return_sampler_fingerprint:
        return embeds, sampler_fingerprint.hexdigest()
    return embeds

def sgrl_train(args, dataset, device, return_embeddings=True, return_online_state=False):
    """
    Training SGRL model.
    Args:
        args (argparse.Namespace): The arguments for SGRL
        dataset (torch_geometric.data.InMemoryDataset): The dataset
        device (torch.device): The device
    Returns:
        torch.Tensor or dict: By default, the embeddings of all nodes in dataset.
        If return_embeddings is False and return_online_state is True, returns the
        online encoder checkpoint path and state_dict for downstream reuse.
    """
    seed_all(getattr(args, 'pretraining_seed', args.seed))
    e1_lr = args.e1_lr
    e2_lr = args.e2_lr
    weight_decay = args.weight_decay
    hidden_dim = args.cl_hid_dim
    activation = args.cl_act_fn
    num_layers = args.cl_gnn_layers
    num_epochs = args.cl_epochs
    dropout = args.cl_dropout
    momentum = args.momentum
    graph_scope = getattr(args, 'sgrl_graph_scope', 'all')
    train_graph_indices = (
        [0] if graph_scope == 'source' else list(range(len(dataset.names)))
    )
    train_graph_names = [dataset.names[index] for index in train_graph_indices]
    train_graph = adaption_for_sgrl(dataset, train_graph_indices)
    train_adj = adj_norm(train_graph)

    #========== model construction ==========#
    num_hop = args.num_hops
    target_update = args.sgrl_pretrain_target_update
    (
        online_model,
        target_model,
        online_optimizer,
        target_optimizer,
    ) = build_sgrl_training_state(args, device)

    best_online_loss = 1e9
    best_target_loss = 1e9

    #========== contrastive learning ==========#
    train_graph_loader = make_sgrl_loader(
        args, train_graph, num_layers, shuffle=True
    )
    # kwargs = {
    # 'batch_size': batch_size, 'shuffle': True, 'num_workers': 8, 
    # 'drop_last': True, 'pin_memory': True
    # }
    # train_graph_loader = ShaDowKHopSampler(
    # data=train_graph, depth=num_layers, 
    # num_neighbors=32, node_idx=None, **kwargs)

    tag = str(time.time())
    model_name = ""
    best_epoch = 0
    cnt_wait = 0

    cache_key = sgrl_cache_fingerprint(args, train_graph_names)
    model_name = os.path.join(
        'pkl',
        'pkl_online',
        f'best_online_{cache_key}.pkl',
    )
    target_model_name = model_name.replace('online', 'target')
    os.makedirs(os.path.dirname(model_name), exist_ok=True)
    os.makedirs(os.path.dirname(target_model_name), exist_ok=True)
  

    if not os.path.exists(model_name):
        print(f"Training SGRL model with name {model_name}...")

        for epoch in range(num_epochs):
            online_optimizer.zero_grad()
            if target_optimizer is not None:
                target_optimizer.zero_grad()
            
            # online_loss = train_online(online_model,online_optimizer,data)  
            # online_loss = train_customonline(online_model, online_optimizer, train_graph)        
            online_loss = train_online_encoder(
                online_model, online_optimizer, 
                train_graph_loader, train_adj,  device)        
            if online_loss < best_online_loss:
                best_online_loss = online_loss
                best_epoch = epoch
                torch.save(online_model.state_dict(), model_name)
                cnt_wait = 0
            
            target_loss = None
            if target_optimizer is not None:
                target_loss = train_target_encoder(
                    target_model, target_optimizer, train_graph_loader, device
                )
                if target_loss < best_target_loss:
                    best_target_loss = target_loss
                    torch.save(target_model.state_dict(), target_model_name)

            target_loss_text = (
                f'{target_loss:.6f}' if target_loss is not None else 'ema_only'
            )
            print(
                f"Epoch:{epoch} online_loss={online_loss:.6f} "
                f"target_loss={target_loss_text}"
            )

            losses_converged = (
                online_loss < -0.99
                and (target_loss is None or target_loss < -0.99)
            )
            if losses_converged or cnt_wait == 20:
                print("Do early stop")
                break
            else:
                cnt_wait += 1

    if return_online_state:
        online_model.load_state_dict(torch.load(
            model_name, map_location=device, weights_only=True
        ))
        if not return_embeddings:
            return {
                'online_model_path': model_name,
                'online_state_dict': state_dict_to_cpu(online_model.state_dict()),
                'cache_key': cache_key,
                'train_graph_names': train_graph_names,
                'target_update': target_update,
            }

    #========== get all node embeddings learnt by SGRL ==========#
    inference_graph = adaption_for_sgrl(
        dataset,
        list(range(len(dataset.names))),
    )
    inference_loader = make_sgrl_loader(
        args, inference_graph, num_layers, shuffle=False
    )
    embeds, embedding_sampler_fingerprint = get_all_contrastive_embed(
        online_model, model_name, inference_graph,
        inference_loader, hidden_dim, num_hop, device,
        inference_seed=getattr(args, 'embedding_inference_seed', args.seed),
        return_sampler_fingerprint=True,
    )

    if return_online_state:
        return {
            'embeddings': embeds,
            'online_model_path': model_name,
            'online_state_dict': state_dict_to_cpu(online_model.state_dict()),
            'cache_key': cache_key,
            'train_graph_names': train_graph_names,
            'target_update': target_update,
            'embedding_sampler_fingerprint': embedding_sampler_fingerprint,
        }

    return embeds

# if __name__ == '__main__':
#     # warnings.filterwarnings("ignore")
#     parser = argparse.ArgumentParser('SGRL')
#     parser.add_argument('--dataset_name', type=str, default='Photo', help='dataset_name')
#     parser.add_argument('--data_dir', type=str, default='../../datasets', help='data_dir')
#     parser.add_argument('--log_dir', type=str, default='./log/log_Photo', help='log_dir')
#     parser.add_argument('--e1_lr', type=float, default=0.001, help='online_learning_rate')
#     parser.add_argument('--e2_lr', type=float, default=0.001, help='target_learning_rate')
#     parser.add_argument('--momentum', type=float, default=0.99, help='EMA')
#     parser.add_argument('--weight_decay', type=float, default=0., help='weight_decay')
#     parser.add_argument('--num_epochs', type=int, default=700, help='num_epochs')
#     parser.add_argument('--seed', type=int, default=66666, help='seed')
#     parser.add_argument('--hidden_dim', type=int, default=1024, help='hidden_dim')
#     parser.add_argument('--num_layers', type=int, default=1, help='num_layers')
#     parser.add_argument('--num_hop', type=int, default=1, help='num_hop')
#     parser.add_argument('--trials', type=int, default=20, help='trials')
#     args = parser.parse_args()  
#     contrastive_train(args)
