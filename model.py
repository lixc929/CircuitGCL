import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from torch_geometric.nn import (
    GCNConv, SAGEConv, GATConv, ResGatedGraphConv, 
    GINEConv, ClusterGCNConv
)
from torch_geometric.nn.models.mlp import MLP
from torch_geometric.data import Data
from sgrl_models import CustomConv

NET = 0
DEV = 1
PIN = 2


def extract_sgrl_encoder_state(online_state_dict):
    prefix = 'online_encoder.'
    if any(key.startswith(prefix) for key in online_state_dict):
        return {
            key[len(prefix):]: value
            for key, value in online_state_dict.items()
            if key.startswith(prefix)
        }
    return online_state_dict


def shape_str(tensor):
    return 'x'.join(str(dim) for dim in tensor.shape)


def build_gnn_layer(model_name, in_dim, out_dim, edge_dim, activation='relu'):
    if model_name == 'clustergcn':
        return ClusterGCNConv(in_dim, out_dim)
    if model_name == 'gcn':
        return GCNConv(in_dim, out_dim)
    if model_name == 'sage':
        return SAGEConv(in_dim, out_dim)
    if model_name == 'gat':
        return GATConv(in_dim, out_dim, heads=1)
    if model_name == 'resgatedgcn':
        return ResGatedGraphConv(in_dim, out_dim, edge_dim=edge_dim)
    if model_name == 'gine':
        mlp = MLP(
            in_channels=in_dim,
            hidden_channels=out_dim,
            out_channels=out_dim,
            num_layers=2,
            norm=None,
            activation=activation,
        )
        return GINEConv(mlp, train_eps=True, edge_dim=edge_dim)
    raise ValueError(f'Unsupported GNN model: {model_name}')


class MergeableLoRALinear(nn.Module):
    """Low-rank linear update that can be folded into its base weight."""

    def __init__(self, base, rank, alpha=None):
        super().__init__()
        if rank <= 0:
            raise ValueError('LoRA rank must be positive.')
        if not hasattr(base, 'weight'):
            raise TypeError('MergeableLoRALinear requires a linear base module.')

        self.base = base
        self.rank = rank
        self.alpha = float(rank if alpha is None else alpha)
        self.scaling = self.alpha / self.rank
        in_features = getattr(base, 'in_features', None)
        if in_features is None:
            in_features = base.in_channels
        out_features = getattr(base, 'out_features', None)
        if out_features is None:
            out_features = base.out_channels
        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_b.weight)
        self.adapter_enabled = True
        self.merged = False

    def forward(self, x):
        output = self.base(x)
        if self.adapter_enabled and not self.merged:
            output = output + self.scaling * self.lora_b(self.lora_a(x))
        return output

    def adapter_parameter_count(self):
        return self.lora_a.weight.numel() + self.lora_b.weight.numel()

    @torch.no_grad()
    def merge_and_unwrap(self):
        if not self.merged:
            delta = self.scaling * (
                self.lora_b.weight @ self.lora_a.weight
            )
            self.base.weight.add_(delta.to(self.base.weight.dtype))
            self.merged = True
        return self.base


class GraphHead(nn.Module):
    """ GNN head for graph-level prediction.

    Implementation adapted from the transductive GraphGPS.

    Args:
        hidden_dim (int): Hidden features' dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
        num_layers (int): Number of layers of GNN model
        layers_post_mp (int): number of layers of head MLP
        use_bn (bool): whether to use batch normalization
        drop_out (float): dropout rate
        activation (str): activation function
        src_dst_agg (str): the way to aggregate src and dst nodes, which can be 'concat' or 'add' or 'pool'
    """
    def __init__(self, args):
        super().__init__()
        self.use_cl = bool(getattr(args, 'use_graph_cl_features', args.sgrl))
        self.use_stats = args.use_stats
        hidden_dim = args.hid_dim
        node_embed_dim = hidden_dim
        self.task = args.task
        self.task_level = args.task_level
        self.net_only = args.net_only
        self.num_classes = args.num_classes
        self.class_boundaries = args.class_boundaries

        
        ## circuit statistics encoder + PE encoder + node&edge type encoders
        if args.use_stats + self.use_cl == 2:
            assert hidden_dim % 3 == 0, \
                "hidden_dim should be divided by 3 (3 types of encoders)"
            node_embed_dim = hidden_dim // 3

        ## circuit statistics/pe encoder + node&edge type encoders
        elif self.use_stats + self.use_cl == 1:
            assert hidden_dim % 2 == 0, \
                "hidden_dim should be divided by 2 (2 types of encoders)"
            node_embed_dim = hidden_dim // 2

        ## only use node&edge type encoders
        else:
            pass

        ## Contrastive learning encoder
        if self.use_cl:
            self.cl_linear = nn.Linear(args.cl_hid_dim, node_embed_dim)

        ## Circuit Statistics encoder, producing matrix C
        if self.use_stats:
            ## add node_attr transform layer for net/device/pin nodes, by shan
            self.net_attr_layers = nn.Linear(17, node_embed_dim, bias=True)
            self.dev_attr_layers = nn.Linear(17, node_embed_dim, bias=True)
            ## pin attributes are {0, 1, 2} for gate pin, source/drain pin, and base pin
            self.pin_attr_layers = nn.Embedding(17, node_embed_dim)
            self.c_embed_dim = node_embed_dim

        ## Node / Edge type encoders.
        ## Node attributes are {0, 1, 2} for net, device, and pin
        self.node_encoder = nn.Embedding(num_embeddings=4,
                                         embedding_dim=node_embed_dim)
        ## Edge attributes are {0, 1} for 'device-pin' and 'pin-net' edges
        self.edge_encoder = nn.Embedding(num_embeddings=4,
                                         embedding_dim=hidden_dim)
        
        # GNN layers
        self.layers = nn.ModuleList()
        self.model = args.model

        for _ in range(args.num_gnn_layers):
            ## the following are examples of using different GNN layers
            if args.model == 'clustergcn':
                self.layers.append(ClusterGCNConv(hidden_dim, hidden_dim))
            elif args.model == 'gcn':
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
            elif args.model == 'sage':
                self.layers.append(SAGEConv(hidden_dim, hidden_dim))
            elif args.model == 'gat':
                self.layers.append(GATConv(hidden_dim, hidden_dim, heads=1))
            elif args.model == 'resgatedgcn':
                self.layers.append(ResGatedGraphConv(hidden_dim, hidden_dim, edge_dim=hidden_dim))
            elif args.model == 'gine':
                mlp = MLP(
                    in_channels=hidden_dim, 
                    hidden_channels=hidden_dim, 
                    out_channels=hidden_dim, 
                    num_layers=2, 
                    norm=None,
                )
                self.layers.append(GINEConv(mlp, train_eps=True, edge_dim=hidden_dim))
            else:
                raise ValueError(f'Unsupported GNN model: {args.model}')
        
        self.src_dst_agg = args.src_dst_agg

        ## Add graph pooling layer
        if args.src_dst_agg == 'pooladd':
            self.pooling_fun = pygnn.pool.global_add_pool
        elif args.src_dst_agg == 'poolmean':
            self.pooling_fun = pygnn.pool.global_mean_pool
        
        ## The head configuration
        head_input_dim = hidden_dim * 2 if self.src_dst_agg == 'concat'  and self.task_level == 'edge' else hidden_dim

        if self.task == 'regression':
            dim_out = 1
        elif self.task =='classification':
            dim_out = args.num_classes
        else:
            raise ValueError('Invalid task')
        
        # head MLP layers
        self.head_layers = MLP(
            in_channels=head_input_dim, 
            hidden_channels=hidden_dim, 
            out_channels=dim_out, 
            num_layers=args.num_head_layers, 
            use_bn=False, dropout=0.0, 
            activation=args.act_fn,
        )

        ## Batch normalization
        self.use_bn = args.use_bn
        self.bn_node_x = nn.BatchNorm1d(hidden_dim)
        if self.use_bn and self.use_cl:
            print("[Warning] Using batch normalization with contrastive learning may cause performance degradation.")

        ## activation setting
        if args.act_fn == 'relu':
            self.activation = nn.ReLU()
        elif args.act_fn == 'elu':
            self.activation = nn.ELU()
        elif args.act_fn == 'tanh':
            self.activation = nn.Tanh()
        elif args.act_fn == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif args.act_fn == 'prelu':
            self.activation = nn.PReLU()
        else:
            raise ValueError('Invalid activation')
        
        ## Dropout setting
        self.drop_out = args.dropout

    def load_sgrl_encoder_init(self, online_state_dict):
        """Initialize compatible GraphHead tensors from a pretrained SGRL encoder."""
        encoder_state = extract_sgrl_encoder_state(online_state_dict)
        target_state = self.state_dict()
        copied = []
        skipped = []
        consumed_sources = set()

        def copy_tensor(source_key, target_key, allow_prefix_rows=False):
            if source_key not in encoder_state:
                skipped.append((source_key, target_key, 'missing source'))
                return
            if target_key not in target_state:
                skipped.append((source_key, target_key, 'missing target'))
                consumed_sources.add(source_key)
                return

            source_tensor = encoder_state[source_key]
            target_tensor = target_state[target_key]
            source_shape = tuple(source_tensor.shape)
            target_shape = tuple(target_tensor.shape)
            consumed_sources.add(source_key)

            with torch.no_grad():
                if source_shape == target_shape:
                    target_tensor.copy_(source_tensor.to(target_tensor.device))
                    copied.append((source_key, target_key, 'full', target_tensor.numel()))
                    return

                if (
                    allow_prefix_rows
                    and source_tensor.ndim == 2
                    and target_tensor.ndim == 2
                    and target_shape[0] <= source_shape[0]
                    and target_shape[1] == source_shape[1]
                ):
                    target_tensor.copy_(
                        source_tensor[:target_shape[0]].to(target_tensor.device)
                    )
                    copied.append((source_key, target_key, 'prefix_rows', target_tensor.numel()))
                    return

            skipped.append((
                source_key,
                target_key,
                f'shape {shape_str(source_tensor)} -> {shape_str(target_tensor)}',
            ))

        copy_tensor('node_type_embed.weight', 'node_encoder.weight', allow_prefix_rows=True)
        copy_tensor('edge_type_embed.weight', 'edge_encoder.weight', allow_prefix_rows=True)

        layer_prefixes = sorted({
            key.split('.')[1]
            for key in encoder_state
            if key.startswith('layers.') and key.count('.') >= 2
        }, key=int)
        for layer_idx in layer_prefixes[:len(self.layers)]:
            source_prefix = f'layers.{layer_idx}.'
            for source_key in sorted(encoder_state):
                if not source_key.startswith(source_prefix):
                    continue
                target_key = source_key
                copy_tensor(source_key, target_key)

        for source_key in ['bn_node_x.weight', 'bn_node_x.bias',
                           'bn_node_x.running_mean', 'bn_node_x.running_var',
                           'bn_node_x.num_batches_tracked',
                           'activation.weight']:
            copy_tensor(source_key, source_key)

        for source_key in sorted(encoder_state):
            if source_key in consumed_sources:
                continue
            if source_key.startswith('projection_head.'):
                skipped.append((source_key, '-', 'projection head not used downstream'))
            elif source_key.startswith('layers.'):
                skipped.append((source_key, source_key, 'no matching downstream layer'))
            else:
                skipped.append((source_key, '-', 'no mapping'))

        copied_tensors = len(copied)
        copied_values = sum(item[3] for item in copied)
        print(
            "SGRL GraphHead init reuse summary: "
            f"copied_tensors={copied_tensors}, copied_values={copied_values}, "
            f"skipped={len(skipped)}."
        )
        if copied:
            preview = '; '.join(
                f"{src}->{dst}({mode})" for src, dst, mode, _ in copied[:8]
            )
            print(f"Copied preview: {preview}")
        if skipped:
            preview = '; '.join(
                f"{src}->{dst}({reason})" for src, dst, reason in skipped[:8]
            )
            print(f"Skipped preview: {preview}")
        if copied_tensors == 0:
            print(
                "[Warning] No SGRL tensors were compatible with GraphHead. "
                "For S4 init reuse, align --model/--cl_model and usually set "
                "--hid_dim close to --cl_hid_dim."
            )
    

    def forward(self, batch, cl_x=None):
        ## Node type / Edge type encoding
        x = self.node_encoder(batch.node_type)
        xe = self.edge_encoder(batch.edge_type)

        ## Contrastive learning encoder
        if self.use_cl:
            if cl_x is None:
                cl_x = batch.x
            xcl = self.cl_linear(cl_x)
            ## concatenate node embeddings and embeddings learned by SGRL
            x = torch.cat((x, xcl), dim=1)

        
        ## If we use circuit statistics encoder
        if self.use_stats:
            net_node_mask = batch.node_type == NET
            dev_node_mask = batch.node_type == DEV
            pin_node_mask = batch.node_type == PIN
            ## circuit statistics embeddings (C in EQ.6)
            node_attr_emb = torch.zeros(
                (batch.num_nodes, self.c_embed_dim), device=batch.x.device
            )
            node_attr_emb[net_node_mask] = \
                self.net_attr_layers(batch.node_attr[net_node_mask])
            node_attr_emb[dev_node_mask] = \
                self.dev_attr_layers(batch.node_attr[dev_node_mask])
            node_attr_emb[pin_node_mask] = \
                self.pin_attr_layers(batch.node_attr[pin_node_mask, 0].long())
            ## concatenate node embeddings and circuit statistics embeddings (C in EQ.6)
            x = torch.cat((x, node_attr_emb), dim=1)

        # GNN layers
        for conv in self.layers:
            ## for models that also take edge_attr as input
            if self.model == 'gine' or self.model == 'resgatedgcn':
                x = conv(x, batch.edge_index, edge_attr=xe)
            else:
                x = conv(x, batch.edge_index)

            if self.use_bn:
                x = self.bn_node_x(x)
            
            x = self.activation(x)

            if self.drop_out > 0.0:
                x = F.dropout(x, p=self.drop_out, training=self.training)

        ## task level : node
        if self.task_level == 'node':
            if self.net_only:
                net_node_mask = batch.node_type == NET
                pred = self.head_layers(x[net_node_mask])
                true_class = batch.y[:, 1][net_node_mask].long()
                true_label = batch.y[net_node_mask]
            else:
                pred = self.head_layers(x)
                true_class = batch.y[:, 1].long()
                true_label = batch.y

        elif self.task_level == 'edge':
            if self.src_dst_agg[:4] == 'pool':
                graph_emb = self.pooling_fun(x, batch.batch)
            ## Otherwise, only 2 embeddings from the anchor nodes are used to final prediction.
            else:
                batch_size = batch.edge_label.size(0)
                ## In the LinkNeighbor loader, the first batch_size nodes in x are source nodes and,
                ## the second 'batch_size' nodes in x are destination nodes. 
                ## Remaining nodes are their '1-hop', '2-hop', 'n-hop' neighbors.
                src_emb = x[:batch_size, :]
                dst_emb = x[batch_size:batch_size*2, :]
                if self.src_dst_agg == 'concat':
                    graph_emb = torch.cat((src_emb, dst_emb), dim=1)
                else:
                    graph_emb = src_emb + dst_emb

            pred = self.head_layers(graph_emb)
            true_class = batch.edge_label[:, 1].long()
            true_label = batch.edge_label
        
        else:
            raise ValueError('Invalid task level')
            
        return pred,true_class,true_label


class OnlineFeatureGraphHead(nn.Module):
    """Feed online SGRL features into the original downstream GraphHead.

    This is the conservative S2 reuse path: the GCL online encoder is used as
    a frozen/eval feature extractor per downstream batch, while the downstream
    GNN layers and prediction head remain the original GraphHead.
    """

    def __init__(self, args):
        super().__init__()
        self.online_encoder = CustomConv(args)
        self.graph_head = GraphHead(args)
        self.online_encoder_frozen = False

    def load_online_encoder_state(self, online_state_dict, freeze=True):
        prefix = 'online_encoder.'
        if any(key.startswith(prefix) for key in online_state_dict):
            encoder_state = {
                key[len(prefix):]: value
                for key, value in online_state_dict.items()
                if key.startswith(prefix)
            }
        else:
            encoder_state = online_state_dict

        load_msg = self.online_encoder.load_state_dict(encoder_state, strict=False)
        print(
            "Loaded SGRL online encoder for online feature reuse "
            f"(missing={len(load_msg.missing_keys)}, "
            f"unexpected={len(load_msg.unexpected_keys)}, freeze={freeze})."
        )

        if freeze:
            self.freeze_online_encoder()

    def freeze_online_encoder(self):
        self.online_encoder_frozen = True
        self.online_encoder.eval()
        for param in self.online_encoder.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.online_encoder_frozen:
            self.online_encoder.eval()
        return self

    def _make_sgrl_batch(self, batch):
        return Data(
            x=batch.node_type.view(-1, 1),
            edge_index=batch.edge_index,
            num_nodes=batch.num_nodes,
        )

    def _online_features(self, batch):
        sgrl_batch = self._make_sgrl_batch(batch)
        if self.online_encoder_frozen:
            with torch.no_grad():
                return self.online_encoder.encode(sgrl_batch)
        return self.online_encoder.encode(sgrl_batch)

    def forward(self, batch):
        cl_x = self._online_features(batch)
        return self.graph_head(batch, cl_x=cl_x)


class SharedGNNBackbone(nn.Module):
    """Lower SGRL online encoder layers reused as a shared downstream backbone."""

    def __init__(self, args, num_layers=None):
        super().__init__()
        self.hidden_dim = args.cl_hid_dim
        self.shared_layers = (
            args.shared_gnn_layers if num_layers is None else num_layers
        )
        self.model = args.cl_model

        if self.shared_layers < 0:
            raise ValueError("--shared_gnn_layers must be non-negative.")
        if self.shared_layers > args.cl_gnn_layers:
            raise ValueError(
                "--shared_gnn_layers cannot exceed --cl_gnn_layers "
                f"({self.shared_layers} > {args.cl_gnn_layers})."
            )

        self.node_type_embed = nn.Embedding(6, self.hidden_dim)
        self.edge_type_embed = nn.Embedding(8, self.hidden_dim)
        self.layers = nn.ModuleList([
            build_gnn_layer(
                args.cl_model,
                self.hidden_dim,
                self.hidden_dim,
                self.hidden_dim,
                activation=args.cl_act_fn,
            )
            for _ in range(self.shared_layers)
        ])

        self.use_bn = args.use_bn
        self.bn_node_x = nn.BatchNorm1d(self.hidden_dim)
        if args.cl_act_fn == 'relu':
            self.activation = nn.ReLU()
        elif args.cl_act_fn == 'elu':
            self.activation = nn.ELU()
        elif args.cl_act_fn == 'tanh':
            self.activation = nn.Tanh()
        elif args.cl_act_fn == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif args.cl_act_fn == 'prelu':
            self.activation = nn.PReLU()
        else:
            raise ValueError('Invalid activation')
        self.drop_out = args.cl_dropout

    def load_online_encoder_state(self, online_state_dict):
        encoder_state = extract_sgrl_encoder_state(online_state_dict)
        target_state = self.state_dict()
        copied = []
        skipped = []

        def copy_tensor(source_key, target_key=None):
            target_key = target_key or source_key
            if source_key not in encoder_state:
                skipped.append((source_key, target_key, 'missing source'))
                return
            if target_key not in target_state:
                skipped.append((source_key, target_key, 'missing target'))
                return

            source_tensor = encoder_state[source_key]
            target_tensor = target_state[target_key]
            if tuple(source_tensor.shape) != tuple(target_tensor.shape):
                skipped.append((
                    source_key,
                    target_key,
                    f'shape {shape_str(source_tensor)} -> {shape_str(target_tensor)}',
                ))
                return

            with torch.no_grad():
                target_tensor.copy_(
                    source_tensor.to(
                        device=target_tensor.device,
                        dtype=target_tensor.dtype,
                    )
                )
            copied.append((source_key, target_key, target_tensor.numel()))

        copy_tensor('node_type_embed.weight')
        copy_tensor('edge_type_embed.weight')

        for layer_idx in range(self.shared_layers):
            source_prefix = f'layers.{layer_idx}.'
            for source_key in sorted(encoder_state):
                if source_key.startswith(source_prefix):
                    copy_tensor(source_key)

        for source_key in [
            'bn_node_x.weight',
            'bn_node_x.bias',
            'bn_node_x.running_mean',
            'bn_node_x.running_var',
            'bn_node_x.num_batches_tracked',
            'activation.weight',
        ]:
            copy_tensor(source_key)

        print(
            "Partial shared backbone load summary: "
            f"shared_layers={self.shared_layers}, "
            f"copied_tensors={len(copied)}, "
            f"copied_values={sum(item[2] for item in copied)}, "
            f"skipped={len(skipped)}."
        )
        if copied:
            preview = '; '.join(
                f"{src}->{dst}" for src, dst, _ in copied[:8]
            )
            print(f"Partial shared copied preview: {preview}")
        if skipped:
            preview = '; '.join(
                f"{src}->{dst}({reason})" for src, dst, reason in skipped[:8]
            )
            print(f"Partial shared skipped preview: {preview}")

    def encode_inputs(self, batch):
        node_type = batch.node_type.view(-1).long()
        x = self.node_type_embed(node_type)

        edge_attr = None
        if hasattr(batch, 'edge_type'):
            edge_type = batch.edge_type.view(-1).long()
            edge_attr = self.edge_type_embed(edge_type)

        return x, edge_attr

    def propagate(self, x, edge_attr, batch):

        for conv in self.layers:
            if self.model == 'gine' or self.model == 'resgatedgcn':
                x = conv(x, batch.edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, batch.edge_index)

            if self.use_bn:
                x = self.bn_node_x(x)

            x = self.activation(x)

            if self.drop_out > 0.0:
                x = F.dropout(x, p=self.drop_out, training=self.training)

        return x

    def forward(self, batch):
        x, edge_attr = self.encode_inputs(batch)
        return self.propagate(x, edge_attr, batch)


class JointSharedEncoder(nn.Module):
    """Two-layer deployment encoder shared by GCL and supervision."""

    def __init__(self, args):
        super().__init__()
        self.hidden_dim = args.cl_hid_dim
        self.gnn = SharedGNNBackbone(
            args,
            num_layers=args.joint_shared_gnn_layers,
        )
        self.use_stats = bool(args.use_stats)
        if self.use_stats:
            self.net_attr_layers = nn.Linear(17, self.hidden_dim, bias=True)
            self.dev_attr_layers = nn.Linear(17, self.hidden_dim, bias=True)
            self.pin_attr_layers = nn.Embedding(17, self.hidden_dim)
            self.stats_residual_scale = nn.Parameter(torch.zeros(1))
        self.lora_rank = getattr(args, 'joint_lora_rank', 0)
        self.lora_alpha = getattr(args, 'joint_lora_alpha', None)
        self.lora_layer = getattr(args, 'joint_lora_layer', -1)
        self.lora_attached = False

    def load_online_encoder_state(self, online_state_dict):
        self.gnn.load_online_encoder_state(online_state_dict)
        if self.lora_rank > 0:
            self.attach_task_lora()

    def _resolved_lora_layer(self):
        layer_count = len(self.gnn.layers)
        layer = self.lora_layer
        if layer < 0:
            layer += layer_count
        if layer < 0 or layer >= layer_count:
            raise ValueError(
                f'joint_lora_layer {self.lora_layer} is out of range for '
                f'{layer_count} GNN layers.'
            )
        return layer

    def attach_task_lora(self):
        if self.lora_attached:
            return
        if self.gnn.model != 'clustergcn':
            raise ValueError(
                'Mergeable joint LoRA currently supports clustergcn only.'
            )
        layer = self.gnn.layers[self._resolved_lora_layer()]
        layer.lin_out = MergeableLoRALinear(
            layer.lin_out,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
        )
        layer.lin_root = MergeableLoRALinear(
            layer.lin_root,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
        )
        self.lora_attached = True
        print(
            'Attached mergeable task LoRA to joint shared backbone '
            f'(layer={self._resolved_lora_layer()}, rank={self.lora_rank}, '
            f'alpha={layer.lin_out.alpha}).'
        )

    def _lora_modules(self):
        return [
            module
            for module in self.modules()
            if isinstance(module, MergeableLoRALinear)
        ]

    def set_task_lora_enabled(self, enabled):
        for module in self._lora_modules():
            module.adapter_enabled = enabled

    def lora_parameter_count(self):
        return sum(
            module.adapter_parameter_count()
            for module in self._lora_modules()
        )

    @torch.no_grad()
    def merge_task_lora(self):
        if not self.lora_attached:
            return 0
        layer = self.gnn.layers[self._resolved_lora_layer()]
        merged_values = self.lora_parameter_count()
        layer.lin_out = layer.lin_out.merge_and_unwrap()
        layer.lin_root = layer.lin_root.merge_and_unwrap()
        self.lora_attached = False
        return merged_values

    def _encode_stats(self, batch):
        node_type = batch.node_type.view(-1)
        net_node_mask = node_type == NET
        dev_node_mask = node_type == DEV
        pin_node_mask = node_type == PIN
        stats_x = torch.zeros(
            (batch.num_nodes, self.hidden_dim),
            device=batch.node_attr.device,
        )
        stats_x[net_node_mask] = self.net_attr_layers(
            batch.node_attr[net_node_mask]
        )
        stats_x[dev_node_mask] = self.dev_attr_layers(
            batch.node_attr[dev_node_mask]
        )
        stats_x[pin_node_mask] = self.pin_attr_layers(
            batch.node_attr[pin_node_mask, 0].long()
        )
        return stats_x

    def forward(self, batch, task_path=True):
        x, edge_attr = self.gnn.encode_inputs(batch)
        if self.use_stats and task_path:
            x = x + self.stats_residual_scale * self._encode_stats(batch)
        self.set_task_lora_enabled(task_path)
        try:
            return self.gnn.propagate(x, edge_attr, batch)
        finally:
            self.set_task_lora_enabled(True)


class JointSharedGraphHead(nn.Module):
    """One deployment backbone optimized by supervised and GCL losses."""

    def __init__(self, args):
        super().__init__()
        self.shared_backbone = JointSharedEncoder(args)
        self.target_backbone = copy.deepcopy(self.shared_backbone)
        for parameter in self.target_backbone.parameters():
            parameter.requires_grad = False
        self.target_backbone.eval()

        hidden_dim = args.cl_hid_dim
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.target_momentum = args.momentum
        self.separate_gcl_base_path = bool(
            getattr(args, 'joint_lora_rank', 0) > 0
        )
        self.task = args.task
        self.task_level = args.task_level
        self.net_only = args.net_only
        self.src_dst_agg = args.src_dst_agg
        self.num_classes = args.num_classes
        self.class_boundaries = args.class_boundaries

        if self.src_dst_agg == 'pooladd':
            self.pooling_fun = pygnn.pool.global_add_pool
        elif self.src_dst_agg == 'poolmean':
            self.pooling_fun = pygnn.pool.global_mean_pool

        head_input_dim = (
            hidden_dim * 2
            if self.src_dst_agg == 'concat' and self.task_level == 'edge'
            else hidden_dim
        )
        dim_out = 1 if self.task == 'regression' else args.num_classes
        self.head_layers = MLP(
            in_channels=head_input_dim,
            hidden_channels=hidden_dim,
            out_channels=dim_out,
            num_layers=args.num_head_layers,
            use_bn=False,
            dropout=0.0,
            activation=args.act_fn,
        )

    def train(self, mode=True):
        super().train(mode)
        self.target_backbone.eval()
        return self

    def load_online_encoder_state(self, online_state_dict):
        self.shared_backbone.load_online_encoder_state(online_state_dict)
        self.target_backbone = copy.deepcopy(self.shared_backbone)
        for parameter in self.target_backbone.parameters():
            parameter.requires_grad = False
        self.target_backbone.eval()
        print(
            'Loaded complete online GNN into joint shared online/EMA backbones '
            f'(layers={len(self.shared_backbone.gnn.layers)}, '
            'stats_residual_scale=0, '
            f'lora_rank={self.shared_backbone.lora_rank}).'
        )

    @torch.no_grad()
    def update_target_backbone(self):
        for target, online in zip(
            self.target_backbone.parameters(),
            self.shared_backbone.parameters(),
        ):
            target.data.mul_(self.target_momentum).add_(
                online.data,
                alpha=1.0 - self.target_momentum,
            )

    def gcl_loss(self, batch, online_x=None):
        if self.separate_gcl_base_path:
            online_x = self.shared_backbone(batch, task_path=False)
        elif online_x is None:
            online_x = self.shared_backbone(batch)
        online_prediction = self.predictor(online_x)
        with torch.no_grad():
            target_x = self.target_backbone(
                batch,
                task_path=not self.separate_gcl_base_path,
            )
        online_prediction = F.normalize(online_prediction, dim=-1, p=2)
        target_x = F.normalize(target_x, dim=-1, p=2)
        return 1.0 - (online_prediction * target_x).sum(dim=-1).mean()

    def deployment_parameter_count(self):
        return sum(
            parameter.numel()
            for module in [self.shared_backbone, self.head_layers]
            for parameter in module.parameters()
        )

    def merged_deployment_parameter_count(self):
        return self.deployment_parameter_count() - \
            self.shared_backbone.lora_parameter_count()

    @torch.no_grad()
    def merge_task_lora_for_deployment(self):
        return self.shared_backbone.merge_task_lora()

    def _predict(self, x, batch):
        if self.task_level == 'node':
            if self.net_only:
                net_node_mask = batch.node_type.view(-1) == NET
                pred = self.head_layers(x[net_node_mask])
                true_class = batch.y[:, 1][net_node_mask].long()
                true_label = batch.y[net_node_mask]
            else:
                pred = self.head_layers(x)
                true_class = batch.y[:, 1].long()
                true_label = batch.y
        elif self.task_level == 'edge':
            if self.src_dst_agg[:4] == 'pool':
                graph_emb = self.pooling_fun(x, batch.batch)
            else:
                batch_size = batch.edge_label.size(0)
                src_emb = x[:batch_size]
                dst_emb = x[batch_size:batch_size * 2]
                if self.src_dst_agg == 'concat':
                    graph_emb = torch.cat((src_emb, dst_emb), dim=1)
                else:
                    graph_emb = src_emb + dst_emb
            pred = self.head_layers(graph_emb)
            true_class = batch.edge_label[:, 1].long()
            true_label = batch.edge_label
        else:
            raise ValueError(f'Invalid task level: {self.task_level}')
        return pred, true_class, true_label

    def forward(self, batch, return_backbone=False):
        x = self.shared_backbone(batch)
        prediction = self._predict(x, batch)
        if return_backbone:
            return (*prediction, x)
        return prediction


class PartialSharedGraphHead(nn.Module):
    """Share lower SGRL online GNN layers and keep a downstream tail/head."""

    def __init__(self, args):
        super().__init__()
        if args.shared_gnn_layers > args.num_gnn_layers:
            raise ValueError(
                "--shared_gnn_layers cannot exceed --num_gnn_layers "
                f"({args.shared_gnn_layers} > {args.num_gnn_layers})."
            )

        self.shared_backbone = SharedGNNBackbone(args)
        self.shared_dim = args.cl_hid_dim
        self.hidden_dim = args.hid_dim
        self.task = args.task
        self.task_level = args.task_level
        self.net_only = args.net_only
        self.num_classes = args.num_classes
        self.class_boundaries = args.class_boundaries
        self.src_dst_agg = args.src_dst_agg
        self.model = args.model
        self.use_stats = bool(
            args.use_stats and args.partial_shared_stats_fusion != 'none'
        )
        self.stats_fusion = (
            args.partial_shared_stats_fusion if self.use_stats else 'none'
        )

        stats_embed_dim = self.shared_dim
        fused_dim = self.shared_dim
        if self.use_stats:
            self.net_attr_layers = nn.Linear(17, stats_embed_dim, bias=True)
            self.dev_attr_layers = nn.Linear(17, stats_embed_dim, bias=True)
            self.pin_attr_layers = nn.Embedding(17, stats_embed_dim)
            if self.stats_fusion == 'concat':
                fused_dim = self.shared_dim + stats_embed_dim
            elif self.stats_fusion in [
                'add',
                'gate',
                'residual_gate',
                'scalar_gate',
                'vector_gate',
            ]:
                fused_dim = self.shared_dim
            else:
                raise ValueError(
                    f'Unsupported partial-shared stats fusion: {self.stats_fusion}'
                )
            if self.stats_fusion in ['gate', 'residual_gate']:
                self.stats_gate = nn.Sequential(
                    nn.Linear(self.shared_dim + stats_embed_dim, self.shared_dim),
                    nn.Sigmoid(),
                )
            elif self.stats_fusion == 'scalar_gate':
                self.stats_gate_logit = nn.Parameter(torch.zeros(1))
            elif self.stats_fusion == 'vector_gate':
                self.stats_gate_logit = nn.Parameter(torch.zeros(self.shared_dim))
            print(
                "Using circuit-statistics adapter after partial shared backbone "
                f"(fusion={self.stats_fusion})."
            )

        self.edge_encoder = nn.Embedding(num_embeddings=4, embedding_dim=self.hidden_dim)
        self.tail_layers = nn.ModuleList()
        tail_layers = args.num_gnn_layers - args.shared_gnn_layers
        current_dim = fused_dim
        for _ in range(tail_layers):
            self.tail_layers.append(
                build_gnn_layer(
                    args.model,
                    current_dim,
                    self.hidden_dim,
                    self.hidden_dim,
                    activation=args.act_fn,
                )
            )
            current_dim = self.hidden_dim

        self.representation_dim = current_dim

        if args.src_dst_agg == 'pooladd':
            self.pooling_fun = pygnn.pool.global_add_pool
        elif args.src_dst_agg == 'poolmean':
            self.pooling_fun = pygnn.pool.global_mean_pool

        head_input_dim = (
            self.representation_dim * 2
            if self.src_dst_agg == 'concat' and self.task_level == 'edge'
            else self.representation_dim
        )

        if self.task == 'regression':
            dim_out = 1
        elif self.task == 'classification':
            dim_out = args.num_classes
        else:
            raise ValueError('Invalid task')

        self.head_layers = MLP(
            in_channels=head_input_dim,
            hidden_channels=self.hidden_dim,
            out_channels=dim_out,
            num_layers=args.num_head_layers,
            use_bn=False,
            dropout=0.0,
            activation=args.act_fn,
        )

        self.use_bn = args.use_bn
        self.tail_bn_node_x = nn.BatchNorm1d(self.hidden_dim)
        if args.act_fn == 'relu':
            self.activation = nn.ReLU()
        elif args.act_fn == 'elu':
            self.activation = nn.ELU()
        elif args.act_fn == 'tanh':
            self.activation = nn.Tanh()
        elif args.act_fn == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif args.act_fn == 'prelu':
            self.activation = nn.PReLU()
        else:
            raise ValueError('Invalid activation')

        self.drop_out = args.dropout

    def load_online_encoder_state(self, online_state_dict):
        self.shared_backbone.load_online_encoder_state(online_state_dict)

    def _encode_stats(self, batch):
        node_type = batch.node_type.view(-1)
        net_node_mask = node_type == NET
        dev_node_mask = node_type == DEV
        pin_node_mask = node_type == PIN
        node_attr_emb = torch.zeros(
            (batch.num_nodes, self.shared_dim), device=batch.node_attr.device
        )
        node_attr_emb[net_node_mask] = \
            self.net_attr_layers(batch.node_attr[net_node_mask])
        node_attr_emb[dev_node_mask] = \
            self.dev_attr_layers(batch.node_attr[dev_node_mask])
        node_attr_emb[pin_node_mask] = \
            self.pin_attr_layers(batch.node_attr[pin_node_mask, 0].long())
        return node_attr_emb

    def _fuse_stats(self, x, batch):
        if not self.use_stats:
            return x

        stats_x = self._encode_stats(batch)
        if self.stats_fusion == 'concat':
            return torch.cat((x, stats_x), dim=1)
        if self.stats_fusion == 'add':
            return x + stats_x
        if self.stats_fusion == 'gate':
            gate = self.stats_gate(torch.cat((x, stats_x), dim=1))
            return gate * x + (1.0 - gate) * stats_x
        if self.stats_fusion == 'residual_gate':
            gate = self.stats_gate(torch.cat((x, stats_x), dim=1))
            return x + gate * stats_x
        if self.stats_fusion == 'scalar_gate':
            gate = torch.sigmoid(self.stats_gate_logit)
            return gate * x + (1.0 - gate) * stats_x
        if self.stats_fusion == 'vector_gate':
            gate = torch.sigmoid(self.stats_gate_logit).view(1, -1)
            return gate * x + (1.0 - gate) * stats_x
        raise ValueError(f'Unsupported partial-shared stats fusion: {self.stats_fusion}')

    def _run_tail(self, x, batch):
        if len(self.tail_layers) == 0:
            return x

        edge_attr = self.edge_encoder(batch.edge_type.view(-1).long())
        for conv in self.tail_layers:
            if self.model == 'gine' or self.model == 'resgatedgcn':
                x = conv(x, batch.edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, batch.edge_index)

            if self.use_bn:
                x = self.tail_bn_node_x(x)

            x = self.activation(x)

            if self.drop_out > 0.0:
                x = F.dropout(x, p=self.drop_out, training=self.training)

        return x

    def forward(self, batch):
        x = self.shared_backbone(batch)
        x = self._fuse_stats(x, batch)
        x = self._run_tail(x, batch)

        if self.task_level == 'node':
            if self.net_only:
                net_node_mask = batch.node_type.view(-1) == NET
                pred = self.head_layers(x[net_node_mask])
                true_class = batch.y[:, 1][net_node_mask].long()
                true_label = batch.y[net_node_mask]
            else:
                pred = self.head_layers(x)
                true_class = batch.y[:, 1].long()
                true_label = batch.y

        elif self.task_level == 'edge':
            if self.src_dst_agg[:4] == 'pool':
                graph_emb = self.pooling_fun(x, batch.batch)
            else:
                batch_size = batch.edge_label.size(0)
                src_emb = x[:batch_size, :]
                dst_emb = x[batch_size:batch_size * 2, :]
                if self.src_dst_agg == 'concat':
                    graph_emb = torch.cat((src_emb, dst_emb), dim=1)
                else:
                    graph_emb = src_emb + dst_emb

            pred = self.head_layers(graph_emb)
            true_class = batch.edge_label[:, 1].long()
            true_label = batch.edge_label

        else:
            raise ValueError('Invalid task level')

        return pred, true_class, true_label


class SgrlBackboneHead(nn.Module):
    """Downstream head that reuses the SGRL online encoder as its backbone."""

    def __init__(self, args):
        super().__init__()
        self.encoder = CustomConv(args)
        self.encoder_frozen = False
        hidden_dim = args.cl_hid_dim
        stats_fusion = getattr(args, 'sgrl_reuse_stats_fusion', 'concat')
        self.use_stats = bool(
            getattr(args, 'use_stats', 0)
            and getattr(args, 'sgrl_reuse_stats', 1)
            and stats_fusion != 'none'
        )
        self.stats_fusion = stats_fusion if self.use_stats else 'none'
        self.task = args.task
        self.task_level = args.task_level
        self.net_only = args.net_only
        self.num_classes = args.num_classes
        self.class_boundaries = args.class_boundaries
        self.src_dst_agg = args.src_dst_agg
        self.stats_embed_dim = hidden_dim
        fused_dim = hidden_dim

        if self.use_stats:
            self.net_attr_layers = nn.Linear(17, self.stats_embed_dim, bias=True)
            self.dev_attr_layers = nn.Linear(17, self.stats_embed_dim, bias=True)
            self.pin_attr_layers = nn.Embedding(17, self.stats_embed_dim)
            if self.stats_fusion == 'concat':
                fused_dim = hidden_dim + self.stats_embed_dim
            elif self.stats_fusion in [
                'add',
                'gate',
                'residual_gate',
                'scalar_gate',
                'vector_gate',
            ]:
                fused_dim = hidden_dim
            else:
                raise ValueError(f'Unsupported SGRL stats fusion: {self.stats_fusion}')
            if self.stats_fusion in ['gate', 'residual_gate']:
                self.stats_gate = nn.Sequential(
                    nn.Linear(hidden_dim + self.stats_embed_dim, hidden_dim),
                    nn.Sigmoid(),
                )
            elif self.stats_fusion == 'scalar_gate':
                self.stats_gate_logit = nn.Parameter(torch.zeros(1))
            elif self.stats_fusion == 'vector_gate':
                self.stats_gate_logit = nn.Parameter(torch.zeros(hidden_dim))
            print(
                "Using circuit-statistics adapter for SGRL backbone reuse "
                f"(fusion={self.stats_fusion})."
            )

        if args.src_dst_agg == 'pooladd':
            self.pooling_fun = pygnn.pool.global_add_pool
        elif args.src_dst_agg == 'poolmean':
            self.pooling_fun = pygnn.pool.global_mean_pool

        head_input_dim = (
            fused_dim * 2
            if self.src_dst_agg == 'concat' and self.task_level == 'edge'
            else fused_dim
        )

        if self.task == 'regression':
            dim_out = 1
        elif self.task == 'classification':
            dim_out = args.num_classes
        else:
            raise ValueError('Invalid task')

        self.head_layers = MLP(
            in_channels=head_input_dim,
            hidden_channels=hidden_dim,
            out_channels=dim_out,
            num_layers=args.num_head_layers,
            use_bn=False,
            dropout=0.0,
            activation=args.act_fn,
        )

    def load_online_encoder_state(self, online_state_dict, freeze=False):
        prefix = 'online_encoder.'
        if any(key.startswith(prefix) for key in online_state_dict):
            encoder_state = {
                key[len(prefix):]: value
                for key, value in online_state_dict.items()
                if key.startswith(prefix)
            }
        else:
            encoder_state = online_state_dict

        load_msg = self.encoder.load_state_dict(encoder_state, strict=False)
        print(
            "Loaded SGRL online encoder into downstream backbone "
            f"(missing={len(load_msg.missing_keys)}, "
            f"unexpected={len(load_msg.unexpected_keys)}, freeze={freeze})."
        )

        if freeze:
            self.freeze_online_encoder()

    def freeze_online_encoder(self):
        self.encoder_frozen = True
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        return self

    def _encode(self, batch):
        if self.encoder_frozen:
            with torch.no_grad():
                x = self.encoder.encode(batch)
        else:
            x = self.encoder.encode(batch)
        return x

    def _encode_stats(self, batch):
        net_node_mask = batch.node_type == NET
        dev_node_mask = batch.node_type == DEV
        pin_node_mask = batch.node_type == PIN
        node_attr_emb = torch.zeros(
            (batch.num_nodes, self.stats_embed_dim), device=batch.node_attr.device
        )
        node_attr_emb[net_node_mask] = \
            self.net_attr_layers(batch.node_attr[net_node_mask])
        node_attr_emb[dev_node_mask] = \
            self.dev_attr_layers(batch.node_attr[dev_node_mask])
        node_attr_emb[pin_node_mask] = \
            self.pin_attr_layers(batch.node_attr[pin_node_mask, 0].long())
        return node_attr_emb

    def _fuse_stats(self, x, batch):
        if not self.use_stats:
            return x

        stats_x = self._encode_stats(batch)
        if self.stats_fusion == 'concat':
            return torch.cat((x, stats_x), dim=1)
        if self.stats_fusion == 'add':
            return x + stats_x
        if self.stats_fusion == 'gate':
            gate = self.stats_gate(torch.cat((x, stats_x), dim=1))
            return gate * x + (1.0 - gate) * stats_x
        if self.stats_fusion == 'residual_gate':
            gate = self.stats_gate(torch.cat((x, stats_x), dim=1))
            return x + gate * stats_x
        if self.stats_fusion == 'scalar_gate':
            gate = torch.sigmoid(self.stats_gate_logit)
            return gate * x + (1.0 - gate) * stats_x
        if self.stats_fusion == 'vector_gate':
            gate = torch.sigmoid(self.stats_gate_logit).view(1, -1)
            return gate * x + (1.0 - gate) * stats_x
        raise ValueError(f'Unsupported SGRL stats fusion: {self.stats_fusion}')

    def forward(self, batch):
        x = self._encode(batch)
        x = self._fuse_stats(x, batch)

        if self.task_level == 'node':
            if self.net_only:
                net_node_mask = batch.node_type == NET
                pred = self.head_layers(x[net_node_mask])
                true_class = batch.y[:, 1][net_node_mask].long()
                true_label = batch.y[net_node_mask]
            else:
                pred = self.head_layers(x)
                true_class = batch.y[:, 1].long()
                true_label = batch.y

        elif self.task_level == 'edge':
            if self.src_dst_agg[:4] == 'pool':
                graph_emb = self.pooling_fun(x, batch.batch)
            else:
                batch_size = batch.edge_label.size(0)
                src_emb = x[:batch_size, :]
                dst_emb = x[batch_size:batch_size * 2, :]
                if self.src_dst_agg == 'concat':
                    graph_emb = torch.cat((src_emb, dst_emb), dim=1)
                else:
                    graph_emb = src_emb + dst_emb

            pred = self.head_layers(graph_emb)
            true_class = batch.edge_label[:, 1].long()
            true_label = batch.edge_label

        else:
            raise ValueError('Invalid task level')

        return pred, true_class, true_label
