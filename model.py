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
            elif self.stats_fusion in ['add', 'gate', 'residual_gate']:
                fused_dim = hidden_dim
            else:
                raise ValueError(f'Unsupported SGRL stats fusion: {self.stats_fusion}')
            if self.stats_fusion in ['gate', 'residual_gate']:
                self.stats_gate = nn.Sequential(
                    nn.Linear(hidden_dim + self.stats_embed_dim, hidden_dim),
                    nn.Sigmoid(),
                )
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
