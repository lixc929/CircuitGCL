import argparse
import torch
from sram_dataset import performat_SramDataset, adaption_for_sgrl
from downstream_train import downstream_train
import os
import json
from sgrl_train import (
    embedding_cache_fingerprint,
    sgrl_cache_fingerprint,
    sgrl_train,
)
import datetime
import sys

from rng_utils import resolve_stage_seeds, seed_all

from run_artifacts import (
    file_sha256,
    finalize_run_artifacts,
    prepare_run_artifacts,
    update_run_config,
    write_json_atomic,
)

if __name__ == "__main__":
    # STEP 0: Parse Arguments ======================================================================= #
    parser = argparse.ArgumentParser(description="CircuitGCL")
    # Task setting
    parser.add_argument("--task_level", type=str, default="edge", help="Task level. 'node' or 'edge'.")
    parser.add_argument("--task", type=str, default="classification", help="Task type. 'classification' or 'regression'.")
    
    # Dataset setting
    parser.add_argument("--dataset", type=str, default="ssram+digtime+timing_ctrl+array_128_32_8t", help="Names of datasets.") # the first dataset is the training dataset
    parser.add_argument('--neg_edge_ratio',type=float,default=0.0,help='The ratio of negative edges.') # 0.0 for classification, 0.5 for regression
    parser.add_argument('--net_only',type=bool,default=True,help='Only use net nodes for node level task or not.')
    parser.add_argument(
        '--protocol',
        choices=['strict_inductive', 'transductive_legacy'],
        default='transductive_legacy',
        help='Scientific data-visibility protocol for this run.',
    )
    parser.add_argument(
        '--sgrl_graph_scope',
        choices=['auto', 'source', 'all'],
        default='auto',
        help='Graphs used to fit SGRL; auto follows --protocol.',
    )
    parser.add_argument(
        '--normalization_scope',
        choices=['auto', 'source', 'all'],
        default='auto',
        help='Graphs used to fit node-feature normalization; auto follows --protocol.',
    )

    # Graph sampling setting
    parser.add_argument("--small_dataset_sample_rates", type=float, default=1.0, help="The sample rate for small dataset.")
    parser.add_argument("--large_dataset_sample_rates", type=float, default=0.1, 
                        help='Target edge num of large dataset. 20 percent for large G')
    parser.add_argument("--num_hops", type=int, default=4, help="Number of hops in subgraph sampling.")
    parser.add_argument('--num_neighbors',type=int,default=64,help='The number of neighbors in subgraph sampling.')
    
    # Training setting
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument(
        '--pretraining_seed',
        type=int,
        default=None,
        help='SGRL pretraining seed. Default: --seed.',
    )
    parser.add_argument(
        '--embedding_inference_seed',
        type=int,
        default=None,
        help='Frozen-embedding neighbor-sampling seed. Default: --seed.',
    )
    parser.add_argument(
        '--downstream_seed',
        type=int,
        default=None,
        help='Downstream model/criterion initialization seed. Default: --seed.',
    )
    parser.add_argument(
        '--train_sampler_seed',
        type=int,
        default=None,
        help='Downstream training neighbor-sampling seed. Default: --seed.',
    )
    parser.add_argument(
        '--relation_sample_seed',
        type=int,
        default=None,
        help='Processed relation-balanced sampling seed. Default: --seed.',
    )
    parser.add_argument(
        '--split_seed',
        type=int,
        default=None,
        help='Source train/validation split seed. Default: --seed.',
    )
    parser.add_argument(
        '--eval_seed',
        type=int,
        default=0,
        help='Fixed validation/test neighbor-sampling seed.',
    )
    parser.add_argument("--num_workers", type=int, default=8, help="The number of workers in data loaders.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index. Default: -1, using cpu.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="The batch size.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=0,
        help='Stop after this many epochs without raw validation-MSE improvement. Zero disables.',
    )
    parser.add_argument(
        '--early_stopping_min_delta',
        type=float,
        default=0.0,
        help='Minimum raw validation-MSE decrease required to reset early stopping.',
    )

    # SGRL arguments
    parser.add_argument('--sgrl', type=int, default=0, help='Enable contrastive learning, i.e., SGRL.')
    parser.add_argument(
        '--sgrl_mode',
        type=str,
        default='static',
        choices=[
            'static',
            'init',
            'freeze',
            'online_feature',
            'online_feature_finetune',
            'init_reuse',
            'partial_shared',
            'joint_shared',
        ],
        help=(
            "How to use SGRL downstream. "
            "'static' keeps the original cached embedding path; "
            "'init' uses the online encoder as the downstream backbone; "
            "'freeze' uses and freezes the online encoder as the downstream backbone; "
            "'online_feature' computes frozen online-encoder features per downstream batch "
            "and feeds them into the original downstream GraphHead; "
            "'online_feature_finetune' also allows supervised gradients to update "
            "the online encoder; "
            "'init_reuse' initializes compatible original GraphHead parameters from "
            "the online encoder and then trains a single downstream GraphHead; "
            "'partial_shared' reuses the online encoder's lower GNN layers as a "
            "shared downstream backbone and keeps task-specific downstream tail/head; "
            "'joint_shared' trains one complete deployment backbone with supervised "
            "and optional EMA-target GCL losses."
        ),
    )
    parser.add_argument(
        '--shared_gnn_layers',
        type=int,
        default=1,
        help='Number of lower SGRL online GNN layers to share in partial_shared mode.',
    )
    parser.add_argument(
        '--partial_shared_stats_fusion',
        type=str,
        default='add',
        choices=[
            'add',
            'gate',
            'residual_gate',
            'scalar_gate',
            'vector_gate',
            'concat',
            'none',
        ],
        help='How to fuse circuit statistics after the shared lower GNN backbone.',
    )
    parser.add_argument(
        '--partial_shared_backbone_lr',
        type=float,
        default=None,
        help=(
            'Optional learning rate for the shared lower GNN backbone in '
            'partial_shared mode. Default: use --lr for all trainable params.'
        ),
    )
    parser.add_argument(
        '--partial_shared_freeze_epochs',
        type=int,
        default=0,
        help=(
            'Freeze the shared lower GNN backbone for this many initial epochs '
            'in partial_shared mode, then unfreeze it without resetting the '
            'optimizer state of downstream parameters.'
        ),
    )
    parser.add_argument(
        '--partial_shared_backbone_eval_policy',
        type=str,
        default='frozen_only',
        choices=['frozen_only', 'always'],
        help=(
            "Control dropout/batchnorm mode independently of gradient freezing. "
            "'frozen_only' keeps the backbone in eval mode only during the freeze "
            "window; 'always' keeps it in eval mode throughout downstream training "
            "while still allowing gradients after unfreezing."
        ),
    )
    parser.add_argument(
        '--partial_shared_audit',
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            'Record fixed-batch representation drift and statistics-fusion '
            'diagnostics for partial_shared runs.'
        ),
    )
    parser.add_argument(
        '--joint_shared_gnn_layers',
        type=int,
        default=2,
        help='Number of GNN layers in the single joint-shared deployment backbone.',
    )
    parser.add_argument(
        '--joint_gcl_lambda',
        type=float,
        default=0.0,
        help='Weight of the EMA-target GCL loss in joint_shared mode.',
    )
    parser.add_argument(
        '--joint_backbone_lr',
        type=float,
        default=None,
        help=(
            'Optional learning rate for pretrained joint-shared base GNN '
            'parameters. LoRA, statistics, predictor, and head keep --lr.'
        ),
    )
    parser.add_argument(
        '--joint_shared_audit',
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            'Record fixed-batch base/task representation drift for '
            'joint_shared runs. Transfer representations are evaluated only '
            'after the best checkpoint is restored.'
        ),
    )
    parser.add_argument(
        '--joint_shared_audit_interval',
        type=int,
        default=5,
        help='Epoch interval for source-validation joint_shared audit records.',
    )
    parser.add_argument(
        '--joint_lora_rank',
        type=int,
        default=0,
        help=(
            'Rank of the mergeable supervised LoRA update on the selected '
            'joint-shared ClusterGCN layer. Zero disables LoRA.'
        ),
    )
    parser.add_argument(
        '--joint_lora_alpha',
        type=float,
        default=None,
        help='LoRA scale numerator. Default: rank, giving unit scaling.',
    )
    parser.add_argument(
        '--joint_lora_layer',
        type=int,
        default=-1,
        help='Zero-based joint GNN layer to adapt; -1 selects the last layer.',
    )
    parser.add_argument(
        '--sgrl_reuse_stats',
        type=int,
        default=1,
        help='0 or 1. Add the downstream circuit-statistics adapter when reusing the SGRL backbone.',
    )
    parser.add_argument(
        '--sgrl_reuse_stats_fusion',
        type=str,
        default='concat',
        choices=[
            'concat',
            'add',
            'gate',
            'residual_gate',
            'scalar_gate',
            'vector_gate',
            'none',
        ],
        help="How to fuse reused SGRL backbone features and circuit-statistics features.",
    )
    parser.add_argument('--e1_lr', type=float, default=1e-6, help='Learning rate for online encoder in SGRL.')
    parser.add_argument('--e2_lr', type=float, default=2e-7, help='Learning rate for target encoder in SGRL.')
    parser.add_argument(
        '--sgrl_pretrain_target_update',
        choices=['sgrl_dual_rsm_ema', 'circuitgcl_text_ema_only'],
        default='sgrl_dual_rsm_ema',
        help=(
            'Target update used only during SGRL pretraining. The dual mode '
            'matches the SGRL author implementation; ema_only is a matched '
            'CircuitGCL-text interpretation ablation.'
        ),
    )
    parser.add_argument(
        '--sgrl_online_lr',
        type=float,
        default=1e-6,
        help='Downstream finetuning learning rate for the SGRL online encoder.',
    )
    parser.add_argument('--momentum', type=float, default=0.99, help='EMA')
    parser.add_argument('--weight_decay', type=float, default=0., help='weight_decay')
    parser.add_argument('--cl_epochs', type=int, default=800, help='cl_epochs')
    parser.add_argument("--cl_model", type=str, default='clustergcn', 
                        choices=['clustergcn', 'resgatedgcn', 'gat', 'gcn', 'sage', 'gine'],
                        help="The gnn model of SGRL encoders.")
    parser.add_argument('--cl_act_fn', default='tanh', choices=['relu', 'elu', 'tanh', 'leakyrelu', 'prelu'], help='Activation function of SGRL encoders')
    parser.add_argument("--cl_gnn_layers", type=int, default=4, help="Number of GNN layers of encoders in SGRL.")
    parser.add_argument('--cl_hid_dim', type=int, default=256, help='hidden_dim for contrastive learning')
    parser.add_argument('--cl_batch_size', type=int, default=1024, 
                        help='Batch size for contrastive learning. 512 for large G')
    parser.add_argument('--cl_num_neighbors', type=int, default=64, 
                        help='Number of neighbors for contrastive learning. 128 for large G')
    parser.add_argument('--cl_dropout', type=float, default=0.3, help='Dropout for SGRL encoders.')
    
    ## Downstream GNN setting
    parser.add_argument("--model", type=str, default='sage', help="The gnn model. Could be 'clustergcn', 'resgatedgcn', 'gat', 'gcn', 'sage', 'gine'.")
    parser.add_argument("--num_gnn_layers", type=int, default=4, help="Number of GNN layers.")
    parser.add_argument("--num_head_layers", type=int, default=2, help="Number of head layers.")
    parser.add_argument("--hid_dim", type=int, default=144, help="Hidden layer dim.")
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout for neural networks.')
    parser.add_argument('--use_bn', type=int, default=0, help='0 or 1. Batch norm for neural networks.')
    parser.add_argument('--act_fn', default='prelu', choices=['relu', 'elu', 'tanh', 'leakyrelu', 'prelu'], help='Activation function')
    parser.add_argument('--use_stats', type=int, default=1, help='0 or 1. Circuit statistics features. Use in node task.')

    # Regression setting
    parser.add_argument('--src_dst_agg', type=str, default='concat', choices=['concat', 'add', 'pooladd', 'poolmean'],
                        help="The way to aggregate nodes. Can be 'concat' or 'add' or 'pooladd' or 'poolmean'.")
    parser.add_argument("--regress_loss", type=str, default='mse', choices=['mse', 'gai', 'bmc', 'bni', 'lds'], help="The loss function for edge regression. Could be 'mse', 'bmc', or 'gai'.")
    
    # Classification setting
    parser.add_argument('--class_loss',type=str,default='bsmCE',choices=['bsmCE','focal','cross_entropy'],help='The loss function for classification.')
    parser.add_argument('--num_classes',type=int,default=5,help='The number of classes for node classification.')
    parser.add_argument('--class_boundaries',type=list,default=[0.2,0.4,0.6,0.8],help='The boundaries for classification.')

    # Balanced MSE setting for GAI implementation
    parser.add_argument("--noise_sigma", type=float, default=0.001, help="The simga_noise of Balanced MSE (EQ 3.6).")
    
    # LDS setting
    parser.add_argument('--lds_kernel', type=str, default='gaussian',
                    choices=['gaussian', 'triang', 'laplace'], help='LDS kernel type')
    parser.add_argument('--lds_ks', type=int, default=9, help='LDS kernel size: should be odd number. 5 0r 9.')
    parser.add_argument('--lds_sigma', type=float, default=0.02, help='LDS gaussian/laplace kernel sigma. 1 or 2.')
    
    parser.add_argument('--log_dir', type=str, default='logs', help='The directory to save the log file.')

    args = parser.parse_args()
    resolved_seeds = resolve_stage_seeds(args)
    if args.sgrl_graph_scope == 'auto':
        args.sgrl_graph_scope = (
            'source' if args.protocol == 'strict_inductive' else 'all'
        )
    if args.normalization_scope == 'auto':
        args.normalization_scope = (
            'source' if args.protocol == 'strict_inductive' else 'all'
        )
    if args.protocol == 'strict_inductive':
        if args.sgrl_graph_scope != 'source':
            raise ValueError(
                'strict_inductive requires --sgrl_graph_scope source.'
            )
        if args.normalization_scope != 'source':
            raise ValueError(
                'strict_inductive requires --normalization_scope source.'
            )
    if args.sgrl == 0 and args.sgrl_mode != 'static':
        print(f"[Warning] --sgrl_mode {args.sgrl_mode} is ignored because --sgrl is 0.")
        args.sgrl_mode = 'static'

    args.use_sgrl_embeds = int(args.sgrl == 1 and args.sgrl_mode == 'static')
    args.use_sgrl_backbone = int(args.sgrl == 1 and args.sgrl_mode in ['init', 'freeze'])
    args.use_sgrl_graph_init = int(args.sgrl == 1 and args.sgrl_mode == 'init_reuse')
    args.use_sgrl_partial_shared = int(args.sgrl == 1 and args.sgrl_mode == 'partial_shared')
    args.use_sgrl_joint_shared = int(args.sgrl == 1 and args.sgrl_mode == 'joint_shared')
    args.use_sgrl_online_features = int(
        args.sgrl == 1
        and args.sgrl_mode in ['online_feature', 'online_feature_finetune']
    )
    args.finetune_sgrl_online = int(args.sgrl == 1 and args.sgrl_mode == 'online_feature_finetune')
    args.use_graph_cl_features = int(
        args.sgrl == 1
        and args.sgrl_mode in ['static', 'online_feature', 'online_feature_finetune']
    )
    if args.use_sgrl_joint_shared:
        if args.task != 'regression':
            raise ValueError('joint_shared currently supports regression tasks only.')
        if args.joint_shared_gnn_layers > args.cl_gnn_layers:
            raise ValueError(
                '--joint_shared_gnn_layers cannot exceed --cl_gnn_layers '
                f'({args.joint_shared_gnn_layers} > {args.cl_gnn_layers}).'
            )
        if args.joint_gcl_lambda < 0.0:
            raise ValueError('--joint_gcl_lambda must be non-negative.')
        if args.joint_backbone_lr is not None and args.joint_backbone_lr < 0.0:
            raise ValueError('--joint_backbone_lr must be non-negative.')
        if args.joint_lora_rank < 0:
            raise ValueError('--joint_lora_rank must be non-negative.')
        if args.joint_lora_rank > 0 and args.cl_model != 'clustergcn':
            raise ValueError(
                'Mergeable joint LoRA currently supports --cl_model '
                'clustergcn only.'
            )
    if args.early_stopping_patience < 0:
        raise ValueError('--early_stopping_patience must be non-negative.')
    if args.early_stopping_min_delta < 0.0:
        raise ValueError('--early_stopping_min_delta must be non-negative.')
    if args.joint_shared_audit_interval <= 0:
        raise ValueError('--joint_shared_audit_interval must be positive.')

    # Preserve legacy dataset/setup behavior before stage-specific boundaries.
    seed_all(args.seed)

    ##set log file
    
    # create log file
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if args.task == 'classification':
        log_filename = os.path.join(args.log_dir, f"{timestamp}_{args.task_level}_{args.task}_{args.dataset}_loss{args.class_loss}_batch{args.batch_size}.txt")
    else: # regression task
        log_filename = os.path.join(args.log_dir, f"{timestamp}_{args.task_level}_{args.task}_{args.dataset}_loss{args.regress_loss}_batch{args.batch_size}.txt")
    prepare_run_artifacts(args, log_filename)
    update_run_config(args, {'resolved_seeds': resolved_seeds})
    log_file = open(log_filename, 'w')
    
    # Redirect standard output to both file and console
    class Tee(object):
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)


    # Check cuda
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    print(
        "CUDA status: "
        f"available={cuda_available}, device_count={cuda_device_count}, "
        f"requested_gpu={args.gpu}"
    )
    if args.gpu != -1 and cuda_available:
        if args.gpu >= cuda_device_count:
            raise ValueError(
                f"Requested GPU {args.gpu}, but only {cuda_device_count} CUDA devices are visible."
            )
        device = torch.device("cuda:{}".format(args.gpu))
        print('Using GPU: {}'.format(args.gpu))
    else:
        device = torch.device("cpu")
        print("Using CPU")

    print(f"============= PID = {os.getpid()} ============= ")
    print(args)


    # STEP 1: Load Dataset =================================================================== #
    dataset = performat_SramDataset(
        name=args.dataset, 
        dataset_dir='./datasets/', 
        neg_edge_ratio=args.neg_edge_ratio,
        to_undirected=True,
        small_dataset_sample_rates=args.small_dataset_sample_rates,
        large_dataset_sample_rates=args.large_dataset_sample_rates,
        task_level=args.task_level,
        net_only=args.net_only,
        class_boundaries=args.class_boundaries,
        relation_sample_seed=args.relation_sample_seed,
    )
    update_run_config(args, {
        'relation_sample_seed': args.relation_sample_seed,
        'processed_caches': dataset.processed_cache_provenance,
    })

    # STEP 2-3: If you do graph contrastive learning, you should add the code here =========== #
    cl_embeds = None
    sgrl_online_state = None
    if args.sgrl == 1:
        train_graph_names = (
            [dataset.names[0]]
            if args.sgrl_graph_scope == 'source'
            else list(dataset.names)
        )
        sgrl_key = sgrl_cache_fingerprint(args, train_graph_names)
        if args.sgrl_mode == 'static':
            embedding_dir = './embeddings/'
            os.makedirs(embedding_dir, exist_ok=True)
            embedding_key = embedding_cache_fingerprint(args, sgrl_key)
            embedding_path = os.path.join(
                embedding_dir,
                f'embeddings_{args.dataset}_{embedding_key}.pt',
            )
            embedding_metadata_path = embedding_path + '.metadata.json'
            embedding_sampler_fingerprint = None
            if os.path.exists(embedding_path):
                cl_embeds = torch.load(
                    embedding_path,
                    map_location='cpu',
                    weights_only=True,
                )
                if os.path.exists(embedding_metadata_path):
                    with open(
                        embedding_metadata_path, encoding='utf-8'
                    ) as metadata_file:
                        embedding_metadata = json.load(metadata_file)
                    cached_seed = int(
                        embedding_metadata['embedding_inference_seed']
                    )
                    if cached_seed != args.embedding_inference_seed:
                        raise RuntimeError(
                            'Embedding cache inference-seed mismatch: '
                            f'{cached_seed} != {args.embedding_inference_seed}.'
                        )
                    embedding_sampler_fingerprint = embedding_metadata.get(
                        'embedding_sampler_fingerprint'
                    )
            else:
                sgrl_result = sgrl_train(
                    args,
                    dataset,
                    device,
                    return_embeddings=True,
                    return_online_state=True,
                )
                cl_embeds = sgrl_result['embeddings']
                embedding_sampler_fingerprint = sgrl_result[
                    'embedding_sampler_fingerprint'
                ]
                torch.save(cl_embeds, embedding_path)
                write_json_atomic(embedding_metadata_path, {
                    'embedding_cache_key': embedding_key,
                    'sgrl_cache_key': sgrl_key,
                    'embedding_inference_seed': args.embedding_inference_seed,
                    'embedding_sampler_fingerprint': (
                        embedding_sampler_fingerprint
                    ),
                })
            update_run_config(args, {
                'sgrl_cache_key': sgrl_key,
                'embedding_cache_key': embedding_key,
                'sgrl_train_graph_names': train_graph_names,
                'sgrl_pretrain_target_update': args.sgrl_pretrain_target_update,
                'pretraining_seed': args.pretraining_seed,
                'embedding_inference_seed': args.embedding_inference_seed,
                'embedding_sampler_fingerprint': (
                    embedding_sampler_fingerprint
                ),
                'embedding_path': os.path.abspath(embedding_path),
                'embedding_sha256': file_sha256(embedding_path),
            })
        else:
            sgrl_result = sgrl_train(
                args,
                dataset,
                device,
                return_embeddings=False,
                return_online_state=True,
            )
            sgrl_online_state = sgrl_result['online_state_dict']
            print(f"Using SGRL online encoder from {sgrl_result['online_model_path']}")
            update_run_config(args, {
                'sgrl_cache_key': sgrl_result['cache_key'],
                'sgrl_train_graph_names': sgrl_result['train_graph_names'],
                'sgrl_pretrain_target_update': sgrl_result['target_update'],
                'pretraining_seed': args.pretraining_seed,
                'sgrl_checkpoint_path': os.path.abspath(
                    sgrl_result['online_model_path']
                ),
                'sgrl_checkpoint_sha256': file_sha256(
                    sgrl_result['online_model_path']
                ),
            })
    # STEP 4: Training Epochs ================================================================ #

    # This boundary makes cache hit/miss and inference graph size irrelevant to
    # downstream construction and model randomness.
    seed_all(args.downstream_seed)
    try:
        downstream_train(args, dataset, device, cl_embeds, sgrl_online_state)
    except BaseException:
        finalize_run_artifacts(args, status='failed')
        raise
    else:
        finalize_run_artifacts(args, status='completed')
    finally:
        sys.stdout = original_stdout
        log_file.close()
    print(f"Finished running and save results to {log_filename}")
