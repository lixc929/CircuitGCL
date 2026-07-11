import copy
import json
import random

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score,
    mean_absolute_error, mean_squared_error,
    root_mean_squared_error, r2_score,
)
import numpy as np
import time
from tqdm import tqdm
from model import (
    GraphHead,
    JointSharedGraphHead,
    OnlineFeatureGraphHead,
    PartialSharedGraphHead,
    SgrlBackboneHead,
)
from sampling import dataset_sampling
from balanced_mse import GAILoss, BMCLoss, BNILoss, train_gmm, WeightedMSE, get_lds_weights, BalancedSoftmax, FocalLoss, compute_class_weights
import os
import matplotlib.pyplot as plt

from run_artifacts import (
    load_best_checkpoint,
    save_best_checkpoint,
    update_best_checkpoint_metrics,
    write_run_metrics,
)

# from torch.utils.data.sampler import SubsetRandomSampler
# from sram_dataset import LinkPredictionDataset
# from sram_dataset import collate_fn, adaption_for_sgrl
# from torch_geometric.data import Batch

# from torch_geometric.loader import NeighborLoader, GraphSAINTRandomWalkSampler, GraphSAINTEdgeSampler, ShaDowKHopSampler

NET = 0
DEV = 1
PIN = 2

class Logger (object):
    """ 
    Logger for printing message during training and evaluation. 
    Adapted from GraphGPS 
    """
    
    def __init__(self, task='classification', max_label=None):
        super().__init__()
        # Whether to run comparison tests of alternative score implementations.
        self.test_scores = False
        self._iter = 0
        self._true = []
        self._pred = []
        self._loss = 0.0
        self._size_current = 0
        self.task = task
        self.max_label = max_label
    def update_stats(self, true, pred, batch_size, loss):
        self._true.append(true)
        self._pred.append(pred)
        self._size_current += batch_size
        self._loss += loss * batch_size
        self._iter += 1

    def write_epoch(self, split=""):
        true, pred_score = torch.cat(self._true), torch.cat(self._pred)
        true = true.numpy()
        pred_score = pred_score.numpy()
        reformat = lambda x: round(float(x), 4)        

        if self.task == 'classification':

            max_label_indices = np.where(true == self.max_label)[0]
            mask = np.zeros_like(true, dtype=bool)
            if len(max_label_indices) > 0:
                selected_max_indices = np.random.choice(
                    max_label_indices, 
                    size=max(1, len(max_label_indices) // 10),  # 至少保留1个样本
                    replace=False
                )
                mask[selected_max_indices] = True
                    
            mask[true != self.max_label] = True

            accuracy = accuracy_score(true[mask], pred_score[mask])
            f1 = f1_score(true[mask], pred_score[mask], average='macro')
            precision = precision_score(true[mask], pred_score[mask], average='macro')
            recall = recall_score(true[mask], pred_score[mask], average='macro')

            res = {
                'loss': reformat(self._loss / self._size_current),
                'accuracy': reformat(accuracy),
                'f1': reformat(f1),
                'precision': reformat(precision),
                'recall': reformat(recall),
            }

        else:  # regression task
            mse_raw = float(mean_squared_error(true, pred_score))
            res = {
                'loss': round(self._loss / self._size_current, 8),
                'mae': reformat(mean_absolute_error(true, pred_score)),
                'mse': reformat(mse_raw),
                'mse_raw': mse_raw,
                'rmse': reformat(root_mean_squared_error(true, pred_score)),
                'r2': reformat(r2_score(true, pred_score)),
                'label_bin_metrics': regression_bin_metrics(true, pred_score),
            }

        # Just print the results to screen
        printable = {
            key: value for key, value in res.items()
            if key != 'label_bin_metrics'
        }
        print(split, printable)
        return res


def regression_bin_metrics(true, prediction, bin_edges=None):
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if true.shape != prediction.shape:
        raise ValueError('True and predicted regression values must have equal shape.')
    if bin_edges is None:
        bin_edges = np.linspace(0.0, 1.0, 11)
    bin_edges = np.asarray(bin_edges, dtype=np.float64)
    records = []
    for index, (lower, upper) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (true >= lower) & (
            true <= upper if index == len(bin_edges) - 2 else true < upper
        )
        count = int(mask.sum())
        record = {
            'lower': float(lower),
            'upper': float(upper),
            'count': count,
            'fraction': float(count / max(true.size, 1)),
            'mse': None,
            'mae': None,
            'bias': None,
        }
        if count:
            error = prediction[mask] - true[mask]
            record.update({
                'mse': float(np.mean(error ** 2)),
                'mae': float(np.mean(np.abs(error))),
                'bias': float(np.mean(error)),
            })
        records.append(record)
    return records

def compute_loss(args, pred, true, criterion):
    """Compute loss and prediction score. 
    Args:
        args (argparse.Namespace): The arguments
        pred (torch.tensor): Unnormalized prediction
        true (torch.tensor): Groud truth label
        criterion (torch.nn.Module): The loss function
    Returns: Loss, normalized prediction score
    """
    assert criterion, "Loss function is not provided!"
    assert pred.size(0) == true.size(0), \
        "Prediction and true label size mismatch!"

    if args.task == 'classification':
        if args.class_loss == 'focal':
            class_weights = compute_class_weights(true, args.num_classes) 
            focal_loss = FocalLoss(gamma=2.0, class_weights=class_weights)
            loss = focal_loss(pred, true)
        else:
            loss = F.cross_entropy(pred, true)
        predict_class = torch.argmax(pred, dim=1)
        return loss, predict_class, true
      
    elif args.task == 'regression':
        ## Size of `pred` must be [N, 1] for regression task
        assert pred.ndim == 1 or pred.size(1) == 1
        pred = pred.view(-1, 1)

        assert (true.size(1) == 2), \
            "true label has two columns [continuous label, discrete label or label weights]!"
        
        ## for LDS loss, the second column of `true` is the weights
        if args.regress_loss == 'lds':
            loss = criterion(
                pred, 
                true[:, 0].view(-1, 1),
                true[:, 1].view(-1, 1) # the weight for each label
            )
            return loss, pred, true[:, 0].view(pred.size())

        ## for other loss func, the second column of true is the discrete label,
        ## which is not in use.
        ## Size of `true[:, 0]` is [N,] for regression task, 
        ## ensuring same sizes of `pred` and `true`
        true = true[:, 0] if true.ndim == 2 else true
        true = true.view(-1, 1)
        
        return criterion(pred, true), pred, true
    
    else:
        raise ValueError(f"Task type {args.task} not supported!")

@torch.no_grad()
def eval_epoch(args, loader, model, device, 
               split='val', criterion=None):
    """ 
    evaluate the model on the validation or test set
    Args:
        args (argparse.Namespace): The arguments
        loader (torch.utils.data.DataLoader): The data loader
        model (torch.nn.Module): The model
        device (torch.device): The device to run the model on
        split (str): The split name, 'val' or 'test'
        criterion (torch.nn.Module): The loss function
    """
    model.eval()
    time_start = time.time()
    logger = Logger(task=args.task)

    for i, batch in enumerate(tqdm(loader, desc="eval_"+split, leave=False)):
        pred, class_true, label_true = model(batch.to(device))
        if args.task == 'regression':
            loss, pred_score, true = compute_loss(args, pred, label_true, criterion=criterion)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)
            logger.update_stats(true=_true,
                                pred=_pred,
                                batch_size=_true.size(0),
                                loss=loss.detach().cpu().item(),
                                )
        elif args.task == 'classification':
            loss, predict_class, true = compute_loss(args, pred, class_true, criterion=criterion)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = predict_class.detach().to('cpu', non_blocking=True)
            logger.update_stats(true=_true,
                                pred=_pred,
                                batch_size=_true.size(0),
                                loss=loss.detach().cpu().item(),
                                )
    return logger.write_epoch(split)


def validation_mse(result):
    """Return the unrounded MSE used for checkpoint selection."""
    return float(result.get('mse_raw', result['mse']))


def validation_improved(best_mse, result, min_delta=0.0):
    return best_mse - validation_mse(result) > min_delta


def has_partial_shared_backbone(args, model):
    return bool(
        getattr(args, 'use_sgrl_partial_shared', 0)
        and hasattr(model, 'shared_backbone')
    )


def set_partial_shared_backbone_trainable(args, model, trainable):
    if not has_partial_shared_backbone(args, model):
        return False

    for param in model.shared_backbone.parameters():
        param.requires_grad = trainable

    state = "trainable" if trainable else "frozen"
    print(f"Partial-shared backbone is now {state}.")
    return True


def apply_partial_shared_backbone_eval_policy(args, model, epoch):
    if not has_partial_shared_backbone(args, model):
        return False

    freeze_epochs = getattr(args, 'partial_shared_freeze_epochs', 0)
    eval_policy = getattr(
        args,
        'partial_shared_backbone_eval_policy',
        'frozen_only',
    )
    is_frozen = freeze_epochs > 0 and epoch < freeze_epochs
    if is_frozen or eval_policy == 'always':
        model.shared_backbone.eval()
        return True
    return False


def add_partial_shared_backbone_optimizer_group(args, model, optimizer):
    existing_param_ids = {
        id(param)
        for group in optimizer.param_groups
        for param in group['params']
    }
    backbone_params = [
        param
        for param in model.shared_backbone.parameters()
        if param.requires_grad and id(param) not in existing_param_ids
    ]
    if not backbone_params:
        return 0

    backbone_lr = getattr(args, 'partial_shared_backbone_lr', None)
    if backbone_lr is None:
        backbone_lr = args.lr
    optimizer.add_param_group({
        'params': backbone_params,
        'lr': backbone_lr,
    })
    return sum(param.numel() for param in backbone_params)


def maybe_unfreeze_partial_shared_backbone(args, model, optimizer, epoch):
    freeze_epochs = getattr(args, 'partial_shared_freeze_epochs', 0)
    if (
        has_partial_shared_backbone(args, model)
        and freeze_epochs > 0
        and epoch == freeze_epochs
    ):
        set_partial_shared_backbone_trainable(args, model, True)
        added_values = add_partial_shared_backbone_optimizer_group(
            args,
            model,
            optimizer,
        )
        print(
            "Added partial-shared backbone to the existing optimizer "
            f"({added_values:,} parameter values); preserved downstream Adam state."
        )
        optimizer.zero_grad()
    return optimizer


def _partial_shared_weight_drift(model, reference_backbone):
    return _module_weight_drift(model.shared_backbone, reference_backbone)


def _module_weight_drift(current_module, reference_module):
    squared_delta = 0.0
    squared_reference = 0.0
    for current, reference in zip(
        current_module.parameters(),
        reference_module.parameters(),
    ):
        current_value = current.detach().float()
        reference_value = reference.detach().float()
        squared_delta += torch.sum(
            (current_value - reference_value).pow(2)
        ).item()
        squared_reference += torch.sum(reference_value.pow(2)).item()
    return (squared_delta ** 0.5) / max(squared_reference ** 0.5, 1e-12)


def _capture_fixed_audit_batches(val_loader, test_loaders):
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.get_rng_state()
    cuda_rng_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    try:
        fixed_batches = {'source_val': next(iter(val_loader)).cpu()}
        for test_name, loader in test_loaders.items():
            fixed_batches[test_name] = next(iter(loader)).cpu()
    finally:
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.set_rng_state(torch_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
    return fixed_batches


def build_partial_shared_audit_context(
        args, model, val_loader, test_loaders, device):
    if not (
        getattr(args, 'partial_shared_audit', 0)
        and has_partial_shared_backbone(args, model)
    ):
        return None

    reference_backbone = copy.deepcopy(model.shared_backbone).to(device)
    reference_backbone.eval()
    for parameter in reference_backbone.parameters():
        parameter.requires_grad = False

    fixed_batches = _capture_fixed_audit_batches(val_loader, test_loaders)

    audit_path = os.path.join(
        args.run_artifact_dir,
        'representation_audit.jsonl',
    )
    print(
        'Partial-shared representation audit enabled: '
        f'splits={list(fixed_batches)}, output={audit_path}'
    )
    return {
        'reference_backbone': reference_backbone,
        'fixed_batches': fixed_batches,
        'device': device,
        'path': audit_path,
    }


@torch.no_grad()
def record_partial_shared_representation_audit(
        args, model, context, epoch):
    if context is None:
        return []

    current_training = model.shared_backbone.training
    model.shared_backbone.eval()
    reference_backbone = context['reference_backbone']
    reference_backbone.eval()
    weight_relative_l2 = _partial_shared_weight_drift(
        model,
        reference_backbone,
    )
    records = []

    for split, cpu_batch in context['fixed_batches'].items():
        batch = cpu_batch.clone().to(context['device'])
        current_x = model.shared_backbone(batch)
        reference_x = reference_backbone(batch)

        if args.task_level == 'edge':
            root_count = 2 * batch.edge_label.size(0)
        else:
            root_count = getattr(batch, 'batch_size', batch.num_nodes)

        current_root = current_x[:root_count]
        reference_root = reference_x[:root_count]
        difference = current_root - reference_root
        cosine = F.cosine_similarity(
            current_root,
            reference_root,
            dim=-1,
            eps=1e-12,
        )
        relative_l2 = difference.norm(dim=-1) / reference_root.norm(
            dim=-1
        ).clamp_min(1e-12)

        stats_x = model._encode_stats(batch) if model.use_stats else None
        fused_x = model._fuse_stats(current_x, batch)
        record = {
            'epoch': epoch,
            'split': split,
            'root_count': int(root_count),
            'weight_relative_l2': weight_relative_l2,
            'root_cosine_mean': cosine.mean().item(),
            'root_cosine_p05': torch.quantile(cosine, 0.05).item(),
            'root_relative_l2_mean': relative_l2.mean().item(),
            'root_mse': difference.pow(2).mean().item(),
            'shared_norm_mean': current_root.norm(dim=-1).mean().item(),
            'fused_norm_mean': fused_x[:root_count].norm(dim=-1).mean().item(),
        }

        if stats_x is not None:
            stats_root = stats_x[:root_count]
            record['stats_norm_mean'] = stats_root.norm(dim=-1).mean().item()

        if model.stats_fusion in ['gate', 'residual_gate']:
            gate = model.stats_gate(
                torch.cat((current_x, stats_x), dim=1)
            )[:root_count]
            record.update({
                'gate_mean': gate.mean().item(),
                'gate_std': gate.std(unbiased=False).item(),
                'gate_below_0_1': (gate < 0.1).float().mean().item(),
                'gate_above_0_9': (gate > 0.9).float().mean().item(),
            })

        records.append(record)
        print(
            'Representation audit '
            f"epoch={epoch} split={split} "
            f"cos={record['root_cosine_mean']:.6f} "
            f"rel_l2={record['root_relative_l2_mean']:.6f} "
            f"weight_rel_l2={weight_relative_l2:.6f}"
        )

    with open(context['path'], 'a', encoding='utf-8') as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + '\n')

    model.shared_backbone.train(current_training)
    return records


def build_joint_shared_audit_context(
        args, model, val_loader, test_loaders, device):
    if not (
        getattr(args, 'joint_shared_audit', 0)
        and getattr(args, 'use_sgrl_joint_shared', 0)
        and hasattr(model, 'shared_backbone')
    ):
        return None

    reference_backbone = copy.deepcopy(model.shared_backbone).to(device)
    reference_backbone.eval()
    for parameter in reference_backbone.parameters():
        parameter.requires_grad = False
    fixed_batches = _capture_fixed_audit_batches(val_loader, test_loaders)
    audit_path = os.path.join(
        args.run_artifact_dir,
        'joint_shared_representation_audit.jsonl',
    )
    print(
        'Joint-shared representation audit enabled: '
        f'source_interval={args.joint_shared_audit_interval}, '
        f'final_splits={list(fixed_batches)}, output={audit_path}'
    )
    return {
        'reference_backbone': reference_backbone,
        'fixed_batches': fixed_batches,
        'device': device,
        'path': audit_path,
    }


def _joint_lora_delta_summary(backbone):
    squared_delta = 0.0
    squared_base = 0.0
    modules = backbone._lora_modules() if hasattr(backbone, '_lora_modules') else []
    for module in modules:
        delta = module.scaling * (
            module.lora_b.weight.detach().float()
            @ module.lora_a.weight.detach().float()
        )
        base = module.base.weight.detach().float()
        squared_delta += delta.square().sum().item()
        squared_base += base.square().sum().item()
    delta_norm = squared_delta ** 0.5
    base_norm = squared_base ** 0.5
    return {
        'lora_modules': len(modules),
        'lora_delta_norm': delta_norm,
        'lora_base_weight_norm': base_norm,
        'lora_relative_delta': delta_norm / max(base_norm, 1e-12),
    }


def _joint_base_weight_drift(current_gnn, reference_gnn):
    reference_parameters = dict(reference_gnn.named_parameters())
    squared_delta = 0.0
    squared_reference = 0.0
    for name, current in current_gnn.named_parameters():
        if '.lora_a.' in name or '.lora_b.' in name:
            continue
        reference = reference_parameters[name]
        current_value = current.detach().float()
        reference_value = reference.detach().float()
        squared_delta += (current_value - reference_value).square().sum().item()
        squared_reference += reference_value.square().sum().item()
    return (squared_delta ** 0.5) / max(squared_reference ** 0.5, 1e-12)


@torch.no_grad()
def record_joint_shared_representation_audit(
        args, model, context, epoch, phase='train', force=False):
    if context is None:
        return []
    interval = getattr(args, 'joint_shared_audit_interval', 5)
    if not force and epoch >= 0 and epoch % interval != 0:
        return []

    current_training = model.shared_backbone.training
    model.shared_backbone.eval()
    reference = context['reference_backbone']
    reference.eval()
    base_weight_drift = _joint_base_weight_drift(
        model.shared_backbone.gnn,
        reference.gnn,
    )
    lora_summary = _joint_lora_delta_summary(model.shared_backbone)
    split_names = (
        list(context['fixed_batches'])
        if phase == 'best'
        else ['source_val']
    )
    records = []
    for split in split_names:
        batch = context['fixed_batches'][split].clone().to(context['device'])
        current_base = model.shared_backbone(batch, task_path=False)
        reference_base = reference(batch, task_path=False)
        current_task = model.shared_backbone(batch, task_path=True)
        root_count = (
            2 * batch.edge_label.size(0)
            if args.task_level == 'edge'
            else getattr(batch, 'batch_size', batch.num_nodes)
        )
        current_base = current_base[:root_count]
        reference_base = reference_base[:root_count]
        current_task = current_task[:root_count]
        base_delta = current_base - reference_base
        task_delta = current_task - current_base
        base_relative_l2 = base_delta.norm(dim=-1) / reference_base.norm(
            dim=-1
        ).clamp_min(1e-12)
        task_relative_l2 = task_delta.norm(dim=-1) / current_base.norm(
            dim=-1
        ).clamp_min(1e-12)
        record = {
            'epoch': epoch,
            'phase': phase,
            'split': split,
            'root_count': int(root_count),
            'base_weight_relative_l2': base_weight_drift,
            'base_cosine_to_initial': F.cosine_similarity(
                current_base,
                reference_base,
                dim=-1,
                eps=1e-12,
            ).mean().item(),
            'base_relative_l2_to_initial': base_relative_l2.mean().item(),
            'task_cosine_to_base': F.cosine_similarity(
                current_task,
                current_base,
                dim=-1,
                eps=1e-12,
            ).mean().item(),
            'task_relative_l2_to_base': task_relative_l2.mean().item(),
            'base_norm_mean': current_base.norm(dim=-1).mean().item(),
            'task_norm_mean': current_task.norm(dim=-1).mean().item(),
            'stats_residual_scale': (
                model.shared_backbone.stats_residual_scale.item()
                if hasattr(model.shared_backbone, 'stats_residual_scale')
                else None
            ),
            **lora_summary,
        }
        records.append(record)
        print(
            'Joint-shared audit '
            f'phase={phase} epoch={epoch} split={split} '
            f"base_cos={record['base_cosine_to_initial']:.6f} "
            f"base_weight_rel_l2={base_weight_drift:.6f} "
            f"task_base_rel_l2={record['task_relative_l2_to_base']:.6f} "
            f"lora_rel={record['lora_relative_delta']:.6f}"
        )

    with open(context['path'], 'a', encoding='utf-8') as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + '\n')
    model.shared_backbone.train(current_training)
    return records

def regress_train(args, regressor, optimizier, criterion,
          train_loader, val_loader, test_loaders, max_label,
          device):
    """
    Train the head model for regression task
    Args:
        args (argparse.Namespace): The arguments
        regressor (torch.nn.Module): The regressor
        optimizier (torch.optim.Optimizer): The optimizer
        criterion (torch.nn.Module): The loss function
        train_loader (torch.utils.data.DataLoader): The training data loader
        val_loader (torch.utils.data.DataLoader): The validation data loader  
        test_laders (list): A list of test data loaders
        device (torch.device): The device to train the model on
    """
    optimizier.zero_grad()
    
    best_results = {
        'best_val_mse': 1e9,
        'best_val_loss': 1e9,
        'best_epoch': -1,
        'test_results': [],
        'test_results_by_name': {},
    }
    epochs_without_improvement = 0
    audit_context = build_partial_shared_audit_context(
        args,
        regressor,
        val_loader,
        test_loaders,
        device,
    )
    record_partial_shared_representation_audit(
        args,
        regressor,
        audit_context,
        epoch=-1,
    )
    joint_audit_context = build_joint_shared_audit_context(
        args,
        regressor,
        val_loader,
        test_loaders,
        device,
    )
    record_joint_shared_representation_audit(
        args,
        regressor,
        joint_audit_context,
        epoch=-1,
    )
    
    for epoch in range(args.epochs):
        optimizier = maybe_unfreeze_partial_shared_backbone(
            args, regressor, optimizier, epoch
        )
        logger = Logger(task=args.task, max_label=max_label)
        regressor.train()
        apply_partial_shared_backbone_eval_policy(args, regressor, epoch)
        joint_gcl_total = 0.0
        joint_gcl_batches = 0

        for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch:{epoch}')):
            optimizier.zero_grad()
            batch = batch.to(device)

            ## Get the prediction from the model
            if getattr(args, 'use_sgrl_joint_shared', 0):
                y_pred, y_class, y, online_x = regressor(
                    batch,
                    return_backbone=True,
                )
            else:
                y_pred, y_class, y = regressor(batch)
                online_x = None
            supervised_loss, pred, true = compute_loss(
                args,
                y_pred,
                y,
                criterion=criterion,
            )
            loss = supervised_loss
            if (
                getattr(args, 'use_sgrl_joint_shared', 0)
                and args.joint_gcl_lambda > 0.0
            ):
                joint_gcl_loss = regressor.gcl_loss(batch, online_x=online_x)
                loss = supervised_loss + args.joint_gcl_lambda * joint_gcl_loss
                joint_gcl_total += joint_gcl_loss.detach().item()
                joint_gcl_batches += 1
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = y_pred.detach().to('cpu', non_blocking=True)

            loss.backward()
            optimizier.step()
            if (
                getattr(args, 'use_sgrl_joint_shared', 0)
                and args.joint_gcl_lambda > 0.0
            ):
                regressor.update_target_backbone()
            
            ## Update the logger and print message to the screen
            logger.update_stats(
                true=_true, pred=_pred, 
                batch_size=_true.squeeze().size(0), 
                loss=supervised_loss.detach().cpu().item()
            )

        logger.write_epoch(split='train')
        if joint_gcl_batches:
            print(
                f'joint_gcl epoch={epoch} lambda={args.joint_gcl_lambda} '
                f'loss={joint_gcl_total / joint_gcl_batches:.8f}'
            )
        ## ========== validation ========== ##
        val_res = eval_epoch(
            args, val_loader, 
            regressor, device, split='val', criterion=criterion
        )

        ## update the best results so far
        if validation_improved(
            best_results['best_val_mse'],
            val_res,
            min_delta=getattr(args, 'early_stopping_min_delta', 0.0),
        ):
            best_results['best_val_mse'] = validation_mse(val_res)
            best_results['best_val_loss'] = val_res['loss']
            best_results['best_epoch'] = epoch
            epochs_without_improvement = 0
            artifact_metrics = {
                'status': 'running',
                'task': args.task,
                'loss': args.regress_loss,
                'joint_gcl_lambda': getattr(args, 'joint_gcl_lambda', 0.0),
                'best_epoch': best_results['best_epoch'],
                'best_val_mse': best_results['best_val_mse'],
                'best_val_loss': best_results['best_val_loss'],
                'test_results': {},
            }
            checkpoint_path = save_best_checkpoint(
                args,
                regressor,
                optimizier,
                epoch,
                artifact_metrics,
                criterion=criterion,
            )
            artifact_metrics['best_checkpoint'] = str(checkpoint_path)
            write_run_metrics(args, artifact_metrics)
            print(f"Saved isolated best checkpoint to {checkpoint_path}")
        else:
            epochs_without_improvement += 1

        record_partial_shared_representation_audit(
            args,
            regressor,
            audit_context,
            epoch=epoch,
        )
        record_joint_shared_representation_audit(
            args,
            regressor,
            joint_audit_context,
            epoch=epoch,
        )

        print( "=====================================")
        print(f" Best epoch: {best_results['best_epoch']}, mse: {best_results['best_val_mse']}, loss: {best_results['best_val_loss']}")
        print(" Test results: deferred until training completes")
        print( "=====================================")

        patience = getattr(args, 'early_stopping_patience', 0)
        if patience > 0 and epochs_without_improvement >= patience:
            print(
                f'Early stopping at epoch {epoch}: no validation-MSE '
                f'improvement greater than '
                f'{getattr(args, "early_stopping_min_delta", 0.0):.2e} '
                f'for {patience} epochs.'
            )
            break

    checkpoint_path = os.path.join(args.run_artifact_dir, 'best_model.pt')
    load_best_checkpoint(
        args,
        regressor,
        map_location=device,
        criterion=criterion,
    )
    print(
        f"Loaded best checkpoint from epoch {best_results['best_epoch']} "
        'for final transfer evaluation.'
    )
    record_joint_shared_representation_audit(
        args,
        regressor,
        joint_audit_context,
        epoch=best_results['best_epoch'],
        phase='best',
        force=True,
    )
    best_validation_result = eval_epoch(
        args,
        val_loader,
        regressor,
        device,
        split='val:best',
        criterion=criterion,
    )
    test_results = []
    test_results_by_name = {}
    for test_name, test_loader in test_loaders.items():
        res = eval_epoch(
            args,
            test_loader,
            regressor,
            device,
            split=f'test:{test_name}',
            criterion=criterion,
        )
        test_results.append(res)
        test_results_by_name[test_name] = res
    best_results['test_results'] = test_results
    best_results['test_results_by_name'] = test_results_by_name

    final_metrics = {
        'status': 'completed',
        'task': args.task,
        'loss': args.regress_loss,
        'joint_gcl_lambda': getattr(args, 'joint_gcl_lambda', 0.0),
        'best_epoch': best_results['best_epoch'],
        'best_val_mse': best_results['best_val_mse'],
        'best_val_loss': best_results['best_val_loss'],
        'validation_results': best_validation_result,
        'test_results': test_results_by_name,
        'best_checkpoint': checkpoint_path,
    }
    update_best_checkpoint_metrics(args, final_metrics)
    write_run_metrics(args, final_metrics)
    return best_results

def class_train(args, classifier,optimizer_classifier, 
          train_loader, val_loader, test_loaders, max_label,
          device):
    """
    Train model for capacitance classification task
    Args:
        args (argparse.Namespace): The arguments
        classifier (torch.nn.Module): The classifier
        optimizer_classifier (torch.optim.Optimizer): The optimizer for the classifier
        train_loader (torch.utils.data.DataLoader): The training data loader
        val_loader (torch.utils.data.DataLoader): The validation data loader  
        test_laders (list): A list of test data loaders
        device (torch.device): The device to train the model on
    """
    # Reset optimizers
    optimizer_classifier.zero_grad()

    # create the directory to save the model
    classifier_save_dir = os.path.join("models_node_cap_classifier")
    
    os.makedirs(classifier_save_dir, exist_ok=True)
    
    # initialize the best model metrics

    best_f1 = 0.0
    
    for epoch in range(args.epochs):
        optimizer_classifier = maybe_unfreeze_partial_shared_backbone(
            args, classifier, optimizer_classifier, epoch
        )
        epoch_start_time = time.time()

        # add the logger for classification task
        logger = Logger(task='classification', max_label=max_label)
        classifier.train()
        apply_partial_shared_backbone_eval_policy(args, classifier, epoch)

        for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch:{epoch}')):
            # Move batch to device
            batch = batch.to(device)
            optimizer_classifier.zero_grad()
            
            ## Get the prediction from the model
            class_logits,true_class, true_label = classifier(batch)
            class_probs = F.softmax(class_logits, dim=1)

            predict_class = torch.argmax(class_probs, dim=1)
            
            ## set the loss function for classification task
            if args.class_loss == 'focal':
                # calculate the class weights
                num_classes = class_logits.size(1)
                class_weights = compute_class_weights(true_class, num_classes)
                # apply Focal Loss
                criterion = FocalLoss(gamma=2.0, class_weights=class_weights)
                class_loss = criterion(class_logits, true_class)
            elif args.class_loss == 'bsmCE':
                ## calculate the number of samples per class
                sample_per_class = []
                for i in range(args.num_classes):
                    sample_per_class.append(torch.sum(true_class == i).item())
                # apply Balanced Softmax CE
                criterion = BalancedSoftmax(sample_per_class)
                class_loss = criterion(class_logits, true_class)

            elif args.class_loss == 'cross_entropy':
                criterion = torch.nn.CrossEntropyLoss()
                class_loss = criterion(class_logits, true_class)
            else:
                raise ValueError(f"Loss function {args.class_loss} not supported!")
                
            class_loss.backward()
            optimizer_classifier.step()
            
            # update the statistics of classification results
            _class_pred = predict_class.detach().to('cpu', non_blocking=True)
            _class_true = true_class.detach().to('cpu', non_blocking=True)
            logger.update_stats(
                true=_class_true, 
                pred=_class_pred, 
                batch_size=_class_true.squeeze().size(0), 
                loss=class_loss if isinstance(class_loss, float) else class_loss.detach().cpu().item()
            )

        print(f"\n===== Epoch {epoch}/{args.epochs} - Elapsed: {time.time() - epoch_start_time:.2f}s =====")
        print("Classification results:")
        logger.write_epoch(split='train')
        
        ## ========== validation ========== ##
        val_class_res = eval_epoch(
            args, val_loader,
            classifier, device, split='val', criterion=criterion
        )
        #visualize_tsne(classifier, val_loader, device, num_samples=2000)

        ## ========== testing on other datasets ========== ##
        test_class_results = {}           
        eval_flag = False

        # if the f1 of the current classifier model is the highest, save the best model
        if val_class_res['f1'] > best_f1:
            best_f1 = val_class_res['f1']
            eval_flag = True
        
        if eval_flag :
            for test_name in test_loaders.keys():
                print(test_name)
                test_class_res = eval_epoch(
                    args, test_loaders[test_name], 
                    classifier, device, split='test', criterion=criterion
                )
                test_class_results[test_name] = test_class_res
        
        print( "=====================================")
        print(f" Best epoch: {epoch}, f1: {val_class_res['f1']}")
        print(f" Test results: {[res for res in test_class_results.values()]}")
        print( "=====================================")


def build_downstream_model(args, sgrl_online_state=None):
    if getattr(args, 'use_sgrl_joint_shared', 0):
        model = JointSharedGraphHead(args)
        if sgrl_online_state is None:
            raise ValueError("SGRL joint-shared mode requires an online encoder state_dict.")
        model.load_online_encoder_state(sgrl_online_state)
        return model

    if getattr(args, 'use_sgrl_partial_shared', 0):
        model = PartialSharedGraphHead(args)
        if sgrl_online_state is None:
            raise ValueError("SGRL partial-shared mode requires an online encoder state_dict.")
        model.load_online_encoder_state(sgrl_online_state)
        return model

    if getattr(args, 'use_sgrl_graph_init', 0):
        model = GraphHead(args)
        if sgrl_online_state is None:
            raise ValueError("SGRL init-reuse mode requires an online encoder state_dict.")
        model.load_sgrl_encoder_init(sgrl_online_state)
        return model

    if getattr(args, 'use_sgrl_online_features', 0):
        model = OnlineFeatureGraphHead(args)
        if sgrl_online_state is None:
            raise ValueError("SGRL online-feature mode requires an online encoder state_dict.")
        model.load_online_encoder_state(
            sgrl_online_state,
            freeze=not getattr(args, 'finetune_sgrl_online', 0),
        )
        return model

    if getattr(args, 'use_sgrl_backbone', 0):
        model = SgrlBackboneHead(args)
        if sgrl_online_state is None:
            raise ValueError("SGRL backbone mode requires an online encoder state_dict.")
        model.load_online_encoder_state(
            sgrl_online_state,
            freeze=args.sgrl_mode == 'freeze',
        )
        return model

    return GraphHead(args)


def trainable_parameters(model):
    return [param for param in model.parameters() if param.requires_grad]


def print_parameter_summary(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    frozen = total - trainable
    print(
        "Model parameters: "
        f"total={total:,}, trainable={trainable:,}, frozen={frozen:,}."
    )
    if hasattr(model, 'deployment_parameter_count'):
        deployment = model.deployment_parameter_count()
        message = f'Deployment parameters: unmerged={deployment:,}'
        if hasattr(model, 'merged_deployment_parameter_count'):
            merged = model.merged_deployment_parameter_count()
            message += f', merged={merged:,}'
        message += '; target backbone and predictor are training-only.'
        print(message)

    child_summaries = []
    for name, child in model.named_children():
        child_total = sum(param.numel() for param in child.parameters())
        child_trainable = sum(
            param.numel() for param in child.parameters()
            if param.requires_grad
        )
        if child_total > 0:
            child_summaries.append(
                f"{name}: total={child_total:,}, trainable={child_trainable:,}"
            )
    if child_summaries:
        print("Top-level parameter summary: " + "; ".join(child_summaries))


def build_optimizer(args, model, criterion=None):
    criterion_params = (
        trainable_parameters(criterion)
        if criterion is not None and hasattr(criterion, 'parameters')
        else []
    )
    joint_backbone_lr = getattr(args, 'joint_backbone_lr', None)
    if (
        getattr(args, 'use_sgrl_joint_shared', 0)
        and joint_backbone_lr is not None
    ):
        base_backbone_params = []
        task_params = list(criterion_params)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            is_base_backbone = (
                name.startswith('shared_backbone.gnn.')
                and '.lora_a.' not in name
                and '.lora_b.' not in name
            )
            if is_base_backbone:
                base_backbone_params.append(param)
            else:
                task_params.append(param)
        print(
            'Using separate optimizer groups for joint-shared base GNN '
            f'(backbone_lr={joint_backbone_lr}, task_lr={args.lr}, '
            f'backbone_values={sum(p.numel() for p in base_backbone_params):,}, '
            f'task_values={sum(p.numel() for p in task_params):,}).'
        )
        param_groups = []
        if base_backbone_params:
            param_groups.append({
                'params': base_backbone_params,
                'lr': joint_backbone_lr,
            })
        if task_params:
            param_groups.append({'params': task_params, 'lr': args.lr})
        return torch.optim.Adam(param_groups, lr=args.lr)

    partial_shared_backbone_lr = getattr(args, 'partial_shared_backbone_lr', None)
    if has_partial_shared_backbone(args, model) and partial_shared_backbone_lr is not None:
        backbone_params = [
            param for name, param in model.named_parameters()
            if param.requires_grad and name.startswith('shared_backbone.')
        ]
        downstream_params = [
            param for name, param in model.named_parameters()
            if param.requires_grad and not name.startswith('shared_backbone.')
        ] + criterion_params
        print(
            "Using separate optimizer groups for partial-shared backbone "
            f"(backbone_lr={partial_shared_backbone_lr}, "
            f"downstream_lr={args.lr})."
        )
        param_groups = []
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': partial_shared_backbone_lr,
            })
        if downstream_params:
            param_groups.append({'params': downstream_params, 'lr': args.lr})
        return torch.optim.Adam(param_groups, lr=args.lr)

    if getattr(args, 'finetune_sgrl_online', 0) and hasattr(model, 'online_encoder'):
        online_params = [
            param for param in model.online_encoder.parameters()
            if param.requires_grad
        ]
        downstream_params = [
            param for name, param in model.named_parameters()
            if param.requires_grad and not name.startswith('online_encoder.')
        ] + criterion_params
        print(
            "Using separate optimizer groups for online feature finetuning "
            f"(online_lr={args.sgrl_online_lr}, downstream_lr={args.lr})."
        )
        return torch.optim.Adam(
            [
                {'params': online_params, 'lr': args.sgrl_online_lr},
                {'params': downstream_params, 'lr': args.lr},
            ],
            lr=args.lr,
        )

    return torch.optim.Adam(
        trainable_parameters(model) + criterion_params,
        lr=args.lr,
    )


def downstream_train(args, dataset, device, cl_embeds=None, sgrl_online_state=None):
    """ downstream task training for link prediction
    Args:
        args (argparse.Namespace): The arguments
        dataset (torch_geometric.data.InMemoryDataset): The dataset
        batch_index (torch.tensor): The batch index for all_node_embeds
        all_node_embeds (torch.tensor): The node embeddings come from the contrastive learning
        device (torch.device): The device to train the model on
    """
    if getattr(args, 'use_sgrl_embeds', args.sgrl):
        dataset.set_cl_embeds(cl_embeds)

    dataset.norm_nfeat([NET, DEV])

    

    
    
    # Subgraph sampling for each dataset graph & PE calculation
    (
        train_loader, val_loader, test_loaders, max_label
    ) = dataset_sampling(args, dataset)


    if args.task == 'regression':
        # set the loss function for regression
        if args.regress_loss == 'gai':
            gmm_path = train_gmm(
                dataset,
                output_path=os.path.join(args.run_artifact_dir, 'gmm.pkl'),
            )
            criterion = GAILoss(init_noise_sigma=args.noise_sigma, gmm=gmm_path, device=device)
        elif args.regress_loss == 'bmc':
            criterion = BMCLoss(init_noise_sigma=args.noise_sigma, device=device)
        elif args.regress_loss == 'bni':
            _, bin_edges, bin_count = get_lds_weights(
                dataset._data.edge_label[:, 1], 
                args.lds_kernel, args.lds_ks, args.lds_sigma
            )
            criterion = BNILoss(args.noise_sigma, bin_edges, bin_count,  device=device)
        elif args.regress_loss == 'mse':
            criterion = torch.nn.MSELoss(reduction='mean')
        elif args.regress_loss == 'lds':
            weights, _, _ = get_lds_weights(
                dataset._data.edge_label[:, 1], 
                args.lds_kernel, args.lds_ks, args.lds_sigma
            )
            dataset._data.edge_label[:, 1] = weights
            criterion = WeightedMSE()
        else:
            raise ValueError(f"Loss func {args.regress_loss} not supported!")
        
        start = time.time()
        model = build_downstream_model(args, sgrl_online_state)
        model = model.to(device)
        if getattr(args, 'partial_shared_freeze_epochs', 0) > 0:
            set_partial_shared_backbone_trainable(args, model, False)
        print_parameter_summary(model)
        optimizier = build_optimizer(args, model, criterion=criterion)
        
        regress_train(args, model, optimizier, criterion,
              train_loader, val_loader, test_loaders, max_label,
              device)
        
    elif args.task == 'classification':
        model = build_downstream_model(args, sgrl_online_state)
        start = time.time()
        model = model.to(device)
        if getattr(args, 'partial_shared_freeze_epochs', 0) > 0:
            set_partial_shared_backbone_trainable(args, model, False)
        print_parameter_summary(model)
        optimizer = build_optimizer(args, model)
        class_train(args, model, optimizer, train_loader, val_loader, test_loaders, max_label,
              device)
    
    else:
        raise ValueError(f"Task type {args.task} not supported!")
        

    elapsed = time.time() - start
    timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed))
    print(f"Done! Training took {timestr}")
   






    
    
    
