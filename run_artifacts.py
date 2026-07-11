import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import torch


def _json_safe(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return str(value)


def _git_commit():
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def file_sha256(path, chunk_size=1024 * 1024):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    with temporary_path.open('w', encoding='utf-8') as output:
        json.dump(_json_safe(payload), output, indent=2, sort_keys=True)
        output.write('\n')
    os.replace(temporary_path, path)


def prepare_run_artifacts(args, log_filename):
    log_path = Path(log_filename).resolve()
    artifact_dir = log_path.parent / f'{log_path.stem}_artifacts'
    artifact_dir.mkdir(parents=True, exist_ok=False)

    args.run_log_path = str(log_path)
    args.run_artifact_dir = str(artifact_dir.resolve())
    args.git_commit = _git_commit()

    config = {
        'status': 'running',
        'started_at': datetime.datetime.now().astimezone().isoformat(),
        'pid': os.getpid(),
        'host': socket.gethostname(),
        'git_commit': args.git_commit,
        'command': [sys.executable, *sys.argv],
        'log_path': args.run_log_path,
        'artifact_dir': args.run_artifact_dir,
        'args': vars(args),
    }
    write_json_atomic(artifact_dir / 'run_config.json', config)
    return artifact_dir


def update_run_config(args, metadata):
    """Atomically merge runtime provenance into the run config."""
    config_path = Path(args.run_artifact_dir) / 'run_config.json'
    with config_path.open(encoding='utf-8') as config_file:
        config = json.load(config_file)
    config.setdefault('runtime_metadata', {}).update(_json_safe(metadata))
    config['args'] = _json_safe(vars(args))
    write_json_atomic(config_path, config)
    return config_path


def _state_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_state_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_state_to_cpu(item) for item in value)
    return value


def save_best_checkpoint(
        args, model, optimizer, epoch, metrics, criterion=None):
    artifact_dir = Path(args.run_artifact_dir)
    checkpoint_path = artifact_dir / 'best_model.pt'
    temporary_path = artifact_dir / 'best_model.pt.tmp'
    payload = {
        'epoch': epoch,
        'git_commit': getattr(args, 'git_commit', None),
        'args': _json_safe(vars(args)),
        'metrics': _json_safe(metrics),
        'model_state_dict': _state_to_cpu(model.state_dict()),
        'optimizer_state_dict': _state_to_cpu(optimizer.state_dict()),
    }
    if criterion is not None and hasattr(criterion, 'state_dict'):
        payload['criterion_state_dict'] = _state_to_cpu(
            criterion.state_dict()
        )
    torch.save(payload, temporary_path)
    os.replace(temporary_path, checkpoint_path)
    return checkpoint_path


def load_best_checkpoint(
        args, model, map_location='cpu', criterion=None):
    checkpoint_path = Path(args.run_artifact_dir) / 'best_model.pt'
    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=True,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    if criterion is not None and 'criterion_state_dict' in checkpoint:
        criterion.load_state_dict(checkpoint['criterion_state_dict'])
    return checkpoint


def update_best_checkpoint_metrics(args, metrics):
    artifact_dir = Path(args.run_artifact_dir)
    checkpoint_path = artifact_dir / 'best_model.pt'
    temporary_path = artifact_dir / 'best_model.pt.tmp'
    checkpoint = torch.load(
        checkpoint_path,
        map_location='cpu',
        weights_only=True,
    )
    checkpoint['metrics'] = _json_safe(metrics)
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, checkpoint_path)
    return checkpoint_path


def write_run_metrics(args, metrics):
    path = Path(args.run_artifact_dir) / 'metrics.json'
    write_json_atomic(path, metrics)
    return path


def finalize_run_artifacts(args, status):
    config_path = Path(args.run_artifact_dir) / 'run_config.json'
    with config_path.open(encoding='utf-8') as config_file:
        config = json.load(config_file)
    config['status'] = status
    config['ended_at'] = datetime.datetime.now().astimezone().isoformat()
    write_json_atomic(config_path, config)
    return config_path
