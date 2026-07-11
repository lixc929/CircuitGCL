"""Stage-scoped random-number streams and stable scientific fingerprints."""

from contextlib import contextmanager
import hashlib
import random

import numpy as np
import torch


STAGE_SEED_FIELDS = (
    'pretraining_seed',
    'embedding_inference_seed',
    'downstream_seed',
    'train_sampler_seed',
    'relation_sample_seed',
    'split_seed',
    'eval_seed',
)


def resolve_stage_seeds(args):
    """Resolve missing stage seeds from the legacy top-level seed in-place."""
    base_seed = int(args.seed)
    for field in STAGE_SEED_FIELDS:
        if getattr(args, field, None) is None:
            setattr(args, field, base_seed)
        else:
            setattr(args, field, int(getattr(args, field)))
    return {field: getattr(args, field) for field in STAGE_SEED_FIELDS}


def seed_all(seed, include_cuda=True):
    """Reset Python, NumPy, Torch CPU, and optionally every visible CUDA RNG."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if include_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@contextmanager
def fixed_rng(seed, include_cuda=False):
    """Use a fixed CPU RNG view and restore caller RNG state on exit."""
    if include_cuda:
        raise ValueError(
            'fixed_rng is CPU-only; CUDA RNG ownership must be explicit.'
        )
    seed = int(seed)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        random.seed(seed)
        np.random.seed(seed)
        torch.random.default_generator.manual_seed(seed)
        try:
            yield seed
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def stable_seed(base_seed, namespace):
    """Derive a stable non-negative Torch-compatible seed for a namespace."""
    digest = hashlib.sha256(str(namespace).encode('utf-8')).digest()
    offset = int.from_bytes(digest[:4], byteorder='little')
    return (int(base_seed) + offset) % (2 ** 31)


class IsolatedRNGStream:
    """A persistent CPU RNG stream that never replaces its caller's stream."""

    def __init__(self, seed):
        self.seed = int(seed)
        with fixed_rng(self.seed, include_cuda=False):
            self.python_state = random.getstate()
            self.numpy_state = np.random.get_state()
            self.torch_state = torch.random.get_rng_state()

    @contextmanager
    def activate(self):
        caller_python_state = random.getstate()
        caller_numpy_state = np.random.get_state()
        caller_torch_state = torch.random.get_rng_state()
        random.setstate(self.python_state)
        np.random.set_state(self.numpy_state)
        torch.random.set_rng_state(self.torch_state)
        try:
            yield
        finally:
            self.python_state = random.getstate()
            self.numpy_state = np.random.get_state()
            self.torch_state = torch.random.get_rng_state()
            random.setstate(caller_python_state)
            np.random.set_state(caller_numpy_state)
            torch.random.set_rng_state(caller_torch_state)


class IsolatedRNGDataLoader:
    """Iterate a loader from a persistent sampler-only RNG stream."""

    def __init__(self, loader, seed):
        self.loader = loader
        self.sampler_seed = int(seed)
        self._rng_stream = IsolatedRNGStream(self.sampler_seed)

    def __iter__(self):
        with self._rng_stream.activate():
            iterator = iter(self.loader)
        while True:
            try:
                with self._rng_stream.activate():
                    batch = next(iterator)
            except StopIteration:
                return
            yield batch

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)


def _update_tensor_hash(digest, name, tensor):
    tensor = tensor.detach().cpu().contiguous()
    digest.update(str(name).encode('utf-8'))
    digest.update(str(tensor.dtype).encode('utf-8'))
    digest.update(str(tuple(tensor.shape)).encode('utf-8'))
    if tensor.dim() == 0:
        tensor = tensor.reshape(1)
    digest.update(tensor.view(torch.uint8).numpy().tobytes())


def batch_fingerprint(batch):
    """Hash the ordered roots and sampled view represented by a PyG batch."""
    digest = hashlib.sha256()
    keys = batch.keys() if callable(getattr(batch, 'keys', None)) else []
    for key in sorted(keys):
        value = getattr(batch, key)
        if torch.is_tensor(value):
            _update_tensor_hash(digest, key, value)
    return digest.hexdigest()


class SamplerFingerprint:
    """Accumulate an ordered, multi-batch sampler/view fingerprint."""

    def __init__(self):
        self._digest = hashlib.sha256()
        self._count = 0

    def update(self, batch):
        fingerprint = batch_fingerprint(batch)
        self._digest.update(str(self._count).encode('utf-8'))
        self._digest.update(fingerprint.encode('ascii'))
        self._count += 1
        return fingerprint

    def hexdigest(self):
        return self._digest.hexdigest()


def state_dict_fingerprint(state_dict):
    """Hash a model state without depending on Torch's pickle serialization."""
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key]
        if torch.is_tensor(value):
            _update_tensor_hash(digest, key, value)
        else:
            digest.update(str(key).encode('utf-8'))
            digest.update(repr(value).encode('utf-8'))
    return digest.hexdigest()


def split_fingerprint(split_indices):
    """Hash split indices and their resolved seed in a serialization-free form."""
    digest = hashlib.sha256()
    digest.update(str(int(split_indices['seed'])).encode('utf-8'))
    _update_tensor_hash(digest, 'train', split_indices['train'])
    _update_tensor_hash(digest, 'val', split_indices['val'])
    return digest.hexdigest()
