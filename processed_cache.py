"""Canonical identity and validation for processed graph caches."""

import hashlib
import json
import os
from pathlib import Path

import torch

from run_artifacts import file_sha256, write_json_atomic


PROCESSED_CACHE_SCHEMA_VERSION = 1


def relation_sampling_distribution(negative_edge_ratio):
    """Describe the existing relation-balanced sampling policy."""
    negative_edge_ratio = float(negative_edge_ratio)
    if negative_edge_ratio == 0.0:
        negative_candidates = 'none'
    elif negative_edge_ratio > 1.0:
        negative_candidates = 'global'
    else:
        negative_candidates = 'structured_per_relation'
    return {
        'name': 'relation_balanced_min_count',
        'version': 1,
        'positive_selection': 'uniform_without_replacement_per_relation',
        'positive_count': 'floor(min_relation_count * sample_rate)',
        'negative_candidates': negative_candidates,
        'negative_selection': 'uniform_without_replacement_per_relation',
        'negative_count': (
            'floor(min_relation_count * negative_edge_ratio * sample_rate)'
        ),
    }


def build_processed_cache_inputs(
        graph_name, raw_sha256, sample_rate, negative_edge_ratio,
        to_undirected, task_level, net_only, class_boundaries,
        relation_sample_seed, graph_relation_sample_seed):
    """Build the complete scientific identity for one processed graph."""
    return {
        'cache_schema_version': PROCESSED_CACHE_SCHEMA_VERSION,
        'graph_name': str(graph_name),
        'raw_sha256': str(raw_sha256),
        'sample_rate': float(sample_rate),
        'negative_edge_ratio': float(negative_edge_ratio),
        'to_undirected': bool(to_undirected),
        'task_level': str(task_level),
        'net_only': bool(net_only),
        'class_boundaries': [float(value) for value in class_boundaries],
        'sampling_distribution': relation_sampling_distribution(
            negative_edge_ratio
        ),
        'relation_sample_seed': int(relation_sample_seed),
        'graph_relation_sample_seed': int(graph_relation_sample_seed),
    }


def canonical_manifest_json(inputs):
    return json.dumps(
        inputs,
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    )


def processed_cache_key(inputs):
    return hashlib.sha256(
        canonical_manifest_json(inputs).encode('utf-8')
    ).hexdigest()


def processed_cache_filename(graph_name, inputs):
    return f'{graph_name}_processed_{processed_cache_key(inputs)}.pt'


def processed_cache_manifest_path(processed_path):
    return f'{processed_path}.manifest.json'


def _validation_error(message, path):
    return RuntimeError(f'Invalid processed cache {path}: {message}')


def validate_processed_cache(processed_path, expected_inputs):
    """Validate provenance and content before a processed tensor is loaded."""
    processed_path = Path(processed_path).resolve()
    manifest_path = Path(
        processed_cache_manifest_path(processed_path)
    ).resolve()
    if not processed_path.is_file():
        raise _validation_error('processed file is missing', processed_path)
    if not manifest_path.is_file():
        raise _validation_error('sidecar manifest is missing', processed_path)

    try:
        with manifest_path.open(encoding='utf-8') as source:
            publication = json.load(source)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise _validation_error(
            f'malformed sidecar manifest ({error})', processed_path
        ) from error

    expected_key = processed_cache_key(expected_inputs)
    if publication.get('cache_key') != expected_key:
        raise _validation_error('cache key mismatch', processed_path)
    if publication.get('inputs') != expected_inputs:
        raise _validation_error('canonical input manifest mismatch', processed_path)

    actual_processed_sha256 = file_sha256(processed_path)
    if publication.get('processed_sha256') != actual_processed_sha256:
        raise _validation_error('processed SHA256 mismatch', processed_path)

    return {
        'processed_path': str(processed_path),
        'cache_key': expected_key,
        'cache_schema_version': expected_inputs['cache_schema_version'],
        'manifest_path': str(manifest_path),
        'manifest_sha256': file_sha256(manifest_path),
        'raw_sha256': expected_inputs['raw_sha256'],
        'processed_sha256': actual_processed_sha256,
        'sampling_distribution': expected_inputs['sampling_distribution'],
        'relation_sample_seed': expected_inputs['relation_sample_seed'],
        'graph_relation_sample_seed': expected_inputs[
            'graph_relation_sample_seed'
        ],
    }


def publish_processed_cache_manifest(processed_path, inputs):
    """Publish a sidecar for an already-written processed tensor."""
    processed_path = Path(processed_path).resolve()
    if not processed_path.is_file():
        raise _validation_error(
            'cannot publish a manifest for a missing file', processed_path
        )
    manifest_path = processed_cache_manifest_path(processed_path)
    if os.path.exists(manifest_path):
        raise _validation_error(
            'refusing to overwrite an existing sidecar manifest',
            processed_path,
        )
    write_json_atomic(manifest_path, {
        'cache_key': processed_cache_key(inputs),
        'inputs': inputs,
        'processed_sha256': file_sha256(processed_path),
    })
    return validate_processed_cache(processed_path, inputs)


def load_validated_processed_cache(processed_path, expected_inputs):
    """Validate a cache hit before invoking the trusted PyTorch loader."""
    provenance = validate_processed_cache(processed_path, expected_inputs)
    payload = torch.load(processed_path)
    return payload, provenance
