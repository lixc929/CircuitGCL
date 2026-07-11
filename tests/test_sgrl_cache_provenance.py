import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from sgrl_train import (
    embedding_cache_fingerprint,
    embedding_cache_identity,
    publish_embedding_metadata,
    publish_sgrl_checkpoint_metadata,
    sgrl_cache_fingerprint,
    sgrl_cache_identity,
    validate_embedding_cache,
    validate_sgrl_checkpoint,
)


def make_args():
    return SimpleNamespace(
        seed=0,
        pretraining_seed=0,
        embedding_inference_seed=20260711,
        sgrl_graph_scope='source',
        sgrl_pretrain_target_update='sgrl_dual_rsm_ema',
        cl_model='clustergcn',
        cl_gnn_layers=2,
        cl_hid_dim=32,
        cl_act_fn='tanh',
        cl_dropout=0.2,
        cl_batch_size=64,
        cl_num_neighbors=10,
        num_hops=2,
        num_workers=0,
        use_bn=0,
        cl_epochs=20,
        e1_lr=1e-4,
        e2_lr=2e-5,
        momentum=0.99,
        weight_decay=0.0,
    )


def cache_ref(graph_name, suffix):
    return {
        'graph_name': graph_name,
        'cache_key': f'cache-{suffix}',
        'raw_sha256': f'raw-{suffix}',
        'processed_sha256': f'processed-{suffix}',
        'relation_sample_seed': 20260711,
        'graph_relation_sample_seed': 100 + len(suffix),
    }


class SgrlCacheProvenanceTest(unittest.TestCase):
    def test_checkpoint_is_source_only_but_embedding_binds_all_graphs(self):
        args = make_args()
        first = [cache_ref('source', 's1'), cache_ref('transfer', 't1')]
        transfer_changed = [
            cache_ref('source', 's1'), cache_ref('transfer', 't2')
        ]
        source_changed = [
            cache_ref('source', 's2'), cache_ref('transfer', 't1')
        ]

        first_key = sgrl_cache_fingerprint(args, ['source'], first)
        self.assertEqual(
            first_key,
            sgrl_cache_fingerprint(args, ['source'], transfer_changed),
        )
        self.assertNotEqual(
            first_key,
            sgrl_cache_fingerprint(args, ['source'], source_changed),
        )
        self.assertNotEqual(
            embedding_cache_fingerprint(args, first_key, first),
            embedding_cache_fingerprint(
                args, first_key, transfer_changed
            ),
        )

    def test_checkpoint_metadata_is_required_and_sha_validated(self):
        args = make_args()
        caches = [cache_ref('source', 's1')]
        identity = sgrl_cache_identity(args, ['source'], caches)
        key = sgrl_cache_fingerprint(args, ['source'], caches)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'online.pt'
            checkpoint.write_bytes(b'checkpoint-v1')
            with self.assertRaisesRegex(RuntimeError, 'metadata is missing'):
                validate_sgrl_checkpoint(str(checkpoint), key, identity)

            published = publish_sgrl_checkpoint_metadata(
                str(checkpoint), key, identity
            )
            self.assertEqual(published['cache_key'], key)

            checkpoint.write_bytes(b'checkpoint-v2')
            with self.assertRaisesRegex(RuntimeError, 'SHA256 mismatch'):
                validate_sgrl_checkpoint(str(checkpoint), key, identity)

    def test_checkpoint_identity_and_schema_mismatches_are_rejected(self):
        args = make_args()
        caches = [cache_ref('source', 's1')]
        identity = sgrl_cache_identity(args, ['source'], caches)
        key = sgrl_cache_fingerprint(args, ['source'], caches)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'online.pt'
            checkpoint.write_bytes(b'checkpoint')
            published = publish_sgrl_checkpoint_metadata(
                str(checkpoint), key, identity
            )
            metadata_path = Path(published['metadata_path'])

            with self.assertRaisesRegex(RuntimeError, 'identity mismatch'):
                validate_sgrl_checkpoint(
                    str(checkpoint), key, {**identity, 'graph_scope': 'all'}
                )

            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            metadata['cache_schema_version'] = 999
            metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'schema mismatch'):
                validate_sgrl_checkpoint(str(checkpoint), key, identity)

    def test_embedding_binds_checkpoint_identity_and_content(self):
        args = make_args()
        caches = [cache_ref('source', 's1'), cache_ref('transfer', 't1')]
        checkpoint_key = sgrl_cache_fingerprint(args, ['source'], caches)
        identity = embedding_cache_identity(args, checkpoint_key, caches)
        key = embedding_cache_fingerprint(args, checkpoint_key, caches)

        with tempfile.TemporaryDirectory() as directory:
            embedding = Path(directory) / 'embedding.pt'
            embedding.write_bytes(b'embedding-v1')
            provenance = publish_embedding_metadata(
                str(embedding), key, identity, 'checkpoint-sha', 'sampler-sha'
            )
            self.assertEqual(
                provenance['embedding_sampler_fingerprint'], 'sampler-sha'
            )

            with self.assertRaisesRegex(
                    RuntimeError, 'checkpoint provenance mismatch'):
                validate_embedding_cache(
                    str(embedding), key, identity, 'different-checkpoint'
                )

            embedding.write_bytes(b'embedding-v2')
            with self.assertRaisesRegex(RuntimeError, 'SHA256 mismatch'):
                validate_embedding_cache(
                    str(embedding), key, identity, 'checkpoint-sha'
                )


if __name__ == '__main__':
    unittest.main()
