#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/lixc/.conda/envs/RCG/bin/python}"
GPU="${GPU:-4}"
SEED="${2:-0}"
CELL="${1:-}"
LOG_ROOT="${LOG_ROOT:-logs/protocol_e0_20260711}"

cd "${ROOT_DIR}"

COMMON_ARGS=(
    --task_level edge
    --task regression
    --dataset ssram+digtime+timing_ctrl+array_128_32_8t
    --net_only True
    --small_dataset_sample_rates 1.0
    --large_dataset_sample_rates 0.1
    --num_hops 2
    --num_neighbors 8
    --num_workers 0
    --epochs 80
    --early_stopping_patience 12
    --early_stopping_min_delta 1e-6
    --batch_size 512
    --lr 0.0001
    --model clustergcn
    --num_gnn_layers 2
    --regress_loss mse
    --seed "${SEED}"
    --split_seed "${SEED}"
    --eval_seed 20260711
    --gpu "${GPU}"
)

STATIC_ARGS=(
    --sgrl 1
    --sgrl_mode static
    --cl_epochs 5
    --cl_model clustergcn
    --cl_gnn_layers 2
    --cl_hid_dim 64
    --cl_batch_size 32768
    --cl_num_neighbors 8
    --hid_dim 63
)

run_static() {
    local protocol="$1"
    local graph_scope="$2"
    local normalization_scope="$3"
    local target_update="$4"
    local name="$5"
    exec env \
        OPENBLAS_NUM_THREADS=4 \
        OMP_NUM_THREADS=4 \
        MKL_NUM_THREADS=4 \
        NUMEXPR_NUM_THREADS=4 \
        "${PYTHON_BIN}" main.py \
        "${COMMON_ARGS[@]}" \
        "${STATIC_ARGS[@]}" \
        --protocol "${protocol}" \
        --sgrl_graph_scope "${graph_scope}" \
        --normalization_scope "${normalization_scope}" \
        --sgrl_pretrain_target_update "${target_update}" \
        --log_dir "${LOG_ROOT}/${name}_seed${SEED}"
}

run_nogcl() {
    local protocol="$1"
    local normalization_scope="$2"
    local name="$3"
    exec env \
        OPENBLAS_NUM_THREADS=4 \
        OMP_NUM_THREADS=4 \
        MKL_NUM_THREADS=4 \
        NUMEXPR_NUM_THREADS=4 \
        "${PYTHON_BIN}" main.py \
        "${COMMON_ARGS[@]}" \
        --protocol "${protocol}" \
        --normalization_scope "${normalization_scope}" \
        --sgrl 0 \
        --hid_dim 64 \
        --log_dir "${LOG_ROOT}/${name}_seed${SEED}"
}

case "${CELL}" in
    A)
        run_static transductive_legacy all all \
            sgrl_dual_rsm_ema static_allgraph_allnorm
        ;;
    B)
        run_static transductive_legacy all source \
            sgrl_dual_rsm_ema static_allgraph_sourcenorm
        ;;
    C)
        run_static transductive_legacy source all \
            sgrl_dual_rsm_ema static_sourcegraph_allnorm
        ;;
    D)
        run_static strict_inductive source source \
            sgrl_dual_rsm_ema static_sourcegraph_sourcenorm
        ;;
    E)
        run_static strict_inductive source source \
            circuitgcl_text_ema_only static_sourcegraph_sourcenorm_emaonly
        ;;
    F)
        run_nogcl transductive_legacy all nogcl_allnorm
        ;;
    G)
        run_nogcl strict_inductive source nogcl_sourcenorm
        ;;
    *)
        echo "Usage: $0 {A|B|C|D|E|F|G} [seed]" >&2
        exit 2
        ;;
esac
