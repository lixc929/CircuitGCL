#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/lixc/.conda/envs/RCG/bin/python}"
LOG_ROOT="${LOG_ROOT:-logs/strict_joint_rank0_lambda005_seed0_20260712}"
GPU="${GPU:-3}"
MIN_FREE_MB="${MIN_FREE_MB:-6500}"
POLL_SECONDS="${POLL_SECONDS:-60}"
STATUS_FILE="${ROOT_DIR}/${LOG_ROOT}/queue_status.tsv"

mkdir -p "${ROOT_DIR}/${LOG_ROOT}"
cd "${ROOT_DIR}"

status() {
    printf '%s\t%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" \
        >> "${STATUS_FILE}"
}

while true; do
    free_mb="$(nvidia-smi -i "${GPU}" \
        --query-gpu=memory.free --format=csv,noheader,nounits)"
    free_mb="${free_mb//[[:space:]]/}"
    if (( free_mb >= MIN_FREE_MB )); then
        break
    fi
    status "waiting gpu=${GPU} free_mb=${free_mb}"
    sleep "${POLL_SECONDS}"
done

status "start name=joint_rank0_lambda005 seed=0 gpu=${GPU} free_mb=${free_mb} commit=$(git rev-parse HEAD)"

env \
    OPENBLAS_NUM_THREADS=4 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 \
    "${PYTHON_BIN}" main.py \
    --task_level edge \
    --task regression \
    --dataset ssram+digtime+timing_ctrl+array_128_32_8t \
    --net_only True \
    --protocol strict_inductive \
    --sgrl_graph_scope source \
    --normalization_scope source \
    --small_dataset_sample_rates 1.0 \
    --large_dataset_sample_rates 0.1 \
    --num_hops 2 \
    --num_neighbors 8 \
    --num_workers 0 \
    --epochs 160 \
    --early_stopping_patience 0 \
    --joint_shared_audit 1 \
    --joint_shared_audit_interval 10 \
    --batch_size 512 \
    --lr 0.0001 \
    --cl_model clustergcn \
    --cl_epochs 5 \
    --cl_gnn_layers 2 \
    --cl_hid_dim 64 \
    --cl_batch_size 32768 \
    --cl_num_neighbors 8 \
    --model clustergcn \
    --num_gnn_layers 2 \
    --regress_loss mse \
    --seed 0 \
    --pretraining_seed 0 \
    --embedding_inference_seed 20260711 \
    --downstream_seed 0 \
    --train_sampler_seed 0 \
    --relation_sample_seed 20260711 \
    --split_seed 0 \
    --eval_seed 20260711 \
    --gpu "${GPU}" \
    --log_dir "${LOG_ROOT}/joint_rank0_lambda005_seed0" \
    --sgrl 1 \
    --sgrl_mode joint_shared \
    --hid_dim 64 \
    --joint_shared_gnn_layers 2 \
    --joint_lora_rank 0 \
    --joint_gcl_lambda 0.05 \
    --sgrl_pretrain_target_update sgrl_dual_rsm_ema
rc=$?

"${PYTHON_BIN}" scripts/summarize_experiments.py \
    "${LOG_ROOT}" --output_dir "${LOG_ROOT}" || true
status "finish name=joint_rank0_lambda005 seed=0 rc=${rc}"
exit "${rc}"
