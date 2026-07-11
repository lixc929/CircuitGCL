#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/lixc/.conda/envs/RCG/bin/python}"
LOG_ROOT="${LOG_ROOT:-logs/s6_p1_p2_protocol_20260711}"
STATUS_FILE="${ROOT_DIR}/${LOG_ROOT}/queue_status.tsv"
MIN_FREE_MB="${MIN_FREE_MB:-8000}"
MAX_UTIL="${MAX_UTIL:-80}"
POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p "${ROOT_DIR}/${LOG_ROOT}"
cd "${ROOT_DIR}"

status() {
    printf '%s\t%s\t%s\n' \
        "$(date '+%Y-%m-%d %H:%M:%S')" "$1" "$2" >> "${STATUS_FILE}"
}

wait_for_gpu() {
    local lane="$1"
    local gpu="$2"
    local free_mb util
    while true; do
        IFS=',' read -r free_mb util < <(
            nvidia-smi -i "${gpu}" \
                --query-gpu=memory.free,utilization.gpu \
                --format=csv,noheader,nounits
        )
        free_mb="${free_mb//[[:space:]]/}"
        util="${util//[[:space:]]/}"
        if (( free_mb >= MIN_FREE_MB && util <= MAX_UTIL )); then
            status "${lane}" "gpu_ready gpu=${gpu} free_mb=${free_mb} util=${util}"
            return
        fi
        status "${lane}" "waiting gpu=${gpu} free_mb=${free_mb} util=${util}"
        sleep "${POLL_SECONDS}"
    done
}

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
    --sgrl 1
    --cl_model clustergcn
    --cl_epochs 5
    --cl_gnn_layers 2
    --cl_hid_dim 64
    --cl_batch_size 32768
    --cl_num_neighbors 8
    --model clustergcn
    --num_gnn_layers 2
    --regress_loss mse
    --seed 0
)

run_experiment() {
    local lane="$1"
    local gpu="$2"
    local name="$3"
    shift 3
    local run_dir="${LOG_ROOT}/${name}"

    wait_for_gpu "${lane}" "${gpu}"
    status "${lane}" "start name=${name} gpu=${gpu}"
    OPENBLAS_NUM_THREADS=16 \
    OMP_NUM_THREADS=16 \
    MKL_NUM_THREADS=16 \
    NUMEXPR_NUM_THREADS=16 \
        "${PYTHON_BIN}" main.py \
        "${COMMON_ARGS[@]}" \
        --gpu "${gpu}" \
        --log_dir "${run_dir}" \
        "$@"
    local rc=$?
    status "${lane}" "finish name=${name} rc=${rc}"
    if (( rc != 0 )); then
        status "${lane}" "queue_stopped failed_name=${name}"
        exit "${rc}"
    fi
}

lane="${1:-}"
case "${lane}" in
    gpu3)
        status "${lane}" "queue_started"
        run_experiment "${lane}" 3 p1_static_mse \
            --sgrl_mode static --hid_dim 63
        run_experiment "${lane}" 3 p1_lora_r8_lambda0 \
            --sgrl_mode joint_shared --joint_shared_gnn_layers 2 \
            --joint_lora_rank 8 --joint_lora_layer -1 \
            --joint_gcl_lambda 0 --hid_dim 64
        run_experiment "${lane}" 3 p2_lora_r8_lambda005_backbone_lr1e6 \
            --sgrl_mode joint_shared --joint_shared_gnn_layers 2 \
            --joint_lora_rank 8 --joint_lora_layer -1 \
            --joint_gcl_lambda 0.05 --joint_backbone_lr 1e-6 --hid_dim 64
        run_experiment "${lane}" 3 p2_lora_r8_lambda0_backbone_lr1e5 \
            --sgrl_mode joint_shared --joint_shared_gnn_layers 2 \
            --joint_lora_rank 8 --joint_lora_layer -1 \
            --joint_gcl_lambda 0 --joint_backbone_lr 1e-5 --hid_dim 64
        status "${lane}" "queue_finished"
        ;;
    gpu4)
        status "${lane}" "queue_started"
        run_experiment "${lane}" 4 p1_joint_lambda0 \
            --sgrl_mode joint_shared --joint_shared_gnn_layers 2 \
            --joint_gcl_lambda 0 --hid_dim 64
        run_experiment "${lane}" 4 p1_lora_r8_lambda005 \
            --sgrl_mode joint_shared --joint_shared_gnn_layers 2 \
            --joint_lora_rank 8 --joint_lora_layer -1 \
            --joint_gcl_lambda 0.05 --hid_dim 64
        run_experiment "${lane}" 4 p2_lora_r8_lambda005_backbone_lr1e5 \
            --sgrl_mode joint_shared --joint_shared_gnn_layers 2 \
            --joint_lora_rank 8 --joint_lora_layer -1 \
            --joint_gcl_lambda 0.05 --joint_backbone_lr 1e-5 --hid_dim 64
        status "${lane}" "queue_finished"
        ;;
    *)
        echo "Usage: $0 {gpu3|gpu4}" >&2
        exit 2
        ;;
esac
