#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/lixc/.conda/envs/RCG/bin/python}"
LOG_ROOT="${LOG_ROOT:-logs/strict_joint_rank0_lambda005_seeds12_audit_20260712}"
STATUS_FILE="${ROOT_DIR}/${LOG_ROOT}/queue_status.tsv"
MIN_FREE_MB="${MIN_FREE_MB:-6500}"
POLL_SECONDS="${POLL_SECONDS:-60}"
GPU_SEED1="${GPU_SEED1:-3}"
GPU_SEED2="${GPU_SEED2:-4}"

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
        if (( free_mb >= MIN_FREE_MB )); then
            status "${lane}" \
                "gpu_ready gpu=${gpu} free_mb=${free_mb} util=${util}"
            return
        fi
        status "${lane}" \
            "waiting gpu=${gpu} free_mb=${free_mb} util=${util}"
        sleep "${POLL_SECONDS}"
    done
}

write_manifest() {
    "${PYTHON_BIN}" -c '
import json
import sys

path, commit = sys.argv[1:]
payload = {
    "schema_version": 1,
    "git_commit": commit,
    "protocol": "strict_inductive",
    "method": "joint_shared_rank0_lambda0.05",
    "seeds": [1, 2],
    "epochs": 160,
    "gradient_audit": {
        "enabled": True,
        "interval": 10,
        "batch_index": 0,
        "scope": "shared_gnn",
        "observational_only": True,
    },
    "comparison_roots": [
        "logs/strict_seed0_seven_20260712_v2",
        "logs/strict_selection_seeds12_20260712",
        "logs/strict_joint_rank0_lambda005_seed0_20260712",
    ],
}
with open(path, "w", encoding="utf-8") as output:
    json.dump(payload, output, indent=2, sort_keys=True)
    output.write("\n")
' "${ROOT_DIR}/${LOG_ROOT}/selection_manifest.json" "$(git rev-parse HEAD)"
}

run_experiment() {
    local lane="$1"
    local gpu="$2"
    local seed="$3"

    wait_for_gpu "${lane}" "${gpu}"
    status "${lane}" "start seed=${seed} gpu=${gpu}"
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
        --joint_gradient_audit 1 \
        --joint_gradient_audit_interval 10 \
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
        --seed "${seed}" \
        --pretraining_seed "${seed}" \
        --embedding_inference_seed 20260711 \
        --downstream_seed "${seed}" \
        --train_sampler_seed "${seed}" \
        --relation_sample_seed 20260711 \
        --split_seed "${seed}" \
        --eval_seed 20260711 \
        --gpu "${gpu}" \
        --log_dir "${LOG_ROOT}/joint_rank0_lambda005_seed${seed}" \
        --sgrl 1 \
        --sgrl_mode joint_shared \
        --hid_dim 64 \
        --joint_shared_gnn_layers 2 \
        --joint_lora_rank 0 \
        --joint_gcl_lambda 0.05 \
        --sgrl_pretrain_target_update sgrl_dual_rsm_ema
    local rc=$?
    status "${lane}" "finish seed=${seed} rc=${rc}"
    return "${rc}"
}

write_manifest
status orchestrator "queue_started commit=$(git rev-parse HEAD)"

run_experiment "seed1_gpu${GPU_SEED1}" "${GPU_SEED1}" 1 &
pid_seed1=$!
sleep 20
run_experiment "seed2_gpu${GPU_SEED2}" "${GPU_SEED2}" 2 &
pid_seed2=$!

rc=0
wait "${pid_seed1}" || rc=1
wait "${pid_seed2}" || rc=1

"${PYTHON_BIN}" scripts/summarize_experiments.py \
    "${LOG_ROOT}" --output_dir "${LOG_ROOT}" || true
status orchestrator "queue_finished rc=${rc}"
exit "${rc}"
