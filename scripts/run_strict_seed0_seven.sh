#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/lixc/.conda/envs/RCG/bin/python}"
LOG_ROOT="${LOG_ROOT:-logs/strict_seed0_seven_20260711}"
STATUS_FILE="${ROOT_DIR}/${LOG_ROOT}/queue_status.tsv"
MIN_FREE_MB="${MIN_FREE_MB:-6500}"
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

COMMON_ARGS=(
    --task_level edge
    --task regression
    --dataset ssram+digtime+timing_ctrl+array_128_32_8t
    --net_only True
    --protocol strict_inductive
    --sgrl_graph_scope source
    --normalization_scope source
    --small_dataset_sample_rates 1.0
    --large_dataset_sample_rates 0.1
    --num_hops 2
    --num_neighbors 8
    --num_workers 0
    --epochs 160
    --early_stopping_patience 0
    --joint_shared_audit 1
    --joint_shared_audit_interval 10
    --batch_size 512
    --lr 0.0001
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
    --pretraining_seed 0
    --embedding_inference_seed 20260711
    --downstream_seed 0
    --train_sampler_seed 0
    --relation_sample_seed 20260711
    --split_seed 0
    --eval_seed 20260711
)

write_manifest() {
    local commit
    commit="$(git rev-parse HEAD)"
    "${PYTHON_BIN}" -c '
import json
import sys

path, commit = sys.argv[1:]
payload = {
    "schema_version": 1,
    "git_commit": commit,
    "protocol": "strict_inductive",
    "dataset": "ssram+digtime+timing_ctrl+array_128_32_8t",
    "selection_target": "source validation MSE",
    "epochs": 160,
    "seeds": {
        "experiment": 0,
        "relation": 20260711,
        "embedding_inference": 20260711,
        "evaluation": 20260711,
    },
    "methods": [
        "no_gcl",
        "static_dual",
        "static_ema",
        "init_reuse",
        "joint_lambda0",
        "lora_r8_lambda0",
        "lora_r8_lambda005",
    ],
    "selection_rule": (
        "Rank by source validation MSE; retain static plus the best compact "
        "lambda=0 and positive-lambda candidates for multi-seed confirmation."
    ),
    "transfer_veto": {
        "mean_relative_degradation": 0.10,
        "any_circuit_relative_degradation": 0.25,
    },
}
with open(path, "w", encoding="utf-8") as output:
    json.dump(payload, output, indent=2, sort_keys=True)
    output.write("\n")
' "${ROOT_DIR}/${LOG_ROOT}/selection_manifest.json" "${commit}"
}

run_experiment() {
    local lane="$1"
    local gpu="$2"
    local name="$3"
    shift 3

    wait_for_gpu "${lane}" "${gpu}"
    status "${lane}" "start name=${name} gpu=${gpu}"
    env \
        OPENBLAS_NUM_THREADS=4 \
        OMP_NUM_THREADS=4 \
        MKL_NUM_THREADS=4 \
        NUMEXPR_NUM_THREADS=4 \
        "${PYTHON_BIN}" main.py \
        "${COMMON_ARGS[@]}" \
        --gpu "${gpu}" \
        --log_dir "${LOG_ROOT}/${name}" \
        "$@"
    local rc=$?
    status "${lane}" "finish name=${name} rc=${rc}"
    return "${rc}"
}

wait_and_record() {
    local phase="$1"
    shift
    local rc=0
    local pid
    for pid in "$@"; do
        if ! wait "${pid}"; then
            rc=1
        fi
    done
    status orchestrator "phase_finished phase=${phase} rc=${rc}"
    return "${rc}"
}

write_manifest
status orchestrator "queue_started commit=$(git rev-parse HEAD)"
status orchestrator "processed_cache_prewarm_started"
if ! "${PYTHON_BIN}" scripts/prewarm_strict_processed_caches.py \
        --relation_sample_seed 20260711 \
        > "${ROOT_DIR}/${LOG_ROOT}/processed_cache_prewarm.log" 2>&1; then
    status orchestrator "queue_stopped reason=processed_cache_prewarm_failure"
    exit 1
fi
status orchestrator "processed_cache_prewarm_finished"

# Phase 1 produces both source-only checkpoint variants. All processed caches
# are validated before these concurrent readers start.
run_experiment phase1_gpu3 3 static_dual \
    --sgrl 1 --sgrl_mode static --hid_dim 63 \
    --sgrl_pretrain_target_update sgrl_dual_rsm_ema &
pid_static_dual=$!
sleep 20
run_experiment phase1_gpu4a 4 static_ema \
    --sgrl 1 --sgrl_mode static --hid_dim 63 \
    --sgrl_pretrain_target_update circuitgcl_text_ema_only &
pid_static_ema=$!
sleep 20
run_experiment phase1_gpu4b 4 no_gcl \
    --sgrl 0 --hid_dim 64 &
pid_no_gcl=$!

if ! wait_and_record phase1 \
        "${pid_static_dual}" "${pid_static_ema}" "${pid_no_gcl}"; then
    status orchestrator "queue_stopped reason=phase1_failure"
    exit 1
fi

# Phase 2 is read-only with respect to the validated dual SGRL checkpoint.
run_experiment phase2_gpu3 3 init_reuse \
    --sgrl 1 --sgrl_mode init_reuse --hid_dim 64 \
    --sgrl_pretrain_target_update sgrl_dual_rsm_ema &
pid_init=$!
sleep 20
run_experiment phase2_gpu4a 4 lora_r8_lambda0 \
    --sgrl 1 --sgrl_mode joint_shared --hid_dim 64 \
    --joint_shared_gnn_layers 2 --joint_lora_rank 8 \
    --joint_lora_layer -1 --joint_gcl_lambda 0 \
    --sgrl_pretrain_target_update sgrl_dual_rsm_ema &
pid_lora0=$!
sleep 20
run_experiment phase2_gpu4b 4 lora_r8_lambda005 \
    --sgrl 1 --sgrl_mode joint_shared --hid_dim 64 \
    --joint_shared_gnn_layers 2 --joint_lora_rank 8 \
    --joint_lora_layer -1 --joint_gcl_lambda 0.05 \
    --sgrl_pretrain_target_update sgrl_dual_rsm_ema &
pid_lora005=$!

if ! wait_and_record phase2a \
        "${pid_init}" "${pid_lora0}" "${pid_lora005}"; then
    status orchestrator "queue_stopped reason=phase2a_failure"
    exit 1
fi

run_experiment phase2_gpu3 3 joint_lambda0 \
    --sgrl 1 --sgrl_mode joint_shared --hid_dim 64 \
    --joint_shared_gnn_layers 2 --joint_lora_rank 0 \
    --joint_gcl_lambda 0 \
    --sgrl_pretrain_target_update sgrl_dual_rsm_ema
rc_joint=$?

"${PYTHON_BIN}" scripts/summarize_experiments.py \
    "${LOG_ROOT}" --output_dir "${LOG_ROOT}" || true
status orchestrator "queue_finished rc=${rc_joint}"
exit "${rc_joint}"
