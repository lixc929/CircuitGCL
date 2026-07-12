#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/lixc/.conda/envs/RCG/bin/python}"
LOG_ROOT="${LOG_ROOT:-logs/strict_selection_seeds12_20260712}"
STATUS_FILE="${ROOT_DIR}/${LOG_ROOT}/queue_status.tsv"
MIN_FREE_MB="${MIN_FREE_MB:-6000}"
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

write_manifest() {
    local commit
    commit="$(git rev-parse HEAD)"
    "${PYTHON_BIN}" -c '
import json
import sys

path, commit = sys.argv[1:]
with open(path, "w", encoding="utf-8") as output:
    json.dump({
        "schema_version": 1,
        "git_commit": commit,
        "protocol": "strict_inductive",
        "epochs": 160,
        "tuning_seeds": [1, 2],
        "fixed_seeds": {
            "relation": 20260711,
            "embedding_inference": 20260711,
            "evaluation": 20260711,
        },
        "methods": [
            "no_gcl",
            "static_dual",
            "init_reuse",
            "joint_lambda0",
            "lora_r8_lambda005",
        ],
        "excluded_seed0": {
            "static_ema": "tied with author dual mode",
            "lora_r8_lambda0": "array degradation exceeded 25 percent veto",
        },
        "selection_order": [
            "paired source validation MSE",
            "deployment parameter count",
            "transfer veto",
            "representation stability",
        ],
    }, output, indent=2, sort_keys=True)
    output.write("\n")
' "${ROOT_DIR}/${LOG_ROOT}/selection_manifest.json" "${commit}"
}

run_experiment() {
    local lane="$1"
    local gpu="$2"
    local seed="$3"
    local name="$4"
    shift 4

    wait_for_gpu "${lane}" "${gpu}"
    status "${lane}" "start name=${name} seed=${seed} gpu=${gpu}"
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
        --seed "${seed}" \
        --pretraining_seed "${seed}" \
        --embedding_inference_seed 20260711 \
        --downstream_seed "${seed}" \
        --train_sampler_seed "${seed}" \
        --relation_sample_seed 20260711 \
        --split_seed "${seed}" \
        --eval_seed 20260711 \
        --gpu "${gpu}" \
        --log_dir "${LOG_ROOT}/${name}_seed${seed}" \
        "$@"
    local rc=$?
    status "${lane}" "finish name=${name} seed=${seed} rc=${rc}"
    return "${rc}"
}

run_seed_lane() {
    local lane="$1"
    local gpu="$2"
    local seed="$3"

    # Static is the single writer for this seed's source-only checkpoint.
    run_experiment "${lane}" "${gpu}" "${seed}" static_dual \
        --sgrl 1 --sgrl_mode static --hid_dim 63 \
        --sgrl_pretrain_target_update sgrl_dual_rsm_ema || return
    run_experiment "${lane}" "${gpu}" "${seed}" no_gcl \
        --sgrl 0 --hid_dim 64 || return
    run_experiment "${lane}" "${gpu}" "${seed}" init_reuse \
        --sgrl 1 --sgrl_mode init_reuse --hid_dim 64 \
        --sgrl_pretrain_target_update sgrl_dual_rsm_ema || return
    run_experiment "${lane}" "${gpu}" "${seed}" joint_lambda0 \
        --sgrl 1 --sgrl_mode joint_shared --hid_dim 64 \
        --joint_shared_gnn_layers 2 --joint_lora_rank 0 \
        --joint_gcl_lambda 0 \
        --sgrl_pretrain_target_update sgrl_dual_rsm_ema || return
    run_experiment "${lane}" "${gpu}" "${seed}" lora_r8_lambda005 \
        --sgrl 1 --sgrl_mode joint_shared --hid_dim 64 \
        --joint_shared_gnn_layers 2 --joint_lora_rank 8 \
        --joint_lora_layer -1 --joint_gcl_lambda 0.05 \
        --sgrl_pretrain_target_update sgrl_dual_rsm_ema
}

write_manifest
status orchestrator "queue_started commit=$(git rev-parse HEAD)"

run_seed_lane seed1_gpu3 3 1 &
pid_seed1=$!
sleep 20
run_seed_lane seed2_gpu4 4 2 &
pid_seed2=$!

rc=0
wait "${pid_seed1}" || rc=1
wait "${pid_seed2}" || rc=1

"${PYTHON_BIN}" scripts/summarize_experiments.py \
    logs/strict_seed0_seven_20260712_v2 "${LOG_ROOT}" \
    --output_dir "${LOG_ROOT}" || true
status orchestrator "queue_finished rc=${rc}"
exit "${rc}"
