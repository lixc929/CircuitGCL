#!/usr/bin/env python3
"""Run the reuse x label-rebalancing development matrix.

The matrix intentionally uses the lightweight SGRL/downstream settings already
used by the local reuse pilots, so results are comparable with prior notes.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


RUNS = [
    ("static", ["--sgrl", "1", "--sgrl_mode", "static"]),
    ("nogcl", ["--sgrl", "0"]),
    (
        "reuse_gate_init",
        [
            "--sgrl",
            "1",
            "--sgrl_mode",
            "init",
            "--sgrl_reuse_stats",
            "1",
            "--sgrl_reuse_stats_fusion",
            "gate",
        ],
    ),
]

LOSSES = ["mse", "gai", "bmc"]

COMMON_ARGS = [
    "--dataset",
    "ssram+digtime+timing_ctrl+array_128_32_8t",
    "--task",
    "regression",
    "--task_level",
    "edge",
    "--seed",
    "42",
    "--epochs",
    "20",
    "--batch_size",
    "512",
    "--num_workers",
    "0",
    "--cl_epochs",
    "5",
    "--cl_gnn_layers",
    "2",
    "--cl_hid_dim",
    "64",
    "--cl_batch_size",
    "32768",
    "--cl_num_neighbors",
    "8",
    "--num_hops",
    "2",
    "--num_neighbors",
    "8",
]


DICT_RE = re.compile(r"\{[^{}]*'mse'[^{}]*\}")
BEST_RE = re.compile(r"Best epoch: ([0-9]+), mse: ([0-9.eE+-]+), loss: ([0-9.eE+-]+)")


def parse_log(log_path: Path) -> dict:
    text = log_path.read_text(errors="replace")
    best_matches = BEST_RE.findall(text)
    best_epoch, best_val_mse, best_val_loss = (None, None, None)
    if best_matches:
        best_epoch, best_val_mse, best_val_loss = best_matches[-1]
        best_epoch = int(best_epoch)
        best_val_mse = float(best_val_mse)
        best_val_loss = float(best_val_loss)

    lines = text.splitlines()
    latest_test_line = ""
    for line in lines:
        if "Test results:" in line:
            latest_test_line = line

    test_results = []
    for match in DICT_RE.findall(latest_test_line):
        # The log dicts are simple Python literals containing numeric values.
        test_results.append(ast.literal_eval(match))

    return {
        "log": str(log_path),
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
        "best_val_loss": best_val_loss,
        "test_results": test_results,
    }


def find_main_log(run_dir: Path) -> Path | None:
    logs = sorted(run_dir.glob("*_edge_regression_*_loss*_batch512.txt"))
    if not logs:
        return None
    return logs[-1]


def run_one(
    python_bin: str,
    repo_root: Path,
    root_log_dir: Path,
    gpu: int,
    run_name: str,
    loss: str,
    mode_args: list[str],
    quiet: bool,
) -> dict:
    label = f"{run_name}_{loss}"
    run_dir = root_log_dir / label
    run_dir.mkdir(parents=True, exist_ok=True)
    supervisor_log = run_dir / "supervisor.log"

    cmd = [
        python_bin,
        "main.py",
        *COMMON_ARGS,
        "--gpu",
        str(gpu),
        "--regress_loss",
        loss,
        "--log_dir",
        str(run_dir),
        *mode_args,
    ]

    meta = {
        "label": label,
        "run_name": run_name,
        "loss": loss,
        "gpu": gpu,
        "cmd": cmd,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (run_dir / "command.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"\n===== RUN {label} =====", flush=True)
    print(" ".join(cmd), flush=True)

    with supervisor_log.open("w") as sink:
        process = subprocess.Popen(
            cmd,
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            if not quiet:
                print(line, end="")
            sink.write(line)
            sink.flush()
        returncode = process.wait()

    meta["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["returncode"] = returncode
    (run_dir / "command.json").write_text(json.dumps(meta, indent=2) + "\n")

    main_log = find_main_log(run_dir)
    result = {
        "label": label,
        "run_name": run_name,
        "loss": loss,
        "returncode": returncode,
        "run_dir": str(run_dir),
        "supervisor_log": str(supervisor_log),
        "main_log": str(main_log) if main_log else None,
    }
    if main_log:
        result.update(parse_log(main_log))
    tests = result.get("test_results") or []
    mse_values = [test.get("mse") for test in tests]
    print(
        "===== DONE {label}: code={code}, best_epoch={epoch}, val_mse={val}, test_mse={tests} =====".format(
            label=label,
            code=returncode,
            epoch=result.get("best_epoch"),
            val=result.get("best_val_mse"),
            tests=mse_values,
        ),
        flush=True,
    )
    return result


def write_summary(results: list[dict], summary_path: Path) -> None:
    summary_path.write_text(json.dumps(results, indent=2) + "\n")

    md_path = summary_path.with_suffix(".md")
    rows = [
        "| Mode | Loss | Status | Best epoch | Val MSE | digtime MSE | timing_ctrl MSE | array MSE | Log |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for res in results:
        tests = res.get("test_results") or []
        mse_values = [test.get("mse") for test in tests]
        while len(mse_values) < 3:
            mse_values.append(None)
        status = "ok" if res.get("returncode") == 0 else f"fail({res.get('returncode')})"
        rows.append(
            "| {mode} | {loss} | {status} | {epoch} | {val} | {digi} | {timing} | {array} | `{log}` |".format(
                mode=res.get("run_name"),
                loss=res.get("loss"),
                status=status,
                epoch=res.get("best_epoch"),
                val=res.get("best_val_mse"),
                digi=mse_values[0],
                timing=mse_values[1],
                array=mse_values[2],
                log=res.get("main_log"),
            )
        )
    md_path.write_text("\n".join(rows) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--log_root", type=Path, default=Path("logs/reuse_rebalance_matrix_20260709"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--only_mode", choices=[run[0] for run in RUNS])
    parser.add_argument("--only_loss", choices=LOSSES)
    parser.add_argument("--summary_only", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Write child output to supervisor logs only.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    root_log_dir = (repo_root / args.log_root).resolve()
    root_log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = root_log_dir / "summary.json"

    if args.summary_only:
        results = []
        for run_name, _mode_args in RUNS:
            for loss in LOSSES:
                if args.only_mode and run_name != args.only_mode:
                    continue
                if args.only_loss and loss != args.only_loss:
                    continue
                run_dir = root_log_dir / f"{run_name}_{loss}"
                main_log = find_main_log(run_dir)
                result = {
                    "label": f"{run_name}_{loss}",
                    "run_name": run_name,
                    "loss": loss,
                    "returncode": 0 if main_log else None,
                    "run_dir": str(run_dir),
                    "main_log": str(main_log) if main_log else None,
                }
                if main_log:
                    result.update(parse_log(main_log))
                results.append(result)
        write_summary(results, summary_path)
        print(summary_path)
        return 0

    results = []
    for run_name, mode_args in RUNS:
        for loss in LOSSES:
            if args.only_mode and run_name != args.only_mode:
                continue
            if args.only_loss and loss != args.only_loss:
                continue
            result = run_one(
                args.python,
                repo_root,
                root_log_dir,
                args.gpu,
                run_name,
                loss,
                mode_args,
                args.quiet,
            )
            results.append(result)
            write_summary(results, summary_path)
            if result["returncode"] != 0:
                print(f"Run failed: {result['label']}", file=sys.stderr)
                return int(result["returncode"])

    write_summary(results, summary_path)
    print(f"Summary: {summary_path}")
    print(f"Markdown: {summary_path.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
