"""Orchestrate hyperparameter sweeps for symbolic regression methods.

The agent samples configurations (optionally from an external search-space file),
launches method-specific training scripts with a per-run timeout, and pushes
metrics/best formulas to Weights & Biases. Runs execute in parallel via a
process pool so we can saturate available CPUs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency for YAML search spaces
    yaml = None

try:
    import wandb
except ImportError:  # pragma: no cover - wandb is optional for offline usage
    wandb = None


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"


METHOD_SCRIPTS: Dict[str, Path] = {
    "pysr": SRC_ROOT / "sr_models" / "pysr_model.py",
    "pysindy": SRC_ROOT / "sr_models" / "pysindy_model.py",
    "dso": SRC_ROOT / "sr_models" / "dso_model.py",
    "kan": SRC_ROOT / "sr_models" / "kan_model.py",
    "aifeynman": SRC_ROOT / "sr_models" / "aifeynman_model.py",
}


DEFAULT_SEARCH_SPACE: Dict[str, Dict[str, Iterable]] = {
    "pysr": {
        "n_iterations": [150, 250],
        "population_size": [30, 60],
        "max_size": [18, 24],
        "parsimony": [1.0, 0.1],
    },
    "pysindy": {
        "alpha": [1e3, 1e4, 1e5],
        "threshold": [1e-3, 1e-4],
    },
    "dso": {
        "n_iterations": [50, 100],
        "n_samples": [128, 256],
        "learning_rate": [5e-4, 1e-3],
    },
    "kan": {
        "n_iterations": [2, 3],
        "steps": [120, 180],
        "threshold": [0.005, 0.01],
        "hidden_width": ["5,7,5,1", "6,8,6,1"],
    },
    "aifeynman": {
        "bf_try_time": [20, 40],
        "nn_epochs": [30, 50],
        "polyfit_degree": [3, 4],
    },
}


METHOD_OUTPUT_SUFFIX = {
    "pysr": "hall_of_fame.csv",
    "dso": "hall_of_fame.tsv",
    "pysindy": "best_formula.txt",
    "kan": "kan_progress.txt",
    "aifeynman": "aifeynman_solution.txt",
}


MethodConfig = Dict[str, Iterable]


@dataclass
class TrialResult:
    method: str
    config: Dict[str, object]
    loss: Optional[float]
    formula: Optional[str]
    output_path: Path
    runtime_seconds: float
    succeeded: bool
    error: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Symbolic regression hyperparameter sweeps")
    parser.add_argument("--dataset", required=True, help="Path to the dataset CSV used by all methods")
    parser.add_argument("--dataset-size", type=int, help="Maximum number of rows to load from the dataset")
    parser.add_argument("--features", type=str, help="Comma-separated list of features to retain")
    parser.add_argument("--methods", type=str, default=",".join(METHOD_SCRIPTS.keys()), help="Comma-separated list of methods to sweep")
    parser.add_argument("--search-space", type=str, help="YAML/JSON file specifying hyperparameter search space")
    parser.add_argument("--output-dir", type=str, default="sweep_runs", help="Directory where temporary outputs are stored")
    parser.add_argument("--project", type=str, default="symbolic-regression-sweep", help="Weights & Biases project name")
    parser.add_argument("--entity", type=str, help="Weights & Biases entity (team or user)")
    parser.add_argument("--wandb-mode", type=str, choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--sweep-id", type=str, help="Optional W&B sweep ID to join (enables server-side orchestration)")
    parser.add_argument("--max-runtime-seconds", type=int, default=900, help="Per-run runtime budget in seconds")
    parser.add_argument("--max-trials", type=int, default=40, help="Maximum total trials across all methods")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum concurrent subprocesses")
    parser.add_argument("--cpus-per-run", type=int, default=2, help="CPUs allocated per subprocess (exported to BLAS/OMP env vars)")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed used for sampling the search space")
    return parser.parse_args()


def load_search_space(path: Optional[str]) -> Dict[str, MethodConfig]:
    if not path:
        return DEFAULT_SEARCH_SPACE

    search_path = Path(path)
    payload = search_path.read_text()
    suffix = search_path.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError(
                "PyYAML is required to read YAML search spaces. Install it or provide JSON."
            )
        data = yaml.safe_load(payload)
    else:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON search space file: {path}\n{exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Search space file must define a mapping, received: {type(data)}")

    return data


def enumerate_trials(
    methods: List[str],
    search_space: Dict[str, MethodConfig],
    max_trials: int,
    seed: int,
) -> List[Tuple[str, Dict[str, object]]]:
    rng = random.Random(seed)
    trials: List[Tuple[str, Dict[str, object]]] = []

    for method in methods:
        space = search_space.get(method)
        if not space:
            continue

        keys = list(space.keys())
        value_lists = [list(space[key]) for key in keys]
        combos = [dict(zip(keys, values)) for values in itertools.product(*value_lists)]
        rng.shuffle(combos)
        trials.extend((method, combo) for combo in combos)

    if len(trials) > max_trials:
        trials = rng.sample(trials, max_trials)

    return trials


def build_command(
    method: str,
    script_path: Path,
    dataset: str,
    dataset_size: Optional[int],
    features: Optional[str],
    temp_file: Path,
    config: Dict[str, object],
    max_runtime_seconds: int,
) -> List[str]:
    cmd = [sys.executable, str(script_path), "--dataset", dataset, "--temp_file", str(temp_file)]

    if dataset_size:
        cmd.extend(["--dataset_size", str(dataset_size)])
    if features:
        cmd.extend(["--features", features])

    cmd.extend(["--max_runtime_seconds", str(max_runtime_seconds)])

    for key, value in config.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])

    return cmd


def parse_best_result(method: str, temp_file: Path) -> Tuple[Optional[str], Optional[float]]:
    if not temp_file.exists():
        return None, math.nan

    try:
        if method in {"pysr", "dso"}:
            df = pd.read_csv(temp_file)
            if df.empty:
                return None, math.nan
            loss_column = "Loss" if "Loss" in df.columns else df.columns[-1]
            equation_column = "Equation" if "Equation" in df.columns else df.columns[0]
            best_row = df.sort_values(by=loss_column).iloc[0]
            return str(best_row[equation_column]), float(best_row[loss_column])

        if method == "kan":
            best_formula, best_loss = None, math.inf
            with open(temp_file) as handle:
                lines = handle.readlines()
            for idx, line in enumerate(lines):
                if line.startswith("Formula:"):
                    formula = line.split("Formula:", 1)[1].strip()
                    if idx + 1 < len(lines) and "Loss:" in lines[idx + 1]:
                        loss = float(lines[idx + 1].split("Loss:", 1)[1].strip())
                        if loss < best_loss:
                            best_loss, best_formula = loss, formula
            if best_formula is not None:
                return best_formula, float(best_loss)
            return None, math.nan

        if method == "pysindy":
            with open(temp_file) as handle:
                text = handle.read()
            formula = None
            loss = math.nan
            for part in text.splitlines():
                if part.startswith("Equation"):
                    formula = part.split("Equation", 1)[1].strip()
                if part.startswith("Loss"):
                    try:
                        loss = float(part.split("Loss", 1)[1].strip())
                    except ValueError:
                        loss = math.nan
            return formula, loss

        if method == "aifeynman":
            with open(temp_file) as handle:
                lines = [line.strip() for line in handle.readlines() if line.strip()]
            formula = lines[0] if lines else None
            loss = None
            for line in lines:
                if line.lower().startswith("loss"):
                    try:
                        loss = float(line.split(":", 1)[1])
                    except (IndexError, ValueError):
                        loss = None
            return formula, loss if loss is not None else math.nan

    except Exception:
        return None, math.nan

    return None, math.nan


def run_trial(
    method: str,
    config: Dict[str, object],
    args: argparse.Namespace,
    run_dir: Path,
) -> TrialResult:
    script_path = METHOD_SCRIPTS[method]
    temp_file = run_dir / METHOD_OUTPUT_SUFFIX[method]
    os.makedirs(temp_file.parent, exist_ok=True)

    cmd = build_command(
        method,
        script_path,
        args.dataset,
        args.dataset_size,
        args.features,
        temp_file,
        config,
        args.max_runtime_seconds,
    )

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", str(args.cpus_per_run))
    env.setdefault("OPENBLAS_NUM_THREADS", str(args.cpus_per_run))
    env.setdefault("MKL_NUM_THREADS", str(args.cpus_per_run))
    env.setdefault("NUMEXPR_NUM_THREADS", str(args.cpus_per_run))

    result = TrialResult(
        method=method,
        config=config,
        loss=math.nan,
        formula=None,
        output_path=temp_file,
        runtime_seconds=0.0,
        succeeded=False,
    )

    start = time.monotonic()
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=args.max_runtime_seconds + 60,
        )
        result.runtime_seconds = time.monotonic() - start
        (run_dir / "stdout.log").write_text(completed.stdout)
        (run_dir / "stderr.log").write_text(completed.stderr)
        if completed.returncode != 0:
            result.error = completed.stderr or completed.stdout
        else:
            result.succeeded = True
    except subprocess.TimeoutExpired as exc:
        result.error = f"External timeout after {exc.timeout} seconds"
        result.runtime_seconds = time.monotonic() - start

    formula, loss = parse_best_result(method, temp_file)
    result.formula = formula
    result.loss = loss

    return result


def log_to_wandb(
    project: str,
    entity: Optional[str],
    mode: str,
    dataset: str,
    result: TrialResult,
    sweep_id: Optional[str] = None,
) -> None:
    if wandb is None or mode == "disabled":
        return

    if mode:
        os.environ.setdefault("WANDB_MODE", mode)

    run = wandb.init(
        project=project,
        entity=entity,
        config={"method": result.method, **result.config, "dataset": dataset},
        settings=wandb.Settings(start_method="thread"),
        reinit=True,
        id=None,
        resume="allow",
        sweep=sweep_id,
    )

    metrics = {
        "loss": result.loss,
        "runtime_seconds": result.runtime_seconds,
        "succeeded": int(result.succeeded),
    }
    if result.error:
        metrics["error"] = result.error[:512]

    wandb.log(metrics)

    if result.formula:
        wandb.summary["best_formula"] = result.formula

    wandb.finish()


def main() -> None:
    args = parse_args()
    args.dataset = str(Path(args.dataset).resolve())

    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    for method in methods:
        if method not in METHOD_SCRIPTS:
            raise ValueError(f"Unknown method requested: {method}")

    search_space = load_search_space(args.search_space)
    trials = enumerate_trials(methods, search_space, args.max_trials, args.seed)

    output_root = Path(args.output_dir).resolve()
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_run: Dict[concurrent.futures.Future, Tuple[str, Dict[str, object], Path]] = {}
        for idx, (method, config) in enumerate(trials):
            run_dir = output_root / f"trial_{idx:03d}_{method}"
            future = executor.submit(run_trial, method, config, args, run_dir)
            future_to_run[future] = (method, config, run_dir)

        for future in concurrent.futures.as_completed(future_to_run):
            result = future.result()
            log_to_wandb(
                project=args.project,
                entity=args.entity,
                mode=args.wandb_mode,
                dataset=args.dataset,
                result=result,
                sweep_id=args.sweep_id,
            )

            status = "OK" if result.succeeded else "FAIL"
            print(f"[{status}] {result.method} loss={result.loss} formula={result.formula}")


if __name__ == "__main__":
    main()

