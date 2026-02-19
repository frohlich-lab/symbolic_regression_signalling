"""Orchestrate hyperparameter sweeps for symbolic regression methods."""

import argparse
import ast
import concurrent.futures
import itertools
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import statistics
import pandas as pd

try:
    import wandb
except ImportError:  # pragma: no cover - wandb is optional in offline mode
    wandb = None
try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency for YAML configs
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

METHOD_SCRIPTS: Dict[str, Path] = {
    "pysr": SRC_ROOT / "sr_models" / "pysr_model.py",
    "pysindy": SRC_ROOT / "sr_models" / "pysindy_model.py",
    "dso": SRC_ROOT / "sr_models" / "dso_model.py",
    "kan": SRC_ROOT / "sr_models" / "kan_model.py",
    "aifeynman": SRC_ROOT / "sr_models" / "aifeynman_model.py",
    "odeformer": SRC_ROOT / "sr_models" / "odeformer_model.py",
}

DEFAULT_SEARCH_SPACE: Dict[str, Dict[str, Iterable]] = {
    "pysr": {
        "n_iterations": [300, 450],
        "population_size": [40, 80],
        "max_size": [14, 20],
        "parsimony": [1.0, 0.3, 0.05],
    },
    "pysindy": {
        "alpha": [1e2, 1e3, 5e3],
        "threshold": [1e-2, 1e-3, 1e-4],
        "finite_difference_order": [1, 2],
    },
    "dso": {
        "n_iterations": [80, 160],
        "n_samples": [96, 192],
        "learning_rate": [3e-4, 6e-4],
        "entropy_weight": [0.02, 0.05],
    },
    "kan": {
        "n_iterations": [2, 4],
        "steps": [80, 160],
        "threshold": [0.003, 0.008],
        "hidden_width": ["4,6,4,1", "5,7,5,1"],
        "grid": [32, 48],
    },
    "aifeynman": {
        "bf_try_time": [15, 30],
        "nn_epochs": [20, 40],
        "polyfit_degree": [2, 3],
        "brute_force": [True, False],
    },
    "odeformer": {
        "beam_size": [32, 50],
        "beam_temperature": [0.1, 0.2],
    },
}

METHOD_OUTPUT_SUFFIX = {
    "pysr": "hall_of_fame.csv",
    "dso": "hall_of_fame.tsv",
    "pysindy": "best_formula.txt",
    "kan": "kan_progress.txt",
    "aifeynman": "aifeynman_solution.txt",
    "odeformer": "odeformer_predictions.csv",
}

CONDA_ENVIRONMENTS: Dict[str, Optional[str]] = {
    # When executed via Snakemake, each sweep rule already activates the
    # appropriate conda environment, so we reuse the current interpreter.
    "pysr": None,
    "pysindy": None,
    "dso": None,
    "kan": None,
    "aifeynman": None,
    "odeformer": None,
}

METHOD_SEED_FLAGS = {
    "pysr": "--seed",
    "pysindy": "--seed",
    "dso": "--seed",
    "kan": "--seed",
    "aifeynman": "--seed",
    "odeformer": "--seed",
}

MethodConfig = Dict[str, Iterable]


class TrialResult:
    def __init__(
        self,
        method: str,
        config: Dict[str, object],
        loss: Optional[float],
        formula: Optional[str],
        output_path: Path,
        runtime_seconds: float,
        succeeded: bool,
        error: Optional[str] = None,
    ) -> None:
        self.method = method
        self.config = config
        self.loss = loss
        self.formula = formula
        self.output_path = output_path
        self.runtime_seconds = runtime_seconds
        self.succeeded = succeeded
        self.error = error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Symbolic regression hyperparameter sweeps")
    parser.add_argument("--dataset", required=True, help="Path to the dataset CSV used by all methods")
    parser.add_argument("--dataset-size", type=int, help="Maximum number of rows to load from the dataset")
    parser.add_argument("--features", type=str, help="Comma-separated list of features to retain")
    parser.add_argument("--methods", type=str, default=",".join(METHOD_SCRIPTS.keys()), help="Comma-separated list of methods to sweep")
    parser.add_argument("--search-space", type=str, help="YAML/JSON file specifying hyperparameter search space")
    parser.add_argument("--output-dir", type=str, default="sweep_runs", help="Directory where temporary outputs are stored")
    parser.add_argument("--project", type=str, default="symbolic-regression-sweep", help="Weights & Biases project name")
    parser.add_argument("--entity", type=str, help="Weights & Biases entity")
    parser.add_argument("--wandb-mode", type=str, choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--sweep-id", type=str, help="Optional W&B sweep ID to join")
    parser.add_argument("--max-runtime-seconds", type=int, default=900, help="Per-run runtime budget in seconds")
    parser.add_argument("--max-trials", type=int, default=40, help="Maximum total trials across all methods")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum concurrent subprocesses")
    parser.add_argument("--cpus-per-run", type=int, default=2, help="CPUs allocated per subprocess")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed used for sampling the search space")
    parser.add_argument("--repeat-seeds", type=str, default="0,1,2", help="Comma-separated seeds for top-config repeats")

    # Backward-compatible aliases with underscores
    parser.add_argument("--dataset_size", dest="dataset_size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--max_runtime_seconds", dest="max_runtime_seconds", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--max_trials", dest="max_trials", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--max_workers", dest="max_workers", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--cpus_per_run", dest="cpus_per_run", type=int, help=argparse.SUPPRESS)

    return parser.parse_args()


def load_search_space(path: Optional[str]) -> Dict[str, MethodConfig]:
    if not path:
        return DEFAULT_SEARCH_SPACE

    search_path = Path(path)
    payload = search_path.read_text()
    suffix = search_path.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        if yaml is not None:
            data = yaml.safe_load(payload)
        else:
            data = _parse_simple_yaml(payload)
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


def parse_repeat_seed_list(spec: str) -> List[int]:
    seeds: List[int] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            seeds.append(int(part))
        except ValueError:
            continue
    return seeds or [0, 1, 2]


def compute_model_size(formula: Optional[str]) -> Optional[int]:
    if not formula:
        return None
    return len("".join(formula.split()))


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
    script_args = [str(script_path), "--dataset", dataset, "--temp_file", str(temp_file)]

    if dataset_size:
        script_args.extend(["--dataset_size", str(dataset_size)])
    if features:
        script_args.extend(["--features", features])

    for key, value in config.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                script_args.append(flag)
        else:
            script_args.extend([flag, str(value)])

    env_name = CONDA_ENVIRONMENTS.get(method)
    if env_name:
        cmd = ["conda", "run", "-n", env_name, "python", *script_args]
    else:
        cmd = [sys.executable, *script_args]

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

        if method == "odeformer":
            df = pd.read_csv(temp_file)
            if df.empty:
                return None, math.nan
            equation_column = "Equation" if "Equation" in df.columns else df.columns[0]
            score_column = None
            for candidate in ("Score", "score", "Loss", "loss"):
                if candidate in df.columns:
                    score_column = candidate
                    break
            best_row = df.iloc[0]
            formula = str(best_row[equation_column])
            if score_column:
                try:
                    raw_score = float(best_row[score_column])
                    # Convert to a minimisation objective (higher score => lower loss).
                    loss = -raw_score
                except Exception:
                    loss = 0.0
            else:
                loss = 0.0
            return formula, loss

    except Exception:
        return None, math.nan

    return None, math.nan


def run_trial(
    method: str,
    config: Dict[str, object],
    args: argparse.Namespace,
    run_dir: Path,
    extra_cli_args: Optional[List[str]] = None,
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

    if extra_cli_args:
        cmd.extend(extra_cli_args)

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
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
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


def write_metadata(args: argparse.Namespace, search_space: Dict[str, Dict[str, Iterable]], output_root: Path) -> None:
    metadata = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "dataset": args.dataset,
        "features": args.features,
        "methods": args.methods,
        "search_space_file": args.search_space,
        "search_space": search_space,
        "project": args.project,
        "entity": args.entity,
        "wandb_mode": args.wandb_mode,
        "sweep_id": args.sweep_id,
        "max_runtime_seconds": args.max_runtime_seconds,
        "max_trials": args.max_trials,
        "max_workers": args.max_workers,
        "cpus_per_run": args.cpus_per_run,
        "seed": args.seed,
        "git_revision": None,
    }

    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT))
        metadata["git_revision"] = rev.decode().strip()
    except Exception:
        pass

    metadata_path = output_root / "sweep_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"[SWEEP] Metadata written to {metadata_path}")


def main() -> None:
    args = parse_args()
    args.dataset = str(Path(args.dataset).resolve())

    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    if len(methods) != 1:
        raise ValueError(
            f"Provide exactly one method per sweep run; received {len(methods)} methods: {methods}"
        )
    method = methods[0]
    if method not in METHOD_SCRIPTS:
        raise ValueError(f"Unknown method requested: {method}")

    search_space = load_search_space(args.search_space)
    trials = enumerate_trials(methods, search_space, args.max_trials, args.seed)

    output_root = Path(args.output_dir).resolve()
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    write_metadata(args, search_space, output_root)
    print(
        "[SWEEP] methods=",
        methods,
        "max_trials=",
        args.max_trials,
        "max_workers=",
        args.max_workers,
        "cpus_per_run=",
        args.cpus_per_run,
    )

    trials_root = output_root / "trials"
    trials_root.mkdir(exist_ok=True)

    best_results: Dict[str, Optional[TrialResult]] = {method: None}

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        method_counters: Dict[str, int] = defaultdict(int)
        futures: Dict[concurrent.futures.Future, Tuple[str, Dict[str, object], Path]] = {}
        for _, (trial_method, config) in enumerate(trials):
            idx = method_counters[trial_method]
            method_counters[trial_method] += 1
            run_dir = trials_root / f"{trial_method}_trial_{idx:03d}"
            future = executor.submit(run_trial, trial_method, config, args, run_dir)
            futures[future] = (trial_method, config, run_dir)

        for future in concurrent.futures.as_completed(futures):
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

            current_best = best_results.get(result.method)
            result_loss = result.loss if result.loss is not None else math.nan
            current_loss = (
                current_best.loss if current_best and current_best.loss is not None else math.nan
            )
            if not math.isnan(result_loss) and (
                current_best is None or math.isnan(current_loss) or result_loss < current_loss
            ):
                best_results[result.method] = result

    best_record = best_results.get(method)
    if best_record is None:
        raise RuntimeError(f"No successful trials produced a valid result for method '{method}'.")

    best_config_path = output_root / "best_config.json"
    best_payload = {
        "method": method,
        "loss": best_record.loss,
        "config": best_record.config,
        "formula": best_record.formula,
        "output_path": str(best_record.output_path),
        "model_size": compute_model_size(best_record.formula),
    }
    best_config_path.write_text(json.dumps(best_payload, indent=2))
    print(f"[SWEEP] Best config saved to {best_config_path}")

    if best_record.formula:
        formula_path = output_root / "best_formula.txt"
        formula_path.write_text(best_record.formula)
        print(f"[SWEEP] Best formula saved to {formula_path}")

    repeat_seeds = parse_repeat_seed_list(args.repeat_seeds)
    repeats_root = output_root / "top_repeats"
    repeats_root.mkdir(exist_ok=True)

    seed_flag = METHOD_SEED_FLAGS.get(method)
    repeat_payload: List[Dict[str, object]] = []

    for seed in repeat_seeds:
        repeat_dir = repeats_root / f"seed_{seed}"
        extra_args = [seed_flag, str(seed)] if seed_flag else None
        repeat_result = run_trial(
            method,
            best_record.config,
            args,
            repeat_dir,
            extra_cli_args=extra_args,
        )
        repeat_payload.append(
            {
                "seed": seed,
                "loss": repeat_result.loss,
                "formula": repeat_result.formula,
                "model_size": compute_model_size(repeat_result.formula),
                "runtime_seconds": repeat_result.runtime_seconds,
                "succeeded": repeat_result.succeeded,
                "error": repeat_result.error,
            }
        )

    valid_losses = [
        entry["loss"]
        for entry in repeat_payload
        if entry["loss"] is not None and not math.isnan(entry["loss"])
    ]
    best_loss = min(valid_losses) if valid_losses else None
    median_loss = statistics.median(valid_losses) if valid_losses else None

    best_repeat_record: Optional[Dict[str, object]] = None
    if valid_losses:
        best_loss_value = min(valid_losses)
        for entry in repeat_payload:
            if entry["loss"] == best_loss_value:
                best_repeat_record = entry
                break

    best_formula = best_repeat_record["formula"] if best_repeat_record else None
    repeat_summary = {
        "seeds": repeat_seeds,
        "results": repeat_payload,
        "best_loss": best_loss,
        "median_loss": median_loss,
        "best_model_size": compute_model_size(best_formula) if best_formula else None,
        "best_formula": best_formula,
    }

    repeats_path = output_root / "best_config_repeats.json"
    repeats_path.write_text(json.dumps(repeat_summary, indent=2))
    print(f"[SWEEP] Repeat summary saved to {repeats_path}")


def _parse_simple_yaml(text: str) -> Dict[str, MethodConfig]:
    result: Dict[str, Dict[str, object]] = {}
    current_key: Optional[str] = None
    indent_prefix = None

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.strip().startswith("#"):
            continue

        if not raw_line.startswith(" "):
            key = raw_line.strip().rstrip(":")
            result[key] = {}
            current_key = key
            indent_prefix = None
            continue

        if current_key is None:
            raise ValueError("Invalid YAML structure: value before key.")

        if indent_prefix is None:
            indent_prefix = len(raw_line) - len(raw_line.lstrip(" "))

        line = raw_line[indent_prefix:].strip()
        if ":" not in line:
            raise ValueError(f"Unsupported YAML line: {raw_line}")
        name, value = line.split(":", 1)
        name = name.strip()
        value = value.strip()
        if not value:
            raise ValueError(f"Missing value in YAML line: {raw_line}")
        try:
            parsed_value = ast.literal_eval(value)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unable to parse value '{value}' in YAML line '{raw_line}'") from exc

        result[current_key][name] = parsed_value

    return result


if __name__ == "__main__":
    main()
