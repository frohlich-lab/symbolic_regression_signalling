"""Bayesian hyperparameter optimisation for the NN surrogate via Optuna."""

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import optuna
import torch.nn as nn
from optuna.samplers import TPESampler

import nn_model
from nn_model import evaluate_model, load_dataset, train_model


ACTIVATION_LOOKUP = {
    "ReLU": nn.ReLU,
    "SiLU": nn.SiLU,
    "Tanh": nn.Tanh,
}


SEARCH_SPACE_SPEC = {
    "learning_rate": {
        "type": "loguniform",
        "low": 5e-5,
        "high": 5e-2,
    },
    "batch_size": {
        "type": "categorical",
        "choices": [256, 512, 1024, 2048],
    },
    "hidden_layers": {
        "type": "categorical",
        "choices": [
            {"layers": [128], "dropout": 0.0},
            {"layers": [256], "dropout": 0.05},
            {"layers": [256, 128], "dropout": 0.1},
            {"layers": [256, 256, 128], "dropout": 0.1},
            {"layers": [512, 256, 128], "dropout": 0.15},
            {"layers": [512, 256, 128, 64], "dropout": 0.2},
        ],
    },
    "dropout_rate": {
        "type": "float",
        "low": 0.0,
        "high": 0.3,
        "step": 0.05,
    },
    "weight_decay": {
        "type": "loguniform",
        "low": 1e-7,
        "high": 1e-2,
    },
    "activation": {
        "type": "categorical",
        "choices": list(ACTIVATION_LOOKUP.keys()),
    },
    "optimizer": {
        "type": "categorical",
        "choices": ["Adam", "AdamW", "RMSprop"],
    },
    "scheduler": {
        "type": "categorical",
        "choices": ["None", "ReduceLROnPlateau", "CosineAnnealingLR", "ExponentialLR"],
    },
    "early_stopping_patience": {
        "type": "int",
        "low": 20,
        "high": 80,
        "step": 10,
    },
    "gradient_clip_norm": {
        "type": "float",
        "low": 0.0,
        "high": 5.0,
        "step": 0.5,
    },
}


def _apply_params(params: Dict[str, Any]) -> None:
    nn_model.LEARNING_RATE = params["learning_rate"]
    nn_model.BATCH_SIZE = params["batch_size"]
    nn_model.HIDDEN_LAYERS = params["hidden_layers"]["layers"]
    nn_model.DROPOUT_RATE = params["hidden_layers"].get("dropout", params["dropout_rate"])
    nn_model.WEIGHT_DECAY = params["weight_decay"]
    nn_model.ACTIVATION = ACTIVATION_LOOKUP[params["activation"]]
    nn_model.OPTIMIZER_NAME = params["optimizer"]
    nn_model.SCHEDULER_NAME = params["scheduler"]
    nn_model.EARLY_STOP_PATIENCE = params["early_stopping_patience"]
    nn_model.GRAD_CLIP_NORM = params["gradient_clip_norm"]


def _serialise_trials(trials: List[optuna.trial.FrozenTrial]) -> List[Dict[str, Any]]:
    serialised: List[Dict[str, Any]] = []
    for trial in trials:
        entry: Dict[str, Any] = {
            "number": trial.number,
            "state": trial.state.name,
            "value": None if trial.value is None or math.isnan(trial.value) else float(trial.value),
            "params": trial.params,
        }
        if trial.datetime_start:
            entry["started_at"] = trial.datetime_start.isoformat()
        if trial.datetime_complete:
            entry["completed_at"] = trial.datetime_complete.isoformat()
        if trial.duration is not None:
            entry["duration_seconds"] = trial.duration.total_seconds()
        if trial.user_attrs:
            entry["user_attrs"] = trial.user_attrs
        serialised.append(entry)
    return serialised


def _write_log(study: optuna.Study, log_path: Path) -> None:
    data: Dict[str, Any] = {
        "search_space": SEARCH_SPACE_SPEC,
        "n_trials": len(study.trials),
        "trials": _serialise_trials(study.trials),
    }
    try:
        best = study.best_trial
    except ValueError:
        best = None
    if best is not None:
        data["best_trial_number"] = best.number
        data["best_params"] = study.best_params
        data["best_value"] = float(study.best_value)
    log_path.write_text(json.dumps(data, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna-based NN hyperparameter optimisation")
    parser.add_argument("--dataset", required=True, help="Path to the processed dataset CSV")
    parser.add_argument("--dataset_size", type=int, default=10_000, help="Samples to draw for tuning (min enforced to 10k)")
    parser.add_argument("--features", type=str, default="all", help="Comma-separated list of feature names or 'all'")
    parser.add_argument("--output-dir", type=str, default="tuning_runs", help="Directory for tuner artefacts")
    parser.add_argument("--log-file", type=str, default=None, help="Path to JSON log file (defaults inside output dir)")
    parser.add_argument("--study-name", type=str, default="nn_optuna", help="Name of the Optuna study")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URI for persistence/resume (e.g. sqlite:///study.db)")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of optimisation trials")
    parser.add_argument("--timeout", type=int, default=None, help="Stop optimisation after this many seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Print per-epoch training diagnostics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log_file) if args.log_file else output_dir / "optuna_trials.json"

    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        storage=args.storage,
        study_name=args.study_name,
        load_if_exists=args.storage is not None,
    )

    train_data, val_data, test_data, full_data = load_dataset(
        args.dataset,
        dataset_size=args.dataset_size,
        features=args.features,
        seed=args.seed,
    )

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", SEARCH_SPACE_SPEC["batch_size"]["choices"]),
            "hidden_layers": trial.suggest_categorical("hidden_layers", SEARCH_SPACE_SPEC["hidden_layers"]["choices"]),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.3, step=0.05),
            "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
            "activation": trial.suggest_categorical("activation", list(ACTIVATION_LOOKUP.keys())),
            "optimizer": trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RMSprop"]),
            "scheduler": trial.suggest_categorical("scheduler", ["None", "ReduceLROnPlateau", "CosineAnnealingLR", "ExponentialLR"]),
            "early_stopping_patience": trial.suggest_int("early_stopping_patience", 20, 80, step=10),
            "gradient_clip_norm": trial.suggest_float("gradient_clip_norm", 0.0, 5.0, step=0.5),
        }

        _apply_params(params)

        trial_seed = args.seed
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            model_path = Path(tmp.name)

        try:
            model = train_model(
                train_data,
                val_data,
                output_path=str(model_path),
                verbose=args.verbose,
                retrain=True,
                seed=trial_seed,
            )
            mae, _ = evaluate_model(model, test_data)
        finally:
            if model_path.exists():
                model_path.unlink(missing_ok=True)

        trial.set_user_attr("params", params)
        return float(mae)

    def log_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        _write_log(study, log_path)

    _write_log(study, log_path)
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        callbacks=[log_callback],
        gc_after_trial=True,
    )
    _write_log(study, log_path)

    best = study.best_trial
    print("Best trial:")
    print(f"  number: {best.number}")
    print(f"  value: {best.value}")
    print("  params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
