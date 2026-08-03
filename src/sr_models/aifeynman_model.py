"""
This modules leverages the AI Feynman library to perform symbolic regression on a given dataset.
It allows for dataset sampling, feature selection, and saves intermediate results to a specified file.

Usage:
    python aifeynman_model.py --dataset <path_to_dataset> --dataset_size <size> --features <feature_list> --temp_file <path_to_temp_file>

Notes on robustness (see run_feynman):
  * AI-Feynman writes all of its scratch output to a ``results/`` directory
    *relative to the current working directory* (not to the dataset dir), and the
    final Pareto front lands in ``results/solution_<filename>``.
  * The generalized-symmetry / gradient-decomposition stage runs UNGUARDED inside
    ``run_AI_all`` whenever the data has more than three columns. On several
    numpy/torch combinations it raises ``'int' object is not callable`` and aborts
    the whole search before any solution file is saved. We neutralize that stage
    (and the equally heavy compositionality stage) so the brute-force / polyfit /
    symmetry Pareto front is still produced and saved.
  * The native solution file is a space-separated table, not the ``Formula:`` /
    ``Error:`` block the downstream parser expects, so we normalize it on the way
    out to ``temp_file``.
"""

import argparse
import random
import pandas as pd
import os
import shutil
import signal
import sys
from pathlib import Path
import tempfile
import numpy as np

AI_FEYNMAN_IMPORT_ERROR = None
try:
    from aifeynman import S_run_aifeynman
except ImportError as exc:  # pragma: no cover - optional dependency
    repo_root = Path(__file__).resolve().parents[2]
    local_pkg = repo_root / "src" / "aifeynman"
    if local_pkg.exists():
        sys.path.insert(0, str(local_pkg))
        try:
            from aifeynman import S_run_aifeynman  # type: ignore
        except ImportError as inner_exc:  # pragma: no cover - defensive
            AI_FEYNMAN_IMPORT_ERROR = inner_exc
    else:  # pragma: no cover - defensive
        AI_FEYNMAN_IMPORT_ERROR = exc

TARGET_COLUMN = 'kcat_cg'

# Hyperparameters for AI Feynman
OPERATORS = '+*-D~ILEA'  # Operators used in symbolic regression
BF_TRY_TIME = 30  # Max time (seconds) per brute-force search step. AI-Feynman
# runs ~10 of these per world transform and re-does them at every symmetry
# level, so the total is BF_TRY_TIME * ~10 * depth. Combined with the full 5000
# sample this can take well over an hour, which is why timeout_duration is 3h -
# a shorter sample/BF completes fast but returns an empty Pareto set.
POLYFIT_DEGREE = 4  # Degree for polynomial fitting
# AI-Feynman's symmetry/separability recursion re-runs the ENTIRE search (NN
# training + ~10 Fortran brute forces) at every level, so NN cost is multiplied
# by the recursion depth. Empirically NN_EPOCHS=100 never finished a single pass
# within the 60-min budget (nothing to salvage), while NN_EPOCHS=40 completes and
# writes a solution. So keep it low - a non-completing run is strictly worse than
# a completing one. The robust SIGTERM salvage above is the backstop, not the plan.
NN_EPOCHS = 40  # Training epochs for neural network stage
DATA_PATHDIR = './data/aifeynman/'  # Directory for dataset
TEST_PERCENTAGE = 20  # Percentage of data for testing
FILENAME = 'mystery.txt'  # Dataset file name

# AI-Feynman writes here (relative to the CWD), regardless of DATA_PATHDIR.
RESULTS_DIR = 'results'

_SALVAGE_TEMP_FILE = None


def _install_sigterm_salvage(temp_file):
    """Salvage the best-so-far Pareto front on a wall-clock SIGTERM, then exit.

    The pipeline wraps this script in `timeout <N> python ...`. We do the salvage
    *inside* the signal handler and then hard-exit with os._exit, because
    AI-Feynman is riddled with bare `except:` clauses that would otherwise swallow
    an exception raised from the handler (so the run would neither salvage nor
    stop). This guarantees temp_file is populated from whatever solution files
    exist on disk, and that the process actually terminates at the deadline.
    """
    global _SALVAGE_TEMP_FILE
    _SALVAGE_TEMP_FILE = temp_file

    def _handler(signum, frame):
        print(f"Received signal {signum}; salvaging best-so-far AI Feynman solution and exiting.")
        try:
            solution_file = find_solution_file()
            if solution_file is not None and _SALVAGE_TEMP_FILE:
                normalize_solution_file(solution_file, _SALVAGE_TEMP_FILE)
        except Exception as exc:  # noqa: BLE001 - best effort in a signal handler
            print(f"Salvage on timeout failed: {exc}")
        os._exit(0)

    try:
        signal.signal(signal.SIGTERM, _handler)
    except (ValueError, OSError):  # pragma: no cover - not in main thread
        pass


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def load_dataset(file_path, dataset_size=None, features=None, seed=None):
    """Loads a dataset from CSV, samples it if specified, and selects columns if features are provided."""
    data = pd.read_csv(file_path)
    target = TARGET_COLUMN if TARGET_COLUMN in data.columns else data.columns[-1]
    if features and features != "all":
        requested = [col.strip() for col in features.split(',')]
        available = [col for col in requested if col in data.columns]
        missing = [col for col in requested if col not in data.columns]
        if missing:
            print(f"Warning: missing features for AI Feynman: {missing}. Using available columns {available}.")
        selected = available + ([target] if target not in available else [])
    else:
        selected = list(data.columns)
    data = data[selected]
    if dataset_size:
        data = data.sample(n=min(dataset_size, len(data)), random_state=seed)

    # Convert inputs back to linear scale while leaving the target in log space
    if not data.empty:
        input_cols = data.columns[:-1]
        data.loc[:, input_cols] = np.exp(data.loc[:, input_cols])
    return data


def neutralize_structural_stages() -> None:
    """Skip AI-Feynman's two NN-gradient structural-discovery stages.

    ``run_AI_all`` runs the generalized-symmetry (``identify_decompositions`` +
    ``brute_force_gen_sym``) and compositionality (``brute_force_comp``) stages
    *unguarded*, so a crash there aborts the whole search before any solution is
    written. We tried running them for real behind guards, but each does a 600s
    Fortran brute force per recursion level, which made a single run exceed an
    hour without finishing, and AI-Feynman's own bare ``except:`` clauses swallow
    the wall-clock SIGTERM so it won't salvage-and-exit. They are the least
    valuable stages for this low-dimensional target, so we replace them with cheap
    no-ops. The retained stages (brute force, polyfit, translational symmetries,
    separabilities) all rely on the trained NN - which is why NN_EPOCHS was raised
    to 800 - and complete quickly. Applied from our own code; no site-packages
    files are edited.
    """
    if AI_FEYNMAN_IMPORT_ERROR is not None:
        return
    srun = S_run_aifeynman

    def _empty_file(name):
        def _fb(*args, **kwargs):
            try:
                open(name, "w").close()
            except OSError:
                pass
        return _fb

    srun.identify_decompositions = lambda *a, **k: np.array([], dtype=int)
    srun.brute_force_gen_sym = _empty_file("results_gen_sym.dat")
    srun.brute_force_comp = _empty_file("results_comp.dat")
    srun.evaluate_derivatives = lambda *a, **k: 0  # -> skip compositionality


def _parse_solution_line(line):
    """Parse one AI-Feynman solution line into ``(error, formula)`` or None.

    The solution table is whitespace-separated with a variable number of leading
    numeric columns followed by the (possibly space-containing) symbolic formula:
      * final file with a held-out test split:
        ``test_error log_err log_err_all complexity fit_error <formula>``
      * final file without a test split:
        ``log_err log_err_all complexity fit_error <formula>``
      * intermediate (pre/first snap) files:
        ``complexity fit_error <formula>``

    We treat the held-out test error (column 0) as the ranking error when at least
    five numeric columns are present, otherwise the trailing fit error.
    """
    tokens = line.split()
    numeric = []
    for tok in tokens:
        try:
            numeric.append(float(tok))
        except ValueError:
            break
    formula = " ".join(tokens[len(numeric):]).strip()
    if not formula or not numeric:
        return None
    error = numeric[0] if len(numeric) >= 5 else numeric[-1]
    if not np.isfinite(error):
        return None
    return error, formula


def normalize_solution_file(solution_path, temp_file):
    """Rewrite an AI-Feynman solution table as ``Formula:`` / ``Error:`` blocks.

    Returns True if at least one valid formula was written.
    """
    if not (os.path.exists(solution_path) and os.path.getsize(solution_path) > 0):
        return False
    entries = []
    with open(solution_path, "r") as handle:
        for line in handle:
            parsed = _parse_solution_line(line)
            if parsed is not None:
                entries.append(parsed)
    if not entries:
        return False
    os.makedirs(os.path.dirname(os.path.abspath(temp_file)), exist_ok=True)
    with open(temp_file, "w") as handle:
        for error, formula in entries:
            handle.write(f"Formula: {formula}\n")
            handle.write(f"Error: {error}\n")
    print(f"Normalized {len(entries)} AI Feynman solution(s) into {temp_file}")
    return True


def find_solution_file(results_dir=RESULTS_DIR, base_filename=FILENAME):
    """Return the best available AI-Feynman solution file, or None.

    Prefers the fully snapped/gradient-descended final solution, then the
    intermediate snapshots that are written earlier in the run, so that a crash
    late in ``run_aifeynman`` still salvages usable results.
    """
    candidates = [
        os.path.join(results_dir, f"solution_{base_filename}"),
        os.path.join(results_dir, f"solution_first_snap_{base_filename}.txt"),
        os.path.join(results_dir, f"solution_before_snap_{base_filename}.txt"),
    ]
    for path in candidates:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
    return None


def run_feynman(
    data,
    temp_file,
    features,
    bf_try_time: int,
    nn_epochs: int,
    polyfit_degree: int,
) -> None:
    """Runs AI Feynman to find the best formula, saving results to a specified file."""
    X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
    data_array = np.concatenate((X, y[:, np.newaxis]), axis=1)
    os.makedirs(DATA_PATHDIR, exist_ok=True)
    np.savetxt(os.path.join(DATA_PATHDIR, FILENAME), data_array)
    np.savetxt(os.path.join(DATA_PATHDIR, '7ops.txt'), [OPERATORS], fmt='%s')

    # Clear stale scratch/output so a previous run's NN checkpoint (trained on a
    # different data shape) cannot trigger a shape mismatch on reload.
    shutil.rmtree(RESULTS_DIR, ignore_errors=True)

    # Skip the crash-prone / very slow NN-gradient structural stages (see docstring).
    neutralize_structural_stages()

    variable_names = (
        [name.strip() for name in features.split(',')]
        if features
        else [f"x_{idx}" for idx in range(X.shape[1])]
    )

    try:
        S_run_aifeynman.run_aifeynman(
            pathdir=DATA_PATHDIR,
            filename=FILENAME,
            BF_try_time=bf_try_time,
            BF_ops_file_type='7ops.txt',
            polyfit_deg=polyfit_degree,
            NN_epochs=nn_epochs,
            vars_name=variable_names,
            test_percentage=TEST_PERCENTAGE,
        )
    except TimeoutError:
        print("AI Feynman process timed out; attempting to salvage partial results.")
    except Exception as e:  # noqa: BLE001 - salvage whatever was produced
        print(f"AI Feynman execution raised an error ({e}); attempting to salvage partial results.")
    finally:
        solution_file = find_solution_file()
        if solution_file is None:
            print("No AI Feynman solution file was produced.")
        elif not normalize_solution_file(solution_file, temp_file):
            print(f"AI Feynman solution file {solution_file} contained no parseable formula.")


def main():
    parser = argparse.ArgumentParser(description='Run AI Feynman for symbolic regression')
    parser.add_argument('--dataset', required=True, help='Path to dataset CSV file')
    parser.add_argument('--dataset_size', type=int, help='Sample size from dataset')
    parser.add_argument('--features', type=str, help='Comma-separated list of features to use')
    parser.add_argument('--temp_file', required=True, help='Path to save intermediate results')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--bf_try_time', type=int, help='Override brute-force search time (seconds)')
    parser.add_argument('--nn_epochs', type=int, help='Override neural network training epochs')
    parser.add_argument(
        '--polyfit_degree',
        type=int,
        help='Override polynomial fit degree used by AI Feynman',
    )

    args = parser.parse_args()
    if AI_FEYNMAN_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Failed to import the AI Feynman package from either the environment or src/aifeynman.\n"
            f"Original error: {AI_FEYNMAN_IMPORT_ERROR}"
        )

    _install_sigterm_salvage(args.temp_file)
    if args.seed is not None:
        seed_everything(args.seed)
    data = load_dataset(args.dataset, args.dataset_size, args.features, args.seed)
    run_feynman(
        data,
        args.temp_file,
        args.features,
        bf_try_time=args.bf_try_time or BF_TRY_TIME,
        nn_epochs=args.nn_epochs or NN_EPOCHS,
        polyfit_degree=args.polyfit_degree or POLYFIT_DEGREE,
    )

if __name__ == '__main__':
    main()
