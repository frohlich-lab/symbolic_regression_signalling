"""Merge raw splits into processed datasets with a shared 80/20 train/test split."""

import argparse
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("preprocessing")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)


def _prepare_dataframe(df: pd.DataFrame, target_feature: str) -> pd.DataFrame:
    """Rename columns and ensure target is last."""
    column_mapping = {
        'K(p=None)': 'K',
        "K(p=None, d=None)": 'K',  # (if present)
        "P(phospho='u', k=None)": 'P_u',
        "P(phospho='p', k=None)": 'P_p',

        # Monomer complexes
        "K(p=1) % P(phospho='u', k=1)": 'KPu',
        "K(p=1) % P(phospho='p', k=1)": 'KPp',
        "K(p=1, d=None) % P(phospho='u', k=1)": 'KPu',
        "K(p=1, d=None) % P(phospho='p', k=1)": 'KPp',
        "S(k=None)": 'P_u',
        "P(k=None)": 'P_p',
        "K(p=1) % S(k=1)": 'KPu',
        "K(p=1) % P(k=1)": 'KPp',

        # Dimers (new)
        "K(p=None, d=1) % K(p=None, d=1)": 'KK',
        "K(p=2, d=1) % K(p=None, d=1) % P(phospho='u', k=2)": 'KKPu',
        "K(p=2, d=1) % K(p=None, d=1) % P(phospho='p', k=2)": 'KKPp',
        "K(p=2, d=1) % K(p=3, d=1) % P(phospho='u', k=2) % P(phospho='u', k=3)": 'KKPuPu',
        "K(p=2, d=1) % K(p=3, d=1) % P(phospho='p', k=3) % P(phospho='u', k=2)": 'KKPpPu',
        "K(p=2, d=1) % K(p=3, d=1) % P(phospho='p', k=2) % P(phospho='p', k=3)": 'KKPpPp',

        # Kinetics
        'koff_substrate': 'k_off',
        'kD_substrate': 'k_D',
        'kcat': 'k_cat',
        'kinact': 'k_inact',

        # Derivatives (if present)
        'dP()': 'dP',
        'dK()': 'dK',
    }

    df = df.rename(column_mapping, axis=1)

    # After renaming, some columns collapse to identical names (e.g., multiple P_p variants).
    # Keep the first occurrence of each logical column to avoid redundant data copies.
    df = df.loc[:, ~df.columns.duplicated()]

    if target_feature in df.columns:
        ordered = [col for col in df.columns if col != target_feature] + [target_feature]
        df = df[ordered]

    return df


KINASE_BASE_COLUMNS = {"K", "KPu", "KPp", "KK", "KKPu", "KKPp", "KKPuPu", "KKPpPu", "KKPpPp"}
KINASE_EXCLUDE_PREFIXES = ("dK",)


def _looks_like_kinase_component(column: str) -> bool:
    """Heuristic to identify log-space kinase species columns."""

    if column in KINASE_BASE_COLUMNS:
        return True
    if not column or column[0] != 'K':
        return False
    if any(column.startswith(prefix) for prefix in KINASE_EXCLUDE_PREFIXES):
        return False
    if '(' in column or '% K(' in column:
        return True
    return False


def _compute_total_tk(df: pd.DataFrame) -> None:
    """Compute log-space total kinase levels by aggregating all detected components."""

    candidate_cols = [col for col in df.columns if _looks_like_kinase_component(col)]
    candidate_cols.sort()
    if len(candidate_cols) < 2:
        missing = sorted(KINASE_BASE_COLUMNS - set(candidate_cols))
        LOGGER.warning(
            "Skipping tK logaddexp: found only %d kinase column(s) (%s) across %d rows.",
            len(candidate_cols),
            ", ".join(candidate_cols) if candidate_cols else "none",
            len(df),
        )
        return

    if len(candidate_cols) < 3:
        LOGGER.info(
            "Computing tK from %s (detected via heuristic) over %d rows.",
            ", ".join(candidate_cols),
            len(df),
        )
    else:
        LOGGER.info(
            "Computing log-space total tK from %d kinase components across %d rows.",
            len(candidate_cols),
            len(df),
        )
        LOGGER.info("Kinase components contributing to tK: %s", ", ".join(candidate_cols))

    components = []
    for col in candidate_cols:
        vals = pd.to_numeric(df[col], errors='coerce').to_numpy(copy=True)
        components.append(np.where(np.isfinite(vals), vals, -np.inf))

    df['tK'] = np.logaddexp.reduce(components)


def _split_dataset(merged: pd.DataFrame, train_fraction: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the merged dataset into deterministic train/test partitions."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    if 'condition_id' in merged.columns:
        unique_ids = merged['condition_id'].unique()
        if len(unique_ids) > 1:
            n_test_ids = max(1, int(np.ceil(len(unique_ids) * (1 - train_fraction))))
            n_test_ids = min(len(unique_ids) - 1, n_test_ids)
            rng = np.random.default_rng(42)
            permuted_ids = rng.permutation(unique_ids)
            split_idx = len(permuted_ids) - n_test_ids
            train_ids = permuted_ids[:split_idx]
            test_ids = permuted_ids[split_idx:]

            LOGGER.info(
                "Condition-based split: %d train IDs / %d test IDs (of %d unique)",
                len(train_ids),
                len(test_ids),
                len(unique_ids),
            )

            train_mask = merged['condition_id'].isin(set(train_ids))
            test_mask = ~train_mask

            train_df = merged.loc[train_mask].reset_index(drop=True)
            test_df = merged.loc[test_mask].reset_index(drop=True)
            if not train_df.empty and not test_df.empty:
                return train_df, test_df

    # Fallback: row-wise sample when we cannot split by condition
    train_df = merged.sample(frac=train_fraction, random_state=42)
    test_df = merged.drop(train_df.index)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, test_df


def merge_and_preprocess_datasets(
    train_path: str,
    test_path: str,
    valid_path: str,
    output_path: str,
    target_feature: str,
    train_output_path: Optional[str] = None,
    test_output_path: Optional[str] = None,
    train_fraction: float = 0.8,
) -> None:
    """Merge raw splits, preprocess columns, and emit consistent train/test files."""
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    valid_raw = pd.read_csv(valid_path)

    total_rows = sum(len(df) for df in (train_raw, test_raw, valid_raw))
    LOGGER.info(
        "Preparing preprocessing for %d rows across train/test/valid splits; "
        "renaming columns and constructing derived features.",
        total_rows,
    )

    processed_frames = []
    for df, suffix in zip((train_raw, test_raw, valid_raw), ("_0", "_1", "_2")):
        prepared = _prepare_dataframe(df, target_feature)
        if 'condition_id' in prepared.columns:
            prepared['condition_id'] = prepared['condition_id'].astype(str) + suffix
        processed_frames.append(prepared)

    merged = pd.concat(processed_frames, ignore_index=True)
    _compute_total_tk(merged)

    # Ensure target remains at the end after deriving features
    if target_feature in merged.columns:
        ordered = [col for col in merged.columns if col != target_feature] + [target_feature]
        merged = merged[ordered]
    merged.to_csv(output_path, index=False)

    if train_output_path or test_output_path:
        LOGGER.info(
            "Creating deterministic train/test split (train_fraction=%.2f); "
            "grouped by condition_id when available.",
            train_fraction,
        )
        train_df, test_df = _split_dataset(merged, train_fraction=train_fraction)
        if train_output_path:
            train_df.to_csv(train_output_path, index=False)
        if test_output_path:
            test_df.to_csv(test_output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Merge and preprocess train, test, and validation datasets.'
    )
    parser.add_argument('--train', required=True, help='Path to the train dataset CSV file')
    parser.add_argument('--test', required=True, help='Path to the test dataset CSV file')
    parser.add_argument('--valid', required=True, help='Path to the validation dataset CSV file')
    parser.add_argument('--output', required=True, help='Path to the output merged dataset CSV file')
    parser.add_argument('--train-output', help='Optional path to save the processed train split (~80%)')
    parser.add_argument('--test-output', help='Optional path to save the processed test split (~20%)')
    parser.add_argument('--target_feature', required=True, help='Name of the target feature to move to the last column')

    args = parser.parse_args()
    merge_and_preprocess_datasets(
        train_path=args.train,
        test_path=args.test,
        valid_path=args.valid,
        output_path=args.output,
        target_feature=args.target_feature,
        train_output_path=args.train_output,
        test_output_path=args.test_output,
    )


if __name__ == '__main__':
    main()
