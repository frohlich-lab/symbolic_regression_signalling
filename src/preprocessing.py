"""Merge raw splits into processed datasets with a shared 80/20 train/test split."""

import argparse
import numpy as np
import pandas as pd
from typing import Optional, Tuple


def _prepare_dataframe(df: pd.DataFrame, target_feature: str) -> pd.DataFrame:
    """Rename columns, construct derived features, and ensure target is last."""
    column_mapping = {
        'K(p=None)': 'K',
        "P(phospho='u', k=None)": 'P_u',
        "P(phospho='p', k=None)": 'P_p',
        "K(p=1) % P(phospho='u', k=1)": 'KPu',
        "S(k=None)": 'P_u',
        "P(k=None)": 'P_p',
        "K(p=1) % S(k=1)": 'KPu',
        "K(p=1) % P(k=1)": 'KP_complex',
        'koff_substrate': 'k_off',
        'kD_substrate': 'k_D',
        'kcat': 'k_cat',
        'kinact': 'k_inact',
        'dP()': 'dP',
        'dK()': 'dK',
    }

    df = df.rename(column_mapping, axis=1)

    if 'K' in df.columns and 'KPu' in df.columns:
        # Values are stored in log-space; use logaddexp for a stable log-sum-exp.
        df['tK'] = np.logaddexp(df['K'], df['KPu'])

    if target_feature in df.columns:
        ordered = [col for col in df.columns if col != target_feature] + [target_feature]
        df = df[ordered]

    return df


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
            test_ids = rng.choice(unique_ids, size=n_test_ids, replace=False)
            train_ids = np.setdiff1d(unique_ids, test_ids)

            train_df = merged[merged['condition_id'].isin(train_ids)].reset_index(drop=True)
            test_df = merged[merged['condition_id'].isin(test_ids)].reset_index(drop=True)
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

    processed_frames = [
        _prepare_dataframe(df, target_feature)
        for df in (train_raw, test_raw, valid_raw)
    ]

    merged = pd.concat(processed_frames, ignore_index=True)
    merged.to_csv(output_path, index=False)

    if train_output_path or test_output_path:
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
