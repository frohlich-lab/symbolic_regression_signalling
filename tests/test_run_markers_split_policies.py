import numpy as np
import pandas as pd

from pipelines.experimental.sr_pipeline.run_markers import (
    choose_bin_split,
    compute_pysr_sample_weights,
)


def test_choose_bin_split_top_gfp_bins_uses_highest_bins():
    df = pd.DataFrame({"GFP_bin": np.repeat(np.arange(50), 2)})

    train_bins, test_bins = choose_bin_split(
        df,
        test_size=0.2,
        random_state=42,
        split_policy="top_gfp_bins",
    )

    assert train_bins == set(range(40))
    assert test_bins == set(range(40, 50))


def test_choose_bin_split_random_bins_returns_disjoint_cover():
    df = pd.DataFrame({"GFP_bin": np.repeat(np.arange(10), 3)})

    train_bins, test_bins = choose_bin_split(
        df,
        test_size=0.2,
        random_state=7,
        split_policy="random_bins",
    )

    assert train_bins.isdisjoint(test_bins)
    assert train_bins | test_bins == set(range(10))
    assert len(test_bins) == 2


def test_compute_pysr_sample_weights_linear_gfp_bin_uses_reference_range():
    train_df = pd.DataFrame({"GFP_bin": [0, 25, 49]})
    reference_df = pd.DataFrame({"GFP_bin": np.arange(50)})

    weights = compute_pysr_sample_weights(
        train_df,
        "gfp_bin_linear",
        2.0,
        reference_df=reference_df,
    )

    assert np.isclose(weights[0], 1.0)
    assert np.isclose(weights[1], 1.0 + (25.0 / 49.0))
    assert np.isclose(weights[2], 2.0)
