"""The drift check: does an upload look like the training data?"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditsurv.drift import (STABLE, category_distance, compare,
                              population_stability_index)
from creditsurv.features.build import FeatureSpec

SPEC = FeatureSpec(numeric=("loan_amnt", "annual_inc", "dti"),
                   categorical=("purpose",), structural_missing=())


def reference(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "loan_amnt": rng.uniform(1000, 35000, n),
        "annual_inc": rng.lognormal(11, 0.4, n),
        "dti": rng.uniform(1, 38, n),
        "purpose": rng.choice(["debt_consolidation", "credit_card", "car"], n,
                              p=[0.6, 0.3, 0.1])})


def test_same_distribution_is_stable():
    ref = reference()
    unshifted = reference(1000, seed=1)          # same generator, new draw
    result = compare(ref, unshifted, SPEC)
    assert result.status == "stable"
    assert result.colour == "green"
    assert result.n_large == 0 and result.n_moderate == 0
    assert (result.table["status"] == "stable").all()
    assert "looks like the training data" in result.headline()


def test_shifted_fixture_is_flagged():
    ref = reference()
    shifted = reference(1000, seed=2)
    shifted["annual_inc"] = shifted["annual_inc"] * 6          # large numeric shift
    shifted["dti"] = shifted["dti"] + 3                        # mild shift
    result = compare(ref, shifted, SPEC)
    assert result.status == "large"
    assert result.colour == "red"
    worst = result.table.iloc[0]
    assert worst["feature"] == "annual_inc" and worst["status"] == "large"
    assert result.table.loc[result.table["feature"] == "loan_amnt", "status"].iloc[0] \
        == "stable"
    assert any("annual_inc" in n for n in result.notes)
    assert "may not be valid" in result.headline()


def test_category_share_change_is_flagged():
    ref = reference()
    shifted = reference(1000, seed=3)
    shifted["purpose"] = "car"                                 # 10% -> 100%
    result = compare(ref, shifted, SPEC)
    row = result.table.loc[result.table["feature"] == "purpose"].iloc[0]
    assert row["measure"] == "TVD" and row["status"] == "large"
    assert "car" in row["detail"]


def test_unseen_levels_are_reported():
    ref = reference()
    shifted = reference(500, seed=4)
    shifted.loc[: len(shifted) // 2, "purpose"] = "moon_landing"
    tvd, unseen, moves = category_distance(ref["purpose"], shifted["purpose"])
    assert unseen > 0.4 and tvd > 0.25
    assert "moon_landing" in moves


def test_absent_feature_is_unknown_not_silently_stable():
    ref = reference()
    upload = reference(500, seed=5).drop(columns=["dti"])
    result = compare(ref, upload, SPEC)
    row = result.table.loc[result.table["feature"] == "dti"].iloc[0]
    assert row["status"] == "unknown"
    assert "absent from the upload" in row["detail"]
    assert result.status in ("moderate", "large")     # never reported as stable


def test_missing_values_shift_the_index():
    ref = pd.Series(np.linspace(0, 100, 2000))
    same = pd.Series(np.linspace(0, 100, 500))
    mostly_missing = pd.Series([np.nan] * 400 + list(np.linspace(0, 100, 100)))
    psi_same, _ = population_stability_index(ref, same)
    psi_missing, _ = population_stability_index(ref, mostly_missing)
    assert psi_same < 0.10 < psi_missing


def test_constant_reference_is_not_comparable():
    ref = pd.Series([5.0] * 100)
    psi, bins = population_stability_index(ref, pd.Series([5.0, 6.0]))
    assert np.isnan(psi) and bins == 0


def test_small_sample_is_not_assessed_rather_than_flagged_red():
    """A tiny file scores high PSI on features that have not moved, so it is
    reported as not assessable instead of as drifted."""
    ref = reference()
    tiny = reference(40, seed=9)
    result = compare(ref, tiny, SPEC)
    assert result.status == "insufficient"
    assert result.colour == "grey"
    assert result.n_rows == 40
    assert result.n_large == 0 and result.n_moderate == 0
    assert (result.table["status"] == "insufficient").all()
    assert "Too few rows to assess drift" in result.headline()
    assert "500" in result.headline()

    # The same tiny file, compared with the minimum lowered, does produce numbers:
    # the guard is about sample size, not about the file being unusable.
    forced = compare(ref, tiny, SPEC, min_rows=10)
    assert forced.status in ("stable", "moderate", "large")


def test_minimum_is_where_psi_noise_falls_below_the_first_band():
    """The documented reason for 500: E[PSI] under the null is about (k-1)/n, so a
    100-row sample sits at the 0.10 band on unshifted data and 500 sits near 0.02."""
    import numpy as np
    from creditsurv.drift import MIN_ROWS, population_stability_index

    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(size=20_000))
    psi_100 = np.median([population_stability_index(
        ref, pd.Series(rng.normal(size=100)))[0] for _ in range(15)])
    psi_min = np.median([population_stability_index(
        ref, pd.Series(rng.normal(size=MIN_ROWS)))[0] for _ in range(15)])
    # Unshifted data, yet a 100-row sample sits within a whisker of the 0.10 band,
    # so individual features routinely cross it; at 500 rows the noise is a fifth
    # of that, which is what makes a reading above 0.10 mean something.
    assert psi_100 > STABLE * 0.7
    assert psi_min < STABLE / 3
    assert psi_100 > 3 * psi_min     # noise falls roughly as 1/n


def test_just_above_the_minimum_is_assessed():
    ref = reference()
    ok = reference(520, seed=11)
    result = compare(ref, ok, SPEC)
    assert result.status != "insufficient"
    assert result.n_rows == 520
