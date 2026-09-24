"""Seeded, parallel, resumable SurvSHAP(t).

The whole point is that none of the three changes the answer: an applicant's
attributions must not depend on who else is in the batch, on how many cores ran it,
or on whether the run was interrupted.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from creditsurv.explain.parallel import (Ledger, explain_rows,
                                         explain_rows_parallel, keep_awake,
                                         row_seed, suggest_workers)
from creditsurv.models.discrete_hazard import DiscreteTimeHazardModel

TIMES = np.array([12.0, 36.0])


@pytest.fixture
def fitted(synthetic_survival_frame):
    df = synthetic_survival_frame
    X = df[["fico_range_low", "dti", "loan_amnt"]].copy()
    model = DiscreteTimeHazardModel(time_bin_months=6, max_horizon_months=36,
                                    num_boost_round=25)
    model.fit(X, df["duration_months"].to_numpy(), df["event"].to_numpy())
    return model, X


def test_row_seed_follows_the_identifier_not_the_position():
    assert row_seed(7, "APP-1") == row_seed(7, "APP-1")
    assert row_seed(7, "APP-1") != row_seed(7, "APP-2")
    assert row_seed(7, "APP-1") != row_seed(8, "APP-1")


def test_seeded_rows_do_not_depend_on_batch_membership(fitted):
    model, X = fitted
    rows = X.iloc[:6]
    ids = [f"APP-{i}" for i in range(6)]
    whole = explain_rows(model, rows, X, TIMES, row_ids=ids, nsamples=40,
                         n_background=8)
    alone = explain_rows(model, rows.iloc[[3]], X, TIMES, row_ids=[ids[3]],
                         nsamples=40, n_background=8)
    np.testing.assert_array_equal(whole.phi[3], alone.phi[0])


def test_parallel_output_is_identical_row_by_row(fitted):
    """The claim that justifies using cores at all."""
    model, X = fitted
    rows = X.iloc[:8]
    ids = [f"APP-{i:03d}" for i in range(8)]
    serial = explain_rows(model, rows, X, TIMES, row_ids=ids, nsamples=40,
                          n_background=8)
    parallel = explain_rows_parallel(model, rows, X, TIMES, row_ids=ids,
                                     nsamples=40, n_background=8, workers=3)
    for i in range(len(rows)):
        np.testing.assert_array_equal(serial.phi[i], parallel.phi[i])
    np.testing.assert_array_equal(serial.prediction, parallel.prediction)
    assert serial.feature_names == parallel.feature_names


def test_a_resumed_run_matches_an_uninterrupted_one(fitted, tmp_path):
    """Kill a run part-way, resume it, and compare with a single clean pass."""
    model, X = fitted
    rows = X.iloc[:6]
    ids = [f"APP-{i}" for i in range(6)]
    clean = explain_rows(model, rows, X, TIMES, row_ids=ids, nsamples=40,
                         n_background=8)

    ledger_path = tmp_path / "explained.jsonl"
    first = Ledger.load(ledger_path)
    explain_rows(model, rows.iloc[:3], X, TIMES, row_ids=ids[:3], nsamples=40,
                 n_background=8, ledger=first)          # "interrupted" after 3
    assert len(Ledger.load(ledger_path).done) == 3

    resumed = explain_rows(model, rows, X, TIMES, row_ids=ids, nsamples=40,
                           n_background=8, ledger=Ledger.load(ledger_path))
    np.testing.assert_allclose(clean.phi, resumed.phi, rtol=0, atol=1e-12)
    assert len(Ledger.load(ledger_path).done) == 6


def test_resuming_skips_work_rather_than_repeating_it(fitted, tmp_path):
    model, X = fitted
    rows, ids = X.iloc[:4], [f"APP-{i}" for i in range(4)]
    path = tmp_path / "ledger.jsonl"
    explain_rows(model, rows, X, TIMES, row_ids=ids, nsamples=40, n_background=8,
                 ledger=Ledger.load(path))
    seen = []
    explain_rows(model, rows, X, TIMES, row_ids=ids, nsamples=40, n_background=8,
                 ledger=Ledger.load(path), progress=lambda i, n: seen.append(i))
    assert seen == []                      # nothing left to do, so nothing was done


def test_a_half_written_ledger_line_is_ignored(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text('{"row_id": "A", "phi": [[1.0, 2.0]]}\n{"row_id": "B", "ph',
                    encoding="utf-8")
    ledger = Ledger.load(path)
    assert ledger.has("A") and not ledger.has("B")


def test_parallel_resumes_from_a_ledger_too(fitted, tmp_path):
    model, X = fitted
    rows, ids = X.iloc[:6], [f"APP-{i}" for i in range(6)]
    path = tmp_path / "ledger.jsonl"
    explain_rows(model, rows.iloc[:2], X, TIMES, row_ids=ids[:2], nsamples=40,
                 n_background=8, ledger=Ledger.load(path))
    serial = explain_rows(model, rows, X, TIMES, row_ids=ids, nsamples=40,
                          n_background=8)
    parallel = explain_rows_parallel(model, rows, X, TIMES, row_ids=ids, nsamples=40,
                                     n_background=8, workers=2,
                                     ledger=Ledger.load(path))
    np.testing.assert_array_equal(serial.phi, parallel.phi)


def test_worker_count_is_bounded_by_the_machine():
    assert suggest_workers(1) == 1                    # never more workers than rows
    assert 1 <= suggest_workers(1000) <= max(1, (__import__("os").cpu_count() or 2))


def test_sleep_request_is_made_and_released():
    granted, note = keep_awake(True)
    try:
        assert isinstance(granted, bool) and note
        if granted:
            # The promise is deliberately limited, and the wording says so.
            assert "closed lid" in note and "resumes" in note
    finally:
        released, release_note = keep_awake(False)
    assert "released" in release_note or not released
