"""Tests for chunked ingest against a synthetic CSV with the real file's quirks."""

from __future__ import annotations

import pandas as pd
import pytest

from creditsurv.io import schema as sch
from creditsurv.io.loaders import (
    find_header_row,
    ingest_accepted,
    ingest_rejected,
    stratified_sample,
)


class TestHeaderDetection:
    def test_skips_prospectus_preamble(self, messy_accepted_csv):
        assert find_header_row(messy_accepted_csv, set(sch.ingest_columns())) == 1

    def test_finds_row_zero_when_there_is_no_preamble(self, messy_rejected_csv):
        assert find_header_row(messy_rejected_csv, set(sch.REJECTED_COLUMNS)) == 0

    def test_raises_on_a_file_with_no_matching_header(self, tmp_path):
        path = tmp_path / "wrong.csv"
        path.write_text("alpha,beta\n1,2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="right file"):
            find_header_row(path, {"loan_amnt", "issue_d"})


class TestIngestAccepted:
    @pytest.fixture
    def ingested(self, messy_accepted_csv, tmp_path):
        out = tmp_path / "accepted.parquet"
        audit = ingest_accepted(messy_accepted_csv, out, chunksize=2)
        return pd.read_parquet(out), audit

    def test_drops_footer_and_blank_rows(self, ingested):
        df, audit = ingested
        assert len(df) == 3
        # pandas skips the truly blank line itself, so only the totals footer
        # reaches the junk filter.
        assert audit["n_junk_rows_dropped"] == 1
        assert audit["n_rows_read"] == 4

    def test_chunking_does_not_change_the_result(self, messy_accepted_csv, tmp_path):
        small = tmp_path / "a.parquet"
        big = tmp_path / "b.parquet"
        ingest_accepted(messy_accepted_csv, small, chunksize=1)
        ingest_accepted(messy_accepted_csv, big, chunksize=10_000)
        pd.testing.assert_frame_equal(pd.read_parquet(small), pd.read_parquet(big))

    def test_strips_thousands_separators(self, ingested):
        df, _ = ingested
        assert df.loc[df["id"] == "1", "annual_inc"].iloc[0] == pytest.approx(60_000.0)

    def test_strips_percent_sign_from_rate(self, ingested):
        df, _ = ingested
        assert df.loc[df["id"] == "1", "int_rate"].iloc[0] == pytest.approx(13.5)

    def test_numeric_columns_are_float32(self, ingested):
        df, _ = ingested
        for col in ("loan_amnt", "dti", "fico_range_low"):
            assert df[col].dtype == "float32"

    def test_reports_requested_columns_absent_from_the_file(self, ingested):
        _, audit = ingested
        # The fixture has only core columns, so extended ones must be reported.
        assert "num_sats" in audit["columns_requested_but_absent"]

    def test_no_leakage_column_is_ingested(self, ingested):
        df, _ = ingested
        assert not (set(df.columns) & sch.LEAKAGE_COLUMNS)

    def test_label_columns_survive_ingest(self, ingested):
        df, _ = ingested
        for col in sch.LABEL_COLUMNS:
            assert col in df.columns

    def test_row_limit_truncates(self, messy_accepted_csv, tmp_path):
        out = tmp_path / "limited.parquet"
        ingest_accepted(messy_accepted_csv, out, chunksize=1, row_limit=1)
        assert len(pd.read_parquet(out)) == 1

    def test_missing_file_raises_clearly(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ingest_accepted(tmp_path / "nope.csv", tmp_path / "o.parquet")


class TestIngestRejected:
    @pytest.fixture
    def ingested(self, messy_rejected_csv, tmp_path):
        out = tmp_path / "rejected.parquet"
        audit = ingest_rejected(messy_rejected_csv, out, chunksize=2)
        return pd.read_parquet(out), audit

    def test_columns_are_renamed_to_accepted_vocabulary(self, ingested):
        df, _ = ingested
        assert set(df.columns) == set(sch.REJECTED_RENAMES.values())

    def test_all_rows_retained(self, ingested):
        df, _ = ingested
        assert len(df) == 3

    def test_non_numeric_risk_score_becomes_nan_not_an_error(self, ingested):
        df, _ = ingested
        assert df["risk_score"].isna().sum() == 2  # blank, and the word "momentum"

    def test_dti_left_as_text_for_stage_4_to_decide(self, ingested):
        df, _ = ingested
        assert df["dti_raw"].dtype == object
        assert "1200%" in set(df["dti_raw"])


class TestLeakageGuard:
    def test_feature_columns_exclude_lc_grade_by_default(self):
        cols = sch.feature_columns()
        assert not (set(cols) & set(sch.LC_ASSESSMENT_COLUMNS))

    def test_feature_columns_can_include_lc_grade_for_the_benchmark(self):
        cols = sch.feature_columns(with_lc_grade=True)
        assert "grade" in cols and "int_rate" in cols

    def test_last_pymnt_d_is_rejected_as_a_feature(self):
        with pytest.raises(sch.LeakageError, match="last_pymnt_d"):
            sch.assert_no_leakage(["loan_amnt", "last_pymnt_d"])

    @pytest.mark.parametrize(
        "col", ["recoveries", "total_rec_prncp", "last_fico_range_low", "hardship_flag"]
    )
    def test_each_known_leak_is_caught(self, col):
        with pytest.raises(sch.LeakageError):
            sch.assert_no_leakage(["loan_amnt", col])

    def test_clean_feature_list_passes(self):
        sch.assert_no_leakage(["loan_amnt", "dti", "fico_range_low", "purpose"])


class TestStratifiedSample:
    @pytest.fixture
    def frame(self):
        return pd.DataFrame(
            {
                "issue_year": [2015] * 90 + [2016] * 9 + [2017],
                "term_months": [36] * 100,
                "x": range(100),
            }
        )

    def test_returns_requested_size(self, frame):
        assert len(stratified_sample(frame, 50, by=("issue_year",), seed=1)) == 50

    def test_returns_everything_when_n_exceeds_population(self, frame):
        assert len(stratified_sample(frame, 500, by=("issue_year",), seed=1)) == 100

    def test_rare_strata_are_not_dropped(self, frame):
        out = stratified_sample(frame, 20, by=("issue_year",), seed=1)
        assert set(out["issue_year"]) == {2015, 2016, 2017}

    def test_is_reproducible_under_a_fixed_seed(self, frame):
        a = stratified_sample(frame, 30, by=("issue_year",), seed=99)
        b = stratified_sample(frame, 30, by=("issue_year",), seed=99)
        pd.testing.assert_frame_equal(a, b)

    def test_falls_back_to_simple_random_when_strata_absent(self, frame):
        out = stratified_sample(frame, 25, by=("nonexistent",), seed=3)
        assert len(out) == 25
