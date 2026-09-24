"""Typed configuration loaded from ``config/config.yaml``.

Every path, seed and threshold the pipeline uses lives in the YAML file so that
a run is reproducible from one artefact. Nothing here reads the environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = ["Paths", "IngestConfig", "SampleConfig", "ModelConfig",
           "ExplainConfig", "DiagnosticConfig", "CleaningConfig", "DecisionConfig",
           "Config",
           "load_config"]


@dataclass(frozen=True)
class Paths:
    accepted_csv: Path | None = None
    rejected_csv: Path | None = None
    data_dir: Path = Path("outputs/data")
    models_dir: Path = Path("outputs/models")
    figures_dir: Path = Path("outputs/figures")
    tables_dir: Path = Path("outputs/tables")

    @property
    def accepted_parquet(self) -> Path:
        return self.data_dir / "accepted_raw.parquet"

    @property
    def rejected_parquet(self) -> Path:
        return self.data_dir / "rejected_raw.parquet"

    @property
    def labeled_parquet(self) -> Path:
        return self.data_dir / "accepted_labeled.parquet"

    @property
    def dev_sample_parquet(self) -> Path:
        return self.data_dir / "dev_sample.parquet"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.models_dir, self.figures_dir, self.tables_dir):
            d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class IngestConfig:
    chunksize: int = 250_000
    extended_features: bool = True
    row_limit: int | None = None
    """Cap on rows read, for smoke-testing ingest against the real file."""


@dataclass(frozen=True)
class SampleConfig:
    dev_sample_size: int = 200_000
    stratify_by: tuple[str, ...] = ("issue_year", "term_months")
    seed: int = 20260921


@dataclass(frozen=True)
class ModelConfig:
    with_lc_grade: bool = False
    """Primary specification excludes Lending Club's own grade / int_rate."""
    test_size: float = 0.2
    time_bin_months: int = 1
    """Discrete-time hazard bin width. 1 on the dev sample; 3 (quarterly) keeps
    the person-period expansion tractable on the full dataset."""
    eval_horizons_months: tuple[int, ...] = (6, 12, 18, 24, 30, 36)
    seed: int = 20260921


@dataclass(frozen=True)
class ExplainConfig:
    n_explain: int = 1_000
    """SurvSHAP(t) is KernelSHAP-based and costs seconds per borrower, so
    explanations are computed on a sample, not the full population."""
    n_background: int = 100
    kernel_nsamples: int = 256
    segment_by: tuple[str, ...] = ("grade", "purpose", "income_band")
    seed: int = 20260921


@dataclass(frozen=True)
class DiagnosticConfig:
    """Pre-registered thresholds for the Stage 4 selection-bias gate.

    Fixed *before* looking at results. At n in the millions every KS p-value is
    ~0, so the gate is effect-size based; p-values are reported but never used
    to decide.
    """

    rejected_sample_size: int = 1_000_000
    smd_notable: float = 0.10
    smd_substantial: float = 0.25
    ks_substantial: float = 0.20
    separability_auc_strong: float = 0.75
    separability_auc_weak: float = 0.60
    min_common_support: float = 0.05
    """If less than this share of rejected applicants falls inside the accepted
    propensity range, reject inference is extrapolation and will not be applied.
    """
    seed: int = 20260921


@dataclass(frozen=True)
class CleaningConfig:
    """Cleaning rules, mirrored into :class:`creditsurv.cleaning.CleaningPolicy`.

    The defaults are the rule set the current trained models were produced under
    (``v1-parity``): read numbers written as text, align categories to the training
    levels, flag anything unusual, change no value. The four switches that would
    alter a model input are off; turning one on changes the training data and
    therefore requires retraining.
    """

    version: str = "v1-parity"
    coerce_numeric_text: bool = True
    align_categories: bool = True
    unseen_category_to_other: bool = False
    rare_category_min_count: int = 0
    clip_numeric: bool = False
    clip_quantiles: tuple[float, float] = (0.001, 0.999)
    range_quantiles: tuple[float, float] = (0.005, 0.995)
    drop_duplicate_ids: bool = False


@dataclass(frozen=True)
class DecisionConfig:
    """The approve/reject policy applied to a scored batch.

    The survival model returns a probability, not a decision. Turning one into
    the other is a *policy* choice and is therefore stated here, in config, shown
    on the dashboard and written into every run summary -- never buried in code.

    The default was chosen on the full model's 451,558-loan test split: rejecting
    at a 36-month default probability of 0.30 declines 17.0% of applicants and
    takes the approved population's observed default rate from 11.8% to 8.9%,
    while rejected applicants default at 25.9%. It also sits just above the 80th
    percentile of predicted risk (0.281). Every borrower in that population had
    already been approved by Lending Club, so a real applicant pool is riskier
    (see the Stage 4 selection-bias result).
    """

    model_tag: str = "full"
    """Which trained bundle the dashboard scores with."""
    model: str = "discrete_hazard"
    horizon_months: int = 36
    reject_at_or_above: float = 0.30
    explain_nsamples: int = 600
    explain_n_background: int = 100
    """Kept at the batch settings: a single applicant costs ~2.7s either way, and
    cheaper settings make the stated reasons unstable."""
    max_explained: int = 0
    """0 = no cap: every rejected applicant is explained, which is what Regulation B
    requires of a file of decisions. A positive value is an explicit override for a
    quick look at a large file; rows beyond it are marked, never left blank."""
    explain_workers: int = 0
    """0 = choose from cores and free memory. Explanations are seeded per applicant,
    so the worker count cannot change the reasons."""
    background_rows: int = 20_000
    """Training rows sampled before k-means summarising to explain_n_background."""
    bulk_explainer: str = "survshap"
    """Which explainer writes the reasons for a bulk scoring run.

    ``"survshap"`` is SurvSHAP(t) everywhere: sampled, about 2.7s per applicant.
    ``"treeshap"`` is the exact, deterministic tree explainer, about 4ms per
    applicant, available only for the discrete-hazard model. ``"auto"`` uses
    TreeSHAP when the model allows it and SurvSHAP(t) otherwise.

    Whichever runs, the explainer that produced each row's reasons is written into
    that row, so an output file always says where its reasons came from. Moving off
    "survshap" is a decision about what a notice is based on, which is why it is a
    config value with a validation behind it (FINDINGS section 7) rather than an
    internal default."""
    chunk_rows: int = 50_000
    """An upload is read, cleaned, scored and explained this many rows at a time, so
    peak memory follows the block size rather than the file size."""
    drift_min_rows: int = 500
    """Below this many rows the drift check reports "too few rows to assess" rather
    than a colour: see creditsurv.drift.MIN_ROWS for the arithmetic."""
    background_above_mb: float = 25.0
    """Uploads at least this large run in a detached background process
    (creditsurv.runner) instead of inside the page, so the browser can be left.
    Small files stay inline: a background run reloads the model and its SHAP
    background from scratch, which costs 20-40s the page already has cached."""
    min_feature_coverage: float = 0.60
    """Below this share of the model's features present in the upload, the run is
    flagged as degraded everywhere it is reported."""


@dataclass(frozen=True)
class Config:
    paths: Paths = field(default_factory=Paths)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)
    label: dict[str, Any] = field(default_factory=dict)
    model: ModelConfig = field(default_factory=ModelConfig)
    explain: ExplainConfig = field(default_factory=ExplainConfig)
    diagnostic: DiagnosticConfig = field(default_factory=DiagnosticConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)


def _as_path(value: Any) -> Path | None:
    return Path(value).expanduser() if value else None


def load_config(path: str | Path = "config/config.yaml") -> Config:
    """Read and validate the YAML config."""
    raw: dict[str, Any] = {}
    p = Path(path)
    if p.exists():
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    pr = raw.get("paths", {}) or {}
    paths = Paths(
        accepted_csv=_as_path(pr.get("accepted_csv")),
        rejected_csv=_as_path(pr.get("rejected_csv")),
        data_dir=Path(pr.get("data_dir", "outputs/data")),
        models_dir=Path(pr.get("models_dir", "outputs/models")),
        figures_dir=Path(pr.get("figures_dir", "outputs/figures")),
        tables_dir=Path(pr.get("tables_dir", "outputs/tables")),
    )

    def section(name: str, cls):
        payload = raw.get(name, {}) or {}
        allowed = {f for f in cls.__dataclass_fields__}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"unknown keys in config section '{name}': {sorted(unknown)}")
        coerced = {
            k: (tuple(v) if isinstance(v, list) else v) for k, v in payload.items()
        }
        return cls(**coerced)

    return Config(
        paths=paths,
        ingest=section("ingest", IngestConfig),
        sample=section("sample", SampleConfig),
        label=raw.get("label", {}) or {},
        model=section("model", ModelConfig),
        explain=section("explain", ExplainConfig),
        diagnostic=section("diagnostic", DiagnosticConfig),
        cleaning=section("cleaning", CleaningConfig),
        decision=section("decision", DecisionConfig),
    )
