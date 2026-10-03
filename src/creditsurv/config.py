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
    registry: Path = Path("config/models.yaml")
    """The model registry (creditsurv.registry): which models may score for
    lending decisions."""

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

    model_tag: str = "full_applicant_nogeo"
    """Which trained bundle scoring uses by default. It must be approved in the
    model registry (config/models.yaml) or the run is refused; see
    creditsurv.registry."""
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
    explain_confirm_above: int = 1000
    """Phase 2 (reasons and notices) starts on its own when a run has at most this
    many rejected applicants. Above it, the dashboard asks first -- explain all, a
    random sample of N, or skip (stamped not for lending decisions) -- and the CLI
    stops after Phase 1 and prints the three commands. Counted in rejected
    applicants, not megabytes, because that is what Phase 2's time is proportional
    to: a large file that is mostly approved is quick to explain. Any file, any
    caller; nothing branches on a name."""
    explain_seconds_each: float = 2.1
    """Wall seconds per rejected applicant for SurvSHAP(t) at the scoring settings on
    this machine (FINDINGS D5 measured 2.06 s with the pool). Used only for the ETA
    shown before Phase 2 starts; while it runs, the ETA is measured."""
    phase2_checkpoint_seconds: float = 20.0
    """How often Phase 2 writes the reasons it has into rejected_applicants.csv and
    the notice zip while it runs (at least; it stretches the gap for a large file so
    rewriting never costs more than a fifth of the time)."""
    fair_lending_review_share: float = 0.05
    """Every run reports the share of rejections in which a non-disclosable feature
    (addr_state, zip_code, policy_code) was among the strongest adverse drivers.
    Above this share the dashboard asks for fair-lending review. 5%: one decline in
    twenty resting partly on a reason the lender may not state is systematic, not
    incidental -- and for a model with no such input the share is 0 by
    construction, so anything above it is itself a finding. Test 1 on the
    deprecated full model measured 65% (166 of 255)."""
    min_feature_coverage: float = 0.60
    """Below this share of the model's features present in the upload, the run is
    flagged as degraded everywhere it is reported."""
    input_quality_max_share: float = 0.05
    """The input-quality check fails the run if any model feature in the file is
    unreadable for more than this share of rows, or missing for more than this
    share *beyond* its missing rate in training (many bureau fields are blank for
    most applicants by design). Decisions are not issued on inputs the model never
    saw. Test 3 read term_months as unreadable for every row and passed.

    It judges only features the file *supplied*. A feature absent altogether is the
    unlearned_missing_* settings below."""
    unlearned_missing_floor: float = 0.01
    """A feature missing in less than this share of the model's training rows has no
    learned route for being absent, so dropping it is an unlearned tree default and
    not a degradation. The second half of the required/optional gate, applied
    whatever the ablation cost says, because ablation measures ranking lost and an
    unlearned default shifts the level instead (FINDINGS 7o)."""
    unlearned_missing_action: str = "fill"
    """What to do when a file omits such a feature.

    ``"fill"``   substitute the value fitted on the training rows -- the median for a
                 numeric feature, the modal level for a categorical one -- and stamp
                 the run and every affected feature by name, so the substitution is
                 visible rather than inferred from a coverage percentage.
    ``"block"``  refuse the file, as for a required feature.

    ``"fill"`` is the default because it is the honest reading of what happens: the
    model is going to use *some* constant for that column either way, and a fitted
    median is a defensible one that the run can name, where the booster's default
    direction is neither. ``"block"`` is for a caller who would rather fix the file
    than score a stamped run."""


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
        registry=Path(pr.get("registry", "config/models.yaml")),
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
