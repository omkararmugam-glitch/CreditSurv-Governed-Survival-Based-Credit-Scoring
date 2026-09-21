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
           "ExplainConfig", "DiagnosticConfig", "Config", "load_config"]


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
class Config:
    paths: Paths = field(default_factory=Paths)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)
    label: dict[str, Any] = field(default_factory=dict)
    model: ModelConfig = field(default_factory=ModelConfig)
    explain: ExplainConfig = field(default_factory=ExplainConfig)
    diagnostic: DiagnosticConfig = field(default_factory=DiagnosticConfig)


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
    )
