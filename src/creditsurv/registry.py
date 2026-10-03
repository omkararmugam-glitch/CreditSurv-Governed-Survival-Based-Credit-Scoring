"""The model registry: which trained models may make lending decisions.

``config/models.yaml`` records, for every model this project has trained, what it
is (features, training data and split, validation metrics), what is known to be
wrong with it (defects, non-disclosable inputs), and one **status**:

==============  ==============================================================
``approved``    may score applicants for lending decisions
``candidate``   proposed for approval; evidence not yet complete
``benchmark``   a research comparison; never for lending decisions
``deprecated``  withdrawn, with the reason recorded
==============  ==============================================================

Scoring used to pick a model from ``decision.model_tag`` alone, so nothing stopped
the ``full`` model -- which has the credit-score defect of FINDINGS 7c and uses
``addr_state`` -- from writing 255 real-looking notices in test 1. Now every scoring
run asks :func:`assess` first, and a model that is not approved is refused unless
the caller overrides explicitly, in which case the run is stamped "not for lending
decisions" everywhere it is reported.

Approval is not a word typed into a file
----------------------------------------
``status: approved`` is accepted only together with an ``approval`` block written
by ``scripts/07_model_registry.py approve``, which refuses to write it unless every
rule in :data:`APPROVAL_RULES` passes. Scoring then re-checks what can be checked
cheaply on every run: the model file is the one approved (SHA-256), its features
are the ones recorded, each evidence file still hashes to what the approval saw,
and the explainer and its settings are ones the evidence covers.

This module reads JSON and YAML only. It never unpickles a model, so the approval
command runs on Windows where lightgbm is blocked.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import yaml

from .provenance import PROJECT_ROOT

__all__ = ["STATUSES", "APPROVAL_RULES", "REPRODUCIBILITY_BARS", "ModelRecord",
           "Registry", "RuleResult", "Assessment", "load_registry", "assess",
           "check_approval_rules", "sha256_of", "RegistryError", "evaluate_rules",
           "approve_model", "ApprovalOutcome", "synced_copy_source", "SYNC_MARKER"]

STATUSES = ("approved", "candidate", "benchmark", "deprecated")

REPRODUCIBILITY_BARS = {"top1_agreement": 0.90, "mean_top4_overlap": 0.85}
"""SurvSHAP(t) against itself under two seeds, at the settings scoring uses.

A notice's reasons must be as reproducible as FINDINGS 7a already requires any
cheaper setting to be (S1 >= 90%, S2 >= 0.85): a model whose reasons change more
than that between two draws states reasons that are partly noise. Fixed here before
``full_applicant_nogeo`` has a measurement; ``full`` measured 95.3% / 0.936."""

APPROVAL_RULES: dict[str, str] = {
    "A1_model_file": "the model file exists and its SHA-256 is the one recorded",
    "A2_features": "the recorded features are the ones the model was trained on "
                   "(02_metrics_<tag>.json)",
    "A3_no_blocking_defect": "no known defect marked blocking",
    "A4_cleaning_values": "cleaning values fitted on this model's whole training "
                          "split, for this model file",
    "A5_ablation": "a measured ablation (03e) for this model file, so required "
                   "columns are measured rather than hand-picked",
    "A6_explainer_validation": "an explainer validation (03d) for this model file, "
                               "with SurvSHAP(t) reproducible at the scoring "
                               "settings: top-1 >= 90% and top-4 overlap >= 0.85",
    "A7_explain_settings": "scoring settings other than the ones A6 measured need a "
                           "passing settings comparison (03f) for this model file "
                           "-- the D3 rule of FINDINGS 7a",
}


class RegistryError(RuntimeError):
    """The registry file is missing or malformed."""


def sha256_of(path: Path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve(p: str | Path, root: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else root / p


@dataclass
class ModelRecord:
    tag: str
    status: str
    summary: str = ""
    model_file: str = ""
    model_sha256: str = ""
    features: dict = field(default_factory=dict)          # numeric / categorical
    training: dict = field(default_factory=dict)
    known_defects: list = field(default_factory=list)     # {id, summary, blocking}
    non_disclosable_features: list = field(default_factory=list)
    validation_metrics: dict = field(default_factory=dict)
    status_reason: str = ""
    approval: dict = field(default_factory=dict)
    """Written only by ``07_model_registry.py approve``: who, when, which FINDINGS
    section, the SHA-256 of every evidence file, and the explainer settings the
    evidence covers."""

    @property
    def all_features(self) -> tuple[str, ...]:
        return tuple(self.features.get("numeric", [])) + \
            tuple(self.features.get("categorical", []))

    @property
    def uses_non_disclosable(self) -> bool:
        return bool(self.non_disclosable_features)

    @property
    def blocking_defects(self) -> list[dict]:
        return [d for d in self.known_defects if d.get("blocking")]

    def to_dict(self) -> dict:
        out = {"status": self.status, "summary": self.summary,
               "status_reason": self.status_reason, "model_file": self.model_file,
               "model_sha256": self.model_sha256, "features": self.features,
               "training": self.training, "known_defects": self.known_defects,
               "uses_non_disclosable_features": self.uses_non_disclosable,
               "non_disclosable_features": self.non_disclosable_features,
               "validation_metrics": self.validation_metrics}
        if self.approval:
            out["approval"] = self.approval
        return out

    @classmethod
    def from_dict(cls, tag: str, raw: dict) -> "ModelRecord":
        status = str(raw.get("status", "")).strip()
        if status not in STATUSES:
            raise RegistryError(f"model {tag!r}: status {status!r} is not one of "
                                f"{', '.join(STATUSES)}")
        return cls(tag=tag, status=status, summary=raw.get("summary", ""),
                   model_file=raw.get("model_file", ""),
                   model_sha256=raw.get("model_sha256", ""),
                   features=raw.get("features") or {},
                   training=raw.get("training") or {},
                   known_defects=list(raw.get("known_defects") or []),
                   non_disclosable_features=list(raw.get("non_disclosable_features")
                                                 or []),
                   validation_metrics=raw.get("validation_metrics") or {},
                   status_reason=raw.get("status_reason", ""),
                   approval=raw.get("approval") or {})


HEADER = """\
# Model registry. Read by every scoring run (creditsurv/registry.py).
#
# status: approved   -- may score applicants for lending decisions
#         candidate  -- proposed; evidence not complete, so NOT for lending decisions
#         benchmark  -- research comparison only
#         deprecated -- withdrawn; status_reason says why
#
# A model is approved only by
#     python scripts/07_model_registry.py approve --model-tag TAG --by NAME --findings 7l
# which checks the approval rules (python scripts/07_model_registry.py rules) and
# writes the approval block. Scoring refuses "approved" without that block, and
# re-checks the model file, its features and every evidence file on each run.
#
# This file is written by 07_model_registry.py; comments other than this header
# are not preserved. Edit status_reason and known_defects by hand if needed, then
# re-run "07_model_registry.py check" to confirm the file still loads.
"""


@dataclass
class Registry:
    path: Path
    models: dict[str, ModelRecord]

    def get(self, tag: str) -> ModelRecord | None:
        return self.models.get(tag)

    def approved(self) -> list[str]:
        return sorted(t for t, r in self.models.items() if r.status == "approved")

    def save(self, path: Path | None = None) -> Path:
        path = Path(path or self.path)
        body = {"models": {t: r.to_dict() for t, r in sorted(self.models.items())}}
        path.write_text(HEADER + "\n" + yaml.safe_dump(body, sort_keys=False,
                                                       width=88),
                        encoding="utf-8")
        return path


def registry_path(cfg=None) -> Path:
    p = getattr(getattr(cfg, "paths", None), "registry", None)
    return _resolve(p or Path("config/models.yaml"), PROJECT_ROOT)


def load_registry(cfg=None, path: Path | None = None) -> Registry:
    path = Path(path) if path else registry_path(cfg)
    if not path.exists():
        raise RegistryError(f"no model registry at {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    models = raw.get("models")
    if not isinstance(models, dict):
        raise RegistryError(f"{path.name} has no 'models:' mapping")
    return Registry(path=path, models={str(t): ModelRecord.from_dict(str(t), r or {})
                                       for t, r in models.items()})


# ------------------------------------------------------------------ rules --

@dataclass
class RuleResult:
    rule: str
    passed: bool
    detail: str
    evidence: dict = field(default_factory=dict)       # file -> sha256


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _model_sha_in(payload: dict | None) -> str | None:
    prov = (payload or {}).get("provenance") or {}
    return ((prov.get("inputs") or {}).get("model") or {}).get("sha256")


def _find_by_model(tables_dir: Path, pattern: str, tag: str) -> list[tuple[Path, dict]]:
    found = []
    for p in sorted(Path(tables_dir).glob(pattern)):
        payload = _read_json(p)
        if payload and payload.get("model_tag") == tag:
            found.append((p, payload))
    return found


def check_approval_rules(record: ModelRecord, *, models_dir: Path, tables_dir: Path,
                         root: Path = PROJECT_ROOT,
                         explain_settings: tuple[int, int] | None = None
                         ) -> list[RuleResult]:
    """Evaluate every rule in :data:`APPROVAL_RULES` for one model, from files.

    ``explain_settings`` is the (nsamples, n_background) scoring will use; A7 asks
    whether the evidence covers it.
    """
    out: list[RuleResult] = []
    tag, sha = record.tag, record.model_sha256
    models_dir, tables_dir = _resolve(models_dir, root), _resolve(tables_dir, root)

    model_path = _resolve(record.model_file or models_dir / f"02_models_{tag}.pkl", root)
    actual = sha256_of(model_path)
    out.append(RuleResult(
        "A1_model_file", bool(actual) and actual == sha,
        f"{model_path.name}: " + ("missing" if not actual else
                                  "matches the recorded SHA-256" if actual == sha else
                                  f"SHA-256 {actual[:12]} is not the recorded "
                                  f"{(sha or 'none')[:12]}")))

    metrics_path = tables_dir / f"02_metrics_{tag}.json"
    metrics = _read_json(metrics_path)
    trained = (metrics or {}).get("features") or {}
    same = (metrics is not None
            and sorted(trained.get("numeric", [])) == sorted(record.features.get("numeric", []))
            and sorted(trained.get("categorical", [])) ==
            sorted(record.features.get("categorical", [])))
    out.append(RuleResult(
        "A2_features", same,
        f"{metrics_path.name}: " + ("missing" if metrics is None else
                                    "features match" if same else
                                    "features differ from the registry"),
        {str(metrics_path): sha256_of(metrics_path)} if metrics else {}))

    blocking = record.blocking_defects
    out.append(RuleResult(
        "A3_no_blocking_defect", not blocking,
        "none" if not blocking else "; ".join(f"{d.get('id')}: {d.get('summary')}"
                                              for d in blocking)))

    cv_path = models_dir / f"02_cleaning_values_{tag}.json"
    cv = _read_json(cv_path)
    n_train = record.training.get("n_train")
    cv_ok = (cv is not None and _model_sha_in(cv) == sha
             and (n_train is None or cv.get("fitted_rows") == n_train))
    if cv is None:
        cv_detail = "missing"
    elif _model_sha_in(cv) != sha:
        cv_detail = "for a different model file"
    elif not cv_ok:
        cv_detail = (f"fitted on {cv.get('fitted_rows')} rows, training split is "
                     f"{n_train}")
    else:
        cv_detail = f"fitted on the whole training split ({cv.get('fitted_rows')} rows)"
    out.append(RuleResult("A4_cleaning_values", cv_ok, f"{cv_path.name}: {cv_detail}",
                          {str(cv_path): sha256_of(cv_path)} if cv else {}))

    ablation = [(p, j) for p, j in _find_by_model(tables_dir, "03e_ablation_*.json", tag)
                if _model_sha_in(j) == sha and j.get("features")]
    out.append(RuleResult(
        "A5_ablation", bool(ablation),
        ablation[-1][0].name if ablation else
        f"no 03e_ablation_*.json for {tag} with this model file",
        {str(ablation[-1][0]): sha256_of(ablation[-1][0])} if ablation else {}))

    validations = [(p, j) for p, j in
                   _find_by_model(tables_dir, "03d_explainer_validation_*.json", tag)
                   if _model_sha_in(j) == sha]
    measured_settings = None
    treeshap_ok = False
    if validations:
        # Every validation of this model file counts, not the latest or the best:
        # a second run that passes does not cancel a first that failed, or the bar
        # could be met by re-running until the sampling noise falls the right way.
        def reproducible(j):
            c = j.get("reference_ceiling_survshap_vs_itself") or {}
            return all(float(c.get(k, 0.0)) >= bar
                       for k, bar in REPRODUCIBILITY_BARS.items())

        failing = [p.name for p, j in validations if not reproducible(j)]
        p, j = validations[-1] if not failing else next(
            (p, j) for p, j in validations if not reproducible(j))
        ceiling = j.get("reference_ceiling_survshap_vs_itself") or {}
        repro = not failing
        measured_settings = (int(j.get("nsamples", 0)), int(j.get("n_background", 0)))
        treeshap_ok = all(v.get("verdict") == "PASS" for _, v in validations)
        extra = (f"; {len(validations)} validations of this model file, "
                 f"{len(failing)} below the bar" if len(validations) > 1 else "")
        out.append(RuleResult(
            "A6_explainer_validation", repro,
            f"{p.name}: SurvSHAP(t) against itself top-1 "
            f"{float(ceiling.get('top1_agreement', 0)):.1%}, top-4 overlap "
            f"{float(ceiling.get('mean_top4_overlap', 0)):.3f} over "
            f"{ceiling.get('n', 0)} applicants at nsamples={measured_settings[0]}, "
            f"n_background={measured_settings[1]}; TreeSHAP stand-in "
            f"{j.get('verdict', 'not assessed')}{extra}",
            {str(q): sha256_of(q) for q, _ in validations}))
    else:
        out.append(RuleResult("A6_explainer_validation", False,
                              f"no 03d_explainer_validation_*.json for {tag} with "
                              f"this model file"))

    approved_settings = [list(measured_settings)] if measured_settings else []
    settings_evidence = {}
    for p, j in _find_by_model(tables_dir, "03f_settings_*.json", tag):
        if _model_sha_in(j) != sha:
            continue
        for c in j.get("candidates", []):
            if c.get("verdict") == "PASS":
                approved_settings.append([int(c["nsamples"]), int(c["n_background"])])
                settings_evidence[str(p)] = sha256_of(p)
    wanted = list(explain_settings) if explain_settings else None
    a7 = wanted is None or wanted in approved_settings
    out.append(RuleResult(
        "A7_explain_settings", a7,
        ("covered settings: " + (", ".join(f"{a}/{b}" for a, b in approved_settings)
                                 or "none"))
        + (f"; scoring uses {wanted[0]}/{wanted[1]}" if wanted else ""),
        settings_evidence))
    # Carried to the approval block rather than re-derived at scoring time.
    out[-1].evidence["_approved_settings"] = approved_settings
    out[-1].evidence["_treeshap_approved"] = treeshap_ok
    return out


# --------------------------------------------------------------- scoring --

@dataclass
class Assessment:
    """What a scoring run needs to know about its model."""

    tag: str
    status: str
    approved: bool
    problems: list[str] = field(default_factory=list)
    record: ModelRecord | None = None

    @property
    def label(self) -> str:
        return "approved" if self.approved else f"{self.status} (not approved)"

    def message(self) -> str:
        if self.approved:
            return f"Model '{self.tag}' is approved in the registry."
        return (f"Model '{self.tag}' is not approved for lending decisions "
                f"(registry status: {self.status}). " + " ".join(self.problems))


def assess(tag: str, cfg=None, *, model_path: Path | None = None,
           spec_features=None, explainer: str = "survshap",
           nsamples: int | None = None, n_background: int | None = None,
           registry: Registry | None = None) -> Assessment:
    """Is ``tag`` approved to make lending decisions, with this explainer?

    Cheap enough for every run: one hash of the model file and one of each evidence
    file, no model loading. Anything unverifiable counts against approval.
    """
    try:
        reg = registry or load_registry(cfg)
    except RegistryError as exc:
        return Assessment(tag, "unregistered", False, [str(exc)])
    rec = reg.get(tag)
    if rec is None:
        return Assessment(tag, "unregistered", False,
                          [f"'{tag}' is not in {reg.path.name}; register it with "
                           f"scripts/07_model_registry.py register --model-tag {tag}."])
    problems: list[str] = []
    if rec.status != "approved":
        problems.append(rec.status_reason or f"status is {rec.status}.")
        return Assessment(tag, rec.status, False, problems, rec)

    approval = rec.approval or {}
    if not approval.get("rules_passed"):
        problems.append("marked approved without an approval block from "
                        "07_model_registry.py approve.")
    if model_path is not None:
        actual = sha256_of(model_path)
        if actual != rec.model_sha256:
            problems.append(f"the model file is not the approved one (SHA-256 "
                            f"{(actual or 'missing')[:12]}, approved "
                            f"{rec.model_sha256[:12]}).")
    if spec_features is not None and set(map(str, spec_features)) != set(rec.all_features):
        extra = sorted(set(map(str, spec_features)) - set(rec.all_features))
        gone = sorted(set(rec.all_features) - set(map(str, spec_features)))
        problems.append(f"the model's features differ from the registry "
                        f"(extra: {extra[:5]}, missing: {gone[:5]}).")
    root = PROJECT_ROOT
    for path, expected in (approval.get("evidence") or {}).items():
        actual = sha256_of(_resolve(path, root))
        if actual != expected:
            problems.append(f"approval evidence {Path(path).name} has "
                            f"{'gone' if actual is None else 'changed'} since approval.")
    if rec.blocking_defects:
        problems.append("a blocking defect is recorded: "
                        + "; ".join(str(d.get("id")) for d in rec.blocking_defects))
    explainers = approval.get("explainers") or []
    if explainer not in explainers:
        problems.append(f"explainer {explainer} is not approved for this model "
                        f"(approved: {', '.join(explainers) or 'none'}).")
    if explainer == "survshap" and nsamples is not None and n_background is not None:
        settings = [list(map(int, s)) for s in approval.get("explain_settings") or []]
        if [int(nsamples), int(n_background)] not in settings:
            problems.append(f"SurvSHAP(t) settings {nsamples}/{n_background} are not "
                            f"covered by this model's evidence (covered: "
                            f"{', '.join(f'{a}/{b}' for a, b in settings) or 'none'}); "
                            f"run 03f for this model first (FINDINGS 7a).")
    return Assessment(tag, rec.status, not problems, problems, rec)


def approval_block(results: list[RuleResult], *, by: str, findings: str,
                   note: str = "") -> dict:
    """The block ``approve`` writes, from rules that all passed."""
    evidence = {}
    settings, treeshap = [], False
    for r in results:
        for k, v in r.evidence.items():
            if k == "_approved_settings":
                settings = v
            elif k == "_treeshap_approved":
                treeshap = bool(v)
            elif v:
                try:
                    k = Path(k).resolve().relative_to(PROJECT_ROOT).as_posix()
                except ValueError:
                    k = Path(k).as_posix()
                evidence[k] = v
    return {"rules_passed": sorted(r.rule for r in results if r.passed),
            "approved_by": by, "approved_on": date.today().isoformat(),
            "approved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "findings_section": findings, "note": note,
            "explainers": ["survshap"] + (["treeshap"] if treeshap else []),
            "explain_settings": settings, "evidence": evidence}


# ------------------------------------------ one check, CLI and page alike --

SYNC_MARKER = ".synced_from_windows"
"""Written by scripts/wsl_launch.sh into the root of the WSL copy it syncs."""


def evaluate_rules(cfg, record: ModelRecord) -> list[RuleResult]:
    """Every approval rule for one model, at the settings scoring will use.

    The single entry point for "does this model pass?": the ``rules`` and
    ``approve`` commands of 07_model_registry.py and the Model registry page all
    call this, so none of them can check a different set of rules or settings.
    """
    d = cfg.decision
    return check_approval_rules(
        record, models_dir=cfg.paths.models_dir, tables_dir=cfg.paths.tables_dir,
        explain_settings=(int(d.explain_nsamples), int(d.explain_n_background)))


def evidence_fingerprint(cfg, registry_file: Path | None = None) -> tuple:
    """Changes whenever a file a rule reads changes -- anything in the models and
    tables folders, or the registry itself -- so a cached rule result never outlives
    the evidence it was computed from."""
    registry_file = Path(registry_file or registry_path(cfg))
    items = []
    for folder in (cfg.paths.models_dir, cfg.paths.tables_dir):
        folder = Path(folder)
        if folder.is_dir():
            items += [(p.name, p.stat().st_mtime_ns, p.stat().st_size)
                      for p in folder.iterdir() if p.is_file()]
    if registry_file.exists():
        items.append(("registry", registry_file.stat().st_mtime_ns))
    return tuple(sorted(items))


def synced_copy_source(root: Path = PROJECT_ROOT) -> str | None:
    """The Windows folder this project was synced from, when it is the WSL copy.

    That copy's ``config/`` is replaced from Windows on every sync, so a registry
    change written there would be silently lost -- the reason approval refuses to
    write into it.
    """
    marker = Path(root) / SYNC_MARKER
    if not marker.is_file():
        return None
    try:
        return marker.read_text(encoding="utf-8").strip() or "Windows"
    except OSError:
        return "Windows"


@dataclass
class ApprovalOutcome:
    approved: bool
    results: list[RuleResult]
    refusal: str = ""

    @property
    def failed(self) -> list[RuleResult]:
        return [r for r in self.results if not r.passed]


def approve_model(cfg, tag: str, *, by: str, findings: str, note: str = "",
                  registry: Registry | None = None) -> ApprovalOutcome:
    """Re-check every rule for ``tag`` and, only if all pass, record the approval.

    What ``07_model_registry.py approve`` and the page's Approve button both run.
    Nothing is written unless every rule passes at this moment -- a result shown
    on a page a minute ago does not count.
    """
    reg = registry or load_registry(cfg)
    rec = reg.get(tag)
    if rec is None:
        return ApprovalOutcome(False, [], f"{tag} is not registered.")
    if rec.status == "deprecated":
        return ApprovalOutcome(False, [], f"{tag} is deprecated ({rec.status_reason}). "
                               f"Withdrawing that decision is a status change of its "
                               f"own: 07_model_registry.py set-status --status "
                               f"candidate --reason ...")
    if not (by or "").strip() or not (findings or "").strip():
        return ApprovalOutcome(False, [], "an approval names who made it and the "
                                          "FINDINGS section recording it.")
    source = synced_copy_source(reg.path.parent.parent)
    if source is not None:
        return ApprovalOutcome(
            False, [],
            f"{reg.path} belongs to the WSL copy synced from {source}. Its config/ is "
            f"replaced from Windows on every sync, so an approval written here would be "
            f"lost. Approve on Windows: from the Model registry page of an app started "
            f"there, or with python scripts/07_model_registry.py approve.")
    results = evaluate_rules(cfg, rec)
    if any(not r.passed for r in results):
        return ApprovalOutcome(False, results, f"{tag} stays {rec.status}: "
                               + ", ".join(r.rule for r in results if not r.passed)
                               + " failed.")
    rec.status = "approved"
    rec.status_reason = (f"approved {findings.strip()}; every approval rule passed"
                         + (f"; {note.strip()}" if note.strip() else ""))
    rec.approval = approval_block(results, by=by.strip(), findings=findings.strip(),
                                  note=note.strip())
    reg.save()
    return ApprovalOutcome(True, results)
