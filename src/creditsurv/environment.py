"""Whether this machine can actually run the models, and a plain answer when it cannot.

Windows Smart App Control blocks a native library that it has no *reputation* for,
regardless of whether the library is signed. On 25 September 2026 it moved itself from
evaluation mode to enforcing on this machine, and from that moment
``lightgbm/bin/lib_lightgbm.dll`` and ``shap/_cext`` stopped loading -- with nothing in
the project having changed. Unpickling a model bundle imports lightgbm, so the failure
surfaces as a raw ``OSError: [WinError 4551]`` from deep inside ``pickle.load``, which
explains nothing to whoever is looking at the screen.

This module turns that into a sentence. It does not work around the policy, change any
security setting, or suggest doing either.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass

__all__ = ["NATIVE_REQUIREMENTS", "ImportCheck", "check_native_imports",
           "blocked_imports", "policy_block_message", "smart_app_control_state",
           "policy_blocked_exception"]

NATIVE_REQUIREMENTS: tuple[tuple[str, str], ...] = (
    ("lightgbm", "the gradient-boosted hazard model that scores applicants, and any "
                 "model bundle, because unpickling one imports it"),
    ("shap", "SurvSHAP(t), which produces the reasons on an adverse-action notice"),
)
"""Native packages the scoring path needs, and what is lost without each."""

_POLICY_MARKERS = ("application control policy", "winerror 4551",
                   "blocked by policy", "code integrity")


@dataclass(frozen=True)
class ImportCheck:
    """The result of trying to import one package."""

    name: str
    needed_for: str
    ok: bool
    error: str = ""

    @property
    def blocked_by_policy(self) -> bool:
        """Whether the import failed because something refused to load the binary.

        Distinguished from a missing package, because the two need opposite advice:
        one is fixed by installing, the other cannot be fixed by installing at all.
        """
        low = self.error.lower()
        return bool(self.error) and any(m in low for m in _POLICY_MARKERS)


def check_native_imports(requirements=NATIVE_REQUIREMENTS) -> list[ImportCheck]:
    """Try each import once and report, without raising."""
    out = []
    for name, needed_for in requirements:
        if name in sys.modules:
            out.append(ImportCheck(name, needed_for, True))
            continue
        try:
            importlib.import_module(name)
        except BaseException as exc:            # an OSError from a DLL loader, too
            out.append(ImportCheck(name, needed_for, False, f"{type(exc).__name__}: {exc}"))
        else:
            out.append(ImportCheck(name, needed_for, True))
    return out


def blocked_imports(requirements=NATIVE_REQUIREMENTS) -> list[ImportCheck]:
    """Only the ones a policy is blocking. Empty means this machine can score."""
    return [c for c in check_native_imports(requirements) if c.blocked_by_policy]


def smart_app_control_state() -> str:
    """Smart App Control's state, read from the registry. Read-only; never written.

    Returns "enforcing", "evaluation", "off", or a short reason it is unknown.
    """
    if sys.platform != "win32":
        return "not applicable (not Windows)"
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SYSTEM\CurrentControlSet\Control\CI\Policy") as key:
            value, _ = winreg.QueryValueEx(key, "VerifiedAndReputablePolicyState")
    except FileNotFoundError:
        return "off (no policy key)"
    except OSError as exc:
        return f"unknown ({exc.__class__.__name__})"
    return {0: "off", 1: "enforcing", 2: "evaluation"}.get(int(value),
                                                           f"unknown code {value}")


def policy_block_message(checks=None) -> str:
    """What is blocked, why it is not the project's fault, and what the options are.

    Written for whoever is looking at the screen, which is why it says what changed
    rather than which exception was raised.
    """
    checks = checks if checks is not None else blocked_imports()
    if not checks:
        return ""
    state = smart_app_control_state()
    found = [c.name for c in checks]
    names = (" and ".join(found) if len(found) < 3
             else ", ".join(found[:-1]) + " and " + found[-1])
    lines = [
        f"**Windows is blocking {names}, so applicants cannot be scored on this "
        f"machine.**",
        "",
        f"Smart App Control is **{state}**. It allows an unsigned native library "
        f"only when Microsoft's cloud reputation service recognises it, and almost "
        f"every binary in this environment is unsigned -- numpy, pandas and scipy "
        f"included. Those are recognised; these two are not.",
        "",
        "Nothing in this project changed. The files were installed on 28 August 2026 "
        "and have not been modified since, and they loaded on this machine earlier "
        "today. So reinstalling them cannot help: the blocked file is the file that "
        "was already working. The verdict is made per file by a service outside this "
        "machine, which is also why it can change without anything local changing -- "
        "in either direction.",
        "",
        "What is unavailable:",
    ]
    lines += [f"- `{c.name}`: {c.needed_for}" for c in checks]
    lines += [
        "",
        "What still works: reading results already on disk, the FINDINGS page, the "
        "run history, and every table and figure produced before now.",
        "",
        "Ways forward, none of which involve weakening Windows security:",
        "- Run the project under **WSL2** (`wsl --install`, then recreate the "
        "environment inside Linux). Linux processes are outside Windows code "
        "integrity policy, so both libraries load normally.",
        "- Run it on a machine where Smart App Control is off by default, for "
        "example a work laptop or a cloud VM.",
        "",
        "Turning Smart App Control off would also fix it, and is deliberately not "
        "recommended here: it cannot be re-enabled without reinstalling Windows.",
    ]
    return "\n".join(lines)


def policy_blocked_exception(exc: BaseException) -> bool:
    """Whether a failure that already happened was a policy block.

    Used where the import is implicit -- ``pickle.load`` on a model bundle imports
    lightgbm to rebuild a booster -- so the failure cannot be anticipated by name but
    can be recognised once it arrives. Preferred over checking up front, which would
    refuse a model that has no native library in it at all.
    """
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if any(m in f"{type(exc).__name__}: {exc}".lower() for m in _POLICY_MARKERS):
            return True
        exc = exc.__cause__ or exc.__context__
    return False
