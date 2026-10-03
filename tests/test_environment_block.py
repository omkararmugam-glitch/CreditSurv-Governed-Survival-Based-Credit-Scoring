"""A blocked native library produces a sentence, not a traceback.

Windows Smart App Control moved itself from evaluation to enforcing on 25 September
2026 and began blocking `lib_lightgbm.dll` and shap's `_cext`. Unpickling a model
bundle imports lightgbm, so the app failed with `OSError: [WinError 4551]` raised
from inside `pickle.load` -- a stack trace that names neither the cause nor anything
the reader can do. These tests pin the explanation, and they run on any platform,
because the message has to be right on the machine that cannot reproduce the block.
"""

from __future__ import annotations

import pytest

from creditsurv import environment as env
from creditsurv.environment import (ImportCheck, blocked_imports,
                                    policy_block_message, smart_app_control_state)

BLOCKED = ImportCheck(
    "lightgbm", "the gradient-boosted hazard model that scores applicants", False,
    "OSError: [WinError 4551] An Application Control policy has blocked this file")
MISSING = ImportCheck("lightgbm", "the model", False,
                      "ModuleNotFoundError: No module named 'lightgbm'")
BROKEN = ImportCheck("shap", "the reasons", False,
                     "ImportError: DLL load failed while importing _internal: An "
                     "Application Control policy has blocked this file.")


def test_a_policy_block_is_told_apart_from_a_missing_package():
    """The two need opposite advice: one is fixed by installing, the other cannot be
    fixed by installing at all, so confusing them sends the reader the wrong way."""
    assert BLOCKED.blocked_by_policy
    assert BROKEN.blocked_by_policy
    assert not MISSING.blocked_by_policy
    assert not ImportCheck("lightgbm", "x", True).blocked_by_policy


def test_the_message_names_the_cause_and_what_is_lost():
    message = policy_block_message([BLOCKED, BROKEN])
    assert "Smart App Control" in message
    assert "lightgbm and shap" in message
    assert "reputation" in message
    # The reader's first guess is a broken install; the message has to close it off.
    assert "reinstalling them cannot help" in message
    assert "28 August 2026" in message
    # And it must not let them expect a stable state either way.
    assert "can change without anything local changing" in message
    for lost in ("scores applicants", "the reasons"):
        assert lost in message


def test_the_message_offers_a_way_forward_and_refuses_the_wrong_one():
    message = policy_block_message([BLOCKED])
    assert "WSL2" in message
    assert "not recommended" in message          # turning the policy off
    assert "reinstalling Windows" in message     # why that door does not reopen
    assert "what still works" in message.lower()


def test_no_message_when_nothing_is_blocked():
    assert policy_block_message([]) == ""


def test_the_message_is_printable_on_a_windows_console():
    """It goes to stderr from 06_score_upload.py, where a non-ASCII character would
    raise UnicodeEncodeError on a cp1252 console and hide the explanation."""
    message = policy_block_message([BLOCKED, BROKEN])
    message.encode("cp1252")                     # raises if it cannot be written
    assert message == message.encode("ascii", "strict").decode()


def test_blocked_imports_reports_the_real_machine():
    """Whatever this machine's state, the call must not raise and must be consistent
    with whether the packages actually import."""
    blocked = blocked_imports()
    names = {c.name for c in blocked}
    for name in ("lightgbm", "shap"):
        try:
            __import__(name)
        except BaseException:
            pass                                  # may or may not be a policy block
        else:
            assert name not in names, f"{name} imports, so it must not be listed"


def test_the_policy_state_is_only_ever_read():
    state = smart_app_control_state()
    assert isinstance(state, str) and state
    import inspect

    source = inspect.getsource(env)
    for writer in ("SetValueEx", "DeleteValue", "CreateKey", "KEY_WRITE",
                   "KEY_SET_VALUE", "subprocess", "os.system"):
        assert writer not in source, f"{writer} has no business in this module"


def test_the_check_does_not_raise_on_a_package_that_explodes(monkeypatch):
    """A DLL loader can raise things that are not ImportError, including SystemExit
    from a badly behaved extension, and none of them may escape the check."""
    def boom(name):
        raise KeyboardInterrupt("a rude extension module")

    monkeypatch.setattr(env.importlib, "import_module", boom)
    monkeypatch.delitem(env.sys.modules, "lightgbm", raising=False)
    checks = env.check_native_imports((("lightgbm", "the model"),))
    assert checks[0].ok is False
    assert "KeyboardInterrupt" in checks[0].error


@pytest.mark.parametrize("path,needle", [
    ("src/creditsurv/api/app.py", "policy_block_message"),
    ("app/views/_common.py", "blocked_message"),
    ("src/creditsurv/batch.py", "policy_blocked_exception"),
    ("scripts/06_score_upload.py", "policy_block_message"),
])
def test_every_scoring_entry_point_explains_itself(path, needle):
    """The ways a model gets loaded: the API (whose explanation every dashboard page
    shows), the library, the CLI. The API and the CLI check up front, because they can say so before any work starts;
    load_context translates the failure instead, so a model with no native library
    in it is never refused for a library it does not use."""
    import pathlib

    assert needle in pathlib.Path(path).read_text(encoding="utf-8"), path


def test_a_failure_that_is_not_a_policy_block_passes_through():
    """load_context must not blame Windows for a missing file."""
    from creditsurv.environment import policy_blocked_exception

    assert not policy_blocked_exception(FileNotFoundError("02_models_x.pkl not found"))
    assert policy_blocked_exception(
        OSError("[WinError 4551] An Application Control policy has blocked this file"))


def test_a_policy_block_is_found_through_a_chained_exception():
    """pickle re-raises, so the interesting exception is often a __context__."""
    from creditsurv.environment import policy_blocked_exception

    try:
        try:
            raise OSError("[WinError 4551] An Application Control policy has "
                          "blocked this file")
        except OSError as inner:
            raise RuntimeError("could not rebuild the booster") from inner
    except RuntimeError as outer:
        assert policy_blocked_exception(outer)
