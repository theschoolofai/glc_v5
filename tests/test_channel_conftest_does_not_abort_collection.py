"""The defensive collection hook must skip one file, not the whole session.

`tests/channels/conftest.py` exists so that one unimportable adapter cannot
"pollute the failure list for every other PR run". It did the opposite.

It called `pytest.skip(..., allow_module_level=True)` from `pytest_collectstart`.
Raising Skipped from that hook is not handled as a per-file skip: pytest reports
INTERNALERROR and abandons the session with "no tests ran".

It also read a channel name out of any `test_<lowercase_words>.py` filename, so a
perfectly ordinary test file was looked up as a channel:

    $ cat > tests/channels/test_owner_pairing_rules.py <<'EOF'
    def test_something_harmless(): assert True
    EOF
    $ uv run pytest tests/channels -q
    INTERNALERROR> Skipped: channel adapter
      glc.channels.catalogue.owner_pairing_rules.adapter failed to import:
      ModuleNotFoundError(...)
    no tests ran in 0.22s

All 105 channel tests stopped running because one file had a plausible name. Any
contributor adding a non-channel test to that directory hits it, and the failure
names a module they never referenced.

These tests run pytest in a subprocess against a copy of the real conftest,
because the defect is in collection itself and cannot be observed from inside a
session the same conftest is already governing.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REAL_CONFTEST = REPO / "tests" / "channels" / "conftest.py"

PYTEST_INTERNAL_ERROR = 3


def _sandbox(tmp_path: Path, filename: str, body: str) -> Path:
    """A minimal tests/channels/ directory governed by the real conftest."""
    channels = tmp_path / "tests" / "channels"
    channels.mkdir(parents=True)
    shutil.copy(REAL_CONFTEST, channels / "conftest.py")
    (channels / filename).write_text(body, encoding="utf-8")
    return channels


def _run_pytest(target: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(target), "-q", "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=REPO, timeout=300,
    )


HARMLESS = "def test_something_harmless() -> None:\n    assert True\n"


def test_a_non_channel_test_file_still_collects(tmp_path: Path) -> None:
    """The exact reproduction: an ordinary name must not be read as a channel."""
    target = _sandbox(tmp_path, "test_owner_pairing_rules.py", HARMLESS)
    result = _run_pytest(target)
    output = result.stdout + result.stderr

    assert "INTERNALERROR" not in output, (
        "collection aborted the whole session for a file that is not a channel "
        f"test:\n{output[-1500:]}"
    )
    assert result.returncode != PYTEST_INTERNAL_ERROR
    assert "1 passed" in output, output[-1500:]


def test_a_real_channel_test_file_still_collects(tmp_path: Path) -> None:
    """A file naming a genuine channel whose adapter imports fine is untouched."""
    target = _sandbox(tmp_path, "test_telegram.py", HARMLESS)
    result = _run_pytest(target)
    output = result.stdout + result.stderr

    assert "INTERNALERROR" not in output, output[-1500:]
    assert "1 passed" in output, output[-1500:]


def test_the_hook_only_claims_names_that_exist_in_the_catalogue() -> None:
    """The guard that stops an ordinary filename being treated as a channel."""
    sys.path.insert(0, str(REPO / "tests" / "channels"))
    try:
        import conftest as channel_conftest
    finally:
        sys.path.pop(0)

    channels = channel_conftest._catalogue_channels()
    assert "telegram" in channels
    assert "imap" in channels
    assert "owner_pairing_rules" not in channels
    assert "send_failure_is_not_success" not in channels
