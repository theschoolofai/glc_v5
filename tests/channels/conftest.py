"""Defensive test collection for channel adapters.

A group's adapter PR may include a syntax error (work-in-progress) or
fail to import for some other reason. Without this hook, a single bad
adapter import error pollutes the failure list for every other PR run
that pulls main.

The hook tries to import each test file's target adapter at collection
time. If the import errors, the matching test file is skipped with a
clear message naming the broken module — the rest of the suite
collects normally.

This only kicks in for a test file named `tests/channels/test_<channel>.py`
whose `<channel>` is an actual package under `glc/channels/catalogue/`.
Filenames that merely look like that pattern are left alone.
"""

from __future__ import annotations

import importlib
import re
import warnings
from functools import lru_cache
from pathlib import Path

# Digits are part of channel and test names alike (twilio_sms, markdownv2), so the
# pattern admits them and the catalogue decides what is really a channel.
_TEST_FILE_RE = re.compile(r"^test_(?P<channel>[a-z0-9_]+)\.py$")


@lru_cache(maxsize=1)
def _catalogue_channels() -> frozenset[str]:
    """Channel names that actually exist, read from the catalogue on disk.

    Without this, any `test_<lowercase_words>.py` in this directory was assumed
    to name a channel: `test_owner_pairing_rules.py` was looked up as a channel
    called `owner_pairing_rules`.
    """
    catalogue = Path(__file__).resolve().parents[2] / "glc" / "channels" / "catalogue"
    if not catalogue.is_dir():
        return frozenset()
    return frozenset(
        entry.name for entry in catalogue.iterdir()
        if entry.is_dir() and (entry / "adapter.py").is_file()
    )


def pytest_ignore_collect(collection_path, config):  # pragma: no cover - pytest hook
    """Ignore one unimportable adapter's test file, and nothing else.

    This was previously done from `pytest_collectstart` with
    `pytest.skip(allow_module_level=True)`. Raising Skipped from that hook is not
    caught as a per-file skip: pytest reports INTERNALERROR and abandons the whole
    session with "no tests ran". The mechanism meant to stop one broken adapter
    from polluting the run was instead the thing that took the run down, and any
    plausibly named non-channel test file triggered it.

    `pytest_ignore_collect` is the hook that deselects a single path, so the rest
    of the suite collects normally, which is what the docstring above promises.
    """
    match = _TEST_FILE_RE.match(Path(collection_path).name)
    if not match:
        return None
    channel = match.group("channel")
    if channel not in _catalogue_channels():
        return None  # an ordinary test file that is not about one channel
    try:
        importlib.import_module(f"glc.channels.catalogue.{channel}.adapter")
    except Exception as error:
        warnings.warn(
            f"skipping {Path(collection_path).name}: channel adapter "
            f"glc.channels.catalogue.{channel}.adapter failed to import: {error!r}",
            stacklevel=1,
        )
        return True
    return None
