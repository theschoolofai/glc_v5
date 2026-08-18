"""install_token is the sole bootstrap secret for the whole control plane —
every WS adapter connection and every /v1/control/* request authenticates
against it (glc/config.py:get_or_create_install_token()). Unlike
channel_secrets.json (see test_channel_setup.py), no test previously
checked that the file it's written to is actually restricted to its owner.
"""

from __future__ import annotations

from tests._acl_assertions import assert_restricted_to_owner


def test_install_token_is_restricted_to_owner(install_token):
    from glc.config import install_token_path

    path = install_token_path()
    assert path.exists()
    assert_restricted_to_owner(path)
