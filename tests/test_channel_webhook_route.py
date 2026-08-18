"""`POST /v1/channels/{name}/webhook` is reachable by anyone who knows the URL.

Every registered adapter is dispatched to from this one route, and each parses
the payload differently. Three things follow, and none of them were covered:

  * `line` reads `raw["events"][0]` and raised `KeyError` out of an
    unauthenticated POST, so the route answered 500.
  * Adapters that do not understand the payload shape mostly do not fail. They
    return an envelope with an empty `channel_user_id`, which flowed on to the
    allowlist and the audit log as though a real message had arrived.
  * `twilio_voice` ships `authenticate_webhook`, documented as "the entry point
    for the deployment HTTP layer", with unit tests including a fail-closed
    case. Nothing in the request path called it. The tests proved the
    algorithm; nothing proved the endpoint was ever protected by it.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from urllib.parse import urlencode


def _twilio_signature(token: str, url: str, form: dict[str, str]) -> str:
    data = url + "".join(f"{k}{form[k]}" for k in sorted(form))
    return base64.b64encode(
        hmac.new(token.encode(), data.encode(), hashlib.sha1).digest()
    ).decode()


FORM = {"CallSid": "CA1", "From": "+15550001111", "To": "+15550002222"}
# urlencode, not manual joining: a bare "+" in a form body decodes to a space,
# so a hand-built body signs different values than the server parses.
BODY = urlencode(FORM)
HEADERS = {"content-type": "application/x-www-form-urlencoded"}


def test_an_unparseable_webhook_is_a_bad_request_not_a_500(app_client):
    """An anonymous POST must not be able to raise inside an adapter."""
    result = app_client.post(
        "/v1/channels/line/webhook", content="not a line event", headers=HEADERS
    )
    assert result.status_code != 500, (
        "an unauthenticated caller who knows the URL could turn this route "
        "into 500s just by posting a body the adapter does not expect"
    )
    assert result.status_code == 400


def test_a_forged_twilio_signature_is_refused(app_client, monkeypatch):
    """The verifier exists and is tested; the route has to actually call it."""
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "the-real-auth-token")

    result = app_client.post(
        "/v1/channels/twilio_voice/webhook",
        content=BODY,
        headers={**HEADERS, "X-Twilio-Signature": "not-the-real-signature"},
    )
    assert result.status_code == 403, (
        "twilio_voice ships authenticate_webhook precisely so a forged webhook "
        "cannot reach on_message; before the fix this returned 200"
    )


def test_a_genuine_twilio_signature_is_not_rejected(app_client, monkeypatch):
    """The check must accept what Twilio would really send."""
    token = "the-real-auth-token"
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", token)
    url = "http://testserver/v1/channels/twilio_voice/webhook"

    result = app_client.post(
        "/v1/channels/twilio_voice/webhook",
        content=BODY,
        headers={**HEADERS, "X-Twilio-Signature": _twilio_signature(token, url, FORM)},
    )
    assert result.status_code != 403, "a correctly signed webhook must pass"


def test_signature_verification_fails_closed_with_no_token(app_client, monkeypatch):
    """No configured token means unverifiable, which is not the same as fine."""
    monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)

    result = app_client.post(
        "/v1/channels/twilio_voice/webhook",
        content=BODY,
        headers={**HEADERS, "X-Twilio-Signature": "anything"},
    )
    assert result.status_code == 403


def test_an_envelope_with_no_sender_is_refused(app_client):
    """A message nobody can be held to is not a message.

    twilio_sms does not raise on the generic payload -- it returns an envelope
    with an empty channel_user_id, which then reached the allowlist and the
    audit log looking like real traffic from an unknown sender. (matrix and
    signal return None here, which is already the correct answer.)
    """
    result = app_client.post(
        "/v1/channels/twilio_sms/webhook", content=BODY, headers=HEADERS
    )
    assert result.status_code == 400, (
        "twilio_sms produced an envelope with no sender and the route took it"
    )


def test_every_refusal_leaves_a_trace(app_client, monkeypatch):
    """A control that prevents work must not also be the least observable thing."""
    from glc.audit import store as audit

    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "the-real-auth-token")
    app_client.post("/v1/channels/line/webhook", content="junk", headers=HEADERS)
    app_client.post(
        "/v1/channels/twilio_voice/webhook",
        content=BODY,
        headers={**HEADERS, "X-Twilio-Signature": "forged"},
    )

    recorded = {row["event_type"] for row in audit.query(limit=50)}
    assert "webhook_unparseable" in recorded
    assert "webhook_signature_rejected" in recorded
