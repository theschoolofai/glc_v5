"""A provider's failure text is read by code, not only by people.

``_backoff_for`` decides how long to bench a key by reading the string a
``ProviderError`` carries. So the cap applied when that string is built is not a
display choice — it decides which facts the router is allowed to see.

At 400 characters, a Gemini 429 arrived like this (measured against a live
gateway, the whole of what the classifier received):

    gemini HTTP 429: {
      "error": {
        "code": 429,
        "message": "You exceeded your current quota, please check your plan and
        billing details. For more information on this error, head to:
        https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your
        current usage, head to: https://ai.dev/rate-limit.
        * Quota exceeded for metric: ...generate_content_free_tier_requests,
        limit: 500

It ends there. Mid-string, unparseable, with the prose and both documentation
URLs intact and every machine-readable fact dropped: the ``QuotaFailure`` naming
which window was exhausted, and the ``RetryInfo`` saying how long to wait.
"""
from __future__ import annotations

import json

from glc.providers import ERROR_BODY_CHARS

#: A RESOURCE_EXHAUSTED body in Google's documented shape. The message and URLs
#: alone are ~380 characters, which is why a 400-character cap keeps exactly the
#: half that carries no information.
GOOGLE_429 = json.dumps({
    "error": {
        "code": 429,
        "message": (
            "You exceeded your current quota, please check your plan and billing details. "
            "For more information on this error, head to: "
            "https://ai.google.dev/gemini-api/docs/rate-limits. "
            "To monitor your current usage, head to: https://ai.dev/rate-limit. \n"
            "* Quota exceeded for metric: "
            "generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 500"
        ),
        "status": "RESOURCE_EXHAUSTED",
        "details": [
            {
                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [{
                    "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                    "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                    "quotaDimensions": {"model": "gemini-3.1-flash-lite", "location": "global"},
                    "quotaValue": "500",
                }],
            },
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "26s"},
        ],
    }
})


def test_the_quota_window_survives_the_cap():
    """Which window was exhausted decides a 60-second bench or an hour-long one."""
    kept = GOOGLE_429[:ERROR_BODY_CHARS]
    assert "PerDay" in kept, (
        "the cap drops the only token naming the window, so a daily exhaustion "
        "cannot be told from a per-minute burst"
    )


def test_the_retry_delay_survives_the_cap():
    """The provider states how long to wait. Keeping it beats inferring it."""
    assert "retryDelay" in GOOGLE_429[:ERROR_BODY_CHARS]


def test_the_kept_body_is_still_parseable():
    """A body cut mid-string cannot be read as anything but a blob of prose."""
    json.loads(GOOGLE_429[:ERROR_BODY_CHARS])


def test_the_old_cap_dropped_all_of_it():
    """What this is fixing, held still so the regression is legible."""
    old = GOOGLE_429[:400]
    assert "PerDay" not in old
    assert "retryDelay" not in old
    try:
        json.loads(old)
        raise AssertionError("400 characters should not have been valid JSON")
    except json.JSONDecodeError:
        pass


def test_the_cap_still_bounds_a_hostile_body():
    """Generous, not unbounded — an upstream returning a megabyte is still cut."""
    assert len(("x" * 50_000)[:ERROR_BODY_CHARS]) == ERROR_BODY_CHARS
    assert ERROR_BODY_CHARS <= 4_000
