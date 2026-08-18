"""A request that asks for a schema must get one, or be told it cannot.

`response_format` has one failure mode that costs a caller far more than a 4xx:
asking for strict structured output, getting prose, and receiving HTTP 200 with
no indication that anything was dropped. The caller then parses the prose as
JSON and fails somewhere else entirely.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from glc.llm_schemas import ChatRequest, ResponseFormat
from glc.providers import OpenAICompatProvider

SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}


def _body_for(response_format) -> dict:
    """What the provider would actually put on the wire."""
    body: dict = {}
    OpenAICompatProvider._apply_response_format(None, body, response_format)  # noqa: SLF001
    return body


# ── the shape that is silently dropped ───────────────────────────────────────


def test_json_schema_without_a_schema_is_refused() -> None:
    """`strict: true` and nothing to be strict about is a contradiction.

    Nothing rejected this. `_apply_response_format` runs neither branch, so the
    provider is sent no `response_format` at all; `_validate_structured` is
    guarded on `schema_` so the output is never checked either. The one thing
    that DOES still happen is `_required_caps` adding "structured", so the
    request is routed to a structured-capable provider and looks, from the
    outside, exactly like a request that worked.
    """
    with pytest.raises(ValidationError, match="schema"):
        ResponseFormat(type="json_schema", strict=True)


def test_the_openai_nested_shape_is_understood() -> None:
    """The most likely way to hit this is to send the shape OpenAI documents.

        {"type": "json_schema", "json_schema": {"name": ..., "schema": {...}}}

    This gateway takes `schema` and `name` flat. `ResponseFormat` did not forbid
    extra fields, so the nested `json_schema` key was accepted and discarded,
    `schema_` stayed None, and the caller got prose. Anyone porting a call from
    OpenAI lands here, and nothing tells them.
    """
    rf = ResponseFormat.model_validate({
        "type": "json_schema",
        "json_schema": {"name": "Person", "schema": SCHEMA, "strict": True},
    })
    assert rf.schema_ == SCHEMA
    assert rf.name == "Person"
    assert _body_for(rf)["response_format"]["json_schema"]["schema"] == SCHEMA


def test_a_request_carrying_the_nested_shape_reaches_the_provider() -> None:
    """End to end through ChatRequest, which is what a caller actually sends."""
    req = ChatRequest(
        messages=[{"role": "user", "content": "one person"}],
        response_format={"type": "json_schema",
                         "json_schema": {"name": "Person", "schema": SCHEMA}},
    )
    assert req.response_format is not None
    assert req.response_format.schema_ == SCHEMA
    assert "response_format" in _body_for(req.response_format)


# ── the shapes that already worked, and must keep working ────────────────────


def test_the_flat_shape_is_unchanged() -> None:
    rf = ResponseFormat(type="json_schema", name="out", schema=SCHEMA)
    assert rf.schema_ == SCHEMA
    assert _body_for(rf)["response_format"]["json_schema"]["schema"] == SCHEMA


def test_json_object_needs_no_schema() -> None:
    """"Give me JSON, I will not tell you which JSON" is a legitimate request."""
    rf = ResponseFormat(type="json_object")
    assert rf.schema_ is None
    assert _body_for(rf)["response_format"] == {"type": "json_object"}


def test_no_response_format_puts_nothing_on_the_wire() -> None:
    assert _body_for(None) == {}
