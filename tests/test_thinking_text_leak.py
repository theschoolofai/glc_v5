"""Thinking / CoT must not leak into the normalised assistant text field."""
from __future__ import annotations

from glc.providers import _assistant_visible_text, _gemini_visible_text, _strip_think_markup


def test_strip_think_markup_removes_qwen_style_blocks():
    raw = "<think>plan the JSON</think>\n{\"ok\": true}"
    assert _strip_think_markup(raw) == '{"ok": true}'


def test_assistant_visible_text_ignores_reasoning_content_field():
    msg = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "I will emit JSON next: {\"a\":1}",
    }
    assert _assistant_visible_text(msg) == ""


def test_assistant_visible_text_keeps_answer_after_think_tags():
    msg = {
        "role": "assistant",
        "content": "<think>scratch</think>\n{\"tasks\":[]}",
    }
    assert _assistant_visible_text(msg) == '{"tasks":[]}'


def test_gemini_visible_text_skips_thought_parts():
    parts = [
        {"text": "hidden chain of thought", "thought": True},
        {"text": "{\"answer\":1}"},
    ]
    assert _gemini_visible_text(parts) == '{"answer":1}'
