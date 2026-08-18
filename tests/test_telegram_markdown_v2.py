"""Telegram rejects the whole message if a reserved character is unescaped.

The adapter declares `parse_mode: MarkdownV2` and passed `reply.text` through
untouched. A full stop is reserved in MarkdownV2, so every ordinary sentence
came back as HTTP 400 "can't parse entities" and no reply was ever delivered.

Inbound worked perfectly throughout, which is what made this expensive to find:
the agent received, planned and answered correctly, and only the bridge's stdout
showed that delivery had failed.
"""

from __future__ import annotations

from glc.channels.catalogue.telegram.adapter import (
    escape_markdown_v2,
    markdown_to_telegram,
)


class TestEscaping:
    def test_a_sentence_ending_in_a_full_stop_is_escaped(self) -> None:
        assert escape_markdown_v2("Done.") == "Done\\."

    def test_every_reserved_character_is_escaped(self) -> None:
        for char in "_*[]()~`>#+-=|{}.!\\":
            assert escape_markdown_v2(char) == "\\" + char, char

    def test_plain_prose_survives_a_round_trip(self) -> None:
        rendered = markdown_to_telegram("Hello! How can I assist you today?")
        assert rendered == "Hello\\! How can I assist you today?"


class TestMarkdownTranslation:
    """The agent emits standard markdown; MarkdownV2 is a different dialect."""

    def test_bold_becomes_markdown_v2_bold_rather_than_literal_asterisks(self) -> None:
        # Standard markdown uses **bold**; MarkdownV2 uses *bold*.
        assert markdown_to_telegram("**urgent**") == "*urgent*"

    def test_headings_become_bold(self) -> None:
        assert markdown_to_telegram("### Notes") == "*Notes*"

    def test_bullets_are_normalised_and_escaped(self) -> None:
        assert markdown_to_telegram("* first") == "\\- first"

    def test_code_spans_are_preserved(self) -> None:
        rendered = markdown_to_telegram("Run `calculate` now.")
        assert "`calculate`" in rendered
        assert rendered.endswith("now\\.")


class TestUnbalancedMarkers:
    """A model regularly emits an unbalanced ** and the reader must not suffer."""

    def test_bold_does_not_span_lines(self) -> None:
        """If bold may cross newlines, a stray marker pairs with a later one,
        bolding everything between and leaving a literal ** behind. Observed on
        a real reply where '**Due This Week' paired with '**Source:**' three
        lines further down."""
        rendered = markdown_to_telegram("**Due This Week\nplain line\n**Source:** here")
        assert "*Source:*" in rendered
        # The stray opener must not have swallowed the intervening lines.
        assert "plain line" in rendered
        assert "**" not in rendered.replace("\\*", "")

    def test_an_unmatched_bold_marker_is_dropped_rather_than_shown(self) -> None:
        rendered = markdown_to_telegram("**Heading with no closing marker")
        assert "*" not in rendered.replace("\\*", "")
        assert "Heading with no closing marker" in rendered


class TestLinks:
    def test_a_local_file_link_collapses_to_its_label(self) -> None:
        """Telegram cannot open a file:// target, and rendering one exposes a
        local filesystem path in the chat."""
        rendered = markdown_to_telegram("[reminders.txt](file://C:/Users/someone/reminders.txt)")
        assert "reminders" in rendered
        assert "Users" not in rendered
        assert "file://" not in rendered

    def test_an_http_link_is_left_intact(self) -> None:
        rendered = markdown_to_telegram("[docs](https://example.com/a)")
        assert "https://example.com/a" in rendered
