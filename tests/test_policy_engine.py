"""Policy engine — rule ordering, deny-wins, condition matching,
defaults, hot reload, malformed-yaml safe default."""

from __future__ import annotations

import signal
from textwrap import dedent

import pytest

from glc.policy.engine import PolicyEngine
from glc.policy.schemas import PolicyConfig, PolicyRule


def _engine(rules):
    return PolicyEngine(PolicyConfig(rules=rules))


def test_first_match_wins():
    eng = _engine(
        [
            PolicyRule(tool="email.send", action="allow", reason="first"),
            PolicyRule(tool="email.send", action="deny", reason="second"),
        ]
    )
    v = eng.evaluate({"name": "email.send", "arguments": {}}, {"channel": "x", "trust_level": "owner_paired"})
    # Both match; deny-on-tie wins.
    assert v.action == "deny"


def test_first_match_when_no_deny():
    eng = _engine(
        [
            PolicyRule(tool="email.send", action="require_approval", reason="A"),
            PolicyRule(tool="email.send", action="allow", reason="B"),
        ]
    )
    v = eng.evaluate({"name": "email.send", "arguments": {}}, {"channel": "x", "trust_level": "owner_paired"})
    assert v.action == "require_approval"
    assert v.reason == "A"


def test_default_allow_for_owner_paired():
    eng = _engine([])
    v = eng.evaluate({"name": "anything", "arguments": {}}, {"channel": "x", "trust_level": "owner_paired"})
    assert v.action == "allow"


def test_default_deny_for_untrusted():
    eng = _engine([])
    v = eng.evaluate({"name": "anything", "arguments": {}}, {"channel": "x", "trust_level": "untrusted"})
    assert v.action == "deny"


def test_glob_condition():
    eng = _engine(
        [
            PolicyRule(
                tool="file.delete",
                condition={"path_glob": "~/Documents/**"},
                action="deny",
                reason="docs are protected",
            ),
        ]
    )
    v = eng.evaluate(
        {"name": "file.delete", "arguments": {"path": "~/Documents/secrets/keys.txt"}},
        {"channel": "x", "trust_level": "owner_paired"},
    )
    assert v.action == "deny"


def test_glob_does_not_match_other_paths():
    eng = _engine(
        [
            PolicyRule(tool="file.delete", condition={"path_glob": "~/Documents/**"}, action="deny"),
        ]
    )
    v = eng.evaluate(
        {"name": "file.delete", "arguments": {"path": "/tmp/junk.txt"}},
        {"channel": "x", "trust_level": "owner_paired"},
    )
    assert v.action == "allow"


def test_command_matches_list():
    eng = _engine(
        [
            PolicyRule(tool="shell.exec", condition={"command_matches": ["sudo", "rm -rf"]}, action="deny"),
        ]
    )
    v = eng.evaluate(
        {"name": "shell.exec", "arguments": {"command": "sudo apt install"}},
        {"channel": "x", "trust_level": "owner_paired"},
    )
    assert v.action == "deny"


def test_untrusted_wildcard_deny():
    eng = _engine(
        [
            PolicyRule(tool="*", trust_level="untrusted", action="deny", reason="no untrusted"),
        ]
    )
    v = eng.evaluate({"name": "anything", "arguments": {}}, {"channel": "x", "trust_level": "untrusted"})
    assert v.action == "deny"


def test_recipient_type_external_requires_approval():
    eng = _engine(
        [
            PolicyRule(
                tool="email.send", condition={"recipient_type": "external"}, action="require_approval"
            ),
        ]
    )
    v = eng.evaluate(
        {"name": "email.send", "arguments": {"recipient_type": "external"}},
        {"channel": "x", "trust_level": "owner_paired"},
    )
    assert v.action == "require_approval"


def test_malformed_yaml_falls_back_to_deny(tmp_path):
    bad = tmp_path / "policy.yaml"
    bad.write_text("rules: not-a-list\n  this is broken: [")
    eng = PolicyEngine.from_yaml(bad)
    v = eng.evaluate({"name": "anything", "arguments": {}}, {"channel": "x", "trust_level": "owner_paired"})
    assert v.action == "deny"


def test_default_policy_yaml_loads_and_matches_lecture():
    """The five rules from §10 of the lecture must load and behave as
    documented."""
    from glc.config import PACKAGED_POLICY

    eng = PolicyEngine.from_yaml(PACKAGED_POLICY)
    deny = eng.evaluate(
        {"name": "file.delete", "arguments": {"path": "~/Documents/x.txt"}},
        {"channel": "telegram", "trust_level": "owner_paired"},
    )
    assert deny.action == "deny"
    approve = eng.evaluate(
        {"name": "email.send", "arguments": {"recipient_type": "external"}},
        {"channel": "slack", "trust_level": "owner_paired"},
    )
    assert approve.action == "require_approval"
    deny_shell = eng.evaluate(
        {"name": "shell.exec", "arguments": {"command": "sudo apt install x"}},
        {"channel": "x", "trust_level": "owner_paired"},
    )
    assert deny_shell.action == "deny"
    deny_untrusted = eng.evaluate(
        {"name": "anything", "arguments": {}},
        {"channel": "x", "trust_level": "untrusted"},
    )
    assert deny_untrusted.action == "deny"


def test_reload_picks_up_new_rules(tmp_path):
    p = tmp_path / "policy.yaml"
    p.write_text(
        dedent("""\
        rules:
          - tool: tool.x
            action: allow
            reason: original
    """)
    )
    eng = PolicyEngine.from_yaml(p)
    v = eng.evaluate({"name": "tool.x", "arguments": {}}, {"channel": "x", "trust_level": "owner_paired"})
    assert v.action == "allow"

    p.write_text(
        dedent("""\
        rules:
          - tool: tool.x
            action: deny
            reason: rewritten
    """)
    )
    eng.reload(p)
    v2 = eng.evaluate({"name": "tool.x", "arguments": {}}, {"channel": "x", "trust_level": "owner_paired"})
    assert v2.action == "deny"
    assert v2.reason == "rewritten"


@pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="SIGHUP only on POSIX")
def test_module_singleton_reloads_via_hook(tmp_path, monkeypatch):
    """The module-level reload_engine() hook is what main.py wires to
    SIGHUP. Verify it picks up file changes."""
    from glc import config

    p = config.CONFIG_DIR / "policy.yaml"
    p.write_text("rules:\n  - {tool: ping, action: allow, reason: v1}\n")
    from glc.policy import engine as eng_mod

    eng_mod._engine = None
    v = eng_mod.evaluate({"name": "ping", "arguments": {}}, {"channel": "x", "trust_level": "owner_paired"})
    assert v.action == "allow"
    p.write_text("rules:\n  - {tool: ping, action: deny, reason: v2}\n")
    eng_mod.reload_engine()
    v2 = eng_mod.evaluate({"name": "ping", "arguments": {}}, {"channel": "x", "trust_level": "owner_paired"})
    assert v2.action == "deny"


# --- ** patterns must stay fnmatch --------------------------------------------
#
# policy.yaml documents condition.path_glob as an fnmatch glob. The `**` branch
# of _matches_glob leaves fnmatch and hand-rolls a regex that only ever unescapes
# `*`, so `?` and `[...]` survive re.escape as literals and the rule matches
# nothing. A deny rule that matches nothing is not inert: evaluate() then falls
# through to default-allow for owner_paired, and the call is permitted.


def test_a_character_class_deny_rule_is_not_silently_bypassed():
    """A deny rule that matches nothing is not a no-op. It is an allow."""
    eng = _engine(
        [
            PolicyRule(
                tool="file.delete",
                condition={"path_glob": "~/[Dd]ocuments/**"},
                action="deny",
                reason="docs are protected",
            ),
        ]
    )
    v = eng.evaluate(
        {"name": "file.delete", "arguments": {"path": "~/Documents/secrets/keys.txt"}},
        {"channel": "x", "trust_level": "owner_paired"},
    )
    assert v.action == "deny"


def test_a_question_mark_deny_rule_is_not_silently_bypassed():
    eng = _engine(
        [
            PolicyRule(
                tool="file.read",
                condition={"path_glob": "~/.ssh/id_?sa**"},
                action="deny",
                reason="private keys are protected",
            ),
        ]
    )
    v = eng.evaluate(
        {"name": "file.read", "arguments": {"path": "~/.ssh/id_rsa"}},
        {"channel": "x", "trust_level": "owner_paired"},
    )
    assert v.action == "deny"


@pytest.mark.parametrize(
    "path,expected",
    [
        ("~/Documents/a.txt", True),
        ("~/documents/a.txt", True),
        ("~/Downloads/a.txt", False),
    ],
)
def test_the_two_star_forms_agree_about_a_character_class(path: str, expected: bool):
    """`*` and `**` differ only in whether they cross a separator. Every other
    metacharacter must mean the same thing in both."""
    from glc.policy.engine import _matches_glob

    assert _matches_glob(path, "~/[Dd]ocuments/*") is expected
    assert _matches_glob(path, "~/[Dd]ocuments/**") is expected


def test_the_two_star_forms_agree_about_a_newline_in_the_value():
    r"""`fnmatch.translate` wraps its output in `(?s:...)`, so `*` spans a
    newline. The hand-rolled `**` regex has no DOTALL, so it does not. A value
    is attacker-shaped: a path argument carrying a newline must not decide
    which half of this function it is judged by."""
    from glc.policy.engine import _matches_glob

    embedded = "/etc/pa" + chr(10) + "sswd"
    assert _matches_glob(embedded, "/etc/*") is True
    assert _matches_glob(embedded, "/etc/**") is True
