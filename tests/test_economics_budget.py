"""The hard controller: admission before the call, refusal on breach, and the
denial-of-wallet invariant (a runaway loop is stopped, not merely reported)."""

from __future__ import annotations

import pytest

from glc import db
from glc.economics import budget as B
from glc.economics import meter as M
from glc.economics import pricing as P


@pytest.fixture(autouse=True)
def _fresh():
    db.init()
    B.init_store()
    P.reload_pricing()
    M.reset_meter()
    B.reset_controller()
    yield
    B.reset_controller()


def _ctl(tmp_path, monkeypatch, yaml_text: str) -> B.BudgetController:
    p = tmp_path / "budgets.yaml"
    p.write_text(yaml_text)
    monkeypatch.setenv("GLC_BUDGETS_YAML", str(p))
    return B.reload_controller()


def _spend(usd_input_mtok: float, reservation_id: int | None = None, **principal_kw):
    """Burn a known number of dollars through the real meter.

    Pass `reservation_id` (an `Admission.reservation_id`) when this spend is
    settling a call `admit()` already reserved a hold for — otherwise the
    hold stays open *and* this write lands, double-counting the same call.
    """
    M.get_meter().record(
        provider="gemini_1",
        model="gemini-3.1-flash-lite",  # $0.25 in / $1.50 out per Mtok
        principal=M.Principal(**principal_kw),
        usage=M.Usage(input_tokens=int(usd_input_mtok / 0.25 * 1_000_000)),
        status="ok",
        reservation_id=reservation_id,
    )


# ── shipped default: nothing is refused ─────────────────────────────────────


def test_default_config_refuses_nothing():
    """A fresh glc_v4 must behave exactly like glc_v3."""
    ctl = B.get_controller()
    assert ctl.policies() == []
    a = ctl.admit(M.Principal(agent="anything", session="s"), projected_usd=1_000_000.0)
    assert a.allowed is True
    assert "no budget policy matches" in a.reason


# ── the headline: refusal at the boundary ───────────────────────────────────


def test_admission_refuses_when_the_projection_would_breach(tmp_path, monkeypatch):
    ctl = _ctl(
        tmp_path,
        monkeypatch,
        "version: 1\npolicies:\n  - principal: 'agent:tightwad'\n    limit_usd: 0.10\n    period: lifetime\n",
    )
    p = M.Principal(agent="tightwad")

    ok = ctl.admit(p, projected_usd=0.05)
    assert ok.allowed is True

    bad = ctl.admit(p, projected_usd=0.5)
    assert bad.allowed is False
    assert bad.breached.principal == "agent:tightwad"
    env = bad.envelope()
    assert env["code"] == B.BUDGET_EXCEEDED
    assert env["limit_usd"] == 0.10
    assert env["projected_usd"] == 0.5
    # `ok`'s $0.05 is a reserved hold now, not merely a refusal that wrote
    # nothing — it counts against `bad`'s check the same way a completed call
    # would, so remaining is $0.05 and the shortfall is 0.5 - 0.05.
    assert env["shortfall_usd"] == pytest.approx(0.45)
    assert "nothing was billed" in env["hint"]


def test_spend_accumulates_and_then_the_gate_closes(tmp_path, monkeypatch):
    ctl = _ctl(
        tmp_path,
        monkeypatch,
        "version: 1\npolicies:\n  - principal: 'session:*'\n    limit_usd: 0.10\n    period: lifetime\n",
    )
    p = M.Principal(session="run-1")
    first = ctl.admit(p, 0.04)
    assert first.allowed is True
    _spend(0.04, session="run-1", reservation_id=first.reservation_id)
    assert ctl.status_for("session", "run-1").spent_usd == pytest.approx(0.04)
    second = ctl.admit(p, 0.04)
    assert second.allowed is True
    _spend(0.04, session="run-1", reservation_id=second.reservation_id)
    # 0.08 spent, 0.02 left: a 0.04 call no longer fits.
    assert ctl.admit(p, 0.04).allowed is False
    # but a small one still does
    assert ctl.admit(p, 0.01).allowed is True


def test_denial_of_wallet_loop_is_stopped_by_the_controller(tmp_path, monkeypatch):
    """S12 invariant 8, made live: an unbounded loop cannot outspend its ceiling.

    The loop never voluntarily stops. What stops it is `admit()` returning
    False, and the total spend stays under the ceiling.
    """
    ctl = _ctl(
        tmp_path,
        monkeypatch,
        "version: 1\npolicies:\n  - principal: 'session:runaway'\n    limit_usd: 0.05\n    period: lifetime\n",
    )
    p = M.Principal(session="runaway")
    admitted = refused = 0
    for _ in range(500):  # an adversary with no intention of stopping
        a = ctl.admit(p, 0.01)
        if a.allowed:
            admitted += 1
            _spend(0.01, session="runaway", reservation_id=a.reservation_id)
        else:
            refused += 1
    total = db.spend_usd("session", "runaway", since=0)
    assert admitted == 5, f"expected exactly 5 admitted calls, got {admitted}"
    assert refused == 495
    assert total <= 0.05 + 1e-9, f"budget breached: spent {total}"


def test_zero_limit_refuses_immediately(tmp_path, monkeypatch):
    """The kill-switch: POST a limit of 0 and the next call cannot happen."""
    ctl = _ctl(
        tmp_path,
        monkeypatch,
        "version: 1\npolicies:\n  - principal: 'agent:stopped'\n    limit_usd: 0\n    period: lifetime\n",
    )
    a = ctl.admit(M.Principal(agent="stopped"), projected_usd=0.000001)
    assert a.allowed is False


# ── matching ────────────────────────────────────────────────────────────────


def test_exact_principal_beats_a_glob(tmp_path, monkeypatch):
    ctl = _ctl(
        tmp_path,
        monkeypatch,
        "version: 1\npolicies:\n"
        "  - principal: 'agent:*'\n    limit_usd: 10.0\n    period: lifetime\n"
        "  - principal: 'agent:special'\n    limit_usd: 0.01\n    period: lifetime\n",
    )
    assert ctl.policy_for("agent", "ordinary").limit_usd == 10.0
    assert ctl.policy_for("agent", "special").limit_usd == 0.01


def test_glob_means_per_principal_not_pooled(tmp_path, monkeypatch):
    ctl = _ctl(
        tmp_path,
        monkeypatch,
        "version: 1\npolicies:\n  - principal: 'user:*'\n    limit_usd: 0.05\n    period: lifetime\n",
    )
    _spend(0.05, user="alice")
    assert ctl.admit(M.Principal(user="alice"), 0.01).allowed is False
    # bob's allowance is his own
    assert ctl.admit(M.Principal(user="bob"), 0.01).allowed is True


def test_the_tightest_of_several_dimensions_binds(tmp_path, monkeypatch):
    ctl = _ctl(
        tmp_path,
        monkeypatch,
        "version: 1\npolicies:\n"
        "  - principal: 'tenant:*'\n    limit_usd: 100.0\n    period: lifetime\n"
        "  - principal: 'session:*'\n    limit_usd: 0.02\n    period: lifetime\n",
    )
    p = M.Principal(tenant="acme", session="s1")
    a = ctl.admit(p, 0.5)
    assert a.allowed is False
    assert a.breached.dimension == "session"
    assert {s.dimension for s in a.checked} == {"tenant", "session"}


def test_dimensions_the_caller_omitted_are_not_checked(tmp_path, monkeypatch):
    ctl = _ctl(
        tmp_path,
        monkeypatch,
        "version: 1\npolicies:\n  - principal: 'tenant:*'\n    limit_usd: 0.0\n    period: lifetime\n",
    )
    # No tenant supplied -> the tenant ceiling cannot apply.
    assert ctl.admit(M.Principal(agent="a"), 1.0).allowed is True
    assert ctl.admit(M.Principal(tenant="t"), 1.0).allowed is False


def test_unmatched_deny_flips_the_default(tmp_path, monkeypatch):
    ctl = _ctl(
        tmp_path,
        monkeypatch,
        "version: 1\ndefaults:\n  unmatched: deny\npolicies: []\n",
    )
    assert ctl.admit(M.Principal(agent="a"), 0.0).allowed is False


def test_master_switch_off_disables_enforcement(tmp_path, monkeypatch):
    ctl = _ctl(
        tmp_path,
        monkeypatch,
        "version: 1\ndefaults:\n  enabled: false\npolicies:\n"
        "  - principal: 'agent:x'\n    limit_usd: 0\n    period: lifetime\n",
    )
    a = ctl.admit(M.Principal(agent="x"), 99.0)
    assert a.allowed is True
    assert a.enforced is False


# ── projection ──────────────────────────────────────────────────────────────


def test_projection_is_worst_case_by_default(tmp_path, monkeypatch):
    ctl = _ctl(tmp_path, monkeypatch, "version: 1\npolicies: []\n")
    # 1M in, 1M max_tokens on flash-lite: 0.25 + 1.50
    assert ctl.project("gemini_1", "gemini-3.1-flash-lite", 1_000_000, 1_000_000) == pytest.approx(1.75)


def test_projection_mode_and_safety_factor_are_config(tmp_path, monkeypatch):
    ctl = _ctl(
        tmp_path,
        monkeypatch,
        "version: 1\nprojection:\n  output_tokens: ratio\n  output_ratio: 0.5\n  safety_factor: 2.0\n"
        "policies: []\n",
    )
    # 1M in, output assumed 0.5M, then doubled: (0.25 + 0.75) * 2
    assert ctl.project("gemini_1", "gemini-3.1-flash-lite", 1_000_000, 999_999) == pytest.approx(2.0)


def test_min_usd_makes_free_models_still_trip_a_ceiling(tmp_path, monkeypatch):
    ctl = _ctl(
        tmp_path,
        monkeypatch,
        "version: 1\nprojection:\n  min_usd: 0.001\npolicies:\n"
        "  - principal: 'session:s'\n    limit_usd: 0.0005\n    period: lifetime\n",
    )
    # nvidia free tier prices at $0, so without min_usd nothing would ever refuse
    assert ctl.project("nvidia", "deepseek-ai/deepseek-v3.2", 10_000, 1_000) == pytest.approx(0.001)
    assert ctl.admit(M.Principal(session="s"), ctl.project("nvidia", "x", 10, 10)).allowed is False


# ── periods ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("period", ["minute", "hour", "day", "month", "lifetime"])
def test_every_documented_period_loads(tmp_path, monkeypatch, period):
    ctl = _ctl(
        tmp_path,
        monkeypatch,
        f"version: 1\npolicies:\n  - principal: 'agent:a'\n    limit_usd: 1.0\n    period: {period}\n",
    )
    st = ctl.status_for("agent", "a")
    assert st.period == period
    assert st.period_start >= 0


def test_period_window_excludes_older_spend(tmp_path, monkeypatch):
    ctl = _ctl(
        tmp_path,
        monkeypatch,
        "version: 1\npolicies:\n  - principal: 'agent:a'\n    limit_usd: 1.0\n    period: minute\n",
    )
    # write a row an hour in the past directly
    with db.conn() as c:
        c.execute(
            "INSERT INTO calls (ts, provider, model, agent, usd, status) VALUES (?,?,?,?,?,?)",
            (db.time.time() - 3600, "gemini_1", "m", "a", 5.0, "ok"),
        )
    assert db.spend_usd("agent", "a", since=0) == pytest.approx(5.0)
    assert ctl.status_for("agent", "a").spent_usd == 0.0  # outside the minute window


# ── runtime overrides (POST /v1/budget) ─────────────────────────────────────


def test_runtime_limit_shadows_the_file(tmp_path, monkeypatch):
    ctl = _ctl(
        tmp_path,
        monkeypatch,
        "version: 1\npolicies:\n  - principal: 'agent:a'\n    limit_usd: 10.0\n    period: lifetime\n",
    )
    assert ctl.policy_for("agent", "a").limit_usd == 10.0
    ctl.set_limit("agent:a", 0.01, period="lifetime")
    pol = ctl.policy_for("agent", "a")
    assert pol.limit_usd == 0.01
    assert pol.source == "runtime"
    assert ctl.admit(M.Principal(agent="a"), 0.02).allowed is False
    # clearing it restores the file policy
    assert ctl.clear_limit("agent:a") is True
    assert ctl.policy_for("agent", "a").limit_usd == 10.0


def test_runtime_limits_survive_a_controller_reload(tmp_path, monkeypatch):
    _ctl(tmp_path, monkeypatch, "version: 1\npolicies: []\n").set_limit(
        "session:persisted", 0.02, period="lifetime"
    )
    fresh = B.reload_controller()
    assert fresh.policy_for("session", "persisted").limit_usd == 0.02


# ── config errors are loud ──────────────────────────────────────────────────


def test_a_typo_raises_rather_than_meaning_no_budget(tmp_path, monkeypatch):
    for bad in (
        "version: 1\npolicies:\n  - principal: 'nonsense:x'\n    limit_usd: 1.0\n",
        "version: 1\npolicies:\n  - principal: 'agent:a'\n",
        "version: 1\npolicies:\n  - principal: 'agent:a'\n    limit_usd: 'lots'\n",
        "version: 1\npolicies:\n  - principal: 'agent:a'\n    limit_usd: 1.0\n    period: fortnight\n",
        "version: 1\npolicies:\n  - principal: 'agent:a'\n    limit_usd: -5\n",
        "version: 1\npolicies: {}\n",
    ):
        p = tmp_path / "bad.yaml"
        p.write_text(bad)
        monkeypatch.setenv("GLC_BUDGETS_YAML", str(p))
        with pytest.raises(B.BudgetConfigError):
            B.BudgetController(p)


def test_affordable_usd_reports_the_tightest_remaining(tmp_path, monkeypatch):
    ctl = _ctl(
        tmp_path,
        monkeypatch,
        "version: 1\npolicies:\n"
        "  - principal: 'tenant:*'\n    limit_usd: 5.0\n    period: lifetime\n"
        "  - principal: 'agent:*'\n    limit_usd: 0.25\n    period: lifetime\n",
    )
    assert ctl.affordable_usd(M.Principal(tenant="t", agent="a")) == pytest.approx(0.25)
    assert ctl.affordable_usd(M.Principal(user="ungoverned")) is None


def test_adding_a_budget_needs_no_python_edit(tmp_path, monkeypatch):
    """No-hardcoding, tested: a dimension/value combination the code has never
    seen is governed purely from YAML."""
    ctl = _ctl(
        tmp_path,
        monkeypatch,
        "version: 1\npolicies:\n"
        "  - principal: 'project:quarterly-report-2027'\n    limit_usd: 0.02\n    period: lifetime\n",
    )
    p = M.Principal(project="quarterly-report-2027")
    assert ctl.admit(p, 0.01).allowed is True
    assert ctl.admit(p, 0.03).allowed is False
