"""Тесты скрипта доказательств (make_evidence.py) и раздела Rule Lab на главной странице. Интернет НЕ нужен.

Данные synthetic, все файлы (evidence/, trials.csv, DECISIONS.md) пишутся во временную папку.
"""
import csv
import re

import numpy as np
import pandas as pd
import pytest

import app as appmod
import data
import make_evidence as me
import rule


@pytest.fixture
def lab(tmp_path, monkeypatch):
    # synthetic: "SPY" - случайный рост с провалом около 2008 года; ставка 2% годовых
    idx = pd.bdate_range("2000-01-03", "2026-11-13")
    returns = np.random.default_rng(1).normal(0.0004, 0.012, len(idx))
    returns[2050:2350] -= 0.003
    spy = 100 * np.cumprod(1 + returns)
    monkeypatch.setattr(data, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(data, "_download_prices", lambda t: pd.DataFrame({t: spy}, index=idx))
    monkeypatch.setattr(data, "_download_risk_free", lambda: pd.DataFrame({data.RF_SERIES: 2.0}, index=idx))
    monkeypatch.setattr(me, "EVIDENCE_DIR", tmp_path / "evidence")
    monkeypatch.setattr(me, "TRIALS_PATH", tmp_path / "trials.csv")
    monkeypatch.setattr(me, "DECISIONS_PATH", tmp_path / "DECISIONS.md")
    monkeypatch.setattr(me, "TEST_STATE_PATH", tmp_path / "evidence" / "test_opened.json")
    monkeypatch.setattr(appmod, "RESULT_CARD_PATH", tmp_path / "RESULT_CARD.md")
    monkeypatch.setattr(rule, "BOOTSTRAP_REPS", 300)   # быстрее; в настоящем запуске 5000
    return tmp_path


def trials(path):
    return list(csv.DictReader((path / "trials.csv").open(encoding="utf-8")))


def test_development_run_keeps_test_closed_and_writes_files(lab):
    ev = me.run()
    assert ev["meta"]["headline_sample"] == "development" and ev["meta"]["test_opened"] == []
    assert "test" not in ev["e8"] and all(row["sample"] != "test" for row in trials(lab))
    for name in ["evidence.json", "evidence_table.md", "result_card_numbers.md",
                 "figures/equity.png", "figures/drawdown.png", "figures/neighbours.png"]:
        assert (lab / "evidence" / name).exists(), name
    assert ev["meta"]["headline_window"] == ["2001-01", "2018-12"] or ev["meta"]["headline_window"] == ("2001-01", "2018-12")


def test_trials_are_appended_never_deleted(lab):
    me.run()
    first = trials(lab)
    assert list(first[0].keys()) == me.TRIALS_HEADER
    assert len(first) == len(rule.NEIGHBOURS) * 2                # train + validation для каждого N
    ev = me.run()
    second = trials(lab)
    assert second[:len(first)] == first and len(second) == 2 * len(first)
    assert ev["e7"]["configurations"] == len(rule.NEIGHBOURS)    # повторный запуск - не новая конфигурация


def test_test_period_opens_once_and_is_recorded(lab):
    ev = me.run(open_test_now=True)
    assert ev["meta"]["headline_sample"] == "full" and len(ev["meta"]["test_opened"]) == 1
    assert "test period 2019-01 to 2026-10 opened (time #1)" in (lab / "DECISIONS.md").read_text(encoding="utf-8")
    with pytest.raises(me.EvidenceError, match="already opened"):
        me.run(open_test_now=True)
    ev = me.run(open_test_now=True, reopen=True)                 # повторное открытие возможно, но записывается
    assert len(ev["meta"]["test_opened"]) == 2
    assert "(time #2)" in (lab / "DECISIONS.md").read_text(encoding="utf-8")


def test_failed_download_does_not_open_the_test(lab, monkeypatch):
    def broken(_):
        raise data.DataError("no internet")
    monkeypatch.setattr(data, "_load_with_fallback", lambda *a, **k: broken(None))
    with pytest.raises(data.DataError):
        me.run(open_test_now=True)
    assert me.test_state()["opened"] == []


def test_same_window_for_rule_and_benchmark(lab):
    ev = me.run()
    head = ev["e1"]["headline"][me.cost_key(0.0)]
    assert head["rule"]["months"] == head["bench"]["months"]


def test_rule_lab_numbers_match_the_evidence_table(lab):
    me.run()
    html = appmod.app.test_client().get("/").get_data(as_text=True)
    table = (lab / "evidence" / "evidence_table.md").read_text(encoding="utf-8")
    ev = me.load_evidence()
    head = ev["e1"]["headline"][me.cost_key(rule.COST_DEFAULT)]
    for value in [me.fmt_pct(head["rule"]["cagr"]), me.fmt_pct(head["rule"]["max_drawdown"]),
                  me.fmt_sr(head["rule"]["sharpe"], head["rule"]["sharpe_se"]),
                  me.fmt_interval(ev["e2"][me.cost_key(rule.COST_DEFAULT)]["bootstrap"])]:
        assert value in html and value in table, value


def test_rule_lab_honest_panel_and_costs_toggle(lab):
    me.run()
    html = appmod.app.test_client().get("/").get_data(as_text=True)
    panel = re.sub(r"<[^>]+>", " ", html.split('id="t-honest"')[1].split("</details>")[0])
    assert "2001-01 to 2018-12" in panel and "configurations looked at" in panel and "not opened yet" in panel
    assert "Team verdict: not written yet" in html
    for cost in rule.COSTS:                                  # переключатель издержек: блок на каждый уровень
        key = me.cost_key(cost)
        assert f'data-cost="{key}"' in html and f"Net of {me.fmt_cost(cost)} per switch" in html, key


def test_rule_lab_shows_team_verdict_from_result_card(lab):
    me.run()
    (lab / "RESULT_CARD.md").write_text("Verdict: the difference is not distinguishable from zero.\n", encoding="utf-8")
    html = appmod.app.test_client().get("/").get_data(as_text=True)
    assert "Team verdict:</b> the difference is not distinguishable from zero." in html


def test_rule_lab_generates_evidence_on_first_visit(lab):
    html = appmod.app.test_client().get("/").get_data(as_text=True)
    assert 'class="error"' not in html and "E1–E8 evidence tables" in html and 'id="rl-data"' in html
    assert (lab / "evidence" / "evidence.json").exists()
