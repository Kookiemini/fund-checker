"""Тесты страницы Forecast (forecast.py и /forecast). Интернет НЕ нужен: данные synthetic."""
import json

import numpy as np
import pandas as pd
import pytest

import app as appmod
import data
import forecast

MONTHS = pd.date_range("2005-01-31", periods=240, freq="ME")


def test_constant_returns_give_one_known_answer():
    # synthetic: каждый месяц ровно +1% -> любой "перемешанный" путь одинаковый: 1.01^36 через 3 года
    returns = pd.Series(0.01, index=MONTHS)
    rows = forecast.summarize(forecast.simulate(returns, 120, paths=200), 1000)
    three = rows[0]
    assert three["years"] == 3
    for value in three["values"].values():
        assert value == pytest.approx(1000 * 1.01 ** 36)
    assert three["loss_chance"] == 0 and three["median_per_year"] == pytest.approx(1.01 ** 12 - 1)


def test_scenarios_are_ordered_repeatable_and_widen_with_time():
    returns = pd.Series(np.random.default_rng(3).normal(0.006, 0.045, len(MONTHS)), index=MONTHS)
    growth = forecast.simulate(returns, 120, paths=2000)
    assert np.array_equal(growth, forecast.simulate(returns, 120, paths=2000))   # тот же seed - те же числа
    rows = forecast.summarize(growth, 1000)
    for r in rows:
        v = [r["values"][p] for p in forecast.PERCENTILES]
        assert v == sorted(v)                                                    # плохой <= средний <= хороший
    spreads = [r["values"][95] - r["values"][5] for r in rows]
    assert spreads == sorted(spreads)                                            # 10 лет шире, чем 3 года
    assert rows[-1]["loss_chance"] <= rows[0]["loss_chance"]                     # дольше держим - реже минус


def test_horizon_longer_than_history_still_works():
    returns = pd.Series(np.random.default_rng(4).normal(0.005, 0.04, 60), index=MONTHS[:60])
    assert forecast.simulate(returns, 120, paths=100).shape == (100, 120)


@pytest.fixture
def site(tmp_path, monkeypatch):
    idx = pd.bdate_range("2000-01-03", "2026-11-13")
    r = np.random.default_rng(1).normal(0.0004, 0.012, len(idx))
    monkeypatch.setattr(data, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(data, "_download_prices", lambda t: pd.DataFrame({t: 100 * np.cumprod(1 + r)}, index=idx))
    monkeypatch.setattr(data, "_download_risk_free", lambda: pd.DataFrame({data.RF_SERIES: 2.0}, index=idx))
    monkeypatch.setattr(forecast, "PATHS", 500)                                  # быстрее; в настоящем запуске 10 000
    return appmod.app.test_client()


def page(site, **q):
    query = {"target": "8", "start": "2023-01-02", "end": "2026-11-13", "amount": "1000", "add": "0"}
    query.update(q)
    return site.get("/", query_string=query).get_data(as_text=True)


def test_one_page_has_all_sections_and_forecast_data(site):
    html = page(site)
    assert 'class="error"' not in html
    for title in ["Funds that gave 8% a year or more", "Forecast: your", "Rule Lab: would a simple rule have helped?",
                  "How we calculate", "Deep analysis", "Honesty panel", 'id="summary"']:
        assert title in html, title
    payload = json.loads(html.split('id="fc-data">')[1].split("</script>")[0])
    assets = [i["asset"] for i in payload["items"]]
    assert assets[-1] == "RULE_MA10" and len(assets) >= 2
    assert [r["years"] for r in payload["items"][0]["rows"]] == [3, 5, 10]
    assert 'href="/forecast"' not in html and 'href="/rule-lab"' not in html      # одна страница, без переходов


def test_monthly_top_up_raises_money_put_in(site):
    one = json.loads(page(site, add="100").split('id="fc-data">')[1].split("</script>")[0])
    row = one["items"][0]["rows"][0]
    assert row["put_in"] == pytest.approx(1000 + 100 * 36)


def test_api_forecast_recalculates_and_checks_inputs(site):
    ok = site.get("/api/forecast?target=10&amount=5000&add=50&assets=SPY,RULE_MA10").get_json()
    assert [i["asset"] for i in ok["items"]] == ["SPY", "RULE_MA10"] and ok["amount"] == 5000
    assert 0 <= ok["items"][0]["rows"][2]["chance_target"] <= 1
    for bad in ["amount=abc", "amount=0", "amount=1e12", "amount=1000&add=-5"]:
        r = site.get(f"/api/forecast?target=10&assets=SPY&{bad}")
        assert r.status_code == 400 and "Enter the" in r.get_json()["error"], bad
    assert site.get("/api/forecast?target=10&amount=1000&assets=BITO").status_code == 400


def test_bad_amount_on_the_page_gives_clear_error(site):
    assert "Enter the amount as a number from 1 to" in page(site, amount="abc")


def test_high_target_shows_warning(site):
    assert "a very high target" in page(site, target="40")
    assert "a very high target" not in page(site, target="5")


def test_fund_row_opens_full_check_without_leaving_the_page(site):
    html = site.get("/api/fund?fund=QQQ&benchmark=SPY&start=2023-01-02&end=2026-11-13&target=8").get_data(as_text=True)
    assert "data-fund-chart" in html and "Compare with" in html and "advert" not in html.lower()
    assert "Please choose a fund from the list." in site.get("/api/fund?fund=BITO&start=2023-01-02&end=2026-11-13").get_data(as_text=True)
