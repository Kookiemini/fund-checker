"""Тесты страницы Forecast (forecast.py и /forecast). Интернет НЕ нужен: данные synthetic."""
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


def test_forecast_page_shows_three_horizons_and_honest_panel(site):
    html = site.get("/forecast?asset=SPY&amount=1000").get_data(as_text=True)
    assert 'class="error"' not in html
    for years in (3, 5, 10):
        assert f"In {years} years" in html
    assert "Honest panel" in html and "Not a promise" in html and "data:image/png;base64," in html


def test_forecast_for_the_rule_and_bad_inputs(site):
    assert 'class="error"' not in site.get("/forecast?asset=RULE_MA10&amount=5000").get_data(as_text=True)
    assert "Please choose a fund from the list." in site.get("/forecast?asset=BITO").get_data(as_text=True)
    for bad in ["abc", "0", "-5", "1e12"]:
        assert "Enter the amount as a number" in site.get(f"/forecast?asset=SPY&amount={bad}").get_data(as_text=True)


def test_menu_links_to_forecast(site):
    assert 'href="/forecast"' in site.get("/").get_data(as_text=True)
