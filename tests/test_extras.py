"""Тесты дополнительных функций: инфляция, "сколько докладывать", смесь фондов, комиссии и CSV.

Интернет НЕ нужен: цены и ставки synthetic, индекс цен CPI подменяется в tests/conftest.py (2.5% в год).
"""
import csv
import io

import numpy as np
import pandas as pd
import pytest

import app as appmod
import data
import extras
import forecast
from conftest import CPI_YEARLY, synthetic_cpi


# ---------- инфляция ----------

def test_yearly_inflation_and_real_return():
    cpi = synthetic_cpi().iloc[:, 0]
    rate, d0, d1 = extras.yearly_inflation(cpi, "2010-01-15", "2020-01-20")
    assert rate == pytest.approx(CPI_YEARLY, abs=1e-4)
    assert (str(d0), str(d1)) == ("2010-01-01", "2020-01-01")
    assert extras.real_return(0.10, 0.025) == pytest.approx(1.10 / 1.025 - 1)
    assert extras.real_return(0.02, 0.02) == pytest.approx(0.0)
    assert extras.todays_money(1025.0, 0.025, 1) == pytest.approx(1000.0)


def test_yearly_inflation_without_enough_data_is_none():
    cpi = synthetic_cpi().iloc[:, 0]
    assert extras.yearly_inflation(cpi, "1999-01-01", "2005-01-01") is None   # раньше начала ряда
    assert extras.yearly_inflation(cpi, "2020-01-10", "2020-03-10") is None   # меньше полугода


# ---------- сколько докладывать ----------

def test_growth_factors_match_the_wealth_simulation():
    monthly = np.random.default_rng(3).normal(0.006, 0.04, (50, 24))
    grown, adds = extras.growth_factors(monthly)
    final = forecast.wealth_paths(monthly, 1000, 75)[:, -1]
    assert np.allclose(1000 * grown + 75 * adds, final)


def test_required_add_with_steady_returns_equals_the_formula():
    returns = pd.Series([0.01] * 120)                       # каждый месяц ровно +1%: все сценарии одинаковы
    pct = (1.01 ** 12 - 1) * 100
    plan = extras.required_add(returns, 1000, 50_000, 10, pct, paths=200)
    assert plan["adds"]["50"] == pytest.approx(plan["steady"], rel=1e-9)
    assert plan["adds"]["50"] == pytest.approx(plan["adds"]["90"], rel=1e-9)
    final = forecast.wealth_paths(np.full((1, 120), 0.01), 1000, plan["adds"]["50"])[0, -1]
    assert final == pytest.approx(50_000)


def test_required_add_is_zero_when_the_goal_is_already_reached():
    returns = pd.Series([0.01] * 120)
    plan = extras.required_add(returns, 10_000, 5_000, 5, 12, paths=100)
    assert plan["adds"] == {"50": 0.0, "75": 0.0, "90": 0.0} and plan["chance_without"] == 1.0
    assert extras.steady_add(1000, 1600, 0, 6) == pytest.approx(100)


def test_higher_chance_needs_a_bigger_top_up():
    returns = pd.Series(np.random.default_rng(5).normal(0.007, 0.05, 240))
    plan = extras.required_add(returns, 1000, 100_000, 10, 8, paths=2000)
    assert plan["adds"]["50"] < plan["adds"]["75"] < plan["adds"]["90"]


# ---------- смесь фондов ----------

def two_funds():
    idx = pd.bdate_range("2020-01-01", "2023-12-29")
    rng = np.random.default_rng(9)
    a = 100 * np.cumprod(1 + rng.normal(0.0005, 0.015, len(idx)))
    b = 50 * np.cumprod(1 + rng.normal(0.0002, 0.006, len(idx)))
    return pd.DataFrame({"AAA": a, "BBB": b}, index=idx)


def test_mix_of_100_percent_one_fund_is_that_fund():
    prices = two_funds()
    mixed = extras.mix_returns(prices, {"AAA": 1.0, "BBB": 0.0})
    assert np.allclose(mixed.values, prices["AAA"].pct_change().dropna().values)


def test_mix_goes_back_to_its_shares_every_month():
    prices = two_funds()
    w = {"AAA": 0.6, "BBB": 0.4}
    mixed = extras.mix_returns(prices, w)
    returns = prices.pct_change().dropna()
    first_days = returns.groupby(returns.index.to_period("M")).head(1)
    expected = 0.6 * first_days["AAA"] + 0.4 * first_days["BBB"]   # в первый день месяца доли ровно 60/40
    assert np.allclose(mixed.loc[first_days.index].values, expected.values)


def test_mix_result_has_stats_fees_and_chart():
    res = extras.mix(two_funds(), {"AAA": 60, "BBB": 40}, 0.02)
    assert set(res["funds"]) == {"AAA", "BBB"} and res["mix"]["fee"] == 0.0
    assert len(res["chart"]["dates"]) == len(res["chart"]["series"]["Mix"]) == 48
    assert res["mix"]["volatility"] < res["funds"]["AAA"]["volatility"]


@pytest.mark.parametrize("funds, weights, message", [
    ("SPY", "100", "2 or 3 funds"),
    ("SPY,TLT", "60,30", "add up to 100%"),
    ("SPY,SPY", "50,50", "different funds"),
    ("SPY,XXX", "50,50", "from the list"),
    ("SPY,TLT", "abc,50", "numbers"),
    ("SPY,TLT", "120,-20", "more than 0%"),
])
def test_mix_form_errors_are_clear(funds, weights, message):
    with pytest.raises(ValueError, match=message):
        extras.parse_mix(funds, weights, appmod.MIX_CHOICES)


def test_fees_cover_every_fund_and_benchmark():
    assert all(extras.expense_ratio(t) is not None for t in appmod.MIX_CHOICES)
    assert extras.fee_cost(0.75, 10_000, 1) == pytest.approx(75)


# ---------- сайт: новые колонки, API и CSV ----------

@pytest.fixture
def site(tmp_path, monkeypatch):
    idx = pd.bdate_range("2000-01-03", "2026-11-13")
    def prices(ticker):
        rng = np.random.default_rng(sum(map(ord, ticker)))
        return pd.DataFrame({ticker: 100 * np.cumprod(1 + rng.normal(0.0006, 0.012, len(idx)))}, index=idx)
    monkeypatch.setattr(data, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(data, "_download_prices", prices)
    monkeypatch.setattr(data, "_download_risk_free", lambda: pd.DataFrame({data.RF_SERIES: 2.0}, index=idx))
    monkeypatch.setattr(forecast, "PATHS", 300)
    return appmod.app.test_client()


PAGE = "/?target=5&amount=1000&add=50&my_fund=&start=2021-01-04&end=2024-12-31"


def test_page_shows_inflation_fees_planner_mix_and_csv_link(site):
    html = site.get(PAGE).get_data(as_text=True)
    for text in ["After inflation", "Fund fee", "How much to add every month?", "Mix funds",
                 "/download.csv?", "CPIAUCSL", "already inside the Adjusted Close price"]:
        assert text in html, text
    assert "US prices rose about 2.50% a year" in html


def test_page_still_works_without_inflation_data(site, monkeypatch):
    def no_key():
        raise data.MissingKeyError("FRED_API_KEY is not set in the .env file.")
    monkeypatch.setattr(data, "_download_cpi", no_key)
    res = site.get(PAGE)
    html = res.get_data(as_text=True)
    assert res.status_code == 200 and 'class="error"' not in html
    assert "After inflation:</b> not available right now" in html


def test_api_plan(site):
    j = site.get("/api/plan?asset=SPY&target=8&amount=1000&goal=50000&years=10").get_json()
    assert j["years"] == 10 and j["adds"]["50"] <= j["adds"]["75"] <= j["adds"]["90"]
    assert j["goal_today"] == pytest.approx(50_000 / (1 + CPI_YEARLY) ** 10, rel=0.01)
    bad = site.get("/api/plan?asset=SPY&target=8&amount=1000&goal=50000&years=99")
    assert bad.status_code == 400 and "years" in bad.get_json()["error"]
    assert site.get("/api/plan?asset=NOPE&target=8&amount=1000&goal=5&years=3").status_code == 400


def test_api_mix(site):
    j = site.get("/api/mix?funds=SPY,TLT&weights=60,40&start=2021-01-04&end=2024-12-31").get_json()
    assert set(j["funds"]) == {"SPY", "TLT"} and j["weights"] == {"SPY": 60.0, "TLT": 40.0}
    assert j["mix"]["fee"] == pytest.approx(0.6 * 0.0945 + 0.4 * 0.15)
    bad = site.get("/api/mix?funds=SPY,TLT&weights=60,30&start=2021-01-04&end=2024-12-31")
    assert bad.status_code == 400 and "100%" in bad.get_json()["error"]


def test_csv_download(site):
    res = site.get(PAGE.replace("/?", "/download.csv?"))
    assert res.status_code == 200 and res.mimetype == "text/csv"
    assert "attachment" in res.headers["Content-Disposition"]
    rows = list(csv.reader(io.StringIO(res.get_data(as_text=True).lstrip("﻿"))))
    head = next(r for r in rows if r and r[0] == "Fund")
    assert "After inflation, %" in head and "Fee per year, %" in head
    funds = [r[0] for r in rows[rows.index(head) + 1:]]
    assert "ARKK" in funds and len(funds) == len(data.FUNDS) - (1 if data.DEFAULT_BENCHMARK in data.FUNDS else 0)
    assert site.get("/download.csv?target=abc").status_code == 400
