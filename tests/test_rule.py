"""Тесты ядра правила (rule.py). Интернет НЕ нужен: все ряды synthetic (искусственные).

1. Toy-тест: маленький пример, который команда считает ВРУЧНУЮ (значения вписывает команда).
2. Canary-тесты: ловят заглядывание в будущее (look-ahead).
3. Тесты с заранее известным ответом: издержки, просадка, кэш, стандартная ошибка, бутстреп, break-even.
"""
import math

import numpy as np
import pandas as pd
import pytest

import rule

MONTHS = pd.date_range("2020-01-31", periods=6, freq="ME")


# ---------- 1. Toy-тест (заполняет КОМАНДА, по ручному расчёту) ----------

# synthetic: 6 месячных цен, средняя за N = 3 месяца, ставка 1.2% годовых (0.1% в месяц) во все месяцы
TOY_PRICES = [100.0, 104.0, 102.0, 99.0, 103.0, 108.0]
TOY_N = 3
TOY_RATE_PERCENT = 1.2
# Команда: посчитайте на бумаге и впишите ожидаемые значения для месяцев 4, 5 и 6
# (до этого позиции нет: средняя за 3 месяца появляется на закрытии месяца 3).
TOY_EXPECTED = {
    "signal": [0.0, 0.0, 1.0, 1.0],    # сигнал на закрытии месяцев 3..6, например [1.0, 0.0, ...]
    "position": [0.0, 0.0, 1.0],  # позиция в месяцах 4..6
    "gross": [0.001, 0.001, 0.048544],     # доходность правила в месяцах 4..6 (доли, 0.01 = 1%)
}


def test_toy_example_matches_hand_calculation():
    if any(v is None for v in TOY_EXPECTED.values()):
        pytest.skip("Team: fill TOY_EXPECTED with your hand-computed values (RED task, no AI).")
    prices = pd.Series(TOY_PRICES, index=MONTHS)
    rate = pd.Series(TOY_RATE_PERCENT, index=MONTHS)
    sig = rule.signal(prices, TOY_N)
    assert sig.iloc[2:].tolist() == pytest.approx(TOY_EXPECTED["signal"])
    table = rule.backtest(prices, rate, TOY_N)
    assert table["position"].tolist() == pytest.approx(TOY_EXPECTED["position"])
    assert table["gross"].tolist() == pytest.approx(TOY_EXPECTED["gross"], abs=1e-6)


# ---------- 2. Canary: заглядывание в будущее ----------

def random_walk(months=1200, seed=7):
    # synthetic: случайное блуждание без тренда и без памяти - никакое честное правило не может его предсказать
    idx = pd.date_range("1900-01-31", periods=months, freq="ME")
    returns = np.random.default_rng(seed).normal(0.0, 0.045, months)
    return pd.Series(100 * np.cumprod(1 + returns), index=idx), pd.Series(0.0, index=idx)


def test_canary_future_prices_do_not_change_past_positions():
    """Меняем ВСЕ цены после месяца k. Позиции и доходности до месяца k включительно не должны измениться."""
    prices, rate = random_walk(240)
    k = 150
    changed = prices.copy()
    changed.iloc[k + 1:] *= np.linspace(0.3, 3.0, len(changed) - k - 1)
    a, b = rule.backtest(prices, rate, 10), rule.backtest(changed, rate, 10)
    cut = prices.index[k]
    pd.testing.assert_series_equal(a.loc[:cut, "position"], b.loc[:cut, "position"])
    pd.testing.assert_series_equal(a.loc[:cut, "gross"], b.loc[:cut, "gross"])
    # и позиция месяца k+1 уже известна на закрытии месяца k
    assert a.loc[prices.index[k + 1], "position"] == b.loc[prices.index[k + 1], "position"]


def test_canary_rule_on_random_walk_has_no_edge_but_a_cheater_would():
    """На случайном блуждании честное правило не даёт преимущества. Версия, которая берёт сигнал
    того же месяца (без сдвига), показывает огромный Sharpe - значит тест умеет ловить look-ahead."""
    prices, rate = random_walk()
    honest = rule.backtest(prices, rate, 10)
    assert abs(rule.sharpe(honest["gross"], honest["rf"])) < 0.3
    cheat_position = rule.signal(prices, 10)                     # БЕЗ shift(1): заглядывание в будущее
    cheat = (cheat_position * prices.pct_change()).dropna()
    cheat_sharpe = rule.sharpe(cheat, pd.Series(0.0, index=cheat.index))
    assert cheat_sharpe > 0.6 and cheat_sharpe - rule.sharpe(honest["gross"], honest["rf"]) > 0.5


def test_position_is_previous_month_signal():
    prices, rate = random_walk(60)
    table = rule.backtest(prices, rate, 10)
    sig = rule.signal(prices, 10)
    assert (table["position"] == sig.shift(1).loc[table.index]).all()


# ---------- 3. Известные ответы ----------

def test_cash_earns_previous_month_rate():
    # synthetic: цена всегда ниже средней после падения -> позиция 0 -> доход = ставка прошлого месяца / 12
    idx = pd.date_range("2020-01-31", periods=8, freq="ME")
    prices = pd.Series([100, 100, 100, 90, 80, 70, 60, 50.0], index=idx)
    rate = pd.Series([1.2, 1.2, 1.2, 2.4, 3.6, 4.8, 6.0, 7.2], index=idx)
    table = rule.backtest(prices, rate, 3)
    out = table[table["position"] == 0]
    assert len(out) >= 3
    for month, row in out.iterrows():
        previous = rate.loc[:month].iloc[-2]
        assert row["gross"] == pytest.approx(previous / 100 / 12)


def test_costs_known_answer():
    # synthetic: позиции 0,1,1,0,1 -> изменения 0,1,0,1,1; c = 1% -> вычитаем 0, 1%, 0, 1%, 1%
    idx = pd.date_range("2020-01-31", periods=5, freq="ME")
    part = pd.DataFrame({"gross": [0.01, 0.02, 0.03, 0.04, 0.05], "turnover": [0, 1, 0, 1, 1.0],
                         "position": [0, 1, 1, 0, 1.0]}, index=idx)
    assert rule.net_returns(part, 0.01).tolist() == pytest.approx([0.01, 0.01, 0.03, 0.03, 0.04])
    assert rule.net_returns(part, 0.0).tolist() == pytest.approx(part["gross"].tolist())


def test_turnover_counts_each_entry_and_exit():
    prices, rate = random_walk(120)
    table = rule.backtest(prices, rate, 10)
    assert (table["turnover"] == table["position"].diff().abs().fillna(0.0)).all()


def test_max_drawdown_known_answer():
    # synthetic: стоимость 1 -> 1.2 -> 0.9 -> 1.1 -> 0.6 -> 0.8: худшее падение от 1.2 до 0.6 = -50%
    wealth = np.array([1.0, 1.2, 0.9, 1.1, 0.6, 0.8])
    returns = pd.Series(wealth[1:] / wealth[:-1] - 1)
    assert rule.max_drawdown(returns) == pytest.approx(-0.5)
    assert rule.max_drawdown(pd.Series([0.01, 0.02])) == 0.0      # только рост - просадки нет
    assert rule.max_drawdown(pd.Series([-0.1])) == pytest.approx(-0.1)  # падение с самого старта


def test_cagr_known_answer():
    # synthetic: +21% за 24 месяца ровно -> 10% в год
    returns = pd.Series([1.21 ** (1 / 24) - 1] * 24)
    assert rule.cagr(returns) == pytest.approx(0.10)


def test_sharpe_se_known_answer():
    # при Sharpe 0 и 120 месяцах: sqrt(1/120) * sqrt(12) = sqrt(0.1)
    assert rule.sharpe_se(0.0, 120) == pytest.approx(math.sqrt(0.1))
    assert rule.sharpe_se(None, 120) is None


def test_var_and_worst_month():
    returns = pd.Series(np.linspace(-0.10, 0.09, 20), index=pd.date_range("2020-01-31", periods=20, freq="ME"))
    s = rule.summary(pd.DataFrame({"bench": returns, "rf": 0.0, "position": 1.0, "turnover": 0.0}), "bench")
    assert s["worst_month"] == pytest.approx(-0.10) and s["worst_month_date"] == "2020-01"
    assert s["var_5"] == pytest.approx(np.quantile(returns, 0.05))


def test_bootstrap_identical_series_gives_zero_and_is_repeatable():
    prices, rate = random_walk(240)
    table = rule.backtest(prices, rate, 10)
    same = table.assign(net=table["bench"])
    iv = rule.sharpe_diff_bootstrap(same, reps=500)
    assert iv["diff"] == pytest.approx(0) and iv["low"] == pytest.approx(0) and iv["high"] == pytest.approx(0)
    other = table.assign(net=table["gross"])
    assert rule.sharpe_diff_bootstrap(other, reps=500) == rule.sharpe_diff_bootstrap(other, reps=500)


def test_break_even_cost_removes_the_advantage():
    prices, rate = random_walk(600, seed=11)
    table = rule.backtest(prices, rate, 10)
    for measure in ("sharpe", "cagr"):
        be = rule.break_even_cost(table, measure)
        if be["status"] == "found":
            assert rule.advantage(table, be["value"], measure) == pytest.approx(0, abs=1e-6)
        else:
            assert be["status"] in ("no_advantage", "above_max")
    # synthetic: правило, которое всегда в рынке, равно бенчмарку -> преимущества нет
    flat = table.assign(gross=table["bench"], turnover=0.0)
    assert rule.break_even_cost(flat)["status"] == "no_advantage"


def test_expected_max_noise_sharpe():
    assert rule.expected_max_noise_sharpe(1, 144) == 0.0
    five, fifty = rule.expected_max_noise_sharpe(5, 144), rule.expected_max_noise_sharpe(50, 144)
    assert 0 < five < fifty                                     # больше попыток -> выше лучший шум


def test_monthly_prices_drop_unfinished_month():
    days = pd.bdate_range("2024-01-01", "2024-03-13")
    monthly = rule.monthly_prices(pd.Series(range(len(days)), index=days, dtype=float))
    assert [d.strftime("%Y-%m") for d in monthly.index] == ["2024-01", "2024-02"]   # март ещё не закончился


def test_frozen_settings():
    assert rule.PARAM_MAIN in rule.NEIGHBOURS
    assert rule.PARAM_ALLOWED[0] <= min(rule.NEIGHBOURS) and max(rule.NEIGHBOURS) <= rule.PARAM_ALLOWED[1]
    assert rule.TRAIN[1] < rule.VALIDATION[0] <= rule.VALIDATION[1] < rule.TEST[0]   # хронологически, без перекрытия
    assert len(rule.COSTS) >= 3 and rule.COST_DEFAULT in rule.COSTS
