"""Тесты для core.py (запуск: pytest).

Три группы тестов:
  A. HAND-CALCULATED - ожидаемое значение посчитано вручную на маленьких числах.
     Шаги расчёта расписаны в комментарии над константой, константа лежит
     вверху файла, тест только сравнивает с ней результат функции из core.py.
  B. SYNTHETIC - искусственные данные с очевидным результатом
     (одинаковые ряды -> бета 1, постоянный ряд -> волатильность 0 и т.д.).
  C. Проверки ошибок и вердикта.
  D. Deep analysis (раздел H внизу файла): "честная" рекламная цифра, нехватка до зелёного,
     близость Sharpe, ближайшие окна дат; проверка дат с минимумом из data.DOWNLOAD_START.
"""
from datetime import date

import numpy as np
import pandas as pd
import pytest

import core

# ======================================================================
# Ожидаемые значения, посчитанные вручную (шаги расчёта - в комментариях)
# ======================================================================

# 1) Накопленная доходность, цены [100, 110, 121]:
#    121 / 100 - 1 = 0.21
HAND_CUMULATIVE_RETURN = 0.21

# 2) Годовая волатильность, дневные доходности [0.01, 0.03]:
#    среднее = 0.02; отклонения = -0.01 и +0.01
#    дисперсия (ddof=1) = (0.0001 + 0.0001) / (2 - 1) = 0.0002
#    годовая волатильность = sqrt(0.0002 * 252) = sqrt(0.0504) ~ 0.2245
HAND_VOLATILITY = 0.2245

# 3) Sharpe, доходности [0.01, 0.03], risk-free = 0.0252 годовых:
#    дневная ставка = 0.0252 / 252 = 0.0001
#    избыточные доходности = [0.0099, 0.0299], среднее = 0.0199
#    годовая избыточная доходность = 0.0199 * 252 = 5.0148
#    Sharpe = 5.0148 / sqrt(0.0504) ~ 22.3377
HAND_SHARPE = 22.3377

# 4) Бета, бенчмарк [0.01, -0.01, 0.02, 0.00], фонд [0.03, -0.02, 0.05, 0.01]:
#    среднее бенчмарка = 0.005; отклонения: 0.005, -0.015, 0.015, -0.005
#    сумма квадратов отклонений бенчмарка = 0.0005
#    среднее фонда = 0.0175; отклонения: 0.0125, -0.0375, 0.0325, -0.0075
#    сумма произведений отклонений = 0.00115
#    бета = 0.00115 / 0.0005 = 2.3  (деление на n-1 сокращается)
HAND_BETA = 2.3

# 5) Risk-free: ставки FRED [4.0, 5.0] процентов -> среднее 4.5% -> 0.045
HAND_RISK_FREE = 0.045

# 6) Арифметическая годовая доходность, дневные доходности [0.001, 0.003, -0.001, 0.001]:
#    сумма = 0.004; среднее = 0.004 / 4 = 0.001
#    годовая = 0.001 * 252 = 0.252
HAND_ARITH_ANNUAL = 0.252

# 7) Арифметическая и геометрическая годовая доходность, 126 дней: 125 дней по 0%, один день +21%:
#    арифметическая: среднее = 0.21 / 126; годовая = (0.21 / 126) * 252 = 0.21 * 2 = 0.42
#    геометрическая: рост за период = 1.21; показатель = 252 / 126 = 2;
#    годовая = 1.21 ** 2 - 1 = 1.4641 - 1 = 0.4641
#    (две доходности за один и тот же период различаются: 0.42 против 0.4641)
HAND_ARITH_ANNUAL_126 = 0.42
HAND_CAGR_126 = 0.4641

# 8) Отрицательная доходность, 126 дней: 125 дней по 0%, один день -19%:
#    рост = 1 - 0.19 = 0.81; показатель = 2; CAGR = 0.81 ** 2 - 1 = 0.6561 - 1 = -0.3439
#    арифметическая = (-0.19 / 126) * 252 = -0.38
HAND_CAGR_NEGATIVE = -0.3439
HAND_ARITH_NEGATIVE = -0.38

# 9) CAPM и альфа. Фонд [0.03, -0.02, 0.05, 0.01], бенчмарк [0.01, -0.01, 0.02, 0.00], rf = 0.02 годовых:
#    арифметическая годовая фонда = среднее * 252 = 0.0175 * 252 = 4.41
#    арифметическая годовая бенчмарка = 0.005 * 252 = 1.26
#    бета = 2.3 (см. пункт 4)
#    capm_expected = rf + бета * (бенчмарк - rf) = 0.02 + 2.3 * (1.26 - 0.02)
#                  = 0.02 + 2.3 * 1.24 = 0.02 + 2.852 = 2.872
#    альфа = 4.41 - 2.872 = 1.538
HAND_CAPM_EXPECTED = 2.872
HAND_ALPHA = 1.538

# 10) Разрыв "реклама минус реальность" (CAGR из пункта 7 = 0.4641):
#    реклама 50%:  gap = 0.50 - 0.4641 = 0.0359  -> 3.59 п.п., реклама приукрашена
#    реклама 40%:  gap = 0.40 - 0.4641 = -0.0641 -> -6.41 п.п., фонд превзошёл рекламу
HAND_GAP_OVERSTATED = 0.0359
HAND_GAP_BEATEN = -0.0641

# 11) Панель "Why this number?" (core.explain_gap), реклама выше реальности:
#    реклама 0.30, реально 0.10 -> gap = 0.30 - 0.10 = 0.20 -> 0.20 * 100 = 20 п.п. (реклама приукрашена)
#    бета 1.5: |1.5 - 1| = 0.5 > 0.05 и 1.5 > 1 -> фонд двигается сильнее бенчмарка ("higher")
#    альфа 0.04 > 0 -> фонд дал больше платы за риск ("above")
HAND_EXPLAIN_POINTS_OVERSTATED = 20.0

# 12) Фонд дал больше рекламы:
#    реклама 0.10, реально 0.30 -> gap = 0.10 - 0.30 = -0.20 -> -20 п.п. (фонд превзошёл рекламу)
#    бета 0.5: |0.5 - 1| = 0.5 > 0.05 и 0.5 < 1 -> "lower"; альфа -0.02 < 0 -> "below"
HAND_EXPLAIN_POINTS_BEATEN = -20.0

# ---- Deep analysis: ожидаемые значения посчитаны вручную (synthetic ряды, шаги расчёта в комментариях) ----

# 13) "Честная" рекламная цифра: CAGR в % вниз до шага 0.1
#    0.0712 -> 7.12% -> вниз до 7.1;  0.10 -> 10.0;  0.0999 -> 9.99% -> 9.9;  отрицательный -> None
HAND_HONEST_7_12 = 7.1
HAND_HONEST_10 = 10.0
HAND_HONEST_9_99 = 9.9

# 14) Нехватка до Sharpe бенчмарка. Доходности [0.01, 0.03], rf = 0, Sharpe бенчмарка = 30:
#    vol = sqrt(0.0504) = 0.2245;  доходность фонда = 0.02 * 252 = 5.04;  Sharpe фонда = 5.04 / 0.2245 = 22.45
#    разница Sharpe = 30 - 22.45 = 7.55
#    нужная доходность при той же волатильности = rf + 30 * vol = 0 + 30 * 0.2245 = 6.73498
#    нехватка = 6.73498 - 5.04 = 1.69498, то есть 169.498 п.п. в год
#    допустимая волатильность при той же доходности = (5.04 - 0) / 30 = 0.168
#    сокращение волатильности = 0.2245 - 0.168 = 0.0565, то есть 5.64994 п.п.
HAND_SHORTFALL_GAP = 7.55006
HAND_SHORTFALL_REQUIRED = 6.73498
HAND_SHORTFALL_PP = 169.498
HAND_SHORTFALL_MAX_VOL = 0.168
HAND_SHORTFALL_CUT_PP = 5.64994

# 15) Доходность фонда НИЖЕ rf. Доходности [0.0001, 0.0003], rf = 0.10, Sharpe бенчмарка = 1:
#    доходность фонда = 0.0002 * 252 = 0.0504 (меньше rf 0.10: одной волатильностью не исправить)
#    std = 0.00014142, vol = 0.00014142 * sqrt(252) = 0.002245
#    нужная доходность = 0.10 + 1 * 0.002245 = 0.102245;  нехватка = (0.102245 - 0.0504) * 100 = 5.1845 п.п.
HAND_LOW_REQUIRED = 0.102245
HAND_LOW_SHORTFALL_PP = 5.1845

# 16) Сокращение окна: выбранный старт 2005-01-01, ряд начинается 2014-10-31.
#    с 2005-01-01 до 2014-01-01 девять лет: 9 * 365 + 2 високосных дня (2008, 2012) = 3287 дней;
#    с 2014-01-01 до 2014-10-31: 31+28+31+30+31+30+31+31+30 = 273 дня до 1 октября, плюс 30 дней = 303;
#    итого 3287 + 303 = 3590 дней
HAND_SHORTENED_DAYS = 3590

# 17) Ближайшее окно (synthetic, 13 торговых дат, 12 доходностей, шаг 3 дня, минимум окна 4 дня):
#    бенчмарк: +0.01, -0.01, +0.01, ... (среднее 0); фонд = бенчмарк - 0.03 на первых 6 днях и + 0.01 на последних 6
#    (общий результат красный: фонд в среднем на 0.01 в день хуже). Ставка 0% на первых 6 датах и 2.52% после.
#    Окно со сдвигом начала на 3 дня красное (в него попали 3 плохих дня), на 6 дней: зелёное.
#    Окно: доходности фонда [0.02, 0, 0.02, 0, 0.02, 0], среднее 0.01, std = sqrt(0.0006 / 5) = 0.0109545,
#    vol = 0.0109545 * sqrt(252) = 0.17390; ставка за ЭТО окно = 2.52% = 0.0252 (за весь период была бы 1.36%)
#    Sharpe фонда = (0.01 * 252 - 0.0252) / 0.17390 = 14.3465
#    Sharpe бенчмарка = (0 * 252 - 0.0252) / 0.17390 = -0.14491 (те же +-0.01, та же vol)
#    сдвиг начала = 6 торговых дат = с 2024-01-01 до 2024-01-09 = 8 дней
HAND_WINDOW_SHARPE_FUND = 14.3465
HAND_WINDOW_SHARPE_BENCH = -0.14491
HAND_WINDOW_SHIFT_DAYS = 8
HAND_WINDOW_SHORT_SHIFT_DAYS = 11   # окно из 3 доходностей (минимум 3): с 2024-01-01 до 2024-01-12 = 11 дней


# ======================================================================
# A. Тесты с ручным расчётом
# ======================================================================

def test_cumulative_return_hand_calculated():
    prices = pd.Series([100, 110, 121])
    assert core.cumulative_return(prices) == pytest.approx(HAND_CUMULATIVE_RETURN, abs=1e-9)


def test_volatility_hand_calculated():
    returns = pd.Series([0.01, 0.03])
    assert core.annualized_volatility(returns) == pytest.approx(HAND_VOLATILITY, abs=1e-4)


def test_sharpe_hand_calculated():
    returns = pd.Series([0.01, 0.03])
    assert core.sharpe_ratio(returns, rf_annual=0.0252) == pytest.approx(HAND_SHARPE, abs=1e-3)


def test_beta_hand_calculated():
    bench = pd.Series([0.01, -0.01, 0.02, 0.00])
    fund = pd.Series([0.03, -0.02, 0.05, 0.01])
    assert core.beta(fund, bench) == pytest.approx(HAND_BETA, abs=1e-9)


def test_risk_free_annual_hand_calculated():
    rf_percent = pd.Series([4.0, 5.0])
    assert core.risk_free_annual(rf_percent) == pytest.approx(HAND_RISK_FREE, abs=1e-9)


def test_daily_returns_hand_calculated():
    # 110/100 - 1 = 0.10;  99/110 - 1 = -0.10
    result = core.daily_returns(pd.Series([100, 110, 99]))
    assert list(result) == pytest.approx([0.10, -0.10], abs=1e-9)


# ======================================================================
# B. Синтетические данные (synthetic)
# ======================================================================

# synthetic: небольшой ряд дневных доходностей, не из реального рынка
SYNTHETIC_RETURNS = pd.Series([0.01, -0.02, 0.015, 0.0, 0.005])


def test_beta_of_identical_series_is_one():
    # synthetic: два одинаковых ряда -> бета = 1
    assert core.beta(SYNTHETIC_RETURNS, SYNTHETIC_RETURNS) == pytest.approx(1.0)


def test_beta_of_double_returns_is_two():
    # synthetic: фонд двигается ровно вдвое сильнее бенчмарка -> бета = 2
    assert core.beta(2 * SYNTHETIC_RETURNS, SYNTHETIC_RETURNS) == pytest.approx(2.0)


def test_volatility_of_constant_series_is_zero():
    # synthetic: постоянные доходности -> разброса нет -> волатильность 0
    constant = pd.Series([0.01] * 10)
    assert core.annualized_volatility(constant) == pytest.approx(0.0)


def test_sharpe_of_constant_series_is_none():
    # synthetic: волатильность 0 -> делить нельзя -> None (а не ошибка)
    constant = pd.Series([0.01] * 10)
    assert core.sharpe_ratio(constant, rf_annual=0.0) is None


def test_beta_with_constant_benchmark_is_none():
    # synthetic: у бенчмарка нулевая дисперсия -> бета не определена
    bench = pd.Series([0.01] * 5)
    assert core.beta(SYNTHETIC_RETURNS, bench) is None


def test_annualized_return_known_growth():
    # synthetic: 252 дня, в первый день +10%, дальше 0% -> за год ровно +10%
    returns = pd.Series([0.10] + [0.0] * 251)
    assert core.annualized_return(returns) == pytest.approx(0.10, abs=1e-9)


def test_higher_risk_free_lowers_sharpe():
    # synthetic: чем выше безрисковая ставка, тем ниже Sharpe
    low = core.sharpe_ratio(SYNTHETIC_RETURNS, rf_annual=0.0)
    high = core.sharpe_ratio(SYNTHETIC_RETURNS, rf_annual=0.05)
    assert high < low


# ======================================================================
# C. Вердикт, сравнение с рекламой и ошибки
# ======================================================================

def test_justifies_risk_verdicts():
    assert core.justifies_risk(1.2, 0.8) is True
    assert core.justifies_risk(0.5, 0.8) is False
    assert core.justifies_risk(None, 0.8) is None


def test_compare_claim_uses_cagr_for_gap_and_arithmetic_for_alpha():
    # synthetic: фонд = бенчмарк + 0.1% в день, реклама обещает 30% годовых
    bench = SYNTHETIC_RETURNS
    fund = SYNTHETIC_RETURNS + 0.001
    result = core.compare_claim(0.30, fund, bench, rf_annual=0.0)
    # gap считается от геометрической доходности (CAGR), альфа - от арифметической
    assert result["gap"] == pytest.approx(0.30 - core.annualized_return(fund))
    assert result["alpha"] == pytest.approx(core.arithmetic_annual_return(fund) - result["capm_expected"])
    assert result["actual_arith"] == pytest.approx(core.arithmetic_annual_return(fund))
    assert result["beta"] == pytest.approx(1.0)  # сдвиг на константу не меняет бету


def test_cumulative_return_needs_two_prices():
    with pytest.raises(ValueError):
        core.cumulative_return(pd.Series([100]))


# ======================================================================
# D. Вспомогательные функции для экрана
# ======================================================================

def test_percent_conversions_roundtrip():
    # 35% -> 0.35 и обратно
    assert core.percent_to_decimal(35) == pytest.approx(0.35)
    assert core.to_percent(0.35) == pytest.approx(35)


def test_growth_of_100_hand_calculated():
    # цены 50 -> 60: рост на 20%, значит 100 превращаются в 120
    result = core.growth_of_100(pd.Series([50, 60]))
    assert list(result) == pytest.approx([100.0, 120.0])


# ======================================================================
# E. Новые тесты с ручным расчётом (значения - в константах вверху файла)
# ======================================================================

# synthetic: 126 дней, один день +21%, остальные 0% (рост 1.21)
RETURNS_126_UP = pd.Series([0.0] * 125 + [0.21])
# synthetic: 126 дней, один день -19%, остальные 0% (рост 0.81)
RETURNS_126_DOWN = pd.Series([0.0] * 125 + [-0.19])


def test_arithmetic_annual_return_hand_calculated():
    returns = pd.Series([0.001, 0.003, -0.001, 0.001])
    assert core.arithmetic_annual_return(returns) == pytest.approx(HAND_ARITH_ANNUAL, abs=1e-9)


def test_arithmetic_and_geometric_differ_hand_calculated():
    assert core.arithmetic_annual_return(RETURNS_126_UP) == pytest.approx(HAND_ARITH_ANNUAL_126, abs=1e-9)
    assert core.annualized_return(RETURNS_126_UP) == pytest.approx(HAND_CAGR_126, abs=1e-9)


def test_negative_returns_hand_calculated():
    assert core.annualized_return(RETURNS_126_DOWN) == pytest.approx(HAND_CAGR_NEGATIVE, abs=1e-9)
    assert core.arithmetic_annual_return(RETURNS_126_DOWN) == pytest.approx(HAND_ARITH_NEGATIVE, abs=1e-9)


def test_capm_expected_and_alpha_hand_calculated():
    # synthetic: те же числа, что в пункте 4 (бета 2.3)
    bench = pd.Series([0.01, -0.01, 0.02, 0.00])
    fund = pd.Series([0.03, -0.02, 0.05, 0.01])
    result = core.compare_claim(0.35, fund, bench, rf_annual=0.02)
    assert result["beta"] == pytest.approx(HAND_BETA, abs=1e-9)
    assert result["capm_expected"] == pytest.approx(HAND_CAPM_EXPECTED, abs=1e-9)
    assert result["alpha"] == pytest.approx(HAND_ALPHA, abs=1e-9)


def test_gap_and_overstated_flag_when_advertisement_is_too_high():
    result = core.compare_claim(0.50, RETURNS_126_UP, RETURNS_126_UP, rf_annual=0.0)
    assert result["gap"] == pytest.approx(HAND_GAP_OVERSTATED, abs=1e-9)
    assert result["overstated"] is True
    assert result["headline"]["direction"] == "overstated"
    assert result["headline"]["points_signed"] == pytest.approx(3.59, abs=1e-7)


def test_gap_and_overstated_flag_when_fund_beat_advertisement():
    result = core.compare_claim(0.40, RETURNS_126_UP, RETURNS_126_UP, rf_annual=0.0)
    assert result["gap"] == pytest.approx(HAND_GAP_BEATEN, abs=1e-9)
    assert result["overstated"] is False
    assert result["headline"]["direction"] == "beat"
    assert result["headline"]["points_signed"] == pytest.approx(-6.41, abs=1e-7)
    assert result["headline"]["points_abs"] == pytest.approx(6.41, abs=1e-7)


def test_gap_headline_matched_when_difference_is_tiny():
    # synthetic: разница 0.001 п.п. на экране выглядит как 0.00 -> "matched", флаг False
    assert core.gap_headline(0.00001)["direction"] == "matched"
    assert core.gap_headline(0.0)["direction"] == "matched"
    assert core.gap_headline(0.01)["direction"] == "overstated"
    assert core.gap_headline(-0.01)["direction"] == "beat"


def test_verdict_code():
    assert core.verdict_code(True) == "justifies"
    assert core.verdict_code(False) == "does_not_justify"
    assert core.verdict_code(None) == "unknown"


# ======================================================================
# F. Граничные случаи
# ======================================================================

def test_single_price_gives_no_returns_and_errors():
    # synthetic: один день цен -> дневных доходностей нет
    returns = core.daily_returns(pd.Series([100.0]))
    assert len(returns) == 0
    with pytest.raises(ValueError):
        core.annualized_return(returns)
    with pytest.raises(ValueError):
        core.arithmetic_annual_return(returns)
    with pytest.raises(ValueError):
        core.cumulative_return(pd.Series([100.0]))


def test_single_return_has_no_volatility_sharpe_or_beta():
    # synthetic: одна доходность -> ddof=1 не определён
    one = pd.Series([0.01])
    with pytest.raises(ValueError):
        core.annualized_volatility(one)
    with pytest.raises(ValueError):
        core.sharpe_ratio(one, rf_annual=0.0)
    with pytest.raises(ValueError):
        core.beta(one, one)


def test_zero_volatility_fund_gives_unknown_verdict():
    # synthetic: постоянная доходность фонда -> Sharpe None -> вердикт "unknown", без ошибки
    constant = pd.Series([0.001] * 10)
    result = core.compare_claim(0.10, constant, pd.Series([0.01, -0.01] * 5), rf_annual=0.0)
    assert result["sharpe_fund"] is None
    assert result["justifies_risk"] is None
    assert result["verdict"] == "unknown"


def test_zero_variance_benchmark_gives_no_beta_capm_or_alpha():
    # synthetic: бенчмарк не меняется -> бета не определена, CAPM и альфа тоже None
    bench = pd.Series([0.001] * 5)
    result = core.compare_claim(0.10, SYNTHETIC_RETURNS, bench, rf_annual=0.01)
    assert result["beta"] is None
    assert result["capm_expected"] is None
    assert result["alpha"] is None
    assert result["gap"] is not None  # разрыв с рекламой всё равно считается


def test_all_negative_returns():
    # synthetic: все доходности отрицательные -> обе годовые доходности отрицательные, CAGR > -1
    negative = pd.Series([-0.01, -0.02, -0.005, -0.01])
    assert core.arithmetic_annual_return(negative) < 0
    assert -1 < core.annualized_return(negative) < 0
    assert core.sharpe_ratio(negative, rf_annual=0.0) < 0


def test_total_loss_gives_cagr_minus_one():
    # synthetic: цена упала до нуля (-100% за день) -> рост 0 -> CAGR = -100%
    assert core.annualized_return(pd.Series([0.0, -1.0])) == pytest.approx(-1.0)


def test_benchmark_beta_constant_equals_beta_of_series_with_itself():
    # synthetic: константа BENCHMARK_BETA = beta(x, x) для любого ряда с разбросом
    assert core.BENCHMARK_BETA == pytest.approx(core.beta(SYNTHETIC_RETURNS, SYNTHETIC_RETURNS))


# ======================================================================
# G. Проверка входных данных и подготовка для экрана
# ======================================================================

TODAY = date(2026, 10, 7)


def test_default_period_is_last_three_years():
    assert core.default_period(TODAY) == (date(2023, 10, 7), TODAY)
    # 29 февраля минус 3 года -> 28 февраля
    assert core.default_period(date(2028, 2, 29)) == (date(2025, 2, 28), date(2028, 2, 29))


def test_validate_period_accepts_good_dates():
    core.validate_period(date(2023, 10, 7), TODAY, TODAY)
    core.validate_period(date(2015, 1, 1), TODAY, TODAY)  # граница включена


def test_validate_period_rejects_bad_dates():
    with pytest.raises(ValueError, match="earlier than the end"):
        core.validate_period(date(2024, 5, 1), date(2024, 5, 1), TODAY)
    with pytest.raises(ValueError, match="earlier than the end"):
        core.validate_period(date(2025, 5, 1), date(2024, 5, 1), TODAY)
    with pytest.raises(ValueError, match="2015-01-01"):
        core.validate_period(date(2014, 12, 31), date(2020, 1, 1), TODAY)
    with pytest.raises(ValueError, match="future"):
        core.validate_period(date(2025, 1, 1), date(2026, 10, 8), TODAY)


def test_parse_advertised_pct_accepts_numbers():
    assert core.parse_advertised_pct("35") == pytest.approx(35)
    assert core.parse_advertised_pct(" 35,5 ") == pytest.approx(35.5)
    assert core.parse_advertised_pct("12%") == pytest.approx(12)
    assert core.parse_advertised_pct("0") == pytest.approx(0)
    assert core.parse_advertised_pct("1000") == pytest.approx(1000)


def test_parse_advertised_pct_rejects_bad_input():
    for bad in ["", "abc", "nan", "inf", "-5", "1000.5", "5000"]:
        with pytest.raises(ValueError):
            core.parse_advertised_pct(bad)


def test_chart_values_and_range():
    # цены 3, 7 -> рост 100 и 233.33 (округление до 2 знаков)
    values = core.chart_values(pd.Series([3.0, 7.0]))
    assert values == [100.0, 233.33]
    assert core.chart_range([100.0, 120.0], [90.0, 110.0]) == (90.0, 120.0)



# ======================================================================
# H. Панель "Why this number?" (core.explain_gap)
# ======================================================================

def test_explain_gap_when_ad_is_overstated():
    # реклама 0.30, реально 0.10, gap 0.20 (расчёт - в комментарии 11 вверху файла)
    result = core.explain_gap(0.30, 0.10, 0.20, 1.5, 0.04)
    assert result["direction"] == "overstated"
    assert result["points_signed"] == pytest.approx(HAND_EXPLAIN_POINTS_OVERSTATED, abs=1e-9)
    assert result["beta_level"] == "higher"
    assert result["alpha_level"] == "above"
    # числа переданы без изменений: на экране они совпадут с таблицей
    assert (result["advertised"], result["actual"], result["beta"], result["alpha"]) == (0.30, 0.10, 1.5, 0.04)


def test_explain_gap_when_fund_beat_the_ad_matches_compare_claim():
    # реклама 0.10, реально 0.30, gap -0.20 (расчёт - в комментарии 12 вверху файла)
    result = core.explain_gap(0.10, 0.30, -0.20, 0.5, -0.02)
    assert result["direction"] == "beat"
    assert result["points_signed"] == pytest.approx(HAND_EXPLAIN_POINTS_BEATEN, abs=1e-9)
    assert result["beta_level"] == "lower"
    assert result["alpha_level"] == "below"
    # synthetic: то же число, что в главном блоке (compare_claim из пункта 10 вверху файла)
    claim = core.compare_claim(0.50, RETURNS_126_UP, RETURNS_126_UP, rf_annual=0.0)
    again = core.explain_gap(claim["advertised"], claim["actual"], claim["gap"], claim["beta"], claim["alpha"])
    assert again["points_signed"] == claim["headline"]["points_signed"]
    assert again["points_signed"] == pytest.approx(HAND_GAP_OVERSTATED * 100, abs=1e-7)
    assert again["direction"] == claim["headline"]["direction"]


def test_explain_gap_edge_cases():
    # synthetic: разрыв 0.00004 (0.004 п.п.) на экране выглядит как 0.00 -> "matched"
    assert core.explain_gap(0.2, 0.19996, 0.00004, 1.0, 0.0)["direction"] == "matched"
    # бета в пределах 1 +- 0.05 -> "similar"; альфа меньше 0.005% -> "zero"
    near = core.explain_gap(0.2, 0.2, 0.0, 1.03, 0.00001)
    assert near["beta_level"] == "similar" and near["alpha_level"] == "zero"
    # заметно отрицательная бета (-0.4 < -0.05) -> "opposite"; бета -0.02 - это шум, не "против бенчмарка"
    assert core.explain_gap(0.2, 0.1, 0.1, -0.4, 0.01)["beta_level"] == "opposite"
    assert core.explain_gap(0.2, 0.1, 0.1, -0.02, 0.01)["beta_level"] == "lower"
    # бета не определена (бенчмарк не менялся) -> обе оценки "unknown", без ошибки
    unknown = core.explain_gap(0.2, 0.1, 0.1, None, None)
    assert unknown["beta_level"] == "unknown" and unknown["alpha_level"] == "unknown"


# ======================================================================
# H. Deep analysis: что не хватает до зелёного (только история, не прогноз)
# ======================================================================

def test_honest_limit_hand_calculated():
    assert core.honest_advertised_limit_pct(0.0712) == pytest.approx(HAND_HONEST_7_12)
    assert core.honest_advertised_limit_pct(0.10) == pytest.approx(HAND_HONEST_10)
    assert core.honest_advertised_limit_pct(0.0999) == pytest.approx(HAND_HONEST_9_99)


def test_honest_limit_edge_cases():
    assert core.honest_advertised_limit_pct(-0.0001) is None      # отрицательный CAGR: честной цифры нет
    assert core.honest_advertised_limit_pct(-0.5) is None
    assert core.honest_advertised_limit_pct(0.0) == pytest.approx(0.0)   # ровно 0%: честная цифра 0.0
    assert core.honest_advertised_limit_pct(0.0004) == pytest.approx(0.0)  # 0.04% вниз до шага 0.1 это 0.0
    assert core.honest_advertised_limit_pct(15.0) == pytest.approx(core.MAX_ADVERTISED_PCT)  # 1500% не выше предела


def test_honest_limit_never_counts_as_overstated():
    # synthetic: для любой цифры из списка "честная" реклама не даёт направления "overstated"
    for cagr in [0.0, 0.0004, 0.0712, 0.0999, 0.10, 0.2798, 1.2345]:
        limit = core.honest_advertised_limit_pct(cagr)
        gap = core.percent_to_decimal(limit) - cagr
        assert core.gap_headline(gap)["direction"] != "overstated"


def test_sharpe_shortfall_hand_calculated():
    returns = pd.Series([0.01, 0.03])
    sharpe_fund = core.sharpe_ratio(returns, 0.0)
    result = core.sharpe_shortfall(returns, 0.0, sharpe_fund, 30.0)
    assert result["status"] == "ok" and result["reason"] == ""
    assert result["sharpe_gap"] == pytest.approx(HAND_SHORTFALL_GAP, abs=1e-4)
    assert result["required_arith_return"] == pytest.approx(HAND_SHORTFALL_REQUIRED, abs=1e-4)
    assert result["return_shortfall_pp"] == pytest.approx(HAND_SHORTFALL_PP, abs=1e-2)
    assert result["max_volatility"] == pytest.approx(HAND_SHORTFALL_MAX_VOL, abs=1e-9)
    assert result["volatility_cut_pp"] == pytest.approx(HAND_SHORTFALL_CUT_PP, abs=1e-3)
    # проверка смысла: при нужной доходности и той же волатильности Sharpe фонда равен Sharpe бенчмарка (30)
    assert (result["required_arith_return"] - 0.0) / result["volatility"] == pytest.approx(30.0, abs=1e-9)


def test_sharpe_shortfall_return_below_risk_free_has_no_volatility_fix():
    returns = pd.Series([0.0001, 0.0003])
    sharpe_fund = core.sharpe_ratio(returns, 0.10)
    result = core.sharpe_shortfall(returns, 0.10, sharpe_fund, 1.0)
    assert result["status"] == "ok"
    assert result["required_arith_return"] == pytest.approx(HAND_LOW_REQUIRED, abs=1e-6)
    assert result["return_shortfall_pp"] == pytest.approx(HAND_LOW_SHORTFALL_PP, abs=1e-3)
    assert result["max_volatility"] is None and result["volatility_cut_pp"] is None
    assert result["max_volatility_reason"] == "return_not_above_risk_free"


def test_sharpe_shortfall_edge_cases():
    returns = pd.Series([0.01, 0.03])
    # synthetic: постоянный ряд, нулевая волатильность -> причина, без падения
    constant = core.sharpe_shortfall(pd.Series([0.01] * 5), 0.0, None, 1.0)
    assert constant["status"] == "unavailable" and constant["reason"] == "no_volatility"
    assert constant["required_arith_return"] is None
    # Sharpe бенчмарка <= 0: планка не положительна
    for bench_sharpe in (0.0, -0.5):
        result = core.sharpe_shortfall(returns, 0.0, -1.0, bench_sharpe)
        assert result["status"] == "unavailable" and result["reason"] == "benchmark_sharpe_not_positive"
        assert result["return_shortfall_pp"] is None
    # фонд уже оправдывает риск: показывать нечего
    assert core.sharpe_shortfall(returns, 0.0, 2.0, 1.0)["status"] == "not_needed"
    # Sharpe не посчитан
    assert core.sharpe_shortfall(returns, 0.0, None, 1.0)["reason"] == "sharpe_missing"
    assert core.sharpe_shortfall(returns, 0.0, 1.0, None)["reason"] == "sharpe_missing"
    # одна точка данных
    assert core.sharpe_shortfall(pd.Series([0.01]), 0.0, None, 1.0)["reason"] == "not_enough_data"


def test_closeness_thresholds():
    assert core.closeness(1.27, 1.10) == "clear"        # разница 0.17 >= 0.10
    assert core.closeness(0.93, 0.94) == "too_close"    # разница 0.01 < 0.10
    assert core.closeness(0.94, 0.93) == "too_close"    # знак не важен
    # ровно 0.10 по определению порога уже "clear" (строго меньше порога = "too_close");
    # 1.0 - 0.9 в float даёт 0.09999999999999998, поэтому разница округляется до 9 знаков
    assert core.closeness(1.5, 1.4) == "clear"
    assert core.closeness(1.0, 0.9) == "clear"
    assert core.closeness(1.0, 0.91) == "too_close"
    assert core.closeness(None, 1.0) == "unknown" and core.closeness(1.0, None) == "unknown"


def test_window_shortening_days_hand_calculated():
    assert core.window_shortening_days(date(2005, 1, 1), date(2014, 10, 31)) == HAND_SHORTENED_DAYS
    # допуск 7 дней на выходные и праздники: не сокращение
    assert core.window_shortening_days(date(2024, 1, 1), date(2024, 1, 3)) == 0
    assert core.window_shortening_days(date(2024, 1, 1), date(2024, 1, 8)) == 0   # ровно 7 дней
    assert core.window_shortening_days(date(2024, 1, 1), date(2024, 1, 9)) == 8   # 8 дней: уже сокращение
    # ряд начался раньше выбранной даты: сокращения нет
    assert core.window_shortening_days(date(2024, 1, 1), date(2000, 1, 3)) == 0


def test_validate_period_with_new_minimum_from_parameter():
    new_min = date(2000, 1, 1)
    core.validate_period(date(2000, 1, 1), TODAY, TODAY, min_date=new_min)      # граница включена
    core.validate_period(date(2005, 3, 1), TODAY, TODAY, min_date=new_min)      # раньше 2015, но не раньше 2000
    with pytest.raises(ValueError, match="2000-01-01"):
        core.validate_period(date(1999, 12, 31), TODAY, TODAY, min_date=new_min)
    # без параметра остаётся прежний минимум (старые вызовы не ломаются)
    with pytest.raises(ValueError, match="2015-01-01"):
        core.validate_period(date(2005, 3, 1), TODAY, TODAY)


# ---- Ближайшие окна дат ----

WIN_INDEX = pd.bdate_range("2024-01-01", periods=13)          # synthetic: дата-основа и 12 дат доходностей
WIN_DATES = WIN_INDEX[1:]
WIN_BENCH = pd.Series([0.01, -0.01] * 6, index=WIN_DATES)     # synthetic: +0.01, -0.01, ...
WIN_RF = pd.Series([0.0] * 6 + [2.52] * 7, index=WIN_INDEX)   # synthetic: 0% на первых 6 датах, 2.52% дальше


def win_fund(shifts):
    """synthetic: фонд = бенчмарк + поправка по дням (список из 12 чисел)."""
    return WIN_BENCH + pd.Series(shifts, index=WIN_DATES)


def test_nearest_windows_smallest_shift_and_rf_averaged_over_window():
    fund = win_fund([-0.03] * 6 + [0.01] * 6)
    windows = core.nearest_green_windows(fund, WIN_BENCH, WIN_RF, step_days=3, min_days=4,
                                         start_date=WIN_INDEX[0].date())
    assert len(windows) == 1                      # сдвиг на 3 дня красный, на 6 дней зелёный; назад зелёных нет
    window = windows[0]
    assert window["direction"] == "later_start"
    assert window["start"] == WIN_INDEX[6].date() and window["end"] == WIN_INDEX[12].date()
    assert window["shift_days"] == HAND_WINDOW_SHIFT_DAYS == (WIN_INDEX[6] - WIN_INDEX[0]).days
    # ставка усреднена по окну (2.52%), а не по всему периоду (1.36%): иначе Sharpe был бы другим
    assert window["sharpe_fund"] == pytest.approx(HAND_WINDOW_SHARPE_FUND, abs=1e-3)
    assert window["sharpe_bench"] == pytest.approx(HAND_WINDOW_SHARPE_BENCH, abs=1e-4)
    assert window["verdict"] == "justifies"


def test_nearest_windows_skip_windows_shorter_than_min_days():
    # synthetic: зелёным становится только окно из последних 3 доходностей
    fund = win_fund([-0.03] * 9 + [0.02] * 3)
    assert core.nearest_green_windows(fund, WIN_BENCH, WIN_RF, step_days=3, min_days=4,
                                      start_date=WIN_INDEX[0].date()) == []
    allowed = core.nearest_green_windows(fund, WIN_BENCH, WIN_RF, step_days=3, min_days=3,
                                         start_date=WIN_INDEX[0].date())
    assert len(allowed) == 1 and allowed[0]["shift_days"] == HAND_WINDOW_SHORT_SHIFT_DAYS
    assert allowed[0]["start"] == WIN_INDEX[9].date()


def test_nearest_windows_empty_when_nothing_to_show():
    # уже зелёный результат при поиске зелёных окон: пустой список
    green = win_fund([0.03] * 12)
    assert core.nearest_green_windows(green, WIN_BENCH, WIN_RF, step_days=3, min_days=4) == []
    # красный везде: зелёных окон нет
    red = win_fund([-0.03] * 12)
    assert core.nearest_green_windows(red, WIN_BENCH, WIN_RF, step_days=3, min_days=4) == []
    # слишком мало данных для окон нужной длины
    assert core.nearest_green_windows(red, WIN_BENCH, WIN_RF, step_days=3, min_days=20) == []


def test_nearest_windows_red_target_shows_fragility_of_green_result():
    # synthetic: общий результат зелёный; окно из последних 6 дней красное, из последних 9 ещё зелёное
    fund = win_fund([0.03] * 6 + [-0.01] * 6)
    windows = core.nearest_green_windows(fund, WIN_BENCH, WIN_RF, step_days=3, min_days=4,
                                         target_verdict="does_not_justify", start_date=WIN_INDEX[0].date())
    assert len(windows) == 1
    assert windows[0]["direction"] == "later_start" and windows[0]["shift_days"] == HAND_WINDOW_SHIFT_DAYS
    assert windows[0]["verdict"] == "does_not_justify"
    # если результат уже красный, красные окна не ищем
    assert core.nearest_green_windows(win_fund([-0.03] * 12), WIN_BENCH, WIN_RF, step_days=3, min_days=4,
                                      target_verdict="does_not_justify") == []


def test_nearest_windows_reject_bad_arguments():
    fund = win_fund([0.0] * 12)
    with pytest.raises(ValueError):
        core.nearest_green_windows(fund, WIN_BENCH, WIN_RF, step_days=0)
    with pytest.raises(ValueError):
        core.nearest_green_windows(fund, WIN_BENCH, WIN_RF, min_days=1)


def long_series():
    """synthetic: 400 доходностей; бенчмарк со средним ровно 0.001 в день и шумом с нулевым средним."""
    index = pd.bdate_range("2023-01-02", periods=401)
    noise = np.random.RandomState(3).normal(0, 0.006, 400)
    bench = pd.Series(0.001 + noise - noise.mean(), index=index[1:])
    return index, bench


def test_nearest_windows_are_reproducible_from_their_dates():
    # Свойство: данные окна, посчитанные заново по его датам (как это сделает приложение по ссылке "Try this
    # window"), дают те же Sharpe, которые показаны рядом со ссылкой. Ставка меняется во времени (1% -> 5%).
    index, bench = long_series()
    fund = bench + pd.Series([-0.003] * 240 + [0.003] * 160, index=index[1:])
    rf = pd.Series(np.linspace(1.0, 5.0, 401), index=index)
    rf_whole = core.risk_free_annual(rf)
    assert core.verdict_code(core.justifies_risk(core.sharpe_ratio(fund, rf_whole),
                                                 core.sharpe_ratio(bench, rf_whole))) == "does_not_justify"
    windows = core.nearest_green_windows(fund, bench, rf, start_date=index[0].date())
    assert windows and windows[0]["direction"] == "later_start"
    for window in windows:
        start, end = pd.Timestamp(window["start"]), pd.Timestamp(window["end"])
        fund_part = fund[(fund.index > start) & (fund.index <= end)]     # доходности после даты-основы
        bench_part = bench[(bench.index > start) & (bench.index <= end)]
        rf_window = core.risk_free_annual(rf.loc[start:end])
        assert core.sharpe_ratio(fund_part, rf_window) == pytest.approx(window["sharpe_fund"], abs=1e-9)
        assert core.sharpe_ratio(bench_part, rf_window) == pytest.approx(window["sharpe_bench"], abs=1e-9)
        assert len(fund_part) >= core.WINDOW_MIN_DAYS
        assert window["verdict"] == "justifies"


def test_deep_analysis_collects_data_for_each_verdict():
    index, bench = long_series()
    rf = pd.Series(np.full(401, 4.5), index=index)
    rf_annual = core.risk_free_annual(rf)

    # красный результат: есть нехватка, зелёные окна, а реклама приукрашена (CAGR положительный)
    red_fund = bench + pd.Series([-0.003] * 240 + [0.003] * 160, index=index[1:])
    red = core.deep_analysis(core.compare_claim(0.35, red_fund, bench, rf_annual), red_fund, bench, rf,
                             rf_annual, start_date=index[0].date())
    assert red["verdict"] == "does_not_justify" and red["overstated"] is True
    assert red["shortfall"]["status"] == "ok"
    assert red["windows"] and all(w["verdict"] == "justifies" for w in red["windows"])
    assert red["honest_limit_pct"] is not None and red["sharpe_margin"] < 0
    assert red["sharpe_margin_abs"] == pytest.approx(-red["sharpe_margin"])

    # зелёный результат: нехватки нет, окна красные, "честная" цифра не нужна (реклама ниже реальности)
    green_fund = bench + pd.Series([0.003] * 240 + [-0.003] * 160, index=index[1:])
    green = core.deep_analysis(core.compare_claim(0.05, green_fund, bench, rf_annual), green_fund, bench, rf,
                               rf_annual, start_date=index[0].date())
    assert green["verdict"] == "justifies" and green["shortfall"] is None and green["honest_limit_pct"] is None
    assert green["windows"] and all(w["verdict"] == "does_not_justify" for w in green["windows"])
    assert green["sharpe_margin"] > 0

    # Sharpe бенчмарка не посчитать (бенчмарк не менялся): вердикта нет, всё пусто, без падения
    flat = pd.Series(0.001, index=index[1:])
    unknown = core.deep_analysis(core.compare_claim(0.35, red_fund, flat, rf_annual), red_fund, flat, rf,
                                 rf_annual, start_date=index[0].date())
    assert unknown["verdict"] == "unknown" and unknown["shortfall"] is None and unknown["windows"] == []
    assert unknown["closeness"] == "unknown" and unknown["sharpe_margin"] is None
