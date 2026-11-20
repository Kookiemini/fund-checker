"""Дополнительные расчёты для сайта: инфляция, "сколько докладывать", смесь фондов, комиссии фондов.

Здесь только расчёты (без Flask и без скачивания). Данные приходят из data.py,
готовые формулы доходности, риска и Sharpe берутся из core.py и rule.py.
"""
import numpy as np
import pandas as pd

import core
import forecast
import rule

# ---------- Комиссия фонда (expense ratio) ----------
# Годовая плата фонду в процентах, из открытых данных фондов (сайты провайдеров и справочники ETF),
# проверено 2026-11-20. Комиссии иногда меняются: перед покупкой смотрите сайт фонда.
# Важно: эта плата УЖЕ вычтена из цены фонда (Adjusted Close), поэтому доходности на сайте - после неё.
EXPENSE_RATIOS_CHECKED = "2026-11-20"
EXPENSE_RATIOS = {
    "ARKK": 0.75, "ARKW": 0.76, "ARKG": 0.75, "QQQ": 0.18, "ICLN": 0.39, "SMH": 0.35, "TAN": 0.70,
    "LIT": 0.75, "GLD": 0.40, "SCHD": 0.06, "EEM": 0.72, "VTI": 0.03, "VUG": 0.03, "XLV": 0.08,
    "VNQ": 0.13, "TLT": 0.15, "SPY": 0.0945, "IWM": 0.19, "ACWI": 0.32, "AGG": 0.03,
}


def expense_ratio(ticker: str):
    """Комиссия фонда в процентах в год или None, если её нет в таблице."""
    return EXPENSE_RATIOS.get(ticker)


def fee_cost(fee_pct: float, amount: float, years: int) -> float:
    """Сколько примерно стоит комиссия за years лет на сумму amount (без роста денег, для наглядности)."""
    return amount * fee_pct / 100 * years


# ---------- Инфляция ----------

def yearly_inflation(cpi: pd.Series, first, last):
    """Средняя инфляция в год между двумя датами по индексу цен CPI (месячный ряд FRED).

    Берём значение индекса на месяц первой и последней даты (или ближайшее раньше).
    Возвращает (инфляция в год как доля, дата начала, дата конца) или None, если данных не хватает.
    """
    cpi = cpi.dropna().sort_index()
    first, last = pd.Timestamp(first), pd.Timestamp(last)
    before_first, before_last = cpi.loc[:first], cpi.loc[:last]
    if before_first.empty or before_last.empty:
        return None
    d0, d1 = before_first.index[-1], before_last.index[-1]
    years = (d1 - d0).days / 365.25
    if years < 0.5:  # меньше полугода - средняя инфляция в год ненадёжна
        return None
    rate = (float(before_last.iloc[-1]) / float(before_first.iloc[-1])) ** (1 / years) - 1
    return rate, d0.date(), d1.date()


def real_return(nominal: float, inflation: float) -> float:
    """Доходность после инфляции: во сколько раз выросла покупательная способность, минус 1."""
    return (1 + nominal) / (1 + inflation) - 1


def todays_money(value: float, inflation: float, years: float) -> float:
    """Сумма через years лет в сегодняшних деньгах при средней инфляции inflation в год."""
    return value / (1 + inflation) ** years


# ---------- "Сколько нужно докладывать каждый месяц?" ----------
# Пополнение add входит в итог линейно: итог = сумма * G + add * A,
# где G - во сколько раз выросли деньги за весь срок, A - сумма ростов каждого пополнения.
# Поэтому для каждого сценария можно точно посчитать, сколько нужно докладывать, чтобы дойти до цели.

CHANCES = (0.5, 0.75, 0.9)
YEARS_MAX = 40
GOAL_MAX = 1_000_000_000.0


def growth_factors(monthly: np.ndarray):
    """G (рост первой суммы) и A (сумма ростов пополнений в конце каждого месяца) для каждого сценария."""
    rest = np.cumprod((1 + monthly)[:, ::-1], axis=1)[:, ::-1]   # рост от месяца t до конца срока
    grown = rest[:, 0]
    adds = np.concatenate([rest[:, 1:], np.ones((monthly.shape[0], 1))], axis=1).sum(axis=1)
    return grown, adds


def steady_add(amount: float, goal: float, pct: float, months: int) -> float:
    """Сколько докладывать в месяц, если деньги растут ровно на pct% в год (без колебаний)."""
    rate = (1 + pct / 100) ** (1 / 12) - 1
    if abs(rate) < 1e-12:
        return max((goal - amount) / months, 0.0)
    grow = (1 + rate) ** months
    return max((goal - amount * grow) * rate / (grow - 1), 0.0)


def required_add(returns: pd.Series, amount: float, goal: float, years: int, pct: float,
                 chances=CHANCES, paths: int = forecast.PATHS) -> dict:
    """Сколько докладывать в месяц, чтобы дойти до goal за years лет с шансом 50%, 75% и 90%."""
    months = int(years) * 12
    grown, adds = growth_factors(forecast.sample_returns(returns, months, paths))
    need = np.maximum((goal - amount * grown) / adds, 0.0)        # нужное пополнение в каждом сценарии
    return {
        "goal": goal, "years": int(years), "amount": amount,
        "adds": {str(int(c * 100)): float(np.quantile(need, c)) for c in chances},
        "chance_without": float((amount * grown >= goal).mean()),  # дойти без пополнений
        "steady": steady_add(amount, goal, pct, months),
        "pct": pct,
    }


# ---------- Смесь фондов (портфель) ----------
# Каждый месяц доли возвращаются к выбранным (ребалансировка в начале месяца),
# внутри месяца каждая часть растёт вместе со своим фондом.

MIX_MIN, MIX_MAX = 2, 3


def mix_returns(prices: pd.DataFrame, weights: dict) -> pd.Series:
    """Дневные доходности смеси фондов с ребалансировкой в начале каждого месяца."""
    returns = prices.pct_change().dropna()
    w = np.array([weights[c] for c in prices.columns], dtype=float)
    parts = []
    for _, block in returns.groupby(returns.index.to_period("M")):
        value = np.cumprod(1 + block.values, axis=0) @ w          # стоимость смеси с начала месяца
        prev = np.concatenate([[w.sum()], value[:-1]])
        parts.append(pd.Series(value / prev - 1, index=block.index))
    return pd.concat(parts)


def stats(returns: pd.Series, rf_annual: float) -> dict:
    """Те же показатели, что в списке фондов (формулы core.py и rule.py)."""
    return {
        "cagr": core.annualized_return(returns),
        "volatility": core.annualized_volatility(returns),
        "sharpe": core.sharpe_ratio(returns, rf_annual),
        "max_drawdown": rule.max_drawdown(returns),
    }


def monthly_growth(returns: pd.Series) -> pd.Series:
    """Во что превратились 100 USD: значение на конец каждого месяца (для графика)."""
    wealth = 100 * (1 + returns).cumprod()
    return wealth.groupby(wealth.index.to_period("M")).last()


def mix(prices: pd.DataFrame, weights: dict, rf_annual: float) -> dict:
    """Смесь и каждый фонд по отдельности на одних и тех же датах."""
    weights = {k: v / 100 for k, v in weights.items()}
    mixed = mix_returns(prices, weights)
    singles = {c: prices[c].pct_change().dropna() for c in prices.columns}
    growth = {"Mix": monthly_growth(mixed)}
    growth.update({c: monthly_growth(r) for c, r in singles.items()})
    fee = sum(weights[c] * (expense_ratio(c) or 0.0) for c in prices.columns)
    return {
        "first": prices.index[0].date().isoformat(), "last": prices.index[-1].date().isoformat(),
        "days": len(mixed),
        "mix": dict(stats(mixed, rf_annual), fee=fee),
        "funds": {c: dict(stats(r, rf_annual), fee=expense_ratio(c)) for c, r in singles.items()},
        "chart": {"dates": [str(p) for p in growth["Mix"].index],
                  "series": {k: [round(float(v), 2) for v in s.values] for k, s in growth.items()}},
    }


def parse_mix(funds_text: str, weights_text: str, allowed) -> dict:
    """Проверка формы смеси: 2-3 разных фонда из списка, доли в процентах, в сумме 100."""
    funds = [f.strip().upper() for f in funds_text.split(",") if f.strip()]
    try:
        weights = [float(w) for w in weights_text.split(",") if w.strip()]
    except ValueError:
        raise ValueError("Shares must be numbers, e.g. 60 and 40.") from None
    if not (MIX_MIN <= len(funds) <= MIX_MAX) or len(funds) != len(weights):
        raise ValueError("Choose 2 or 3 funds and a share for each.")
    if len(set(funds)) != len(funds):
        raise ValueError("Choose different funds.")
    if any(f not in allowed for f in funds):
        raise ValueError("Please choose funds from the list.")
    if any(not (0 < w <= 100) or np.isnan(w) for w in weights):
        raise ValueError("Each share must be more than 0% and at most 100%.")
    if abs(sum(weights) - 100) > 0.01:
        raise ValueError(f"The shares must add up to 100% (now {sum(weights):g}%).")
    return dict(zip(funds, weights))

