"""rule.py - ядро бэктеста правила (Assignment 2: Rule Lab). Только расчёты, без экрана и файлов.

Правило (Rule Card, R3-R5 команда описывает своими словами):
- данные помесячные: цена закрытия последнего торгового дня месяца (Adjusted Close, с дивидендами);
- сигнал на закрытии месяца t: 1, если цена выше простой средней за N месяцев (включая месяц t), иначе 0;
- позиция в месяце t+1 = сигнал месяца t (СДВИГ НА ОДИН МЕСЯЦ: решение принимается только по тому,
  что уже известно на закрытии месяца t);
- позиция 1: держим SPY и получаем его доходность за месяц t+1;
  позиция 0: деньги в кэше и получают безрисковую ставку (FRED DGS3MO на закрытии месяца t, / 12);
- издержки: c за каждую единицу изменения позиции (вход 0 -> 1 или выход 1 -> 0 стоит c).

Бенчмарк: "купить и держать" SPY за тот же месяц, тот же период, без издержек.

Правила этого файла те же, что у core.py: никакого интернета, файлов и print; одинаковый вход -> одинаковый выход.
Все доходности и ставки - ДОЛИ (0.05 = 5%), кроме входной ставки FRED (в процентах годовых).
"""
import math
from statistics import NormalDist

import numpy as np
import pandas as pd

# ---------- Замороженные настройки правила (G2). Любое изменение - запись в DECISIONS.md и новая версия ----------
RULE_VERSION = "v1"
ASSET = "SPY"
PARAM_MAIN = 10                      # N месяцев: значение из литературы (правило 10-месячной средней), не подобрано
PARAM_ALLOWED = (6, 14)              # допустимый диапазон N (Rule Card R7)
NEIGHBOURS = [8, 9, 10, 11, 12]      # соседи: N +- 1 и +- 2 шага (E5)
DATA_START = "2000-01-01"            # первые 12 месяцев уходят на разогрев средней
TRAIN = ("2001-01", "2012-12")       # разработка: обучение
VALIDATION = ("2013-01", "2018-12")  # разработка: проверка
TEST = ("2019-01", "2026-10")        # финальный тест: открывается ОДИН раз
COSTS = [0.0, 0.001, 0.0025, 0.005]  # c за единицу изменения позиции: 0%, 0.10%, 0.25%, 0.50%
COST_DEFAULT = 0.001                 # что показывает Rule Lab по умолчанию
BOOTSTRAP_BLOCK = 12                 # длина блока бутстрепа, месяцев
BOOTSTRAP_REPS = 5000                # число повторов бутстрепа
BOOTSTRAP_SEED = 20261114            # фиксированный seed: одинаковый результат при каждом запуске
CONFIDENCE = 0.95                    # уровень интервалов
BREAKEVEN_MAX = 0.10                 # break-even ищем в диапазоне c от 0 до 10% за переключение

PERIODS = 12                         # месяцев в году
EULER_GAMMA = 0.5772156649


# ---------- 1. Помесячные данные ----------

def monthly_prices(daily: pd.Series) -> pd.Series:
    """Цена закрытия последнего торгового дня каждого ПОЛНОГО месяца.

    Последний месяц выбрасываем, если он ещё не закончился (иначе "месяц" был бы обрезанным).
    """
    daily = daily.dropna()
    monthly = daily.resample("ME").last().dropna()
    last_day = daily.index.max()
    month_last_business_day = last_day + pd.offsets.BMonthEnd(0)
    if last_day < month_last_business_day:  # месяц последней даты не закончен
        monthly = monthly.iloc[:-1]
    return monthly


def monthly_rate(daily_percent: pd.Series) -> pd.Series:
    """Ставка FRED (% годовых) на последний день каждого месяца, когда она была опубликована."""
    return daily_percent.dropna().resample("ME").last().dropna()


# ---------- 2. Сигнал, позиция, доходность стратегии ----------

def signal(prices: pd.Series, n: int) -> pd.Series:
    """Сигнал на закрытии месяца t: 1.0, если цена выше средней за n месяцев (включая t), иначе 0.0.

    Пока средняя не посчитана (первые n-1 месяцев), сигнала нет (NaN).
    """
    sma = prices.rolling(n).mean()
    return (prices > sma).astype(float).where(sma.notna())


def position(sig: pd.Series) -> pd.Series:
    """Позиция в месяце t+1 = сигнал месяца t. Это и есть защита от заглядывания в будущее."""
    return sig.shift(1)


def backtest(prices: pd.Series, rate_percent: pd.Series, n: int, cost: float = 0.0) -> pd.DataFrame:
    """Помесячная таблица правила и бенчмарка.

    prices        - помесячные цены (monthly_prices)
    rate_percent  - ставка на конец месяца, % годовых (monthly_rate)
    Колонки: price, signal, position, asset_return, cash_return, turnover, gross, net, bench, rf.
    Строки без позиции (разогрев средней) выброшены.
    """
    table = pd.DataFrame({"price": prices})
    table["signal"] = signal(prices, n)
    table["position"] = position(table["signal"])
    table["asset_return"] = prices.pct_change()
    # кэш в месяце t+1 получает ставку, известную на закрытии месяца t
    rate = rate_percent.reindex(table.index, method="ffill")
    table["rf"] = rate.shift(1) / 100 / PERIODS
    table["cash_return"] = table["rf"]
    # единицы изменения позиции; первая позиция после разогрева считается без издержек
    table["turnover"] = table["position"].diff().abs().fillna(0.0)
    table["gross"] = table["position"] * table["asset_return"] + (1 - table["position"]) * table["cash_return"]
    table["net"] = table["gross"] - cost * table["turnover"]
    table["bench"] = table["asset_return"]
    return table.dropna(subset=["position", "asset_return", "rf"])


def window(table: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Строки с месяца start по месяц end включительно ('YYYY-MM')."""
    return table.loc[start:end]


# ---------- 3. Показатели (помесячные доходности -> годовые) ----------

def cagr(returns: pd.Series) -> float:
    """Годовой темп роста: (произведение (1 + r)) ^ (12 / число месяцев) - 1."""
    return float(np.prod(1 + returns.values) ** (PERIODS / len(returns)) - 1)


def volatility(returns: pd.Series) -> float:
    """Годовая волатильность: стандартное отклонение месячных доходностей * sqrt(12)."""
    return float(returns.std(ddof=1) * math.sqrt(PERIODS))


def sharpe(returns: pd.Series, rf: pd.Series):
    """Годовой Sharpe: среднее (r - rf) / стандартное отклонение (r - rf) * sqrt(12). None, если разброса нет."""
    excess = returns - rf
    sd = excess.std(ddof=1)
    if not sd > 1e-12:
        return None
    return float(excess.mean() / sd * math.sqrt(PERIODS))


def sharpe_se(sharpe_annual, months: int):
    """Стандартная ошибка годового Sharpe (Lo, 2002, iid): sqrt((1 + SR_m^2 / 2) / T) * sqrt(12),
    где SR_m - месячный Sharpe, T - число месяцев."""
    if sharpe_annual is None:
        return None
    monthly = sharpe_annual / math.sqrt(PERIODS)
    return math.sqrt((1 + 0.5 * monthly ** 2) / months) * math.sqrt(PERIODS)


def max_drawdown(returns: pd.Series) -> float:
    """Максимальная просадка: худшее падение стоимости от предыдущего максимума (отрицательное число).
    Стартовая стоимость 1 тоже считается максимумом."""
    wealth = np.concatenate([[1.0], np.cumprod(1 + returns.values)])
    peaks = np.maximum.accumulate(wealth)
    return float((wealth / peaks - 1).min())


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Просадка на конец каждого месяца (для графика)."""
    wealth = (1 + returns).cumprod()
    return wealth / np.maximum(wealth.cummax(), 1.0) - 1


def var_5(returns: pd.Series) -> float:
    """Исторический VaR 5%: месячная доходность, хуже которой было 5% месяцев (отрицательное число)."""
    return float(np.quantile(returns.values, 0.05))


def summary(part: pd.DataFrame, column: str) -> dict:
    """E1 и E8 для одной колонки ('net', 'gross' или 'bench') на уже вырезанном окне."""
    returns, rf = part[column], part["rf"]
    is_rule = column != "bench"
    months = len(returns)
    sr = sharpe(returns, rf)
    worst = returns.idxmin()
    return {
        "cagr": cagr(returns),
        "volatility": volatility(returns),
        "sharpe": sr,
        "sharpe_se": sharpe_se(sr, months),
        "max_drawdown": max_drawdown(returns),
        "time_in_market": float(part["position"].mean()) if is_rule else 1.0,
        "switches": int((part["turnover"] > 0).sum()) if is_rule else 0,
        "turnover_per_year": float(part["turnover"].sum() / (months / PERIODS)) if is_rule else 0.0,
        "worst_month": float(returns.min()),
        "worst_month_date": worst.strftime("%Y-%m"),
        "var_5": var_5(returns),
        "months": months,
    }


# ---------- 4. Разница Sharpe: интервалы (E2, E4) ----------

def _z():
    return NormalDist().inv_cdf(0.5 + CONFIDENCE / 2)


def sharpe_diff_analytic(part: pd.DataFrame, column: str = "net") -> dict:
    """Парный аналитический интервал разницы Sharpe (Jobson-Korkie с поправкой Memmel, 2003):
    Var(SR1 - SR2) = (2 - 2*rho + (SR1^2 + SR2^2 - 2*SR1*SR2*rho^2) / 2) / T  (месячные SR), затем * 12."""
    ex1, ex2 = part[column] - part["rf"], part["bench"] - part["rf"]
    s1, s2 = sharpe(part[column], part["rf"]), sharpe(part["bench"], part["rf"])
    if s1 is None or s2 is None:
        return {"diff": None, "low": None, "high": None, "method": "analytic"}
    m1, m2 = s1 / math.sqrt(PERIODS), s2 / math.sqrt(PERIODS)
    rho = float(np.corrcoef(ex1, ex2)[0, 1])
    months = len(part)
    var_monthly = (2 - 2 * rho + 0.5 * (m1 ** 2 + m2 ** 2 - 2 * m1 * m2 * rho ** 2)) / months
    se = math.sqrt(max(var_monthly, 0.0)) * math.sqrt(PERIODS)
    diff = s1 - s2
    return {"diff": diff, "low": diff - _z() * se, "high": diff + _z() * se, "se": se, "method": "analytic"}


def _block_indices(months: int, block: int, reps: int, rng) -> np.ndarray:
    """Индексы кругового блочного бутстрепа: reps строк по months индексов."""
    blocks = math.ceil(months / block)
    starts = rng.integers(0, months, size=(reps, blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % months
    return idx.reshape(reps, -1)[:, :months]


def _sharpe_rows(excess: np.ndarray) -> np.ndarray:
    sd = excess.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 1e-12, excess.mean(axis=1) / sd * math.sqrt(PERIODS), np.nan)


def sharpe_diff_bootstrap(part: pd.DataFrame, column: str = "net", reps: int = None,
                          block: int = BOOTSTRAP_BLOCK, seed: int = BOOTSTRAP_SEED) -> dict:
    """Парный круговой блочный бутстреп разницы Sharpe (правило минус бенчмарк).

    Парный: для правила и бенчмарка берутся ОДНИ И ТЕ ЖЕ месяцы, поэтому их связь сохраняется.
    Интервал - перцентили 2.5% и 97.5% разниц.
    """
    ex1 = (part[column] - part["rf"]).values
    ex2 = (part["bench"] - part["rf"]).values
    s1, s2 = sharpe(part[column], part["rf"]), sharpe(part["bench"], part["rf"])
    if s1 is None or s2 is None:
        return {"diff": None, "low": None, "high": None, "method": "bootstrap"}
    reps = reps or BOOTSTRAP_REPS
    idx = _block_indices(len(ex1), block, reps, np.random.default_rng(seed))
    diffs = _sharpe_rows(ex1[idx]) - _sharpe_rows(ex2[idx])
    diffs = diffs[~np.isnan(diffs)]
    tail = (1 - CONFIDENCE) / 2 * 100
    low, high = np.percentile(diffs, [tail, 100 - tail])
    return {"diff": s1 - s2, "low": float(low), "high": float(high), "se": float(diffs.std(ddof=1)),
            "method": f"paired circular block bootstrap, block {block} months, {reps} repetitions, seed {seed}"}


def zero_inside(interval: dict):
    """True, если 0 внутри интервала (разница НЕ отличима от нуля). None, если интервала нет."""
    if interval["low"] is None:
        return None
    return interval["low"] <= 0 <= interval["high"]


# ---------- 5. Издержки (E3) ----------

def net_returns(part: pd.DataFrame, cost: float) -> pd.Series:
    """Доходность правила после издержек c за единицу изменения позиции."""
    return part["gross"] - cost * part["turnover"]


def advantage(part: pd.DataFrame, cost: float, measure: str = "sharpe"):
    """Преимущество правила над бенчмарком после издержек: разница Sharpe или разница CAGR."""
    net = net_returns(part, cost)
    if measure == "cagr":
        return cagr(net) - cagr(part["bench"])
    s1, s2 = sharpe(net, part["rf"]), sharpe(part["bench"], part["rf"])
    return None if s1 is None or s2 is None else s1 - s2


def break_even_cost(part: pd.DataFrame, measure: str = "sharpe", high: float = BREAKEVEN_MAX):
    """Издержки c, при которых преимущество правила исчезает (бинарный поиск: преимущество падает с ростом c).

    Возвращает {"value": c или None, "status": "found" | "no_advantage" | "above_max"}.
    """
    at_zero = advantage(part, 0.0, measure)
    if at_zero is None or at_zero <= 0:
        return {"value": None, "status": "no_advantage"}
    if advantage(part, high, measure) > 0:
        return {"value": None, "status": "above_max"}
    low = 0.0
    for _ in range(60):
        middle = (low + high) / 2
        if advantage(part, middle, measure) > 0:
            low = middle
        else:
            high = middle
    return {"value": (low + high) / 2, "status": "found"}


# ---------- 6. Попытки (E7) ----------

def expected_max_noise_sharpe(trials: int, months: int):
    """Sharpe, которого в среднем достигает ЛУЧШЕЕ из `trials` правил-шумов (истинный Sharpe = 0),
    на выборке из `months` месяцев (Bailey и Lopez de Prado, 2014):
    E[max] = SE * ((1 - g) * Z(1 - 1/K) + g * Z(1 - 1/(K e))), SE = sqrt(12 / T), g = 0.5772.
    При одной попытке ожидание равно 0."""
    if trials < 1 or months < 2:
        return None
    if trials == 1:
        return 0.0
    se = math.sqrt(PERIODS / months)
    z = NormalDist().inv_cdf
    return se * ((1 - EULER_GAMMA) * z(1 - 1 / trials) + EULER_GAMMA * z(1 - 1 / (trials * math.e)))
