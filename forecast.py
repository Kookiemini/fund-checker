"""forecast.py - сценарии на 3, 5 и 10 лет по историческим месячным доходностям. Только расчёты.

Это НЕ предсказание одной цифры. Будущее неизвестно, поэтому считаем диапазон того, что могло бы
случиться, если будущее будет похоже на прошлое:
- берём реальные месячные доходности (с 2000 года или с начала истории фонда);
- тысячи раз собираем из них "будущее" кусками по 12 месяцев подряд (круговой блочный бутстреп,
  та же схема, что в rule.py): так сохраняются серии плохих месяцев, то есть кризисы;
- для каждого горизонта смотрим, сколько стало бы денег: плохой (5%), осторожный (25%), средний (50%),
  хороший (75%) и очень хороший (95%) сценарии, и как часто итог ниже вложенной суммы.

Правила те же, что у core.py и rule.py: без интернета, файлов и print; одинаковый вход -> одинаковый выход.
"""
import math

import numpy as np
import pandas as pd

import rule

HORIZONS = [3, 5, 10]                      # лет
PATHS = 10_000                             # сколько вариантов "будущего" собираем
BLOCK = 12                                 # длина куска истории, месяцев
SEED = 20261119                            # фиксированный seed: одинаковый результат при каждом запуске
PERCENTILES = [5, 25, 50, 75, 95]          # плохой ... очень хороший сценарий
MIN_MONTHS = 36                            # меньше 3 лет истории - сценарии не считаем
AMOUNT_DEFAULT = 1000.0
AMOUNT_MAX = 100_000_000.0


def monthly_returns(daily_prices: pd.Series) -> pd.Series:
    """Месячные доходности из дневных цен (полные месяцы, как в rule.py)."""
    return rule.monthly_prices(daily_prices).pct_change().dropna()


def simulate(returns: pd.Series, months: int, paths: int = PATHS, block: int = BLOCK, seed: int = SEED) -> np.ndarray:
    """Матрица роста 1 доллара: paths строк, months столбцов (рост на конец каждого месяца)."""
    values = returns.values
    idx = rule._block_indices(len(values), block, paths, np.random.default_rng(seed))
    while idx.shape[1] < months:                                   # горизонт длиннее истории: добираем кусками
        idx = np.concatenate([idx, rule._block_indices(len(values), block, paths,
                                                       np.random.default_rng(seed + idx.shape[1]))], axis=1)
    return np.cumprod(1 + values[idx[:, :months]], axis=1)


def summarize(growth: np.ndarray, amount: float, horizons=HORIZONS) -> list:
    """Итоги по каждому горизонту: суммы по сценариям, шанс оказаться в минусе, средний рост в год."""
    rows = []
    for years in horizons:
        final = growth[:, years * 12 - 1]
        values = np.percentile(final, PERCENTILES)
        rows.append({
            "years": years,
            "values": {p: float(v * amount) for p, v in zip(PERCENTILES, values)},
            "median_per_year": float(values[2] ** (1 / years) - 1),
            "loss_chance": float((final < 1).mean()),
            "half_loss_chance": float((final < 0.5).mean()),
        })
    return rows


def history_summary(returns: pd.Series) -> dict:
    """Что было в истории, на которой строятся сценарии (для честной панели)."""
    wealth = np.concatenate([[1.0], np.cumprod(1 + returns.values)])
    return {
        "start": returns.index[0].strftime("%Y-%m"),
        "end": returns.index[-1].strftime("%Y-%m"),
        "months": len(returns),
        "cagr": rule.cagr(returns),
        "volatility": rule.volatility(returns),
        "max_drawdown": float((wealth / np.maximum.accumulate(wealth) - 1).min()),
        "worst_month": float(returns.min()),
    }


def band(growth: np.ndarray, amount: float) -> dict:
    """Полосы для графика-веера: процентили суммы на конец каждого месяца."""
    q = np.percentile(growth, PERCENTILES, axis=0) * amount
    return {p: q[i] for i, p in enumerate(PERCENTILES)}


def fan_chart_png(growth: np.ndarray, amount: float, title: str) -> bytes:
    """График-веер (PNG): медиана и коридоры 25-75% и 5-95% на 10 лет."""
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b = band(growth, amount)
    years = np.arange(1, growth.shape[1] + 1) / 12
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(years, b[5], b[95], color="#c9daf2", label="5% to 95% of scenarios")
    ax.fill_between(years, b[25], b[75], color="#8fb1e3", label="25% to 75% of scenarios")
    ax.plot(years, b[50], color="#1f5fbf", linewidth=2, label="Middle scenario (median)")
    ax.axhline(amount, color="#5b6b7a", linestyle="--", linewidth=1, label="Amount invested")
    for h in HORIZONS:
        ax.axvline(h, color="#dde3ea", linewidth=1)
    ax.set_xlabel("Years from now")
    ax.set_ylabel("USD")
    ax.set_title(title)
    ax.set_xlim(0, growth.shape[1] / 12)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    out = io.BytesIO()
    fig.savefig(out, format="png", dpi=110)
    plt.close(fig)
    return out.getvalue()
