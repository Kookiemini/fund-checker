"""core.py - только чистые функции-формулы.

Правила этого файла:
- никакого интернета, никаких файлов, никакого вывода на экран (print);
- функция получает числа или таблицы pandas и возвращает числа;
- одинаковые входные данные всегда дают одинаковый результат.

Все доходности и ставки здесь - ДОЛИ (0.05 = 5%), кроме функции
risk_free_annual, которая как раз переводит проценты в доли.

Две годовые доходности (важно не путать):
- арифметическая = средняя дневная доходность * 252. Её используют Sharpe,
  CAPM и альфа Йенсена: эти формулы построены на средних (ожидаемых) доходностях;
- геометрическая (CAGR) = рост с учётом сложного процента. Её используют ТОЛЬКО
  для сравнения с рекламной цифрой (gap): реклама обычно называет именно
  годовой темп роста вложенных денег.
У волатильного фонда (например ARKK) эти два числа заметно расходятся
(арифметическая больше геометрической), поэтому их нельзя смешивать в одной формуле.
"""
import math
from datetime import date

import numpy as np
import pandas as pd

TRADING_DAYS = 252  # торговых дней в году (стандартное допущение)
EPSILON = 1e-12     # всё, что меньше, считаем нулём (из-за округления float 0 бывает 1e-18)

# Значение по умолчанию для validate_period, если минимальную дату не передали. Приложение ВСЕГДА
# передаёт её параметром min_date из data.DOWNLOAD_START (единственный источник правды), так что
# эта константа живёт только ради старых вызовов validate_period(start, end, today).
MIN_DATE = date(2015, 1, 1)
MAX_ADVERTISED_PCT = 1000.0       # разумный предел рекламной доходности, % в год
DEFAULT_ADVERTISED_PCT = 35       # значение по умолчанию для поля на экране, % в год
PP_ROUNDING = 0.005               # разрыв меньше 0.005 п.п. на экране выглядит как 0.00: считаем его "совпадением"
BETA_TOLERANCE = 0.05             # бета в пределах 1 +- 0.05 считаем "почти как у бенчмарка"
ALPHA_ROUNDING = 0.00005          # альфа меньше 0.005% (0.00005 в долях) на экране выглядит как 0.00%: считаем её нулём

# Бета бенчмарка к самому себе равна 1 по определению: cov(x, x) / var(x) = 1.
# Поэтому в таблице сравнения для бенчмарка берём эту константу, а не считаем.
BENCHMARK_BETA = 1.0

# ---- Пороги Deep analysis: СОБСТВЕННЫЕ решения команды, это не формулы курса ----
FRAGILE_SHARPE_DIFF = 0.10   # разница Sharpe меньше 0.10 считаем "слишком близко, чтобы судить"
WINDOW_STEP_DAYS = 21        # шаг сдвига окна: 21 торговый день, примерно месяц
WINDOW_MIN_DAYS = 126        # окна короче 126 торговых дней (примерно полгода) не рассматриваем
HONEST_STEP_PCT = 0.1        # шаг "честной" рекламной цифры, % годовых
WINDOW_TOLERANCE_DAYS = 7    # данные могут начаться на несколько дней позже выбранной даты (выходные, праздники): это не сокращение


# ---------- 1. Доходность ----------

def daily_returns(prices: pd.Series) -> pd.Series:
    """Дневная доходность: r_t = P_t / P_(t-1) - 1. Первая строка пропадает (нет вчера)."""
    return prices.pct_change().dropna()


def cumulative_return(prices: pd.Series) -> float:
    """Накопленная доходность за период: P_конец / P_начало - 1."""
    if len(prices) < 2:
        raise ValueError("At least 2 prices are required.")
    return float(prices.iloc[-1] / prices.iloc[0] - 1)


def annualized_return(returns: pd.Series) -> float:
    """Геометрическая годовая доходность (CAGR) из дневных доходностей.

    Перемножаем (1 + r) за все дни, получаем рост за период,
    затем приводим к году: рост ** (252 / число_дней) - 1.
    Используется ТОЛЬКО для сравнения с рекламной цифрой (gap).
    """
    if len(returns) < 1:
        raise ValueError("No returns available.")
    growth = (1 + returns).prod()
    return float(growth ** (TRADING_DAYS / len(returns)) - 1)


def arithmetic_annual_return(returns: pd.Series) -> float:
    """Арифметическая годовая доходность: средняя дневная доходность * 252.

    Используется в Sharpe, CAPM и альфе (формулы на средних доходностях).
    """
    if len(returns) < 1:
        raise ValueError("No returns available.")
    return float(returns.mean() * TRADING_DAYS)


# ---------- 2. Волатильность ----------

def annualized_volatility(returns: pd.Series) -> float:
    """Годовая волатильность: стандартное отклонение дневных доходностей * sqrt(252)."""
    if len(returns) < 2:
        raise ValueError("At least 2 returns are required.")
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


# ---------- Risk-free rate ----------

def risk_free_annual(rf_percent: pd.Series) -> float:
    """Средняя безрисковая ставка за период в ДОЛЯХ годовых.

    FRED отдаёт проценты (4.5 = 4.5%), поэтому делим на 100.
    Берём среднее за период: простое и понятное допущение.
    """
    if len(rf_percent) < 1:
        raise ValueError("No risk-free rate data available.")
    return float(rf_percent.mean() / 100)


# ---------- 3. Sharpe ratio ----------

def sharpe_ratio(returns: pd.Series, rf_annual: float):
    """Годовой Sharpe ratio = (годовая избыточная доходность) / (годовая волатильность).

    Числитель = арифметическая годовая доходность минус rf (mean * 252 - rf).
    Избыточная доходность = дневная доходность минус дневная безрисковая ставка
    (rf_annual / 252). Если волатильность равна 0, делить нельзя: возвращаем None.
    """
    if len(returns) < 2:
        raise ValueError("At least 2 returns are required.")
    excess = returns - rf_annual / TRADING_DAYS       # избыточная доходность за день
    annual_excess = excess.mean() * TRADING_DAYS      # приводим среднее к году
    volatility = annualized_volatility(returns)
    if volatility < EPSILON:
        return None
    return float(annual_excess / volatility)


# ---------- 4. Бета ----------

def beta(fund_returns: pd.Series, bench_returns: pd.Series):
    """Бета = ковариация(фонд, бенчмарк) / дисперсия(бенчмарк).

    Бета 1 - фонд двигается как рынок, больше 1 - сильнее рынка, меньше 1 - слабее.
    Если дисперсия бенчмарка равна 0, возвращаем None.
    """
    pair = pd.concat([fund_returns, bench_returns], axis=1).dropna()  # только общие даты
    if len(pair) < 2:
        raise ValueError("At least 2 common dates are required.")
    fund, bench = pair.iloc[:, 0], pair.iloc[:, 1]
    variance = bench.var(ddof=1)
    if variance < EPSILON:
        return None
    return float(fund.cov(bench) / variance)


# ---------- 5. Рекламируемая доходность против реальности ----------

def justifies_risk(sharpe_fund, sharpe_bench):
    """Фонд оправдывает риск, если его Sharpe выше, чем у бенчмарка.

    Возвращает True / False, а если Sharpe посчитать нельзя (None), то None.
    """
    if sharpe_fund is None or sharpe_bench is None:
        return None
    return sharpe_fund > sharpe_bench


def verdict_code(justifies) -> str:
    """Переводит результат justifies_risk в код вердикта для экрана.

    "justifies" / "does_not_justify" / "unknown". Текст для пользователя подбирает app.py.
    """
    if justifies is True:
        return "justifies"
    if justifies is False:
        return "does_not_justify"
    return "unknown"


def gap_headline(gap: float) -> dict:
    """Главное число: разрыв "реклама минус реальность" в процентных пунктах в год.

    gap - в долях (0.2798 = 27.98 п.п.). Возвращает словарь:
      points_signed - разрыв в п.п. со знаком (положительный: реклама приукрашена);
      points_abs    - модуль разрыва в п.п. (для текста "на X п.п.");
      direction     - "overstated" (реклама выше реальности), "beat" (фонд превзошёл
                      рекламу) или "matched" (разница меньше 0.005 п.п., на экране это 0.00).
    """
    points = gap * 100
    if abs(points) < PP_ROUNDING:
        direction = "matched"
    elif points > 0:
        direction = "overstated"
    else:
        direction = "beat"
    return {"points_signed": points, "points_abs": abs(points), "direction": direction}


def compare_claim(advertised_return: float, fund_returns: pd.Series,
                  bench_returns: pd.Series, rf_annual: float) -> dict:
    """Сравнивает рекламную цифру с реальностью, скорректированной на риск.

    advertised_return - рекламируемая годовая доходность в долях (0.35 = 35%).
    Возвращает словарь:
      advertised    - рекламируемая доходность
      actual        - ГЕОМЕТРИЧЕСКАЯ годовая доходность фонда (CAGR); нужна для gap
      bench_actual  - геометрическая годовая доходность бенчмарка (CAGR)
      actual_arith  - АРИФМЕТИЧЕСКАЯ годовая доходность фонда (mean * 252)
      bench_arith   - арифметическая годовая доходность бенчмарка
      gap           - advertised - actual (CAGR); положительное = реклама приукрашивает
      headline      - главное число в п.п. (см. gap_headline)
      overstated    - True, если реклама выше реальности (иначе False)
      beta          - бета фонда к бенчмарку
      capm_expected - доходность, которую "заслуживает" такой риск (модель CAPM,
                      на арифметических доходностях): rf + бета * (bench_arith - rf)
      alpha         - actual_arith - capm_expected (альфа Йенсена)
      sharpe_fund, sharpe_bench - Sharpe фонда и бенчмарка (арифметический)
      justifies_risk - True / False / None (см. justifies_risk)
      verdict       - код вердикта (см. verdict_code)
    """
    actual = annualized_return(fund_returns)                 # CAGR - только для gap
    bench_actual = annualized_return(bench_returns)
    actual_arith = arithmetic_annual_return(fund_returns)    # для CAPM и альфы
    bench_arith = arithmetic_annual_return(bench_returns)
    fund_beta = beta(fund_returns, bench_returns)
    sharpe_fund = sharpe_ratio(fund_returns, rf_annual)
    sharpe_bench = sharpe_ratio(bench_returns, rf_annual)

    capm_expected = None
    alpha = None
    if fund_beta is not None:
        capm_expected = rf_annual + fund_beta * (bench_arith - rf_annual)
        alpha = actual_arith - capm_expected

    gap = advertised_return - actual
    headline = gap_headline(gap)
    justifies = justifies_risk(sharpe_fund, sharpe_bench)

    return {
        "advertised": advertised_return,
        "actual": actual,
        "bench_actual": bench_actual,
        "actual_arith": actual_arith,
        "bench_arith": bench_arith,
        "gap": gap,
        "headline": headline,
        "overstated": headline["direction"] == "overstated",
        "beta": fund_beta,
        "capm_expected": capm_expected,
        "alpha": alpha,
        "sharpe_fund": sharpe_fund,
        "sharpe_bench": sharpe_bench,
        "justifies_risk": justifies,
        "verdict": verdict_code(justifies),
    }


def explain_gap(advertised: float, actual: float, gap: float, beta_value, alpha) -> dict:
    """Данные для панели "Why this number?": как получилось главное число.

    Получает УЖЕ посчитанные значения (формулы здесь не повторяются), все в долях:
      advertised - рекламная доходность (0.30 = 30%);
      actual     - реальная геометрическая годовая доходность фонда (CAGR);
      gap        - advertised - actual, из compare_claim;
      beta_value - бета фонда к бенчмарку (или None, если не определена);
      alpha      - альфа Йенсена (или None, если бета не определена).
    Возвращает словарь:
      advertised, actual, beta, alpha - те же числа без изменений (для показа на экране);
      points_signed - разрыв в п.п. со знаком, ровно то же число, что и в главном блоке;
      direction     - "overstated" (реклама выше реальности), "beat" (фонд дал больше
                      рекламы) или "matched" (разница меньше 0.005 п.п.): по нему выбирается фраза;
      beta_level    - "higher" (бета выше 1), "lower" (ниже 1), "similar" (в пределах
                      BETA_TOLERANCE от 1), "opposite" (бета меньше -BETA_TOLERANCE) или "unknown" (None);
      alpha_level   - "above" (фонд дал больше платы за риск), "below" (меньше),
                      "zero" (в пределах округления) или "unknown" (None).
    """
    headline = gap_headline(gap)  # то же правило округления, что и у главного числа

    if beta_value is None:
        beta_level = "unknown"
    elif beta_value < -BETA_TOLERANCE:  # чуть ниже нуля (шум) не считаем "против бенчмарка"
        beta_level = "opposite"
    elif abs(beta_value - BENCHMARK_BETA) <= BETA_TOLERANCE:
        beta_level = "similar"
    elif beta_value > BENCHMARK_BETA:
        beta_level = "higher"
    else:
        beta_level = "lower"

    if alpha is None:
        alpha_level = "unknown"
    elif abs(alpha) < ALPHA_ROUNDING:
        alpha_level = "zero"
    elif alpha > 0:
        alpha_level = "above"
    else:
        alpha_level = "below"

    return {
        "advertised": advertised,
        "actual": actual,
        "points_signed": headline["points_signed"],
        "direction": headline["direction"],
        "beta": beta_value,
        "alpha": alpha,
        "beta_level": beta_level,
        "alpha_level": alpha_level,
    }


# ---------- Проверка входных данных ----------

def default_period(today: date):
    """Период по умолчанию: последние 3 года. Возвращает (start, end).

    29 февраля минус 3 года даёт несуществующую дату, тогда берём 28 февраля.
    """
    try:
        start = today.replace(year=today.year - 3)
    except ValueError:
        start = today.replace(year=today.year - 3, day=28)
    return start, today


def validate_period(start: date, end: date, today: date, min_date: date = None) -> None:
    """Проверяет даты. При ошибке бросает ValueError с понятным текстом для экрана.

    today и min_date передаются параметрами, чтобы функция оставалась чистой (и проверялась тестом).
    min_date приложение берёт из data.DOWNLOAD_START; без неё действует MIN_DATE.
    """
    lowest = MIN_DATE if min_date is None else min_date
    if start >= end:
        raise ValueError("The start date must be earlier than the end date.")
    if start < lowest:
        raise ValueError(f"The start date cannot be earlier than {lowest.isoformat()}: "
                         "the app only has data from that date.")
    if end > today:
        raise ValueError("The end date cannot be in the future.")


def parse_advertised_pct(text: str) -> float:
    """Превращает строку из поля "advertised return" в число процентов в год.

    Принимает "35", "35.5", "35,5", "35%". Бросает ValueError с понятным текстом, если
    это не число, оно отрицательное или больше MAX_ADVERTISED_PCT.
    """
    cleaned = (text or "").strip().replace(",", ".").rstrip("%").strip()
    try:
        value = float(cleaned)
    except ValueError:
        raise ValueError("Enter the advertised return as a number of percent per year, e.g. 35.")
    if value != value or value in (float("inf"), float("-inf")):  # nan и бесконечность
        raise ValueError("Enter the advertised return as a number of percent per year, e.g. 35.")
    if value < 0:
        raise ValueError("The advertised return cannot be negative. Enter 0 or a positive number.")
    if value > MAX_ADVERTISED_PCT:
        raise ValueError(f"The advertised return looks too high (more than {MAX_ADVERTISED_PCT:.0f}% "
                         "per year). Please check the number.")
    return value


# ---------- Вспомогательные функции для экрана ----------
# Они лежат здесь, а не в app.py, потому что app.py сам ничего не считает.

def percent_to_decimal(percent: float) -> float:
    """Проценты в доли: 35 -> 0.35 (так вводит пользователь рекламную цифру)."""
    return percent / 100


def to_percent(decimal: float) -> float:
    """Доли в проценты: 0.35 -> 35 (для показа на экране)."""
    return decimal * 100


def growth_of_100(prices: pd.Series) -> pd.Series:
    """Во что превратились бы 100 вложенных в первый день (для графика)."""
    return prices / prices.iloc[0] * 100


def chart_values(prices: pd.Series) -> list:
    """Ряд "рост 100" для графика: список чисел, округлённых до 2 знаков (только для передачи на страницу)."""
    return [round(float(v), 2) for v in growth_of_100(prices)]


def chart_range(*series) -> tuple:
    """Минимум и максимум по всем рядам графика (для масштаба осей). Возвращает (min, max)."""
    values = [v for one in series for v in one]
    return min(values), max(values)


# ---------- Deep analysis: что не хватает до зелёного (только история, не прогноз) ----------

def window_shortening_days(requested_start: date, first_date: date,
                           tolerance_days: int = WINDOW_TOLERANCE_DAYS) -> int:
    """На сколько дней окно сокращено, если ряд начинается позже выбранной даты начала.

    Возвращает 0, если ряд начинается не позже выбранной даты или позже её не более чем на
    tolerance_days (выходные и праздники). Иначе число календарных дней между датами.
    """
    late = (first_date - requested_start).days
    return late if late > tolerance_days else 0


def honest_advertised_limit_pct(cagr: float):
    """Наибольшая рекламная цифра (% годовых, шаг 0.1, вниз), при которой реклама НЕ приукрашена.

    Это CAGR фонда, округлённый вниз до 0.1. Если CAGR отрицательный, возвращает None:
    даже реклама "0%" была бы приукрашиванием. Предел MAX_ADVERTISED_PCT не превышается,
    чтобы ссылка "Try X%" проходила обычную проверку формы.
    """
    if cagr < 0:
        return None
    steps = math.floor(to_percent(cagr) / HONEST_STEP_PCT + 1e-9)  # 1e-9 гасит шум float (7.1999.. вместо 7.2)
    return min(round(steps * HONEST_STEP_PCT, 1), MAX_ADVERTISED_PCT)


def _shortfall_unavailable(reason: str) -> dict:
    """Пустой результат sharpe_shortfall: все числа None и понятная причина (код)."""
    return {"status": "unavailable", "reason": reason, "sharpe_gap": None, "volatility": None,
            "fund_arith": None, "required_arith_return": None, "return_shortfall_pp": None,
            "max_volatility": None, "volatility_cut_pp": None, "max_volatility_reason": ""}


def sharpe_shortfall(fund_returns: pd.Series, rf_annual: float, sharpe_fund, sharpe_bench) -> dict:
    """Чего фонду не хватило, чтобы сравняться с бенчмарком по Sharpe (алгебра из определения Sharpe).

    Sharpe = (a - rf) / vol, где a - арифметическая годовая доходность, vol - годовая волатильность.
    Приравниваем Sharpe фонда к Sharpe бенчмарка (sb) и получаем две границы:
      required_arith_return = rf + sb * vol  - какая доходность нужна при ТОЙ ЖЕ волатильности;
      max_volatility        = (a - rf) / sb  - какая волатильность допустима при ТОЙ ЖЕ доходности.
    На границе Sharpe равны, а "оправдывает риск" требует СТРОГО выше: нужно "at least" эту
    доходность (и чуть выше, чтобы пройти), либо чуть ниже эту волатильность.

    Возвращает словарь с ключами status ("ok", "not_needed", "unavailable"), reason (код причины
    при "unavailable"), sharpe_gap, volatility, fund_arith, required_arith_return (доли),
    return_shortfall_pp и volatility_cut_pp (процентные пункты), max_volatility (доли или None)
    и max_volatility_reason. Нулевая волатильность, Sharpe бенчмарка <= 0, нехватка данных:
    "unavailable" с причиной, без падения.
    """
    if len(fund_returns) < 2:
        return _shortfall_unavailable("not_enough_data")
    volatility = annualized_volatility(fund_returns)
    if volatility < EPSILON:
        return _shortfall_unavailable("no_volatility")
    if sharpe_fund is None or sharpe_bench is None:
        return _shortfall_unavailable("sharpe_missing")
    if sharpe_fund > sharpe_bench:
        result = _shortfall_unavailable("")
        result["status"] = "not_needed"      # фонд уже оправдывает риск: показывать нечего
        return result
    if sharpe_bench <= 0:
        return _shortfall_unavailable("benchmark_sharpe_not_positive")

    fund_arith = arithmetic_annual_return(fund_returns)
    required = rf_annual + sharpe_bench * volatility
    max_volatility, cut_pp, max_reason = None, None, ""
    if fund_arith > rf_annual:
        max_volatility = (fund_arith - rf_annual) / sharpe_bench
        cut_pp = to_percent(volatility - max_volatility)
    else:
        max_reason = "return_not_above_risk_free"   # при доходности не выше rf Sharpe не спасти одной волатильностью
    return {"status": "ok", "reason": "", "sharpe_gap": sharpe_bench - sharpe_fund,
            "volatility": volatility, "fund_arith": fund_arith, "required_arith_return": required,
            "return_shortfall_pp": to_percent(required - fund_arith), "max_volatility": max_volatility,
            "volatility_cut_pp": cut_pp, "max_volatility_reason": max_reason}


def closeness(sharpe_fund, sharpe_bench) -> str:
    """"too_close", если Sharpe отличаются меньше чем на FRAGILE_SHARPE_DIFF, иначе "clear".

    Разница ровно 0.10 уже "clear" (строго меньше порога = "too_close"). Перед сравнением
    разницу округляем до 9 знаков, чтобы шум float (0.1000000000000009) не менял ответ.
    Если Sharpe не посчитан (None), возвращает "unknown".
    """
    if sharpe_fund is None or sharpe_bench is None:
        return "unknown"
    difference = round(abs(sharpe_fund - sharpe_bench), 9)
    return "too_close" if difference < FRAGILE_SHARPE_DIFF else "clear"


def _window_verdict(fund: pd.Series, bench: pd.Series, rf_percent: pd.Series):
    """Sharpe фонда и бенчмарка и код вердикта для одного окна (rf = среднее за окно).

    Возвращает (sharpe_fund, sharpe_bench, verdict) или None, если за окно нет данных по ставке.
    """
    if len(rf_percent) < 1:
        return None
    rf_annual = risk_free_annual(rf_percent)
    sharpe_fund = sharpe_ratio(fund, rf_annual)
    sharpe_bench = sharpe_ratio(bench, rf_annual)
    return sharpe_fund, sharpe_bench, verdict_code(justifies_risk(sharpe_fund, sharpe_bench))


def nearest_green_windows(fund_returns: pd.Series, bench_returns: pd.Series, rf_percent: pd.Series,
                          step_days: int = WINDOW_STEP_DAYS, min_days: int = WINDOW_MIN_DAYS,
                          target_verdict: str = "justifies", start_date: date = None) -> list:
    """Ближайшие окна ВНУТРИ уже загруженного периода, где вердикт равен target_verdict.

    Ищет в двух направлениях с шагом step_days торговых дней (окна короче min_days не берутся):
      "later_start" - начало сдвигается вперёд, конец прежний;
      "earlier_end" - конец сдвигается назад, начало прежнее.
    Для каждого окна Sharpe считается заново, rf = среднее значение ставки за окно (rf_percent в процентах).
    В каждом направлении возвращается окно с наименьшим сдвигом; результат отсортирован по сдвигу.
    Если вердикт ВСЕГО периода уже равен target_verdict (или не определён) либо подходящих окон нет,
    возвращает []. target_verdict="does_not_justify" ищет красные окна для зелёного результата.
    start_date - дата первой цены периода (нужна для сдвига и ссылки; без неё берётся первая дата доходностей).
    Каждое окно: direction, start, end (даты цен, их можно ввести в форму), shift_days, sharpe_fund,
    sharpe_bench, verdict. Сеть не используется.
    """
    if step_days < 1 or min_days < 2:
        raise ValueError("step_days must be at least 1 and min_days at least 2.")
    pair = pd.concat([fund_returns, bench_returns], axis=1).dropna()
    count = len(pair)
    if count < min_days:
        return []
    fund, bench, dates = pair.iloc[:, 0], pair.iloc[:, 1], pair.index
    first_price_date = start_date if start_date is not None else dates[0].date()
    last_date = dates[-1].date()

    whole = _window_verdict(fund, bench, rf_percent)
    if whole is None or whole[2] == target_verdict:
        return []

    found = []
    # Начало вперёд: доходности [i:], цена-основа лежит на дате dates[i-1], конец прежний
    for i in range(step_days, count - min_days + 1, step_days):
        outcome = _window_verdict(fund.iloc[i:], bench.iloc[i:], rf_percent.loc[dates[i - 1]:])
        if outcome is not None and outcome[2] == target_verdict:
            start = dates[i - 1].date()
            found.append({"direction": "later_start", "start": start, "end": last_date,
                          "shift_days": (start - first_price_date).days,
                          "sharpe_fund": outcome[0], "sharpe_bench": outcome[1], "verdict": outcome[2]})
            break
    # Конец назад: доходности [:j], последняя дата окна dates[j-1], начало прежнее
    for j in range(count - step_days, min_days - 1, -step_days):
        outcome = _window_verdict(fund.iloc[:j], bench.iloc[:j], rf_percent.loc[:dates[j - 1]])
        if outcome is not None and outcome[2] == target_verdict:
            end = dates[j - 1].date()
            found.append({"direction": "earlier_end", "start": first_price_date, "end": end,
                          "shift_days": (last_date - end).days,
                          "sharpe_fund": outcome[0], "sharpe_bench": outcome[1], "verdict": outcome[2]})
            break
    return sorted(found, key=lambda window: window["shift_days"])


def deep_analysis(claim: dict, fund_returns: pd.Series, bench_returns: pd.Series, rf_percent: pd.Series,
                  rf_annual: float, start_date: date = None) -> dict:
    """Собирает все данные раздела "Deep analysis" (экран только подставляет их в тексты).

    claim - результат compare_claim. Возвращает словарь:
      verdict, overstated   - из claim;
      honest_limit_pct      - "честная" рекламная цифра (только если реклама приукрашена, иначе None);
      shortfall             - результат sharpe_shortfall (только для красного вердикта, иначе None);
      windows               - ближайшие зелёные окна (для красного) или красные (для зелёного), иначе [];
      closeness             - "clear" / "too_close" / "unknown";
      sharpe_margin         - Sharpe фонда минус Sharpe бенчмарка (или None), sharpe_margin_abs - его модуль;
      min_window_days       - минимальная длина окна, для текста на экране.
    """
    verdict = claim["verdict"]
    margin = None
    if claim["sharpe_fund"] is not None and claim["sharpe_bench"] is not None:
        margin = claim["sharpe_fund"] - claim["sharpe_bench"]

    honest_limit = honest_advertised_limit_pct(claim["actual"]) if claim["overstated"] else None
    shortfall, windows = None, []
    if verdict == "does_not_justify":
        shortfall = sharpe_shortfall(fund_returns, rf_annual, claim["sharpe_fund"], claim["sharpe_bench"])
        windows = nearest_green_windows(fund_returns, bench_returns, rf_percent,
                                        target_verdict="justifies", start_date=start_date)
    elif verdict == "justifies":
        windows = nearest_green_windows(fund_returns, bench_returns, rf_percent,
                                        target_verdict="does_not_justify", start_date=start_date)
    return {
        "verdict": verdict,
        "overstated": claim["overstated"],
        "honest_limit_pct": honest_limit,
        "shortfall": shortfall,
        "windows": windows,
        "closeness": closeness(claim["sharpe_fund"], claim["sharpe_bench"]),
        "sharpe_margin": margin,
        "sharpe_margin_abs": None if margin is None else abs(margin),
        "min_window_days": WINDOW_MIN_DAYS,
    }
