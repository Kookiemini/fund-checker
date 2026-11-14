"""make_evidence.py - генерирует пакет доказательств E1-E8 для правила из rule.py.

Запуск (из папки проекта, с включённым .venv):
    python make_evidence.py               # период разработки: train + validation (test остаётся закрытым)
    python make_evidence.py --open-test   # ОДИН раз: открыть финальный test-период
    python make_evidence.py --open-test --reopen   # повторное открытие (будет записано и посчитано)

Что получается в папке evidence/:
    evidence.json           - все числа (их же показывает страница Rule Lab, поэтому числа совпадают)
    evidence_table.md       - таблицы E1-E8 для отчёта
    result_card_numbers.md  - числа для RESULT_CARD.md (слова вердикта пишет команда)
    figures/*.png           - графики
Каждый запуск дописывает строки в trials.csv (строки никогда не удаляются).

Данные берутся через data.py (Yahoo Finance + FRED, офлайн-режим и кэш те же, что у сайта).
Расчёты - только rule.py. Этот файл лишь собирает их и записывает файлы.
"""
import argparse
import csv
import json
import statistics
from datetime import date
from pathlib import Path

import pandas as pd

import data
import rule

ROOT = Path(__file__).resolve().parent
EVIDENCE_DIR = ROOT / "evidence"
TRIALS_PATH = ROOT / "trials.csv"
DECISIONS_PATH = ROOT / "DECISIONS.md"
TEST_STATE_PATH = EVIDENCE_DIR / "test_opened.json"
TRIALS_HEADER = ["date", "config_id", "rule_version", "param", "window_start", "window_end",
                 "sharpe", "cagr", "mdd", "sample", "comment"]


class EvidenceError(Exception):
    """Понятная ошибка для экрана и консоли."""


# ---------- Формат чисел (один и тот же для evidence_table.md и для Rule Lab) ----------

def fmt_pct(value, digits: int = 2) -> str:
    return "-" if value is None else f"{value * 100:.{digits}f}%"


def fmt_num(value) -> str:
    return "-" if value is None else f"{value:.2f}"


def fmt_sr(value, se) -> str:
    return "-" if value is None else f"{value:.2f} ± {se:.2f}"


def fmt_interval(iv: dict) -> str:
    return "-" if iv["low"] is None else f"{iv['diff']:+.2f} [{iv['low']:+.2f}, {iv['high']:+.2f}]"


def cost_key(cost: float) -> str:
    return f"{cost:.4f}"


def fmt_cost(cost: float) -> str:
    return f"{cost * 100:.2f}%"


# ---------- Состояние test-периода ----------

def test_state() -> dict:
    if TEST_STATE_PATH.exists():
        return json.loads(TEST_STATE_PATH.read_text(encoding="utf-8"))
    return {"opened": []}


def check_can_open(reopen: bool) -> None:
    """Второе открытие test-периода - только с --reopen (тогда оно записывается и считается)."""
    state = test_state()
    if state["opened"] and not reopen:
        raise EvidenceError(f"The test period was already opened on {state['opened'][0]}. "
                            "Opening it again must be declared: run with --open-test --reopen "
                            "(it will be recorded in DECISIONS.md and counted).")


def open_test(reopen: bool, today: date) -> dict:
    """Отмечает открытие test-периода (в test_opened.json и в DECISIONS.md)."""
    check_can_open(reopen)
    state = test_state()
    state["opened"].append(today.isoformat())
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    TEST_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    times = len(state["opened"])
    line = (f"\n- {today.isoformat()}: test period {rule.TEST[0]} to {rule.TEST[1]} opened "
            f"(time #{times}) with `python make_evidence.py --open-test{' --reopen' if reopen else ''}`. "
            f"Rule {rule.RULE_VERSION}, N = {rule.PARAM_MAIN}. Who decided: [team fills in].\n")
    with DECISIONS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line)
    return state


# ---------- Данные ----------

def load_monthly(today: date):
    """Помесячные цены SPY и ставка FRED через data.py (с офлайн-режимом). Возвращает (prices, rate, sources)."""
    end = min(pd.Period(rule.TEST[1]).end_time.date(), today)
    start = date.fromisoformat(rule.DATA_START)
    prices, price_info = data.get_prices(rule.ASSET, start, end)
    rates, rate_info = data.get_risk_free(start, end)
    monthly = rule.monthly_prices(prices[rule.ASSET])
    rate = rule.monthly_rate(rates.iloc[:, 0])
    sources = []
    for role, info, source in [(f"Prices of {rule.ASSET}", price_info,
                                "Yahoo Finance (yfinance), Adjusted Close (auto_adjust=True), daily -> last close of month"),
                               (f"Risk-free rate {data.RF_SERIES}", rate_info,
                                "FRED API, DGS3MO (3-Month Treasury, % per year), daily -> last value of month")]:
        sources.append({"series": role, "source": source, "pull_date": str(info["pull_date"]),
                        "file": info["file"], "offline": info["offline"], "message": info["message"],
                        "first_date": str(info["first_date"]), "last_date": str(info["last_date"])})
    return monthly, rate, sources


# ---------- Сборка доказательств ----------

def _with_cost(part: pd.DataFrame, cost: float) -> pd.DataFrame:
    return part.assign(net=rule.net_returns(part, cost))


def _e1(part: pd.DataFrame) -> dict:
    return {cost_key(c): {"rule": rule.summary(_with_cost(part, c), "net"), "bench": rule.summary(part, "bench")}
            for c in rule.COSTS}


def _window_check(part: pd.DataFrame, headline_diff, headline_zero) -> dict:
    """E4: одна проверка окна при издержках по умолчанию."""
    p = _with_cost(part, rule.COST_DEFAULT)
    iv = rule.sharpe_diff_analytic(p)
    zero = rule.zero_inside(iv)
    same_sign = None if iv["diff"] is None or headline_diff is None else (iv["diff"] > 0) == (headline_diff > 0)
    return {"start": part.index[0].strftime("%Y-%m"), "end": part.index[-1].strftime("%Y-%m"),
            "months": len(part), "sharpe_rule": rule.sharpe(p["net"], p["rf"]),
            "sharpe_bench": rule.sharpe(p["bench"], p["rf"]), "interval": iv, "zero_inside": zero,
            "same_sign": same_sign, "same_verdict": None if zero is None else (same_sign and zero == headline_zero)}


def build(tables: dict, samples: dict, headline: str) -> dict:
    """Все разделы E1-E8 из таблиц бэктеста (tables: N -> таблица с издержками 0)."""
    main = tables[rule.PARAM_MAIN]
    parts = {name: rule.window(main, *bounds) for name, bounds in samples.items()}
    head = parts[headline]
    default = _with_cost(head, rule.COST_DEFAULT)

    e2 = {}
    for c in rule.COSTS:
        p = _with_cost(head, c)
        boot, analytic = rule.sharpe_diff_bootstrap(p), rule.sharpe_diff_analytic(p)
        e2[cost_key(c)] = {"bootstrap": boot, "analytic": analytic, "zero_inside": rule.zero_inside(boot)}
    head_iv = e2[cost_key(rule.COST_DEFAULT)]

    e3 = {"rows": [], "break_even_sharpe": rule.break_even_cost(head, "sharpe"),
          "break_even_cagr": rule.break_even_cost(head, "cagr")}
    for c in rule.COSTS:
        net = rule.net_returns(head, c)
        e3["rows"].append({"cost": c, "cagr_net": rule.cagr(net), "sharpe_net": rule.sharpe(net, head["rf"]),
                           "cagr_advantage": rule.advantage(head, c, "cagr"),
                           "sharpe_advantage": rule.advantage(head, c, "sharpe")})

    first_year = head.index[0].year
    starts = [rule.window(main, f"{first_year + k}-{head.index[0].month:02d}", head.index[-1].strftime("%Y-%m"))
              for k in (0, 2, 4, 6)]
    thirds = [head.iloc[i * len(head) // 3:(i + 1) * len(head) // 3] for i in range(3)]
    e4 = {"cost": rule.COST_DEFAULT,
          "start_dates": [_window_check(p, head_iv["bootstrap"]["diff"], head_iv["zero_inside"]) for p in starts],
          "subperiods": [_window_check(p, head_iv["bootstrap"]["diff"], head_iv["zero_inside"]) for p in thirds]}

    dev = samples["development"]
    e5_rows = []
    for n in rule.NEIGHBOURS:
        p = rule.window(tables[n], *dev)
        gross_sr = rule.sharpe(p["gross"], p["rf"])
        e5_rows.append({"n": n, "main": n == rule.PARAM_MAIN, "sharpe_gross": gross_sr,
                        "sharpe_net": rule.sharpe(rule.net_returns(p, rule.COST_DEFAULT), p["rf"]),
                        "sharpe_bench": rule.sharpe(p["bench"], p["rf"]), "cagr_gross": rule.cagr(p["gross"]),
                        "max_drawdown": rule.max_drawdown(p["gross"])})
    values = [r["sharpe_gross"] for r in e5_rows if r["sharpe_gross"] is not None]
    main_sr = next(r["sharpe_gross"] for r in e5_rows if r["main"])
    others = [r["sharpe_gross"] for r in e5_rows if not r["main"] and r["sharpe_gross"] is not None]
    e5 = {"sample": "development", "window": dev, "rows": e5_rows,
          "range": (max(values) - min(values)) if values else None,
          "main_is_best": bool(values) and main_sr == max(values),
          "main_minus_median_others": (main_sr - statistics.median(others)) if others and main_sr is not None else None}

    e8 = {name: {"rule": rule.summary(_with_cost(p, rule.COST_DEFAULT), "net"), "bench": rule.summary(p, "bench")}
          for name, p in parts.items()}

    return {
        "e1": {"headline": _e1(head), "by_sample": {name: _e1(p) for name, p in parts.items()}},
        "e2": e2, "e3": e3, "e4": e4, "e5": e5, "e8": e8,
        "series": {"dates": [d.strftime("%Y-%m") for d in head.index],
                   "rule": (1 + default["net"]).cumprod().tolist(), "bench": (1 + head["bench"]).cumprod().tolist(),
                   "dd_rule": rule.drawdown_series(default["net"]).tolist(),
                   "dd_bench": rule.drawdown_series(head["bench"]).tolist(),
                   "position": head["position"].tolist()},
    }


# ---------- trials.csv ----------

def append_trials(tables: dict, samples: dict, today: date, comment: str, test_samples: bool) -> None:
    """Одна строка на каждую конфигурацию и выборку в этом запуске. Существующие строки не трогаем."""
    new_file = not TRIALS_PATH.exists() or TRIALS_PATH.stat().st_size == 0
    names = ["train", "validation"] + (["test"] if test_samples else [])
    with TRIALS_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(TRIALS_HEADER)
        for n in rule.NEIGHBOURS:
            for name in names:
                p = rule.window(tables[n], *samples[name])
                sr = rule.sharpe(p["gross"], p["rf"])
                writer.writerow([today.isoformat(), f"MA{n}", rule.RULE_VERSION, n,
                                 p.index[0].strftime("%Y-%m"), p.index[-1].strftime("%Y-%m"),
                                 "" if sr is None else f"{sr:.4f}", f"{rule.cagr(p['gross']):.4f}",
                                 f"{rule.max_drawdown(p['gross']):.4f}", name,
                                 comment + ("" if n != rule.PARAM_MAIN else "; main configuration")])


def trials_summary(train_months: int) -> dict:
    """E7: сколько конфигураций смотрели, лучший и медианный in-sample (train) Sharpe, ожидаемый максимум шума."""
    rows = list(csv.DictReader(TRIALS_PATH.open(encoding="utf-8"))) if TRIALS_PATH.exists() else []
    latest = {}
    for row in rows:
        if row["sample"] == "train" and row["sharpe"]:
            latest[(row["config_id"], row["rule_version"])] = float(row["sharpe"])
    configs = {(row["config_id"], row["rule_version"]) for row in rows}
    values = list(latest.values())
    return {"lines": len(rows), "configurations": len(configs),
            "best_in_sample_sharpe": max(values) if values else None,
            "median_in_sample_sharpe": statistics.median(values) if values else None,
            "expected_max_noise_sharpe": rule.expected_max_noise_sharpe(len(configs), train_months),
            "train_months": train_months}


# ---------- Файлы отчёта ----------

def write_figures(evidence: dict) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = EVIDENCE_DIR / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    s = evidence["series"]
    dates = pd.to_datetime(s["dates"])
    window_text = f"{s['dates'][0]} to {s['dates'][-1]}, monthly"
    cost_text = fmt_cost(rule.COST_DEFAULT)
    files = []

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(dates, s["rule"], label=f"Rule MA{rule.PARAM_MAIN} (net of {cost_text} per switch)", color="#1f5fbf")
    ax.plot(dates, s["bench"], label=f"Buy and hold {rule.ASSET}", color="#8a96a3")
    ax.set_yscale("log"); ax.set_title(f"Growth of 1 USD, {window_text}"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(folder / "equity.png", dpi=120); plt.close(fig); files.append("equity.png")

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.fill_between(dates, s["dd_bench"], 0, color="#8a96a3", alpha=.35, label=f"Buy and hold {rule.ASSET}")
    ax.plot(dates, s["dd_rule"], color="#1f5fbf", label=f"Rule MA{rule.PARAM_MAIN}")
    ax.set_title(f"Drawdown, {window_text}"); ax.legend(loc="lower left"); ax.grid(alpha=.3)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout(); fig.savefig(folder / "drawdown.png", dpi=120); plt.close(fig); files.append("drawdown.png")

    rows = evidence["e5"]["rows"]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar([str(r["n"]) for r in rows], [r["sharpe_gross"] or 0 for r in rows],
           color=["#1f5fbf" if r["main"] else "#9db7e0" for r in rows])
    ax.axhline(rows[0]["sharpe_bench"] or 0, color="#5b6b7a", linestyle="--", label=f"Buy and hold {rule.ASSET}")
    w = evidence["e5"]["window"]
    ax.set_title(f"Sharpe by N (gross), {w[0]} to {w[1]}"); ax.set_xlabel("N, months"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(folder / "neighbours.png", dpi=120); plt.close(fig); files.append("neighbours.png")
    return files


def _summary_rows(block: dict) -> list:
    r, b = block["rule"], block["bench"]
    return [
        ("CAGR", fmt_pct(r["cagr"]), fmt_pct(b["cagr"])),
        ("Volatility", fmt_pct(r["volatility"]), fmt_pct(b["volatility"])),
        ("Sharpe ± SE", fmt_sr(r["sharpe"], r["sharpe_se"] or 0), fmt_sr(b["sharpe"], b["sharpe_se"] or 0)),
        ("Max drawdown", fmt_pct(r["max_drawdown"]), fmt_pct(b["max_drawdown"])),
        ("Time in market", fmt_pct(r["time_in_market"], 0), fmt_pct(b["time_in_market"], 0)),
        ("Switches", str(r["switches"]), str(b["switches"])),
        ("Turnover per year", fmt_num(r["turnover_per_year"]), fmt_num(b["turnover_per_year"])),
    ]


def _be_text(be: dict) -> str:
    return {"found": lambda: fmt_cost(be["value"]),
            "no_advantage": lambda: "no advantage even before costs",
            "above_max": lambda: f"above {fmt_cost(rule.BREAKEVEN_MAX)}"}[be["status"]]()


def write_table(ev: dict) -> None:
    """evidence_table.md: те же числа и тот же формат, что на странице Rule Lab."""
    m = ev["meta"]
    head = ev["e1"]["headline"]
    lines = [f"# Evidence: rule MA{rule.PARAM_MAIN} on {rule.ASSET}, monthly ({rule.RULE_VERSION})", "",
             f"Generated by `make_evidence.py` on {m['generated']}. Do not edit by hand.", "",
             f"**Headline window: {m['headline_window'][0]} to {m['headline_window'][1]} ({m['headline_sample']}), "
             f"the same for the rule and the benchmark in every row.**", "",
             f"Test period: {'opened ' + ', '.join(m['test_opened']) if m['test_opened'] else 'not opened yet'}.", ""]
    for c in rule.COSTS:
        lines += [f"## E1. Headline table, costs {fmt_cost(c)} per unit of position change", "",
                  "| | Rule | Benchmark |", "|---|---|---|"]
        lines += [f"| {a} | {b} | {d} |" for a, b, d in _summary_rows(head[cost_key(c)])] + [""]
    lines += ["## E2. Difference in Sharpe (rule minus benchmark)", "",
              "| Costs | Paired bootstrap 95% | Paired analytic 95% | Zero inside? |", "|---|---|---|---|"]
    for c in rule.COSTS:
        e = ev["e2"][cost_key(c)]
        lines.append(f"| {fmt_cost(c)} | {fmt_interval(e['bootstrap'])} | {fmt_interval(e['analytic'])} | "
                     f"{'yes' if e['zero_inside'] else 'no' if e['zero_inside'] is not None else '-'} |")
    lines += ["", f"Bootstrap: {ev['e2'][cost_key(0.0)]['bootstrap'].get('method', '-')}.", "",
              "## E3. Costs", "", "| c per switch | Net CAGR | Net Sharpe | CAGR advantage | Sharpe advantage |",
              "|---|---|---|---|---|"]
    for row in ev["e3"]["rows"]:
        lines.append(f"| {fmt_cost(row['cost'])} | {fmt_pct(row['cagr_net'])} | {fmt_num(row['sharpe_net'])} | "
                     f"{fmt_pct(row['cagr_advantage'])} | {fmt_num(row['sharpe_advantage'])} |")
    lines += ["", f"Break-even cost (Sharpe advantage disappears): {_be_text(ev['e3']['break_even_sharpe'])}.",
              f"Break-even cost (CAGR advantage disappears): {_be_text(ev['e3']['break_even_cagr'])}.", "",
              f"## E4. Start dates and subperiods (costs {fmt_cost(ev['e4']['cost'])}, paired analytic 95%)", "",
              "| Kind | Window | Sharpe rule | Sharpe bench | Difference [interval] | Zero inside? | Same sign? |",
              "|---|---|---|---|---|---|---|"]
    for kind, key in [("Start date", "start_dates"), ("Subperiod", "subperiods")]:
        for w in ev["e4"][key]:
            lines.append(f"| {kind} | {w['start']} to {w['end']} | {fmt_num(w['sharpe_rule'])} | "
                         f"{fmt_num(w['sharpe_bench'])} | {fmt_interval(w['interval'])} | "
                         f"{'yes' if w['zero_inside'] else 'no'} | {'yes' if w['same_sign'] else 'no'} |")
    e5 = ev["e5"]
    lines += ["", f"## E5. Neighbours of N (development sample {e5['window'][0]} to {e5['window'][1]})", "",
              "| N | Sharpe gross | Sharpe net | Benchmark Sharpe | CAGR gross | Max drawdown |", "|---|---|---|---|---|---|"]
    for r in e5["rows"]:
        lines.append(f"| {r['n']}{' (main)' if r['main'] else ''} | {fmt_num(r['sharpe_gross'])} | "
                     f"{fmt_num(r['sharpe_net'])} | {fmt_num(r['sharpe_bench'])} | {fmt_pct(r['cagr_gross'])} | "
                     f"{fmt_pct(r['max_drawdown'])} |")
    lines += ["", f"Range of Sharpe across neighbours: {fmt_num(e5['range'])}. Main N is the best of its neighbours: "
              f"{'yes' if e5['main_is_best'] else 'no'}. Main minus median of the others: "
              f"{fmt_num(e5['main_minus_median_others'])}.", "",
              "## E6. Development and evaluation", "", "| Sample | From | To |", "|---|---|---|"]
    for name, (a, b) in m["samples_all"].items():
        lines.append(f"| {name} | {a} | {b} |")
    lines += ["", f"Test opened: {', '.join(m['test_opened']) if m['test_opened'] else 'not yet'} "
              "(dates are also recorded in DECISIONS.md).", "", "## E7. Trial ledger", ""]
    t = ev["e7"]
    lines += [f"- Lines in trials.csv: {t['lines']}; configurations looked at: {t['configurations']}.",
              f"- Best in-sample (train) Sharpe: {fmt_num(t['best_in_sample_sharpe'])}; "
              f"median: {fmt_num(t['median_in_sample_sharpe'])}.",
              f"- Sharpe expected from the best of {t['configurations']} noise rules over {t['train_months']} "
              f"months: {fmt_num(t['expected_max_noise_sharpe'])} (Bailey and Lopez de Prado, 2014).", "",
              f"## E8. Downside (costs {fmt_cost(rule.COST_DEFAULT)})", "",
              "| Sample | Window | Worst month rule | Worst month bench | 5% VaR rule | 5% VaR bench |",
              "|---|---|---|---|---|---|"]
    for name, block in ev["e8"].items():
        r, b = block["rule"], block["bench"]
        a, z = m["samples"][name]
        lines.append(f"| {name} | {a} to {z} | {fmt_pct(r['worst_month'])} ({r['worst_month_date']}) | "
                     f"{fmt_pct(b['worst_month'])} ({b['worst_month_date']}) | {fmt_pct(r['var_5'])} | {fmt_pct(b['var_5'])} |")
    lines += ["", "## Data", ""]
    for s in m["sources"]:
        lines.append(f"- {s['series']}: {s['source']}; data {s['first_date']} to {s['last_date']}; "
                     f"pulled {s['pull_date']}{' (saved file, offline)' if s['offline'] else ''}.")
    lines += ["", "Historical data. Not a forecast. Not investment advice.", ""]
    (EVIDENCE_DIR / "evidence_table.md").write_text("\n".join(lines), encoding="utf-8")


def write_result_numbers(ev: dict) -> None:
    """Числа для RESULT_CARD.md. Слова вердикта и ограничений команда пишет сама."""
    m, c = ev["meta"], rule.COST_DEFAULT
    head = ev["e1"]["headline"][cost_key(c)]
    e2 = ev["e2"][cost_key(c)]
    adv = next(r for r in ev["e3"]["rows"] if r["cost"] == c)
    text = f"""# Numbers for RESULT_CARD.md (generated {m['generated']}, do not edit)

Copy the numbers; the team writes the words of the verdict and the limits.

- Rule: MA{rule.PARAM_MAIN} (10-month moving average) on {rule.ASSET}, {m['headline_window'][0]} to {m['headline_window'][1]}, monthly ({m['headline_sample']} sample).
- The rule earned {fmt_pct(head['rule']['cagr'])} a year (net of {fmt_cost(c)} per switch) against {fmt_pct(head['bench']['cagr'])} for the benchmark.
- Maximum drawdown {fmt_pct(head['rule']['max_drawdown'])} against {fmt_pct(head['bench']['max_drawdown'])}.
- Net of costs of {fmt_cost(c)} per switch the CAGR advantage is {fmt_pct(adv['cagr_advantage'])} and the Sharpe advantage is {fmt_num(adv['sharpe_advantage'])}.
- Break-even cost: {_be_text(ev['e3']['break_even_sharpe'])} (Sharpe), {_be_text(ev['e3']['break_even_cagr'])} (CAGR).
- Difference in Sharpe: {fmt_interval(e2['bootstrap'])} (paired block bootstrap, 95%); zero inside: {'yes' if e2['zero_inside'] else 'no'}.
- Start dates: the sign of the difference is the same in {sum(w['same_sign'] for w in ev['e4']['start_dates'])} of {len(ev['e4']['start_dates'])} start dates and {sum(w['same_sign'] for w in ev['e4']['subperiods'])} of {len(ev['e4']['subperiods'])} subperiods.
- Neighbours: Sharpe range {fmt_num(ev['e5']['range'])} across N = {rule.NEIGHBOURS[0]}..{rule.NEIGHBOURS[-1]}; main N is the best: {'yes' if ev['e5']['main_is_best'] else 'no'}.
- Configurations tried: {ev['e7']['configurations']} ({ev['e7']['lines']} lines in trials.csv); test period opened {len(m['test_opened'])} time(s).
"""
    (EVIDENCE_DIR / "result_card_numbers.md").write_text(text, encoding="utf-8")


# ---------- Главная функция ----------

def run(open_test_now: bool = False, reopen: bool = False, today: date = None) -> dict:
    """Один запуск: данные -> бэктест -> E1-E8 -> файлы evidence/ и строки trials.csv. Возвращает evidence."""
    today = today or date.today()
    if open_test_now:
        check_can_open(reopen)
    # сначала данные: если загрузка не удалась, test-период НЕ считается открытым
    prices, rate, sources = load_monthly(today)
    if open_test_now:
        open_test(reopen, today)
    opened = test_state()["opened"]

    tables = {n: rule.backtest(prices, rate, n) for n in rule.NEIGHBOURS}
    last = tables[rule.PARAM_MAIN].index[-1].strftime("%Y-%m")
    samples_all = {"train": rule.TRAIN, "validation": rule.VALIDATION,
                   "development": (rule.TRAIN[0], rule.VALIDATION[1]), "test": rule.TEST,
                   "full": (rule.TRAIN[0], rule.TEST[1])}
    samples = {k: v for k, v in samples_all.items() if opened or k not in ("test", "full")}
    for name, (a, b) in samples.items():
        part = rule.window(tables[rule.PARAM_MAIN], a, b)
        if len(part) < 24:
            raise EvidenceError(f"Too few months of data for the {name} sample ({a} to {b}): {len(part)}. "
                                f"The data end in {last}; run once with internet to download the full history.")
    headline = "full" if opened else "development"

    evidence = build(tables, samples, headline)
    comment = "make_evidence run" + (" (test opened now)" if open_test_now else
                                     " (re-computation of the opened test, same configuration)" if opened else "")
    append_trials(tables, samples, today, comment, test_samples=bool(opened))
    train_months = len(rule.window(tables[rule.PARAM_MAIN], *rule.TRAIN))
    evidence["e7"] = trials_summary(train_months)
    used = {name: (rule.window(tables[rule.PARAM_MAIN], *b).index[0].strftime("%Y-%m"),
                   rule.window(tables[rule.PARAM_MAIN], *b).index[-1].strftime("%Y-%m")) for name, b in samples.items()}
    evidence["meta"] = {
        "generated": today.isoformat(), "rule_version": rule.RULE_VERSION, "asset": rule.ASSET,
        "param": rule.PARAM_MAIN, "neighbours": rule.NEIGHBOURS, "costs": rule.COSTS, "cost_default": rule.COST_DEFAULT,
        "headline_sample": headline, "headline_window": used[headline], "samples": used,
        "samples_all": {k: list(v) for k, v in samples_all.items()}, "test_opened": opened,
        "data_last_month": last, "sources": sources,
        "offline": any(s["offline"] for s in sources),
    }

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    evidence["meta"]["figures"] = write_figures(evidence)
    (EVIDENCE_DIR / "evidence.json").write_text(json.dumps(evidence, indent=2, default=str), encoding="utf-8")
    write_table(evidence)
    write_result_numbers(evidence)
    return evidence


def load_evidence():
    """Готовые доказательства для Rule Lab (None, если скрипт ещё не запускали)."""
    path = EVIDENCE_DIR / "evidence.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the evidence pack E1-E8 for the rule in rule.py.")
    parser.add_argument("--open-test", action="store_true", help="open the final test period (once)")
    parser.add_argument("--reopen", action="store_true", help="declare and count a repeated opening of the test")
    args = parser.parse_args()
    try:
        ev = run(open_test_now=args.open_test, reopen=args.reopen)
    except (EvidenceError, data.DataError) as err:
        raise SystemExit(f"Error: {err}")
    head = ev["e1"]["headline"][cost_key(rule.COST_DEFAULT)]
    print(f"Evidence written to {EVIDENCE_DIR} (headline {ev['meta']['headline_sample']} "
          f"{ev['meta']['headline_window'][0]} to {ev['meta']['headline_window'][1]}).")
    print(f"Sharpe rule {fmt_sr(head['rule']['sharpe'], head['rule']['sharpe_se'])}, "
          f"benchmark {fmt_sr(head['bench']['sharpe'], head['bench']['sharpe_se'])}; "
          f"difference {fmt_interval(ev['e2'][cost_key(rule.COST_DEFAULT)]['bootstrap'])}.")
    print(f"trials.csv: {ev['e7']['lines']} lines, {ev['e7']['configurations']} configurations.")


if __name__ == "__main__":
    main()
