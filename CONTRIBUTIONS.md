# Contributions

Every member must be able to explain every line of `rule.py`, the tests and every number on the Rule Lab page without a computer.

| Member | What I built | What I checked | What I can explain without a computer |
|---|---|---|---|
| Aiman | Chose the rule (10-month moving average on SPY); the toy example in `tests/test_rule.py` (`TOY_EXPECTED`, computed by hand); ran `make_evidence.py` and the app; set the train / validation / test split. | The toy test passes (125 passed); the Rule Lab page shows the same numbers as `evidence/evidence_table.md`; `trials.csv` only grows. | Signal, position and the one-month lag; the toy example step by step; why the test period is opened only once; the expected best-of-noise Sharpe (E7). |
| Maksat | Data part: SPY prices from Yahoo and the 3-month Treasury rate from FRED, monthly conversion; cost levels; the fund finder pages and `REUSE_CHANGES.md`. | Data sources, dates and offline status on the Honest panel; net vs gross returns in E3; the start page in the browser. | Where the data come from and why Adjusted Close; how costs per switch work and what the break-even cost is (E3); why the headline window is development before the test and full after it. |
| Yersultan | N = 10 fixed in advance and the neighbours 8–12 (E5); the Sharpe intervals (bootstrap and analytic); `DECISIONS.md` and `AI_USE.md`. | The canary tests fail when the lag is removed; the neighbour table (plateau or spike); the start-date and subperiod check (E4). | Sharpe and its standard error; the paired bootstrap interval and what "zero inside" means; the canary tests; why N was not chosen by the best Sharpe. |
