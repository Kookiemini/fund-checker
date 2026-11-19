# Decisions

Format: date — decision — alternatives considered — who decided. Never delete a line; add new ones at the end.
Openings of the test period are appended automatically by `make_evidence.py --open-test`.

- 2026-11-14 — Rule: 10-month moving average on SPY, monthly, cash at the 3-month Treasury rate when out. — Alternatives: stop-loss after an X% fall; stock/bond rebalancing threshold. Chosen because the lecture gives reference numbers to sanity-check the core. — Who decided: Aiman.
- 2026-11-14 — Data from 2000-01-01 (the app's existing download start), evaluation 2001-01 to 2026-10. — Alternative: ^GSPC from 1927 (longer, but a price index without dividends). — Who decided: Maksat.
- 2026-11-14 — N = 10 fixed in advance; neighbours 8–12 looked at for robustness; allowed range 6–14. — Alternative: choose N by the best train Sharpe (rejected: that would be fitting). — Who decided: Yersultan.
- 2026-11-14 — Samples: train 2001-01 to 2012-12, validation 2013-01 to 2018-12, test 2019-01 to 2026-10 (opened once). — Alternative: a single full-period backtest (rejected: no out-of-sample check). — Who decided: Aiman.
- 2026-11-14 — Costs 0%, 0.10%, 0.25%, 0.50% per unit of position change; 0.10% shown by default. — Who decided: Maksat.
- 2026-11-14 — Sharpe interval: paired circular block bootstrap (block 12 months, 5000 repetitions, seed 20261114) as the headline; paired analytic interval (Jobson–Korkie with the Memmel correction) next to it and for subperiods. Standard error of a single Sharpe: Lo (2002). — Who decided: Yersultan. Check against the Week 11 slides.
- 2026-11-14 — Expected best-of-noise Sharpe (E7): Bailey and López de Prado (2014). — Who decided: Aiman. Check against the formula in the Week 11 slides.
- 2026-11-14 — Before the test is opened the headline is the development sample (train + validation); after it is opened, the full window 2001-01 to 2026-10. Neighbours (E5) always on the development sample. — Who decided: Maksat.
- 2026-11-14 — We already saw the development-period results (train + validation) on 2026-11-14, when we ran `python make_evidence.py` before writing R8; R8 was written after that, and the test period 2019-01 to 2026-10 has not been opened yet. — Who decided: Yersultan.
- 2026-11-18: test period 2019-01 to 2026-10 opened (time #1) with `python make_evidence.py --open-test`. Rule v1, N = 10. Who decided: [Aiman, Maksat, Yersultan].
