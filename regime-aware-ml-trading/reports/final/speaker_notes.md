# Speaker Notes — 10-Minute BEM Defense

Supervisor: Hadházi Dániel | Budapest University of Technology and Economics | 2026

---

## Slide 1 — Title (30 sec)

"My project investigates machine learning for equity trading — specifically,
whether technical pattern detection combined with event-based learning can
predict trade outcomes on SPY daily data. I'll present three research
questions: can we beat a random baseline, do classification-optimal
parameters also maximise profit, and how stable are results across time?"

---

## Slide 2 — The Problem (55 sec)

"Let me start with why this is hard. [Point to COVID chart] This is SPY
during COVID — a 34% crash in 23 days. Daily returns swing from minus 12%
to plus 9%. Most of these bars are pure noise.

[Point to volatility chart] Over 15 years, volatility regimes shift
dramatically. A model trained in the calm 2017 market faces completely
different dynamics during COVID. This non-stationarity is the fundamental
challenge. Naive models learn time artefacts, not trading edges."

---

## Slide 3 — Research Design (40 sec)

"The pipeline wasn't designed all at once. Each step was introduced to fix
a specific problem. [Walk through chain] Bars are noisy, so we filter with
detectors. Patterns are subjective, so we formalise them. Direction labels
are naive, so we use triple-barrier. Fixed stops are arbitrary, so we
optimise them. Accuracy is misleading, so we simulate trading. And standard
CV leaks future data, so we use walk-forward. Every decision has a
concrete justification."

---

## Slide 4 — Event Detection (60 sec)

"[Point to SPY with events] Out of 4,023 bars, we flag only 104 as trading
candidates — that's 3.3%. The rest is noise we deliberately ignore.

[Point to S/R case study] Here's a real example. The red triangle is where
our detector fired. The green and red lines are the TP and SL barriers.
You can see the support level the detector identified.

[Point to multiple-top case] And here's a reversal pattern — the rolling
high hits a ceiling while the close trend turns negative. Four detectors
working together give us diverse signal types."

---

## Slide 5 — Features & Indicators (50 sec)

"[Point to indicator panel] This shows what the model actually sees for a
2022 period. Price, RSI with oversold zones highlighted, normalised MACD
showing momentum shifts, and Bollinger width capturing the volatility spike.

[Point to grouped importance] Trend features like MA distances dominate.
Momentum features like returns are second. Interestingly, pattern geometry
is secondary — the market context matters more than the pattern shape.

We deliberately removed features that leak temporal info — raw ATR, absolute
SMAs, cumulative OBV. Every feature is normalised."

---

## Slide 6 — Labeling + Failures (50 sec)

"For each event, three barriers race: take-profit, stop-loss, and time.
First hit wins. [Point to label distribution] With our best params, we
get roughly balanced classes.

[Point to failed prediction] But not every signal works. Here's a real
failure — the model predicted one outcome but the price went the other
way. This is why we need robust validation. Showing failures increases
scientific credibility."

---

## Slide 7 — Validation (55 sec)

"[Point to walk-forward diagram] Walk-forward CV always trains before
testing. The key insight is in the timeline below — each test fold spans
a different market regime. Fold 1 covers a 2018 correction, fold 3 covers
the 2022 bear. Results SHOULD vary if the model is honest.

K-fold mixes time periods and hides this regime dependence. We include it
as a diagnostic, but walk-forward is the real evaluation."

---

## Slide 8 — Central Finding (65 sec)

"Now the main result. [Point to scatter plot] Each dot is one parameter
configuration. The horizontal axis is F1, vertical is return. Notice:
the rightmost dots (highest F1) are NOT the highest dots (most profitable).
Green dots — wide stops — cluster at higher returns.

[Point to equity comparison] Left: tight stops. More trades, better F1,
but small gains. Right: wide stops. Fewer trades, worse F1, but the
winners are much larger. Tight stops get knocked out by normal volatility.
Wide stops let winners breathe.

This is the central finding: classification and profitability optimise at
different parameter settings."

---

## Slide 9 — Optimization Landscape (50 sec)

"[Point to heatmap] The divergence is visible in the full landscape.
The brightest F1 cells don't overlap with the highest return cells.

The default config gives F1 of just 0.16. After optimisation, best F1
reaches 0.569 — a 3.6x improvement. But the most profitable config has
only F1 of 0.39. It uses wider stops and longer holding. Barrier parameters
don't just tune the model — they redefine the task itself."

---

## Slide 10 — Profitability Deep Dive (55 sec)

"[Point to equity curve] Here's the test-set equity curve. Green dots are
winners, red are losers. A few large winners dominate the total return.
The drawdown panel shows the strategy gives back gains periodically.

[Point to confusion matrix] The model over-predicts long — which makes
sense for SPY's historical upward bias. Short predictions are rare and
often wrong.

[Point to fold equities] Per-fold equity curves show extreme variation.
One fold profits, another barely breaks even. This is the reality with
17 test events per fold."

---

## Slide 11 — Generalization & F-Beta (50 sec)

"[Point to variability bars] F1 is reasonably stable at 0.28, but
cumulative return swings from near-zero to 7%. Sharpe ratio varies 0 to
0.3. The signal exists but it's thin.

[Point to F-beta] Precision is 0.30, recall is 0.32 — nearly balanced.
F0.5 and F2 scores are similar, meaning the model isn't strongly biased
toward either type of error. In trading terms: we're not clearly better
at avoiding bad trades vs capturing good ones.

The walk-forward variance is the most important diagnostic. It tells us
the 25.9% return is a peak, not a stable estimate."

---

## Slide 12 — Conclusion (45 sec)

"Four findings. One: event filtering works — 3.6x F1 over baseline.
Two: classification doesn't equal profitability — this is robust across
folds. Three: walk-forward CV reveals what k-fold hides. Four: the signal
is real but fragile.

The contribution is the framework and the finding, not a deployable
strategy. Future work needs more data, more assets, and transaction costs.
Thank you — I'm happy to take questions."

---

## Timing

| Slide | Topic | Time | Cumulative |
|-------|-------|------|-----------|
| 1 | Title | 0:30 | 0:30 |
| 2 | Problem | 0:55 | 1:25 |
| 3 | Design | 0:40 | 2:05 |
| 4 | Detection | 1:00 | 3:05 |
| 5 | Features | 0:50 | 3:55 |
| 6 | Labeling | 0:50 | 4:45 |
| 7 | Validation | 0:55 | 5:40 |
| 8 | Central finding | 1:05 | 6:45 |
| 9 | Optimization | 0:50 | 7:35 |
| 10 | Profitability | 0:55 | 8:30 |
| 11 | Generalization | 0:50 | 9:20 |
| 12 | Conclusion | 0:45 | 10:05 |
