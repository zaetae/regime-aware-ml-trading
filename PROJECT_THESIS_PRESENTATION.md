# Regime-Aware ML Trading: Full Technical Thesis Presentation

## 1. Project Motivation

This project is solving a core problem in financial machine learning:

- Financial price series are extremely noisy.
- Most bars are not meaningful trading opportunities.
- Naive price prediction often learns noise instead of structure.

Instead of “predict next price,” the project asks:

- “When price is at a technically meaningful structure, can we predict the outcome of a trade-sized move?”

### Why financial ML / technical pattern detection matters

Technical trading seeks decision points where the market is more likely to behave predictably.

- Patterns like support/resistance, triangles, channels, and multiple tops/bottoms are classic structures where traders expect a reaction.
- Combining these structures with ML is meant to focus learning on moments with real signal.

### Why naive price prediction is difficult

Raw price prediction fails because:

- Markets are almost random in the short term.
- Price movements are affected by regime shifts, news, and structural market changes.
- Most price bars do not contain a repeatable signal.
- A model trained on every bar usually learns to exploit price level or time trends rather than actionable setups.

### Why event-based learning was chosen

Event-based learning was chosen because:

- It reduces the problem from “predict every bar” to “predict only candidate trade bars.”
- It focuses the model on actual trading decisions.
- It allows labeling based on trade outcomes instead of arbitrary future direction.
- It is more robust to noise and regime shifts.

---

## 2. Financial & Technical Analysis Foundations

The project builds on standard charting concepts. Below are the concepts explained intuitively and technically.

### OHLC candles

Intuition:

- Every trading bar represents an interval of activity.
- Open/High/Low/Close summarize the range and direction of that interval.

Technical:

- `Open`: first traded price
- `High`: maximum price
- `Low`: minimum price
- `Close`: last traded price
- All indicators and patterns are computed from these values.

### Trends

Intuition:

- Trends are the market’s direction.
- Uptrend = higher highs and higher lows.
- Downtrend = lower highs and lower lows.

Technical:

- Identified by moving averages or slope of recent closes.
- Trend context matters because the same pattern can have different implications depending on trend.

### Support / Resistance

Intuition:

- Support is a price level where buyers tend to arrive.
- Resistance is a level where sellers tend to arrive.

Technical:

- In this project, support = rolling minimum of `Low` over a window.
- Resistance = rolling maximum of `High`.
- A bar is “near” a level if its `Close` is within `ATR * 0.3`.
- Stability filters require the level to be unchanged for several bars.

### Channels

Intuition:

- A channel is a trend bounded by two parallel lines.
- Price oscillates inside the channel.

Technical:

- The channel detector fits upper and lower lines through chunked extremes.
- It uses chunked highs and lows, linear regression, and adjustments so the lines wrap the data.
- It requires:
  - minimum touches on both boundaries,
  - good containment,
  - parallel slopes,
  - reasonable width in ATR units,
  - price touching the boundary with rejection.

### Triangles

Intuition:

- A triangle is price compression between converging lines.
- Breakouts tend to happen near the apex.

Technical:

- The triangle detector finds swing highs/lows and fits regression lines.
- It checks:
  - line fit quality (`|r| >= 0.9`),
  - convergence of the range,
  - containment of recent bars,
  - breakout beyond recent highs/lows by `0.3 * ATR`.
- It classifies ascending, descending, and symmetric triangles.

### Multiple tops/bottoms

Intuition:

- These are repeated tests of the same ceiling or floor.
- They suggest exhaustion of buyers or sellers.

Technical:

- Identified using rolling high/low extremes and close-slope confirmation.
- It uses a longer confirmation window and cooldown to avoid noise.

### ATR

Intuition:

- ATR measures how much price moves, on average.

Technical:

- `ATR(14)` is computed from true range:
  `max(High−Low, |High−Close_prev|, |Low−Close_prev|)`
- It is used to normalize barrier widths and proximity bands.

### RSI

Intuition:

- RSI measures whether price is overbought or oversold.

Technical:

- Computed over 14 bars.
- `RSI = 100 - 100 / (1 + RS)` where `RS = avg gain / avg loss`.

### MACD

Intuition:

- MACD measures momentum divergence between fast and slow averages.

Technical:

- `MACD = EMA12(Close) - EMA26(Close)`
- `Signal = EMA9(MACD)`
- `Histogram = MACD - Signal`
- In this project, MACD is normalized by `Close` to avoid price-level leakage.

### Bollinger Bands

Intuition:

- Bands measure volatility around a moving average.
- When price touches a band, momentum may be extreme.

Technical:

- `Upper = SMA20 + 2*StdDev20`
- `Lower = SMA20 - 2*StdDev20`
- Features include width and `%B`.

### Moving averages

Intuition:

- MAs smooth price and identify trend direction.

Technical:

- Used windows: 10, 20, 50, 100, 200.
- Distance features: `(Close − SMA) / SMA`.
- MA spreads are gaps between key MAs.

### Volatility

Intuition:

- Volatility is price variability.

Technical:

- Measured by ATR ratio and rolling std of log returns.
- It determines the appropriate scale for profit targets and stop losses.

### Momentum

Intuition:

- Momentum measures how fast price is moving.

Technical:

- Features include returns over 1, 5, 10, 20 bars and momentum ratios.

### What “signals” mean in trading

Intuition:

- A signal is a candidate entry trigger.

Technical:

- In this project, signals are pattern detections or touches.
- They are not trade orders yet — they are events for labeling.

### What “events” mean in this project

Intuition:

- An event is the unit the model trains on: one moment when a signal occurs.

Technical:

- Each event row has:
  - event date,
  - event type,
  - features at that moment,
  - a triple-barrier label outcome.

---

## 3. Initial System Architecture

The project pipeline is:

1. Data collection
2. Detector layer
3. Event generation
4. Feature engineering
5. Labeling
6. ML training
7. Validation

### Why each stage exists

- **Data collection** ensures clean OHLCV input.
- **Detector layer** finds candidate bars that are meaningful.
- **Event generation** converts pattern hits into rows.
- **Feature engineering** creates model inputs from past data.
- **Labeling** defines a tradable outcome.
- **ML training** learns the mapping from features to labels.
- **Validation** estimates how the model performs out-of-sample.

This structure prevents the model from seeing irrelevant bars and enforces a consistent trading interpretation.

---

## 4. Pattern Detectors

The repository contains four detectors.

### Support / resistance

- **Goal:** identify bars near stable horizontal levels.
- **Logic:** rolling max/min over 50 bars plus ATR proximity.
- **Thresholds:** `0.3 * ATR` proximity; stability over 5 bars; cooldown 10 bars.
- **Validation:** level must remain unchanged for `stability_window`.
- **Localization:** event bar is the bar near the level.
- **Initial problem:** raw proximity could flag trending levels.
- **Improvement:** stability filter and cooldown.
- **Why it matters:** it defines genuine tests rather than moving levels.

### Channels

- **Goal:** identify parallel upper and lower boundaries that contain price.
- **Logic:** chunk windowed highs/lows, fit lines, adjust intercepts, choose tightest channel.
- **Thresholds:**
  - minimum upper touches=2, lower touches=3
  - width between 1 and 6 ATR
  - containment ≥ 70%
  - slope parallelism tolerance 0.25
  - boundary touch within `0.3 * ATR`
- **Validation:** swing pivot touch count, containment ratio, rejection check.
- **Localization:** yellow diamond on the bar touching the channel boundary and rejecting.
- **Initial problem:** too many weak or sloppy channels.
- **Improvements:** added swing pivot anchoring, confidence scoring, rejection penalty.
- **Why it matters:** channels are actionable only when the boundary is credible.

### Triangles

- **Goal:** detect compressing price ranges before breakouts.
- **Logic:** find swing highs/lows, fit trendlines, require convergence and breakout.
- **Thresholds:**
  - `|r| >= 0.9` on each trendline
  - compression ≥ 5%
  - containment ≥ 80%
  - breakout threshold `0.3 * ATR`
- **Validation:** line fit quality, containment, breakout confirmation.
- **Localization:** diamond on the breakout bar or descending triangle upper-test bar.
- **Initial problem:** weak fit triangles and premature signals.
- **Improvements:** stronger quality gating and ATR-normalized slopes.
- **Why it matters:** triangle breakouts signal directional resolution.

### Multiple tops/bottoms

- **Goal:** identify repeated high or low tests with reversal bias.
- **Logic:** rolling extreme checks plus recent close slope confirmation.
- **Thresholds:**
  - window 20
  - confirmation bars 5
  - cooldown 10
- **Validation:** the close trend must slope opposite the tested extreme.
- **Localization:** event bar is where the repeated extreme and slope reversal align.
- **Initial problem:** too much noisy clustering.
- **Improvements:** longer confirmation window and cooldown.
- **Why it matters:** captures classic reversal formations.

---

## 5. Signal Localization

### What the yellow diamond means

The yellow diamond indicates the exact event bar where the setup is considered to occur.

It is not just a pattern region; it is the timing of the decision.

### How localization differs per detector

- **Support/resistance:** the bar closest to the level.
- **Channels:** the bar where price touches and rejects from the boundary.
- **Triangles:** the breakout bar or upper-test bar.
- **Multiple tops/bottoms:** the bar where the pattern becomes confirmed.

### Why this became important after supervisor feedback

Supervisor feedback emphasized:

- start sequences from direct touches of trend lines
- align event timing with the actionable moment

This forced the project to tighten signal localization, not just detect generic pattern structure.

### Why event timing matters for labeling and ML

If timing is wrong:

- feature values do not describe the real decision point
- labels may be associated with the wrong bar
- the model learns false patterns
- trading simulation becomes unrealistic

So localization is essential for correct label alignment.

---

## 6. Feature Engineering

The feature matrix is built from:

- trend features
- volatility features
- momentum features
- volume features
- geometry features
- event metadata
- binary technical signals

### Trend features

- **Measures:** current trend and mean reversion tension.
- **Calculated by:** SMA values and relative distances; MA spreads.
- **Why it helps:** situates the event within trend context.
- **Example:** if price is far above SMA200, long signals at resistance are different.
- **Weakness:** absolute SMA values were dropped to avoid leakage.

### Volatility features

- **Measures:** market variability.
- **Calculated by:** ATR ratio, rolling volatility, Bollinger band width.
- **Why it helps:** volatility scales risk and target size.
- **Example:** a wide ATR means larger barriers needed.
- **Weakness:** volatility features can be regime-specific.

### Momentum features

- **Measures:** recent price momentum.
- **Calculated by:** returns and momentum ratios over 5, 10, 20 bars; RSI.
- **Why it helps:** shows whether the market is already moving.
- **Example:** a channel test with rising momentum is more likely to break.
- **Weakness:** momentum can reverse suddenly.

### Volume features

- **Measures:** trading activity strength.
- **Calculated by:** volume relative to 20-bar average and normalized std.
- **Why it helps:** volume can confirm or deny moves.
- **Example:** a breakout with low volume may be suspicious.
- **Weakness:** volume data is noisy and can be lagging.

### Geometry features

- **Measures:** pattern quality.
- **Calculated by:** slopes, containment ratios, touch counts, width in ATR, mean errors.
- **Why it helps:** distinguishes strong patterns from weak patterns.
- **Example:** a channel with many touches and tight containment is more reliable.
- **Weakness:** only available for events with geometry details.

### Event metadata

- **Measures:** event type identity.
- **Calculated by:** one-hot encoding of `event_type`.
- **Why it helps:** allows different treatment of channels, triangles, etc.
- **Example:** `channel_up` and `near_support` can behave differently.
- **Weakness:** splits the dataset into smaller categories.

### Binary technical signals

- **Measures:** simple technical conditions.
- **Calculated by:** Bollinger touches, SMA crossovers, SMA proximity, RSI extremes.
- **Why it helps:** adds interpretable filters.
- **Example:** `rsi_overbought` is a simple momentum warning.
- **Weakness:** binary signals can be sparse and redundant.

---

## 7. Data Leakage & Preprocessing

Leakage is the fatal hidden enemy in financial ML.

### What leakage means

- Feature values encode future information or time-specific trends.
- It makes backtests unrealistically optimistic.

### Why trend leakage was dangerous

- Market price levels trend upward over time.
- A model can cheat by using price-level proxies instead of real signal.
- This is especially bad with a single asset like SPY.

### Why `atr_14`, `event_atr`, `obv_norm` were problematic

- `atr_14` raw scales with price and carries a time trend.
- `event_atr` directly ties to the barrier size and future outcome.
- `obv_norm` is cumulative and therefore accumulates a trend.

These were removed or avoided.

### Why raw MACD was problematic

- Raw MACD values also scale with price level.
- If SPY drifts higher, raw MACD drifts too.

### How normalization fixed this

- `macd_norm = MACD / Close`
- `atr_ratio = ATR / Close`
- `sma_dist = (Close - SMA) / SMA`

Normalization removes explicit level dependence and makes features comparable across history.

### How leakage was prevented throughout the project

- only current/past bar data is used for features
- raw ETF price-level features are dropped
- entry price and event ATR are excluded from model inputs
- labeling uses a separate future walk-forward loop
- train/test splits are chronological

---

## 8. Triple-Barrier Labeling

This is the project’s core labeling method.

### Why standard labels are weak

- Binary “up/down next day” ignores risk and holding period.
- It merges trades of all sizes and durations.
- It is not directly tradable.

### How triple-barrier labeling works

For each event:

- `upper_barrier = entry_price + pt_mult * ATR`
- `lower_barrier = entry_price - sl_mult * ATR`
- `time barrier = event bar + max_holding bars`

The algorithm walks forward:

- if `High` hits upper first → `long`
- if `Low` hits lower first → `short`
- if neither in time → `no_trade`

If both occur the same bar, the Close decides direction.

### TP/SL/max_holding logic

- `pt_mult`: profit target multiple
- `sl_mult`: stop loss multiple
- `max_holding`: maximum bars held before expiry

These parameters define the trade’s risk/reward structure.

### ATR-based barriers

- ATR makes the barrier adaptive to volatility.
- In quiet markets, the barrier is narrow.
- In volatile markets, it is wider.

### Walk-forward labeling

- Labels are created by scanning forward from each event.
- It uses only future bars after the event.
- It avoids feature contamination.

### Why this creates better ML targets

- Labels now represent a tradable decision.
- They encode both direction and whether the move was significant enough.
- They reduce noise compared to simple direction labels.

### Later realization

- TP/SL/max_holding should be hyperparameters.
- The choice of label definition changes what the model can learn.
- A label set that is easy to classify may not be the most profitable.

---

## 9. Machine Learning Models

### Why Random Forest and Bagging were selected

- They are robust to heterogeneous tabular data.
- They handle nonlinear interactions automatically.
- They are less likely to overfit than a single decision tree.
- They are easier to interpret than some neural networks.

### How tree models work intuitively

- Each tree partitions feature space with rules.
- Example: “if RSI > 70 and channel width < 2 ATR, then …”
- Trees capture simple decision logic.

### Ensemble learning

- Multiple trees are combined.
- Random Forest: bootstrap samples + random feature subsets.
- Bagging: bootstrap samples + full features.
- Voting across trees reduces variance.

### Feature splits

- Trees select thresholds that best separate class labels.
- This is useful for technical indicators, which often have threshold-based meaning.

### Voting

- Each tree casts a vote.
- The ensemble prediction is the majority class.
- This stabilizes the output.

### Why trees suit tabular financial data

- No need for heavy feature scaling.
- Works with mixture of numerical and binary features.
- Handles missing values and nonlinear interactions well.

---

## 10. Validation Methods

The project uses several validation strategies.

### Chronological split

- train = first 60%
- validation = next 20%
- test = last 20%

Why:

- preserves time order
- prevents future data leaking into training

Strengths:

- realistic
- simple

Weaknesses:

- may not sample all regimes evenly
- one unlucky final period can dominate test results

### Walk-forward CV

- dataset split into 5 chronological folds
- each fold uses all prior data to train and the next fold to test

Why:

- simulates the way models are updated over time
- provides multiple out-of-sample windows

Strengths:

- temporal realism
- better stability estimate

Weaknesses:

- small event counts can make fold results noisy

### Event-level 5-fold CV

- contiguous folds across events
- rotates each fold as test

Why:

- complementary diagnostic to measure generalization
- more data-efficient

Strengths:

- uses more data for validation
- reduces sampling noise

Weaknesses:

- does not fully respect time ordering
- not suitable as the final evaluation

### Train/validation/test confusion matrices

- used to inspect classification quality on each split
- allows detection of overfitting and label bias

### Per-fold evaluation

- evaluates performance fold by fold
- reveals stability or inconsistency across time periods

Supervisor feedback emphasized:

- use walk-forward validation
- do not rely on a single static train/test split
- include profitability metrics in each fold

---

## 11. Individual Tree Diagnostics

### Why evaluating individual trees matters

- It reveals whether ensemble strength comes from many trees or a few strong ones.
- It helps diagnose whether diversity is real.

### What was discovered

- ensemble accuracy can exceed the mean individual tree accuracy
- this indicates that averaging weak learners improves robustness
- if tree accuracies are widely spread, the ensemble may be relying on variance reduction

### Ensemble diversity

- Random feature selection and bootstrapping create diversity.
- Diversity is beneficial because different trees make different mistakes.

### Why ensemble performance can exceed individual trees

- because errors are averaged out
- because the ensemble can capture more complex patterns than any single tree

---

## 12. Touch Events Expansion

### Why dataset size was a limitation

- The original detector-only pipeline produced only about 104 events.
- This is a very small sample for ML.

### Why touch-events were introduced

- Supervisor feedback urged more direct touch-based sequence starts.
- Touch events expand the dataset without changing the core labeling scheme.

### How they are generated

- Support/resistance touches use a tighter ATR band (`0.2 * ATR`).
- Channel touches detect when price touches the channel trendline.
- A cooldown prevents clustered repeated touch events.

### Advantages

- dataset increases from 104 to 142 events
- adds more real entry moments
- improves opportunities for the model to learn

### Risks / noise tradeoff

- touch events are inherently weaker signals than full patterns
- they can introduce noise
- careful cooling and filtering are needed

---

## 13. Hyperparameter Optimization

### Why TP/SL/max_holding became hyperparameters

- They define the labeling problem.
- Different values create different label distributions.
- The model’s efficacy depends on label difficulty as much as feature quality.

### Optuna / grid search logic

- the project precomputes indicators and pattern scans once
- only the labeling step is re-run per candidate parameter set
- this is efficient because the expensive pattern detection is reused

### Optimization objectives

- classification: `accuracy`, `f1_macro`
- profitability: `cumulative_return`, `sharpe`, `profit_factor`, `win_rate`

### Classification vs profitability tradeoff

- best classification parameters were:
  - `pt_mult = 2.0`
  - `sl_mult = 1.5`
  - `max_holding = 10`
- best profit parameters were:
  - `pt_mult = 2.5`
  - `sl_mult = 3.0`
  - `max_holding = 20`

This proves that the objective matters: the most learnable labels are not always the most profitable.

---

## 14. Trading Simulation / Backtesting

### Why accuracy alone is insufficient

- a model can have good classification but still lose money
- trade outcomes matter more than class labels

### How trading simulation works

- entry at the event bar close
- set TP and SL based on ATR
- walk forward up to `max_holding` bars
- exit on take profit, stop loss, or time expiry
- compute returns per trade

### Profitability metrics

- `cumulative return`: sum of trade returns
  - financial meaning: total edge from the strategy
- `Sharpe ratio`: mean return / std deviation
  - financial meaning: risk-adjusted return
- `win rate`: percentage of winning trades
  - financial meaning: hit rate, but not enough alone
- `drawdown`: largest peak-to-trough loss in cumulative return
  - financial meaning: risk of losing streaks
- `profit factor`: gross profit / gross loss
  - financial meaning: how well winners cover losers

These metrics ensure that evaluation is aligned with real trading.

---

## 15. Evolution of the Project

### Initially

- the project aimed to build a regime-aware event-driven ML strategy.
- main components were pattern detectors and triple-barrier labeling.

### Major problems discovered

- small event set
- weak test performance
- leakage from raw features
- gap between classification and profit

### Supervisor feedback

- add direct touch events
- localize signals precisely
- evaluate with walk-forward backtesting
- treat target barriers as tunable

### Fixes implemented

- added touch-based event generation
- normalized leak-prone indicators
- introduced more robust detector validation
- added hyperparameter optimization over label parameters
- improved validation methodology

### Architecture evolution

- from a static signal-label-predict pipeline to a dynamic one where label definitions are also tuned
- from pattern-only events to pattern + touch events
- from classification-only evaluation to combined profitability evaluation

---

## 16. Current Final System

### Final workflow

1. Load SPY OHLCV data.
2. Compute support/resistance, triangles, channels, multiple tops/bottoms.
3. Optionally generate touch-based events.
4. Compute a broad indicator set.
5. Build event features at each signal bar.
6. Label events using triple-barrier logic.
7. Train Random Forest and Bagging classifiers.
8. Validate using chronological splits, walk-forward CV, and event-level CV.
9. Simulate trades and compute profitability metrics.

This is the current architecture supported by the repo.

---

## 17. Final Experimental Results

### Event counts

From the outputs:

- Near support: 5
- Near resistance: 37
- Triangles: 22
- Multiple top/bottom: 63
- Channels: 12

### Touch events growth

- Original detector events: 104
- Touch events added: 38
- Total events: 142

### Validation results

From `model_comparison.csv`:

- Random Forest validation accuracy: 0.4815
- Random Forest validation F1 macro: 0.3926
- Random Forest test accuracy: 0.2857
- Random Forest test F1 macro: 0.1616
- Bagging test accuracy: 0.3571
- Bagging test F1 macro: 0.2753
- Baseline test accuracy: 0.4643
- Baseline test F1 macro: 0.16

### RF vs baseline

- The baseline outperformed RF on test accuracy.
- Bagging was stronger than RF but still below baseline accuracy.
- This suggests the signal is weak and the model struggles to generalize.

### Best hyperparameters

- Classification best: `pt_mult=2.0`, `sl_mult=1.5`, `max_holding=10`
- Profit best: `pt_mult=2.5`, `sl_mult=3.0`, `max_holding=20`

### Confusion matrix conclusions

- the model struggles with multi-class balance
- class confusion and low F1 indicate weak predictive structure
- baseline performance is hard to beat

### Overfitting / generalization

- the drop from validation to test indicates possible overfitting or regime mismatch.
- the dataset is small, so generalization is difficult.

### Classification vs profitability findings

- the project’s own results confirm:
  - classification optimization is not the same as profitability optimization
  - label design matters as much as model selection

---

## 18. Key Scientific Findings

- Best classification parameters do not equal best profitability parameters.
- Leakage strongly affects financial ML and must be managed explicitly.
- Event quality matters more than quantity: weak touch events can increase size but also noise.
- Profitability evaluation is critical, not optional.
- Ensemble methods add stability but do not guarantee profit.
- Label design influences both learnability and trading outcome.

---

## 19. Remaining Limitations

The honest limitations are:

- very small dataset
- only one asset (SPY)
- no transaction costs included
- possible overfitting from hyperparameter search
- touch events introduce noise as well as volume
- limited coverage of market regimes
- regime-aware modeling is conceptually present but not fully implemented in source code

---

## 20. Future Improvements

A sensible roadmap includes:

- adding more assets for generalization
- introducing explicit regime modeling or HMM-based regime features
- using purged or embargoed cross-validation
- modeling transaction costs and slippage
- exploring sequential models like transformers or LSTMs
- trying reinforcement learning or online learning
- making TP/SL dynamic based on regime or event quality

---

## 21. Final Thesis-Style Conclusion

What was achieved:

- A complete event-driven pipeline from pattern detection to profitability simulation.
- A disciplined feature engineering process with leakage control.
- A demonstration that label definition is part of the model.
- A careful validation strategy combining time splits, walk-forward CV, and profitability metrics.

What the system demonstrates:

- It is possible to frame financial ML as a prediction problem over structured technical events rather than raw price bars.
- It is essential to evaluate models by actual trade outcomes, not just accuracy.

What was learned:

- Technical patterns provide useful candidate events, but they are not automatic profits.
- Normalization and leakage prevention are essential.
- Barrier parameters are hyperparameters too.
- The best model for classification is not necessarily the best for trading.

This work is a realistic project defense narrative: it shows where the approach works, where it does not, and what was learned scientifically from the empirical results.
