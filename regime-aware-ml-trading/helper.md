# Regime-Aware ML Trading Project

## 1. Project Overview

This project builds an event-driven machine learning system for intraday trading on SPY, focusing on regime-aware technical pattern detection, event labeling, and profitability-aware validation.

The goal was to move beyond bar-by-bar forecasting and instead predict whether a detected pattern event will produce a profitable trade signal under a triple-barrier regime.

---

## 2. What I Did

### 2.1 Data and Preparation
- Loaded SPY daily price data from `data/raw/spy.csv`.
- Built a clean data pipeline in `src/data/` with download, load, and indicator utilities.
- Computed ATR, returns, moving averages, volume ratios, RSI, MACD, Bollinger Bands, and other technical features.

### 2.2 Pattern Detection
- Implemented four complementary pattern detectors in `src/patterns/`:
  - `support_resistance.py` for near-support and near-resistance signals
  - `triangles.py` for ascending/descending/contraction triangles
  - `channels.py` for bullish/bearish channel formations
  - `multiple_tops_bottoms.py` for repeated extreme patterns
- Added a scanner in `src/patterns/scanner.py` to unify all detectors and produce a single event set.
- Added touch-event generation in `src/patterns/touch_events.py` to expand the dataset by capturing boundary touches and failed breakout attempts.

### 2.3 Event Labeling
- Implemented triple-barrier labeling in `src/labeling/label_events.py`.
- Each event is labeled as `long`, `short`, or `no_trade` based on which barrier is hit first:
  - Take profit barrier
  - Stop loss barrier
  - Maximum holding time expiration
- Tuned barrier parameters (`pt_mult`, `sl_mult`, `max_holding`) as hyperparameters rather than fixed constants.

### 2.4 Feature Engineering
- Built the event feature matrix in `src/features/build_features.py`.
- Included pattern geometry features, momentum and volatility signals, trend proximity, and binary technical filters.
- Removed obvious leakage sources:
  - Raw ATR, absolute SMA values, event ATR, and cumulative OBV were excluded.
  - Normalized indicators were used to avoid price-level regime leakage.

### 2.5 Model Training and Validation
- Implemented the training pipeline in `src/models/train.py`.
- Added multiple validation strategies:
  - Chronological train/validation/test split
  - Walk-forward cross-validation for temporal robustness
  - 5-fold event-level CV as a diagnostic complement
  - Train confusion matrices and individual tree diagnostics
- Trained and compared:
  - Random Forest
  - Bagging classifier
  - Dummy baseline model

### 2.6 Profitability Evaluation
- Built backtesting logic in `src/backtest/simulator.py`.
- Evaluated predictions not only by classification metrics, but also by:
  - Cumulative return
  - Sharpe ratio
  - Win rate
  - Maximum drawdown
  - Profit factor
- Linked profitability evaluation to hyperparameter selection.

---

## 3. What I Changed and Why

### 3.1 Cleaner Signal Definition
- Changed pattern detectors to return a precise signal bar for each event.
- Reason: avoid fuzzy event timing and align features with a consistent entry point.

### 3.2 Tunable Label Parameters
- Converted TP/SL/holding-time from fixed values to tunable hyperparameters.
- Reason: label design is part of the learning problem and affects both classification quality and profit outcomes.

### 3.3 Stronger Validation
- Added walk-forward CV and event-level 5-fold CV.
- Reason: a single chronological split is insufficient for time-series models, and walk-forward validation better reflects real trading deployment.

### 3.4 Leakage Prevention
- Removed features that could reveal future or regime information indirectly.
- Reason: avoid models learning from price level or trend artifacts instead of genuine event structure.

### 3.5 Dataset Expansion
- Added touch-based events to increase effective sample size.
- Result: dataset grew from ~104 detector-only events to ~142 total events (+37%).
- Reason: more events improve model training while still retaining pattern relevance.

### 3.6 Profit-Oriented Evaluation
- Added trading simulation and profit metrics.
- Reason: classification performance alone is not enough; the ultimate objective is profitable trading.

---

## 4. Implementation Details

### 4.1 Detector Improvements
- Support/Resistance: used ATR-normalized proximity, cooldown filters, and event stabilization.
- Triangles: identified swing pivots, regression lines, containment ratio, and breakout timing.
- Channels: used chunk-based extreme detection, trendline fitting, and boundary touch scoring.
- Multiple tops/bottoms: detected repeated extremes with slope confirmation.

### 4.2 Feature Set
- Trend features: SMA differences, momentum direction, price distance from moving averages.
- Volatility features: ATR ratio, Bollinger band width, realized volatility.
- Momentum features: returns over multiple horizons, RSI, MACD normalization.
- Volume and structure: volume ratios, daily volume normalization, touch counts.
- Binary technical filters: overbought/oversold, band touches, crossovers, breakout-confirmation flags.

### 4.3 Machine Learning Pipeline
- Selected tree-based ensemble models for robustness on tabular financial data.
- Used `RandomForestClassifier` and `BaggingClassifier`.
- Measured:
  - Accuracy and F1 score
  - Confusion matrices for train/validation/test
  - Individual tree performance and ensemble diversity

### 4.4 Model Diagnostics
- Individual-tree diagnostics show whether ensemble gains are coming from real diversity.
- Confusion matrices highlight class imbalance and where the model misclassifies long/short/no_trade.
- Walk-forward CV shows how model performance evolves over time and identifies temporal instability.

---

## 5. Results and Statistics Learned

### 5.1 Performance Findings
- Models are able to learn event outcomes, but performance is sensitive to label parameter choice.
- The best classification parameters are not always the most profitable.
- Profit-based optimization tends to prefer wider stop losses and longer holds compared to raw F1 optimization.

### 5.2 Data and Event Statistics
- Event dataset expanded by approximately 37% through touch-event augmentation.
- Most valuable events are those with clear geometric structure and consistent barrier spacing.
- The combined detector approach provides richer event coverage than any single pattern type.

### 5.3 Validation Insights
- Walk-forward CV reduced over-optimism compared to a single train/test split.
- Event-level 5-fold CV is useful for diagnostic comparison but should not replace temporal validation.
- Train confusion matrices show the magnitude of overfitting and the need for regularization.

### 5.4 Profitability Lessons
- A model can have good classification metrics but still fail to produce positive net trading results.
- Profit factor, Sharpe ratio, and drawdown are essential secondary metrics.
- Simple label structure with realistic exits is critical for translating predictions into tradable signals.

---

## 6. What I Learned

- Event-driven financial ML requires discipline: detector quality, label design, and leakage control are all equally important.
- Validation must reflect deployment: temporal splits and walk-forward tests are more trustworthy than random cross-validation.
- Profitability is the true objective; classification metrics are necessary but not sufficient.
- Data augmentation through touch events can help with small-sample regimes, but it must be done carefully.
- Raw technical signals and pattern geometry must be balanced to avoid overfitting to obvious price moves.

---

## 7. Project Structure

### Core folders
- `src/data/` — data loading, download, indicators
- `src/patterns/` — detectors, touch events, scanner
- `src/labeling/` — triple-barrier label generation
- `src/features/` — feature matrix construction and leakage prevention
- `src/models/` — training, validation, diagnostics
- `src/backtest/` — trading simulator and profitability metrics
- `reports/` — report and presentation generation

### Key notebooks
- `notebooks/10_model_training.ipynb`
- `notebooks/12_hyperparameter_profitability.ipynb`
- `notebooks/11_experiment_summary.ipynb`

---

## 8. Conclusions

- This project successfully built a regime-aware event classification pipeline for SPY using pattern detection and triple-barrier labeling.
- The largest improvements came from better event timing, tunable labeling, stronger validation, and profit-aware evaluation.
- The project now supports both technical learning and practical trading analysis, with clear diagnostics for model behavior.
- The next step is to continue optimizing label/hyperparameter space and verify results with out-of-sample walk-forward backtests.
