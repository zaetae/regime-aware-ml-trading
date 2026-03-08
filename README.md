Project Title

Regime-Aware Machine Learning for Event-Driven Trading on Equity Markets

Project Summary

This project aims to investigate whether combining market regime information with event-driven technical trading signals can improve out-of-sample trading performance.

Instead of training machine learning models on all time periods, the project will focus on selected market situations where trading decisions are more meaningful, such as near horizontal support and resistance levels, at the end of triangle patterns, around multiple tops and bottoms, and near channel boundaries. These technical patterns will be used to identify candidate trading events and reduce the influence of noisy market sequences.

To model broader market conditions, a Hidden Markov Model (HMM) will be applied to return and volatility data in order to infer latent market regimes. The detected regime information will then be incorporated into the decision process, both as a predictive input and as a contextual filter for trading opportunities.

For each preselected event, machine learning models including Logistic Regression and Random Forest will be trained to predict whether to open a long position, a short position, or no position. The feature set will combine technical indicators, event-specific pattern information, and regime-related variables.

The strategy will be evaluated on SPY data using walk-forward backtesting to preserve time ordering and avoid lookahead bias. Transaction costs will be included to produce realistic performance estimates. Results will be compared with simpler baselines, including a regime-agnostic model and buy-and-hold, using metrics such as Sharpe ratio, Sortino ratio, maximum drawdown, and turnover.

The main research question is whether regime-aware, event-driven machine learning can produce more robust and realistic trading performance than models trained without event selection or regime context.

