"""Simple trading strategy simulator for profitability evaluation.

Simulates trades based on model predictions using triple-barrier exit logic:
- LONG: enter at signal-bar close, TP = +a*ATR, SL = -b*ATR, time exit after c bars
- SHORT: symmetric opposite
- NO_TRADE: skip

Assumptions (clearly documented):
- Entry price = Close of the signal bar (the bar where the event was detected).
  This is a simplification; in live trading you would enter at the next bar's Open.
  Using signal-bar Close is consistent with the labeling pipeline.
- Transaction costs are NOT modeled (can be added via the `cost_per_trade` parameter).
- Slippage is NOT modeled.
- Position sizing is equal-weight (1 unit per trade).
- No compounding — returns are simple arithmetic.

References
----------
Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*, Ch. 3.
"""

import numpy as np
import pandas as pd

from src.data.utils import compute_atr


def simulate_trades(df, labeled_df, predictions, pt_mult=2.0, sl_mult=2.0,
                    max_holding=10, atr_window=14, cost_per_trade=0.0):
    """Simulate trades based on model predictions and triple-barrier exits.

    Parameters
    ----------
    df : pd.DataFrame
        Full OHLCV dataset with DatetimeIndex.
    labeled_df : pd.DataFrame
        Labeled events DataFrame (must have 'event_date' column).
    predictions : array-like
        Model predictions aligned with labeled_df rows.
        Values: "long", "short", "no_trade".
    pt_mult : float
        Profit target in ATR multiples.
    sl_mult : float
        Stop loss in ATR multiples.
    max_holding : int
        Maximum bars to hold before time exit.
    atr_window : int
        ATR lookback period.
    cost_per_trade : float
        Round-trip cost per trade as a fraction (e.g. 0.001 = 0.1%).

    Returns
    -------
    trades : pd.DataFrame
        One row per trade with entry/exit details and P&L.
    """
    atr = compute_atr(df, window=atr_window)
    predictions = np.asarray(predictions)

    trades = []
    for i, (_, event) in enumerate(labeled_df.iterrows()):
        pred = predictions[i]
        if pred == "no_trade":
            continue

        event_date = event["event_date"]
        if event_date not in df.index:
            continue

        pos = df.index.get_loc(event_date)
        entry_price = df["Close"].iloc[pos]
        atr_val = atr.iloc[pos]

        if pd.isna(atr_val) or atr_val <= 0:
            continue

        # Set barriers based on direction
        if pred == "long":
            tp_price = entry_price + pt_mult * atr_val
            sl_price = entry_price - sl_mult * atr_val
            direction = 1
        else:  # short
            tp_price = entry_price - pt_mult * atr_val
            sl_price = entry_price + sl_mult * atr_val
            direction = -1

        # Walk forward from pos+1
        exit_price = None
        exit_date = None
        exit_reason = None
        bars_held = 0

        end_pos = min(pos + max_holding, len(df) - 1)

        for j in range(pos + 1, min(pos + max_holding + 1, len(df))):
            high_j = df["High"].iloc[j]
            low_j = df["Low"].iloc[j]

            if pred == "long":
                hit_tp = high_j >= tp_price
                hit_sl = low_j <= sl_price
            else:  # short
                hit_tp = low_j <= tp_price
                hit_sl = high_j >= sl_price

            if hit_tp and hit_sl:
                # Both hit on same bar — use close relative to entry
                exit_price = df["Close"].iloc[j]
                exit_date = df.index[j]
                exit_reason = "both_barriers"
                bars_held = j - pos
                break
            elif hit_tp:
                exit_price = tp_price
                exit_date = df.index[j]
                exit_reason = "take_profit"
                bars_held = j - pos
                break
            elif hit_sl:
                exit_price = sl_price
                exit_date = df.index[j]
                exit_reason = "stop_loss"
                bars_held = j - pos
                break

        if exit_price is None:
            # Time exit
            exit_price = df["Close"].iloc[end_pos]
            exit_date = df.index[end_pos]
            exit_reason = "time_exit"
            bars_held = end_pos - pos

        # Calculate return
        raw_return = direction * (exit_price - entry_price) / entry_price
        net_return = raw_return - cost_per_trade

        trades.append({
            "event_date": event_date,
            "direction": pred,
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "tp_price": round(tp_price, 2),
            "sl_price": round(sl_price, 2),
            "exit_date": exit_date,
            "exit_reason": exit_reason,
            "bars_held": bars_held,
            "atr": round(atr_val, 4),
            "raw_return": round(raw_return, 6),
            "net_return": round(net_return, 6),
            "pnl_dollars": round(direction * (exit_price - entry_price), 2),
        })

    return pd.DataFrame(trades)


def compute_metrics(trades_df):
    """Compute trading performance metrics from a trades DataFrame.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Output of simulate_trades().

    Returns
    -------
    dict
        Trading metrics including cumulative return, win rate, Sharpe, etc.
    """
    if trades_df is None or len(trades_df) == 0:
        return {
            "n_trades": 0,
            "cumulative_return": 0.0,
            "avg_trade_return": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "avg_bars_held": 0.0,
            "n_long": 0,
            "n_short": 0,
            "n_tp": 0,
            "n_sl": 0,
            "n_time_exit": 0,
        }

    returns = trades_df["net_return"].values
    n_trades = len(returns)

    # Basic return metrics
    cumulative_return = float(np.sum(returns))
    avg_return = float(np.mean(returns))

    # Win rate
    winners = returns > 0
    win_rate = float(np.mean(winners)) if n_trades > 0 else 0.0

    # Profit factor = gross profit / gross loss
    gross_profit = float(np.sum(returns[returns > 0])) if np.any(returns > 0) else 0.0
    gross_loss = float(np.abs(np.sum(returns[returns < 0]))) if np.any(returns < 0) else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0.0
    )

    # Sharpe ratio (simple: mean/std of trade returns, not annualized)
    if n_trades > 1 and np.std(returns) > 0:
        sharpe = float(np.mean(returns) / np.std(returns))
    else:
        sharpe = 0.0

    # Max drawdown on cumulative equity curve
    cum_returns = np.cumsum(returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = running_max - cum_returns
    max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Trade breakdown
    n_long = int((trades_df["direction"] == "long").sum())
    n_short = int((trades_df["direction"] == "short").sum())
    n_tp = int((trades_df["exit_reason"] == "take_profit").sum())
    n_sl = int((trades_df["exit_reason"] == "stop_loss").sum())
    n_time = int((trades_df["exit_reason"] == "time_exit").sum())

    return {
        "n_trades": n_trades,
        "cumulative_return": round(cumulative_return, 6),
        "avg_trade_return": round(avg_return, 6),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else "inf",
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_drawdown, 6),
        "avg_bars_held": round(float(trades_df["bars_held"].mean()), 2),
        "n_long": n_long,
        "n_short": n_short,
        "n_tp": n_tp,
        "n_sl": n_sl,
        "n_time_exit": n_time,
    }


def evaluate_profitability(df, labeled_df, predictions, pt_mult=2.0,
                           sl_mult=2.0, max_holding=10, atr_window=14,
                           cost_per_trade=0.0):
    """One-call wrapper: simulate trades and compute metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Full OHLCV data.
    labeled_df : pd.DataFrame
        Labeled events with 'event_date'.
    predictions : array-like
        Model predictions ("long", "short", "no_trade").
    pt_mult, sl_mult, max_holding, atr_window, cost_per_trade
        Forwarded to simulate_trades().

    Returns
    -------
    metrics : dict
        Trading performance metrics.
    trades : pd.DataFrame
        Individual trade records.
    """
    trades = simulate_trades(
        df, labeled_df, predictions,
        pt_mult=pt_mult, sl_mult=sl_mult,
        max_holding=max_holding, atr_window=atr_window,
        cost_per_trade=cost_per_trade,
    )
    metrics = compute_metrics(trades)
    return metrics, trades
