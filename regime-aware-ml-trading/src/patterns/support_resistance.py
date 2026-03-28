import pandas as pd

from src.data.utils import compute_atr


def calculate_support_resistance(df, window=50, atr_mult=0.3,
                                  cooldown=10, stability_window=5):
    """Calculate dynamic support and resistance levels.

    Uses rolling max of High / rolling min of Low over *window* bars to set
    resistance and support.  Flags when price is within *atr_mult* × ATR(14)
    of either level.

    Two filters reduce false positives:
    1. **Level stability** — only flag when the level has been flat (unchanged)
       for at least *stability_window* bars.  A continuously rising resistance
       in an uptrend is not a real level being tested.
    2. **Cooldown** — after a signal fires, suppress the same signal type for
       *cooldown* bars to avoid flagging the same approach as multiple events.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: High, Low, Close
    window : int
        Rolling window size (default 50 for ~2.5 months of trading days).
    atr_mult : float
        ATR multiplier for the proximity band (default 0.3).
    cooldown : int
        Minimum bars between consecutive signals of the same type (default 10).
    stability_window : int
        Resistance/support must be unchanged for this many bars to count
        as a real level (default 5).

    Returns
    -------
    pd.DataFrame
        Original df with added columns:
        - support: rolling low-based support level
        - resistance: rolling high-based resistance level
        - near_support: True when Close is within atr_mult × ATR of support
        - near_resistance: True when Close is within atr_mult × ATR of resistance
    """
    df = df.copy()

    df["resistance"] = df["High"].rolling(window=window).max()
    df["support"] = df["Low"].rolling(window=window).min()

    atr = compute_atr(df)
    band = atr_mult * atr

    # --- Level stability filter ---
    # A level is "stable" if it hasn't changed over the last stability_window bars.
    # When resistance keeps making new highs, price is trending — not at resistance.
    res_stable = df["resistance"] == df["resistance"].shift(stability_window)
    sup_stable = df["support"] == df["support"].shift(stability_window)

    # Raw proximity conditions
    raw_near_res = ((df["Close"] - df["resistance"]).abs() <= band) & res_stable
    raw_near_sup = ((df["Close"] - df["support"]).abs() <= band) & sup_stable

    # --- Cooldown filter ---
    # After a signal fires, suppress for *cooldown* bars.
    df["near_resistance"] = _apply_cooldown(raw_near_res, cooldown)
    df["near_support"] = _apply_cooldown(raw_near_sup, cooldown)

    return df


def _apply_cooldown(signal: pd.Series, cooldown: int) -> pd.Series:
    """Keep only the first True in each cluster, then suppress for *cooldown* bars."""
    result = pd.Series(False, index=signal.index)
    bars_since_last = cooldown + 1  # allow first signal to fire

    for i in range(len(signal)):
        if signal.iloc[i] and bars_since_last > cooldown:
            result.iloc[i] = True
            bars_since_last = 0
        else:
            bars_since_last += 1

    return result
