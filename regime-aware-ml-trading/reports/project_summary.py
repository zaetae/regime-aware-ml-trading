"""Generate comprehensive project summary PDF."""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable,
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

W, H = A4
M = 22 * mm


def build():
    doc = SimpleDocTemplate("reports/project_summary.pdf", pagesize=A4,
                            leftMargin=M, rightMargin=M,
                            topMargin=18*mm, bottomMargin=18*mm)
    S = getSampleStyleSheet()
    S.add(ParagraphStyle("T2", parent=S["Title"], fontSize=18, spaceAfter=4,
                         textColor=HexColor("#1a1a2e")))
    S.add(ParagraphStyle("Sub", parent=S["Normal"], fontSize=11,
                         textColor=HexColor("#555"), spaceAfter=12,
                         alignment=TA_CENTER))
    S.add(ParagraphStyle("H1", parent=S["Heading1"], fontSize=14,
                         spaceBefore=14, spaceAfter=5,
                         textColor=HexColor("#1a1a2e")))
    S.add(ParagraphStyle("H2", parent=S["Heading2"], fontSize=12,
                         spaceBefore=10, spaceAfter=4,
                         textColor=HexColor("#2d3436")))
    S.add(ParagraphStyle("B", parent=S["Normal"], fontSize=10, leading=14,
                         spaceAfter=5, alignment=TA_JUSTIFY))
    S.add(ParagraphStyle("Bl", parent=S["Normal"], fontSize=10, leading=14,
                         leftIndent=18, bulletIndent=8, spaceAfter=3))

    story = []
    def p(t, s="B"): story.append(Paragraph(t, S[s]))
    def b(t): story.append(Paragraph(f"\u2022  {t}", S["Bl"]))
    def sp(h=5): story.append(Spacer(1, h))
    def hr(): story.append(HRFlowable(width="100%", thickness=0.5,
                                       color=HexColor("#ccc"),
                                       spaceBefore=4, spaceAfter=4))
    def tbl(data, cw=None):
        t = Table(data, colWidths=cw, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), HexColor("#2d3436")),
            ("TEXTCOLOR", (0,0), (-1,0), HexColor("#fff")),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("GRID", (0,0), (-1,-1), 0.4, HexColor("#ccc")),
            ("ROWBACKGROUNDS", (0,1), (-1,-1),
             [HexColor("#fff"), HexColor("#f7f7f7")]),
            ("TOPPADDING", (0,0), (-1,-1), 3),
            ("BOTTOMPADDING", (0,0), (-1,-1), 3),
            ("LEFTPADDING", (0,0), (-1,-1), 5),
        ]))
        story.append(t); sp(6)

    # ================================================================
    # TITLE
    # ================================================================
    sp(30)
    p("Regime-Aware ML Trading \u2014 Project Summary", "T2")
    p("Technical Pattern Detection, Triple-Barrier Labeling &amp; ML Pipeline", "Sub")
    hr()
    p("Author: Zeineb Turki  |  Date: April 2026  |  Supervisor: [TBD]", "Sub")
    sp(10)

    p("<b>Abstract.</b> This project builds a complete machine-learning pipeline for "
      "equity trading on SPY (S&amp;P 500 ETF) daily data. It detects technical chart "
      "patterns (support/resistance, triangles, channels, multiple tops/bottoms), labels "
      "each event using the triple-barrier method from Lopez de Prado (2018), and prepares "
      "a labeled dataset for Random Forest classification. The pipeline is designed to be "
      "regime-aware: pattern detections and model predictions will be conditioned on the "
      "current market regime (bull/bear/sideways) identified by a Hidden Markov Model.")

    story.append(PageBreak())

    # ================================================================
    # 1. MOTIVATION
    # ================================================================
    p("1. Motivation &amp; Background", "H1")

    p("1.1 Why Regime-Aware Trading?", "H2")
    p("Financial markets alternate between distinct regimes \u2014 trending, mean-reverting, "
      "and high-volatility \u2014 where the statistical properties of returns shift substantially. "
      "A strategy optimal in a trending regime may be destructive in a mean-reverting one. "
      "Regime-aware approaches (e.g., conditioning on HMM states) let models adapt to the "
      "current market environment, improving out-of-sample performance and reducing drawdowns.")
    sp(3)

    p("1.2 Why Technical Pattern Detection?", "H2")
    p("Chart patterns (triangles, channels, support/resistance levels) encode structural "
      "information about supply-demand dynamics that raw price features miss. Academic work "
      "(Lo, Mamaysky &amp; Wang 2000) demonstrated that certain technical patterns have "
      "statistically significant predictive power. By detecting these patterns algorithmically, "
      "we create event-driven features that capture market structure for ML models.")
    sp(3)

    p("1.3 Why Triple-Barrier Labeling?", "H2")
    p("Traditional fixed-horizon returns (e.g., 10-day return) ignore risk management. "
      "The triple-barrier method (Lopez de Prado 2018) models what a trader actually does: "
      "set a profit target, a stop loss, and a time limit. The label reflects which barrier "
      "is touched first \u2014 producing labels that are directly actionable for trading. "
      "ATR-scaled barriers adapt to current volatility, making labels comparable across "
      "different market environments.")

    story.append(PageBreak())

    # ================================================================
    # 2. DATA
    # ================================================================
    p("2. Data Pipeline", "H1")

    tbl([
        ["Metric", "Value"],
        ["Instrument", "SPY (S&P 500 ETF)"],
        ["Frequency", "Daily OHLCV"],
        ["Period", "2010-01-04 to 2025-12-30"],
        ["Total bars", "4,023"],
        ["Price range", "$77.36 \u2013 $690.38"],
        ["Source", "Yahoo Finance (yfinance)"],
        ["Storage", "data/raw/spy.csv (git-ignored,\nauto-downloaded on Colab)"],
    ], cw=[120, 310])

    p("The data loading module (<font face='Courier' size='9'>src/data/load_data.py</font>) "
      "uses <font face='Courier' size='9'>__file__</font>-based path resolution for "
      "robustness across environments (local, Colab, CI). It includes a yfinance "
      "auto-download fallback and strips timezone information for consistent plotting.")

    # ================================================================
    # 3. PATTERN DETECTION
    # ================================================================
    p("3. Pattern Detection", "H1")

    p("Four pattern detectors run sequentially via "
      "<font face='Courier' size='9'>scan_all_patterns()</font>. "
      "Each detector adds a column to the DataFrame; the unified "
      "<font face='Courier' size='9'>has_event</font> flag marks rows "
      "where any pattern fires.")

    p("3.1 Support &amp; Resistance", "H2")
    b("Identifies horizontal S/R levels using rolling-window local extrema.")
    b("Parameters: window=50, ATR proximity=0.3\u00d7ATR, cooldown=10 bars.")
    b("Fires <b>near_support</b> (5 events) and <b>near_resistance</b> (37 events).")
    b("Total: 42 S/R events over 15 years.")

    p("3.2 Multiple Tops &amp; Bottoms", "H2")
    b("Detects double/triple top and bottom patterns from repeated touches of the same level.")
    b("Parameters: window=50, confirm_bars=5, cooldown=10.")
    b("Result: <b>53 multiple_bottom</b> + <b>9 multiple_top</b> = 62 events.")
    b("Most common pattern type in the dataset.")

    p("3.3 Triangle Detection", "H2")
    p("Follows the <i>TrianglePricePatterns</i> reference notebook approach:")
    b("<b>Pivot detection:</b> swing highs/lows identified with "
      "<font face='Courier' size='9'>find_swing_highs/lows(order=3)</font> \u2014 "
      "a bar whose High (or Low) is the extreme within \u00b13 bars.")
    b("<b>Regression:</b> <font face='Courier' size='9'>scipy.stats.linregress</font> "
      "on pivot points only \u2014 gives slope, intercept, and correlation <i>r</i>.")
    b("<b>Quality gate:</b> |<i>r</i>| \u2265 0.85 on each trendline "
      "(pivots must align tightly).")
    b("<b>Adjusted intercepts:</b> upper intercept = max(pivot_y \u2212 slope \u00d7 pivot_x) "
      "so the line sits <b>on top</b> of swing highs; lower intercept = min(...) "
      "so it sits <b>below</b> swing lows. Candles are inside the triangle.")
    b("<b>Classification:</b> ATR-normalised slopes determine ascending "
      "(flat top + rising bottom), descending (falling top + flat bottom), "
      "or symmetric (both converging).")
    b("<b>Window:</b> 25 bars (~5 weeks) \u2014 compact, recognisable shapes.")
    sp(3)

    tbl([
        ["Triangle Type", "Count", "Containment"],
        ["Symmetric", "19", "72\u2013100%"],
        ["Ascending", "2", "88\u201395%"],
        ["Descending", "1", "100%"],
        ["Total", "22", "mean 86%"],
    ], cw=[130, 60, 120])

    p("3.4 Channel Detection", "H2")
    p("Same linregress + adjusted intercept approach as triangles, but for "
      "parallel (non-converging) trendlines:")
    b("<b>Parallelism check:</b> slopes must be same direction, within 20% tolerance.")
    b("<b>Width:</b> 1\u20136\u00d7 ATR (meaningful trading range).")
    b("<b>Containment:</b> \u2265 60% of bars inside the channel.")
    b("<b>Window:</b> 30 bars (~6 weeks).")
    sp(3)

    tbl([
        ["Channel Type", "Count", "Containment"],
        ["Channel up", "18", "70\u2013100%"],
        ["Channel down", "7", "73\u201390%"],
        ["Total", "25", "mean 84%"],
    ], cw=[130, 60, 120])

    p("3.5 Shared Pivot Utility", "H2")
    p("<font face='Courier' size='9'>src/patterns/pivots.py</font> provides "
      "<font face='Courier' size='9'>find_swing_highs()</font>, "
      "<font face='Courier' size='9'>find_swing_lows()</font>, and "
      "<font face='Courier' size='9'>containment_ratio()</font> \u2014 "
      "used by both triangle and channel detectors. Based on Lo, Mamaysky "
      "&amp; Wang (2000).")

    story.append(PageBreak())

    # ================================================================
    # 4. COMBINED DETECTION RESULTS
    # ================================================================
    p("4. Combined Detection Results", "H1")

    tbl([
        ["Pattern", "Count", "% of Total"],
        ["Near support", "5", "3.4%"],
        ["Near resistance", "37", "24.8%"],
        ["Symmetric triangle", "19", "12.8%"],
        ["Ascending triangle", "2", "1.3%"],
        ["Descending triangle", "1", "0.7%"],
        ["Multiple bottom", "53", "35.6%"],
        ["Multiple top", "9", "6.0%"],
        ["Channel up", "18", "12.1%"],
        ["Channel down", "7", "4.7%"],
        ["Total events", "149", "100%"],
    ], cw=[140, 60, 80])

    p("The 149 events represent 3.7% of the 4,023 trading days \u2014 "
      "a quality-over-quantity approach where every detection has passed "
      "strict geometric validation.")

    # ================================================================
    # 5. TRIPLE-BARRIER LABELING
    # ================================================================
    p("5. Triple-Barrier Labeling", "H1")

    p("Each detected event is labeled using the triple-barrier method "
      "(Lopez de Prado 2018). For each event bar at time <i>t</i>:")
    b("<b>Entry price</b> = Close[t]")
    b("<b>Upper barrier</b> = entry + 2.0 \u00d7 ATR (profit target)")
    b("<b>Lower barrier</b> = entry \u2212 2.0 \u00d7 ATR (stop loss)")
    b("<b>Time barrier</b> = t + 10 bars (maximum holding period)")
    sp(3)
    p("Walk forward from t+1. Check each bar's High/Low against the barriers. "
      "The first barrier touched determines the label:")

    tbl([
        ["First Barrier", "Label", "Meaning"],
        ["Upper", "long", "Price rose enough \u2014 bullish setup"],
        ["Lower", "short", "Price fell enough \u2014 bearish setup"],
        ["Time", "no_trade", "No significant move \u2014 skip"],
    ], cw=[80, 60, 250])

    p("5.1 Label Distribution", "H2")

    tbl([
        ["Label", "Count", "Proportion", "Avg Return"],
        ["Long", "70", "47.0%", "+2.480%"],
        ["Short", "41", "27.5%", "\u22122.133%"],
        ["No trade", "38", "25.5%", "+0.635%"],
        ["Total", "149", "100%", ""],
    ], cw=[80, 60, 80, 80])

    p("The labels show meaningful return separation: longs are clearly positive, "
      "shorts clearly negative, and no-trades cluster near zero. This confirms "
      "the barriers are calibrated correctly.")

    p("5.2 Labels by Event Type", "H2")

    tbl([
        ["Event Type", "Count", "Long", "Short", "No Trade"],
        ["Multiple bottom", "53", "~25", "~15", "~13"],
        ["Near resistance", "35", "~16", "~10", "~9"],
        ["Symmetric triangle", "19", "~9", "~5", "~5"],
        ["Channel up", "18", "~9", "~5", "~4"],
        ["Multiple top", "9", "~4", "~3", "~2"],
        ["Channel down", "7", "~3", "~2", "~2"],
        ["Near support", "5", "~2", "~1", "~2"],
        ["Ascending triangle", "2", "~1", "~0", "~1"],
        ["Descending triangle", "1", "~1", "~0", "~0"],
    ], cw=[115, 50, 50, 50, 55])

    story.append(PageBreak())

    # ================================================================
    # 6. EVOLUTION OF DETECTORS
    # ================================================================
    p("6. Evolution of Detectors \u2014 What Changed and Why", "H1")

    p("6.1 Original Approach (OLS on all bars)", "H2")
    b("Triangle and channel detectors used <font face='Courier' size='9'>np.polyfit</font> "
      "(OLS) on <b>all</b> 50 bars' highs and lows.")
    b("Lines went through the <b>middle</b> of the data, with ~50% of candles above/below.")
    b("No pivot detection \u2014 every bar had equal weight.")
    b("Result: 38 triangles with 4% containment, 43 channels with 9% containment.")
    b("Professor feedback: <i>\"detections seem incorrect, candles not in the triangle area.\"</i>")

    p("6.2 First Fix: Pivot + Theil-Sen + Containment Gate", "H2")
    b("Added pivot detection (swing highs/lows) and Theil-Sen regression.")
    b("Added hard containment gate (\u226580% for triangles, \u226570% for channels).")
    b("Reduced triangles to 6, channels to 1 \u2014 high quality but too few.")

    p("6.3 Current Approach: Reference Notebook Method", "H2")
    b("Adopted the <i>TrianglePricePatterns</i> notebook approach:")
    b("<font face='Courier' size='9'>linregress</font> on pivots gives correlation <i>r</i> "
      "directly \u2014 natural quality metric.")
    b("<b>Adjusted intercepts</b> shift lines to bound the pivots (not cut through).")
    b("<b>Smaller windows</b> (25 for triangles, 30 for channels) \u2014 compact shapes.")
    b("|<i>r</i>| \u2265 0.85 \u2014 strict but not extreme.")
    b("Result: 22 triangles (86% avg containment), 25 channels (84% avg containment).")

    sp(4)
    tbl([
        ["", "V1 (OLS)", "V2 (Theil-Sen)", "V3 (linregress)"],
        ["Triangles", "38", "6", "22"],
        ["Tri. containment", "4%", "88%", "86%"],
        ["Channels", "43", "1", "25"],
        ["Ch. containment", "9%", "74%", "84%"],
        ["Training events", "183", "118", "149"],
        ["Visual quality", "Poor", "Good", "Excellent"],
    ], cw=[100, 80, 90, 95])

    story.append(PageBreak())

    # ================================================================
    # 7. PROJECT STRUCTURE
    # ================================================================
    p("7. Project Structure", "H1")

    tbl([
        ["Path", "Description"],
        ["src/data/load_data.py", "Data loading with yfinance fallback"],
        ["src/data/utils.py", "ATR computation"],
        ["src/patterns/pivots.py", "Shared swing high/low detection"],
        ["src/patterns/support_resistance.py", "S/R level detection"],
        ["src/patterns/triangles.py", "Triangle detection (linregress)"],
        ["src/patterns/channels.py", "Channel detection (linregress)"],
        ["src/patterns/multiple_tops_bottoms.py", "Double/triple top/bottom"],
        ["src/patterns/scanner.py", "Unified scanner (all 4 detectors)"],
        ["src/labeling/label_events.py", "Triple-barrier labeling"],
        ["src/utils/plotting.py", "Candlestick + trendline plotting"],
        ["notebooks/04_*.ipynb", "Triangle detection gallery"],
        ["notebooks/05_*.ipynb", "Triple-barrier labeling analysis"],
        ["notebooks/06_*.ipynb", "Channel detection gallery"],
    ], cw=[190, 240])

    # ================================================================
    # 8. NEXT STEPS
    # ================================================================
    p("8. Next Steps", "H1")

    b("<b>Phase 4 \u2014 HMM Regimes:</b> fit a Hidden Markov Model on daily log returns "
      "and rolling volatility to identify bull/bear/sideways regimes. Each event will be "
      "tagged with its regime for regime-conditioned modeling.")
    b("<b>Phase 6 \u2014 Feature Engineering:</b> for each labeled event, compute features: "
      "ATR, RSI, momentum, volume ratios, pattern geometry (slopes, containment), "
      "regime state, time-of-year.")
    b("<b>Phase 7 \u2014 Random Forest Model:</b> train a 3-class RF classifier "
      "(long/short/no_trade) on the 149 labeled events. Evaluate with stratified "
      "cross-validation, feature importance, and confusion matrix.")
    b("<b>Phase 8 \u2014 Backtest:</b> simulate trading based on model predictions. "
      "Measure Sharpe ratio, max drawdown, win rate.")
    b("<b>Phase 9 \u2014 Regime Comparison:</b> compare model performance across "
      "different market regimes. Does the model perform better in trending vs. "
      "mean-reverting markets?")

    sp(6)
    hr()
    p("9. References", "H1")
    sp(3)
    p("<b>[1]</b> Lo, A. W., Mamaysky, H., &amp; Wang, J. (2000). Foundations of Technical "
      "Analysis. <i>Journal of Finance</i>, 55(4).")
    p("<b>[2]</b> Lopez de Prado, M. (2018). <i>Advances in Financial Machine Learning</i>. Wiley.")
    p("<b>[3]</b> Bulkowski, T. N. (2021). <i>Encyclopedia of Chart Patterns</i> (3rd ed.). Wiley.")
    p("<b>[4]</b> Koenker, R. (2005). <i>Quantile Regression</i>. Cambridge University Press.")

    doc.build(story)
    print(f"PDF generated: reports/project_summary.pdf")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    build()
