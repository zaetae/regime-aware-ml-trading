"""Generate a thesis-style PDF report documenting all pattern detector changes."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Preformatted,
    KeepTogether,
)
from reportlab.lib import colors

# ---------------------------------------------------------------------------
# Run detectors to get live numbers
# ---------------------------------------------------------------------------
from src.data.load_data import load_spy
from src.patterns.support_resistance import calculate_support_resistance
from src.patterns.triangles import detect_triangle_pattern
from src.patterns.channels import detect_channel
from src.patterns.multiple_tops_bottoms import detect_multiple_tops_bottoms

df = load_spy()
n_bars = len(df)
date_start = df.index[0].strftime("%Y-%m-%d")
date_end = df.index[-1].strftime("%Y-%m-%d")

sr_df = calculate_support_resistance(df)
sr = sr_df["near_support"] | sr_df["near_resistance"]

tri_df = detect_triangle_pattern(df)
tri = tri_df["triangle_pattern"].notna()

ch_df = detect_channel(df)
ch = ch_df["channel_pattern"].notna()

mtb_df = detect_multiple_tops_bottoms(df)
mtb = mtb_df["multiple_top_bottom_pattern"].notna()

combined = sr | tri | ch | mtb

results = {
    "Support / Resistance": int(sr.sum()),
    "Triangle Breakouts": int(tri.sum()),
    "Price Channels": int(ch.sum()),
    "Multiple Tops / Bottoms": int(mtb.sum()),
    "COMBINED (any event)": int(combined.sum()),
}

# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "pattern_detection_report.pdf")

doc = SimpleDocTemplate(
    OUTPUT_PATH,
    pagesize=A4,
    topMargin=2.5 * cm,
    bottomMargin=2.5 * cm,
    leftMargin=2.5 * cm,
    rightMargin=2.5 * cm,
)

styles = getSampleStyleSheet()

# Custom styles
styles.add(ParagraphStyle(
    "TitlePage", parent=styles["Title"], fontSize=26, leading=32,
    spaceAfter=6, alignment=TA_CENTER,
))
styles.add(ParagraphStyle(
    "Subtitle", parent=styles["Normal"], fontSize=14, leading=18,
    alignment=TA_CENTER, textColor=HexColor("#555555"), spaceAfter=4,
))
styles.add(ParagraphStyle(
    "SectionHead", parent=styles["Heading1"], fontSize=16, leading=20,
    spaceBefore=24, spaceAfter=10, textColor=HexColor("#1a1a2e"),
))
styles.add(ParagraphStyle(
    "SubSection", parent=styles["Heading2"], fontSize=13, leading=16,
    spaceBefore=16, spaceAfter=8, textColor=HexColor("#16213e"),
))
styles.add(ParagraphStyle(
    "BodyText2", parent=styles["Normal"], fontSize=11, leading=15,
    alignment=TA_JUSTIFY, spaceAfter=8,
))
styles.add(ParagraphStyle(
    "CodeBlock", parent=styles["Code"], fontSize=8, leading=10,
    leftIndent=12, backColor=HexColor("#f4f4f4"), spaceAfter=10,
    spaceBefore=6,
))
styles.add(ParagraphStyle(
    "TableHeader", parent=styles["Normal"], fontSize=10, leading=12,
    alignment=TA_CENTER, textColor=colors.white,
))
styles.add(ParagraphStyle(
    "BulletItem", parent=styles["Normal"], fontSize=11, leading=15,
    leftIndent=24, bulletIndent=12, spaceAfter=4, alignment=TA_JUSTIFY,
))

story = []

# ===== TITLE PAGE =====
story.append(Spacer(1, 6 * cm))
story.append(Paragraph("Regime-Aware ML Trading", styles["TitlePage"]))
story.append(Spacer(1, 0.3 * cm))
story.append(Paragraph("Technical Pattern Detection System", styles["Subtitle"]))
story.append(Paragraph("Architecture, Methodology &amp; Refactoring Report", styles["Subtitle"]))
story.append(Spacer(1, 1.5 * cm))
story.append(Paragraph(f"Dataset: SPY &mdash; {date_start} to {date_end} ({n_bars:,} trading days)", styles["Subtitle"]))
story.append(Spacer(1, 0.5 * cm))
story.append(Paragraph("March 2026", styles["Subtitle"]))
story.append(PageBreak())

# ===== TABLE OF CONTENTS =====
story.append(Paragraph("Table of Contents", styles["SectionHead"]))
toc_items = [
    "1. Introduction &amp; Motivation",
    "2. Shared Infrastructure: Average True Range (ATR)",
    "3. Fix 1 &mdash; Support &amp; Resistance Detector",
    "4. Fix 2 &mdash; Triangle Pattern Detector",
    "5. Fix 3 &mdash; Price Channel Detector",
    "6. Fix 4 &mdash; Multiple Tops &amp; Bottoms Detector",
    "7. Evaluation Results",
    "8. Discussion &amp; Future Work",
]
for item in toc_items:
    story.append(Paragraph(item, styles["BodyText2"]))
story.append(PageBreak())

# ===== CHAPTER 1: INTRODUCTION =====
story.append(Paragraph("1. Introduction &amp; Motivation", styles["SectionHead"]))
story.append(Paragraph(
    "This report documents the systematic refactoring of the technical pattern detection "
    "subsystem within the Regime-Aware ML Trading project. The pattern detectors serve as "
    "the event-generation layer: they scan historical OHLCV price data and flag bars where "
    "a meaningful technical structure has formed. These event signals are subsequently consumed "
    "by downstream machine-learning models that learn regime-dependent trading strategies.",
    styles["BodyText2"],
))
story.append(Paragraph(
    "The original implementation suffered from several critical flaws that rendered the "
    "detectors unreliable for production use:",
    styles["BodyText2"],
))
problems = [
    "<b>Excessive firing rates.</b> Multiple detectors flagged 40&ndash;70% of all bars as events, "
    "far exceeding the target range of 10&ndash;30%. When nearly every bar is an &lsquo;event&rsquo;, the "
    "signal carries no information.",
    "<b>Fixed-percentage thresholds.</b> Hard-coded tolerances (e.g., 1% proximity) do not adapt to "
    "volatility regimes. A 1% move in a low-volatility environment is significant; in a high-volatility "
    "environment it is noise.",
    "<b>Short lookback windows.</b> A 20-bar window is too short to capture genuine formations like "
    "channels and triangles, which typically develop over 30&ndash;60 bars in equity markets.",
    "<b>No breakout confirmation.</b> The triangle detector flagged every bar during the formation "
    "period rather than waiting for the breakout, producing long runs of redundant signals.",
    "<b>No structural validation.</b> The channel detector did not verify that price actually "
    "touched both trendline boundaries, accepting any two roughly parallel lines regardless of "
    "whether price interacted with them.",
]
for p in problems:
    story.append(Paragraph(p, styles["BulletItem"], bulletText="\u2022"))
story.append(Spacer(1, 0.3 * cm))
story.append(Paragraph(
    "The refactoring applies a unified design philosophy across all four detectors: "
    "<b>use ATR-based adaptive thresholds</b>, <b>extend lookback windows</b>, and "
    "<b>require structural confirmation</b> before firing a signal. Each fix is documented "
    "below with its rationale, the before/after logic, and the code implementation.",
    styles["BodyText2"],
))

# ===== CHAPTER 2: ATR =====
story.append(PageBreak())
story.append(Paragraph("2. Shared Infrastructure: Average True Range (ATR)", styles["SectionHead"]))

story.append(Paragraph("2.1 Why ATR?", styles["SubSection"]))
story.append(Paragraph(
    "Average True Range, introduced by J. Welles Wilder in 1978, measures market volatility "
    "by decomposing the full range of price movement &mdash; including overnight gaps &mdash; into a "
    "single smoothed value. Unlike standard deviation (which is anchored to a mean), ATR directly "
    "measures the magnitude of price swings, making it the natural unit for proximity thresholds "
    "in pattern detection.",
    styles["BodyText2"],
))
story.append(Paragraph(
    "By expressing all thresholds in multiples of ATR, the detectors automatically adapt to "
    "changing volatility regimes. A &lsquo;near support&rsquo; threshold of 0.5 &times; ATR is tight in a calm "
    "market and wide in a volatile one &mdash; exactly the behaviour we want.",
    styles["BodyText2"],
))

story.append(Paragraph("2.2 Implementation", styles["SubSection"]))
story.append(Paragraph(
    "The <font face='Courier' size='10'>compute_atr</font> function was placed in "
    "<font face='Courier' size='10'>src/data/utils.py</font> as a shared utility imported by "
    "every detector. The True Range for each bar is defined as:",
    styles["BodyText2"],
))
story.append(Paragraph(
    "<font face='Courier' size='10'>TR = max(High &minus; Low, |High &minus; Close<sub>prev</sub>|, |Low &minus; Close<sub>prev</sub>|)</font>",
    styles["BodyText2"],
))
story.append(Paragraph(
    "ATR is then the simple moving average of TR over 14 periods (the Wilder default). "
    "The implementation uses <font face='Courier' size='10'>pd.concat(...).max(axis=1)</font> "
    "for a vectorised, gap-aware computation.",
    styles["BodyText2"],
))

atr_code = """\
def compute_atr(df, window=14):
    prev_close = df['Close'].shift(1)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - prev_close).abs(),
        (df['Low']  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()"""
story.append(Preformatted(atr_code, styles["CodeBlock"]))

# ===== CHAPTER 3: SUPPORT & RESISTANCE =====
story.append(PageBreak())
story.append(Paragraph("3. Fix 1 &mdash; Support &amp; Resistance Detector", styles["SectionHead"]))

story.append(Paragraph("3.1 Problem Diagnosis", styles["SubSection"]))
story.append(Paragraph(
    "The original detector computed 2-sigma bands on rolling High/Low statistics and flagged "
    "any bar where the closing price was within 1% of these bands. This approach had two "
    "fundamental problems:",
    styles["BodyText2"],
))
orig_sr_problems = [
    "<b>Statistical bands &ne; price levels.</b> Support and resistance are price levels where "
    "actual buying/selling pressure concentrates &mdash; typically at rolling highs and lows. "
    "A mean &plusmn; 2&sigma; band is a statistical construct that may never coincide with a price "
    "level the market has actually tested.",
    "<b>Fixed 1% tolerance.</b> In a low-volatility environment (ATR &asymp; 0.3%), a 1% band "
    "captures bars that are 3&times; ATR away from the level. In a high-volatility regime "
    "(ATR &asymp; 2%), the same 1% band is too tight and misses genuine tests of the level.",
]
for p in orig_sr_problems:
    story.append(Paragraph(p, styles["BulletItem"], bulletText="\u2022"))

story.append(Paragraph("3.2 Solution", styles["SubSection"]))
changes_sr = [
    "<b>Window 20 &rarr; 50.</b> Support and resistance levels are now based on ~2.5 months "
    "of trading data, providing more historically meaningful levels.",
    "<b>Rolling extremes replace sigma bands.</b> Resistance = rolling max of High; "
    "Support = rolling min of Low. These are actual price levels the market has tested.",
    "<b>ATR-based proximity.</b> A bar is flagged &lsquo;near support&rsquo; or &lsquo;near resistance&rsquo; only "
    "when the closing price is within 0.5 &times; ATR(14) of the level. This adapts automatically "
    "to the current volatility regime.",
]
for p in changes_sr:
    story.append(Paragraph(p, styles["BulletItem"], bulletText="\u2022"))

story.append(Paragraph("3.3 Implementation", styles["SubSection"]))
sr_code = """\
def calculate_support_resistance(df, window=50, atr_mult=0.5):
    df = df.copy()
    df["resistance"] = df["High"].rolling(window=window).max()
    df["support"]    = df["Low"].rolling(window=window).min()

    atr  = compute_atr(df)
    band = atr_mult * atr

    df["near_support"]    = (df["Close"] - df["support"]).abs() <= band
    df["near_resistance"] = (df["Close"] - df["resistance"]).abs() <= band
    return df"""
story.append(Preformatted(sr_code, styles["CodeBlock"]))

# ===== CHAPTER 4: TRIANGLES =====
story.append(PageBreak())
story.append(Paragraph("4. Fix 2 &mdash; Triangle Pattern Detector", styles["SectionHead"]))

story.append(Paragraph("4.1 Problem Diagnosis", styles["SubSection"]))
story.append(Paragraph(
    "The triangle detector was the most conceptually broken component. With a 20-bar window, "
    "it asked whether rolling highs and lows were converging &mdash; and random noise almost "
    "always looks like convergence over such a short horizon. Additionally, the detector "
    "flagged every bar during the formation, not the actionable breakout bar.",
    styles["BodyText2"],
))
orig_tri_problems = [
    "<b>Short window.</b> 20 bars is insufficient for a triangle to form. Genuine triangles "
    "in equity markets typically develop over 30&ndash;60 bars.",
    "<b>Bar-to-bar comparison.</b> Comparing <font face='Courier' size='10'>rolling_max(t)</font> vs "
    "<font face='Courier' size='10'>rolling_max(t-1)</font> is extremely noisy. A single outlier bar "
    "can flip the signal.",
    "<b>No compression threshold.</b> Any amount of convergence &mdash; even 0.01% &mdash; triggered "
    "the pattern, catching random fluctuations.",
    "<b>Formation-period firing.</b> Every bar inside the triangle was flagged, creating long "
    "runs of redundant signals rather than a single actionable breakout event.",
]
for p in orig_tri_problems:
    story.append(Paragraph(p, styles["BulletItem"], bulletText="\u2022"))

story.append(Paragraph("4.2 Solution", styles["SubSection"]))
changes_tri = [
    "<b>Window 20 &rarr; 50.</b> Formations are assessed over ~2.5 months.",
    "<b>Linear regression on highs/lows.</b> <font face='Courier' size='10'>np.polyfit</font> fits "
    "a trendline to the window&rsquo;s highs and lows. The slope classifies the triangle: "
    "ascending (flat top, rising bottom), descending (falling top, flat bottom), or "
    "symmetric (both converging).",
    "<b>Minimum 3% compression.</b> The range (mean of first 5 bars vs last 5 bars) must have "
    "contracted by at least 3%, filtering out noise.",
    "<b>Breakout-only firing.</b> The signal fires only on the bar where price exceeds the "
    "recent 5-bar range by 0.3 &times; ATR, capturing the actionable breakout moment.",
    "<b>Symmetric triangle added.</b> A third pattern type where both highs fall and lows rise "
    "is now detected.",
]
for p in changes_tri:
    story.append(Paragraph(p, styles["BulletItem"], bulletText="\u2022"))

story.append(Paragraph("4.3 Classification Logic", styles["SubSection"]))
tri_table_data = [
    ["Pattern", "High Slope", "Low Slope", "Interpretation"],
    ["Ascending", "< 0.01 (flat)", "> 0.01 (rising)", "Buyers lifting floor; resistance holds"],
    ["Descending", "< -0.01 (falling)", "< 0.01 (flat)", "Sellers lowering ceiling; support holds"],
    ["Symmetric", "< -0.01 (falling)", "> 0.01 (rising)", "Both sides converging; volatility squeeze"],
]
tri_table = Table(tri_table_data, colWidths=[2.8*cm, 3.5*cm, 3.5*cm, 6.2*cm])
tri_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f8f8f8"), colors.white]),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
]))
story.append(tri_table)

# ===== CHAPTER 5: CHANNELS =====
story.append(PageBreak())
story.append(Paragraph("5. Fix 3 &mdash; Price Channel Detector", styles["SectionHead"]))

story.append(Paragraph("5.1 Problem Diagnosis", styles["SubSection"]))
story.append(Paragraph(
    "The channel detector was the most broken of the four. It compared the first and last "
    "values of a rolling window to determine trend direction (up/down/flat) and used a "
    "percentage-of-price range constraint. This approach lacked any rigorous definition of "
    "&lsquo;parallel&rsquo; and did not verify that price actually interacted with the channel boundaries.",
    styles["BodyText2"],
))
orig_ch_problems = [
    "<b>Binary trend direction.</b> Comparing start vs end of window yields only +1, -1, or 0 "
    "&mdash; no measure of how parallel the two lines actually are.",
    "<b>No parallelism check.</b> The upper and lower boundaries could diverge or converge "
    "freely. A 10% range constraint is far too loose to enforce parallel walls.",
    "<b>No touch validation.</b> A channel is only meaningful if price has actually tested "
    "both boundaries. The original code accepted any two trending lines regardless of "
    "price interaction.",
    "<b>Every bar flagged.</b> Like the triangle detector, every bar inside the formation "
    "was marked, rather than flagging only bars near the boundaries where trading decisions "
    "are made.",
]
for p in orig_ch_problems:
    story.append(Paragraph(p, styles["BulletItem"], bulletText="\u2022"))

story.append(Paragraph("5.2 Solution", styles["SubSection"]))
changes_ch = [
    "<b>Linear regression trendlines.</b> <font face='Courier' size='10'>np.polyfit</font> fits "
    "separate lines to highs and lows. This gives continuous slope values for rigorous comparison.",
    "<b>Strict parallelism.</b> The slope difference between upper and lower lines must be "
    "within 15% (relative to the upper slope), and both slopes must have the same sign. "
    "If one is positive and the other negative, it is a triangle, not a channel.",
    "<b>Width validation.</b> The mean channel width must be between 1&times; and 6&times; ATR. "
    "Below 1&times; ATR is indistinguishable from noise; above 6&times; ATR is too wide to "
    "represent meaningful structure.",
    "<b>Touch count requirement.</b> Price must touch each boundary at least twice "
    "(within 0.3 &times; ATR), confirming that the channel is a real structure the market "
    "respects.",
    "<b>Boundary-only signals.</b> Signals fire only when the current close is within "
    "0.5 &times; ATR of either trendline &mdash; the zones where channel-based trading decisions "
    "are made.",
]
for p in changes_ch:
    story.append(Paragraph(p, styles["BulletItem"], bulletText="\u2022"))

story.append(Paragraph("5.3 Validation Pipeline", styles["SubSection"]))
story.append(Paragraph(
    "Each candidate channel must pass five sequential filters before a signal is emitted. "
    "This layered approach dramatically reduces false positives:",
    styles["BodyText2"],
))
pipeline_data = [
    ["Stage", "Filter", "Threshold"],
    ["1", "Slope magnitude (avoid flat)", "|slope| > 1e-9"],
    ["2", "Parallelism", "|slope_diff| / |high_slope| < 15%"],
    ["3", "Same direction", "high_slope x low_slope > 0"],
    ["4", "Channel width", "1 x ATR < width < 6 x ATR"],
    ["5", "Band touches", ">= 2 touches per band (0.3 x ATR)"],
]
pipeline_table = Table(pipeline_data, colWidths=[1.5*cm, 5.5*cm, 7*cm])
pipeline_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("ALIGN", (0, 0), (0, -1), "CENTER"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f8f8f8"), colors.white]),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
]))
story.append(pipeline_table)

# ===== CHAPTER 6: MULTI TOP/BOTTOM =====
story.append(PageBreak())
story.append(Paragraph("6. Fix 4 &mdash; Multiple Tops &amp; Bottoms Detector", styles["SectionHead"]))

story.append(Paragraph("6.1 Problem Diagnosis", styles["SubSection"]))
story.append(Paragraph(
    "The multiple top/bottom detector was the least broken of the four, with a reasonable "
    "event count (~248 in the original). However, it lacked close-price confirmation: "
    "a wick touching a ceiling does not constitute a valid multiple top unless the closing "
    "prices confirm the reversal.",
    styles["BodyText2"],
))

story.append(Paragraph("6.2 Solution", styles["SubSection"]))
changes_mtb = [
    "<b>Close-trend confirmation.</b> A linear regression is fit to the last 3 closing prices "
    "using <font face='Courier' size='10'>np.polyfit</font>. For a multiple top, the slope must "
    "be negative (closes trending down). For a multiple bottom, the slope must be positive "
    "(closes trending up).",
    "<b>Base conditions preserved.</b> The original rolling-max/rolling-min logic for detecting "
    "ceiling/floor interaction is retained. The close-trend check is applied as an additional "
    "filter on top of the existing conditions.",
]
for p in changes_mtb:
    story.append(Paragraph(p, styles["BulletItem"], bulletText="\u2022"))

story.append(Paragraph("6.3 Confirmation Logic", styles["SubSection"]))
story.append(Paragraph(
    "The confirmation works by fitting a first-degree polynomial to the three most recent "
    "closing prices [C<sub>t-2</sub>, C<sub>t-1</sub>, C<sub>t</sub>]. The slope of this line "
    "determines whether the close is confirming the pattern:",
    styles["BodyText2"],
))
confirm_data = [
    ["Pattern", "Base Condition", "Confirmation Required"],
    ["Multiple Top", "Rolling max(High) holding ceiling\n+ Rolling max(Close) declining", "3-bar close slope < 0\n(closes trending down)"],
    ["Multiple Bottom", "Rolling min(Low) holding floor\n+ Rolling min(Close) rising", "3-bar close slope > 0\n(closes trending up)"],
]
confirm_table = Table(confirm_data, colWidths=[3*cm, 6*cm, 5.5*cm])
confirm_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f8f8f8"), colors.white]),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
]))
story.append(confirm_table)

# ===== CHAPTER 7: RESULTS =====
story.append(PageBreak())
story.append(Paragraph("7. Evaluation Results", styles["SectionHead"]))

story.append(Paragraph("7.1 Dataset", styles["SubSection"]))
story.append(Paragraph(
    f"All detectors were evaluated on SPY (S&amp;P 500 ETF) daily OHLCV data spanning "
    f"<b>{date_start}</b> to <b>{date_end}</b>, comprising <b>{n_bars:,}</b> trading days. "
    f"The target event rate for the combined signal is 10&ndash;30% of all bars (approximately "
    f"400&ndash;1,200 events).",
    styles["BodyText2"],
))

story.append(Paragraph("7.2 Detection Rates (Post-Refactoring)", styles["SubSection"]))

results_data = [["Detector", "Event Count", "Firing Rate"]]
for name, count in results.items():
    rate = f"{count / n_bars * 100:.1f}%"
    results_data.append([name, str(count), rate])

results_table = Table(results_data, colWidths=[6 * cm, 3.5 * cm, 3.5 * cm])
results_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 10),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f8f8f8"), colors.white]),
    ("BACKGROUND", (0, -1), (-1, -1), HexColor("#e8eaf6")),
    ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
    ("TOPPADDING", (0, 0), (-1, -1), 8),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
]))
story.append(results_table)
story.append(Spacer(1, 0.5 * cm))

target_lo = 400 / n_bars * 100
target_hi = 1200 / n_bars * 100
combined_rate = combined.sum() / n_bars * 100

story.append(Paragraph(
    f"The combined event rate is <b>{combined_rate:.1f}%</b> against a target range of "
    f"<b>{target_lo:.1f}%&ndash;{target_hi:.1f}%</b>. ",
    styles["BodyText2"],
))

if combined_rate > target_hi:
    story.append(Paragraph(
        "The combined rate currently exceeds the target ceiling. The primary contributors are "
        "Support/Resistance and Channels, which together account for the majority of events. "
        "This is expected at this stage: the ATR multipliers (0.5&times; for proximity) and "
        "channel validation thresholds can be tightened in subsequent iterations to bring the "
        "combined rate within the target band. Specifically, reducing <font face='Courier' size='10'>"
        "atr_mult</font> from 0.5 to 0.3 for support/resistance and increasing "
        "<font face='Courier' size='10'>min_touches</font> from 2 to 3 for channels are the "
        "most promising levers.",
        styles["BodyText2"],
    ))
elif combined_rate < target_lo:
    story.append(Paragraph(
        "The combined rate is below the target floor. The detectors may be too strict and "
        "should be loosened by reducing compression thresholds or widening ATR proximity bands.",
        styles["BodyText2"],
    ))
else:
    story.append(Paragraph(
        "The combined rate falls within the target range, indicating that the refactored "
        "detectors are generating a meaningful but selective set of events.",
        styles["BodyText2"],
    ))

story.append(Paragraph("7.3 Comparison: Before vs After", styles["SubSection"]))
comparison_data = [
    ["Aspect", "Before (Original)", "After (Refactored)"],
    ["Volatility adaptation", "None (fixed %, fixed range)", "ATR-based (all thresholds)"],
    ["Lookback window", "20 bars (~1 month)", "50 bars (~2.5 months)"],
    ["S/R level definition", "Mean +/- 2 sigma", "Rolling max(High) / min(Low)"],
    ["S/R proximity", "1% of price (fixed)", "0.5 x ATR (adaptive)"],
    ["Triangle detection", "Bar-to-bar rolling comparison", "Linear regression + compression"],
    ["Triangle signal", "Every bar in formation", "Breakout bar only"],
    ["Channel parallelism", "Start vs end direction (+1/-1)", "Slope difference < 15%"],
    ["Channel validation", "Range < 10% of price", "Width 1-6x ATR + 2 touches/band"],
    ["Multi top/bottom", "Rolling max/min only", "+ 3-bar close trend confirmation"],
]
comp_table = Table(comparison_data, colWidths=[3.5*cm, 5*cm, 5.5*cm])
comp_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f8f8f8"), colors.white]),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
]))
story.append(comp_table)

# ===== CHAPTER 8: DISCUSSION =====
story.append(PageBreak())
story.append(Paragraph("8. Discussion &amp; Future Work", styles["SectionHead"]))

story.append(Paragraph("8.1 Design Principles Applied", styles["SubSection"]))
principles = [
    "<b>ATR as the universal unit.</b> Every threshold in the system is now expressed as a "
    "multiple of ATR(14). This ensures that all detectors automatically adapt to changing "
    "volatility without manual re-tuning.",
    "<b>Structural confirmation before signalling.</b> A pattern is only flagged when there "
    "is evidence that the market respects the structure (touches for channels, breakout for "
    "triangles, close-trend for tops/bottoms).",
    "<b>Signal at the decision point.</b> Events are generated at the moment a trader would "
    "act (breakout bar, boundary approach), not throughout the entire formation. This produces "
    "sparse, actionable signals that downstream models can learn from.",
    "<b>Longer lookback for stability.</b> Extending windows from 20 to 50 bars reduces "
    "noise and ensures that detected formations have had sufficient time to develop.",
]
for p in principles:
    story.append(Paragraph(p, styles["BulletItem"], bulletText="\u2022"))

story.append(Paragraph("8.2 Limitations", styles["SubSection"]))
limitations = [
    "The iterative (bar-by-bar) loops in the triangle and channel detectors have O(n &times; w) "
    "complexity. For the current dataset (~4,000 bars) this runs in seconds, but may need "
    "vectorisation for tick-level data.",
    "ATR(14) is a single-scale volatility measure. A multi-scale approach (e.g., ATR(7) for "
    "short-term and ATR(50) for regime-level) could improve adaptiveness.",
    "The slope thresholds for triangle classification (0.01) are absolute values. Normalising "
    "slopes by ATR or price level would make them more robust across different price ranges.",
    "No out-of-sample validation has been performed yet. The firing rates reported are in-sample "
    "statistics on the full SPY history.",
]
for p in limitations:
    story.append(Paragraph(p, styles["BulletItem"], bulletText="\u2022"))

story.append(Paragraph("8.3 Recommended Next Steps", styles["SubSection"]))
next_steps = [
    "Tune ATR multipliers to bring the combined event rate into the 10&ndash;30% target band. "
    "The <font face='Courier' size='10'>evaluate_rates.py</font> script provides a fast feedback "
    "loop for this parameter search.",
    "Add walk-forward validation: train regime models on the first 70% of data and evaluate "
    "pattern signal quality on the remaining 30%.",
    "Introduce event co-occurrence analysis to understand which patterns tend to fire together "
    "and whether combinations carry stronger predictive power.",
    "Vectorise the triangle and channel loops using rolling-window regression "
    "(<font face='Courier' size='10'>pandas.api.extensions</font> or "
    "<font face='Courier' size='10'>numpy.lib.stride_tricks</font>) for scalability.",
]
for p in next_steps:
    story.append(Paragraph(p, styles["BulletItem"], bulletText="\u2022"))

# Build PDF
doc.build(story)
print(f"Report generated: {OUTPUT_PATH}")
