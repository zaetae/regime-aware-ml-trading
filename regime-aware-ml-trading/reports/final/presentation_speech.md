# Presentation Speech for "Regime-Aware ML for SPY 4 Deck"

13 slides. 10 minutes talk + 5 minutes Q&A.

---

## SLIDE 1 — Title [30 sec]

Good morning. My name is Zeineb Turki and my supervisor is Hadházi Dániel.

My project applies machine learning to financial trading on SPY, which is the S&P 500 ETF tracking US large-cap stocks. I used daily price bars from 2010 to 2025.

The key idea is event-based learning. Instead of predicting every daily bar, the system detects specific chart patterns and predicts trade outcomes only at those moments. The project also studies the trade-off between classification accuracy and actual trading profitability using a method called triple-barrier labeling.

---

## SLIDE 2 — Why financial ML refuses to behave [50 sec]

So why is financial prediction so difficult?

Over about 4,000 SPY daily bars in our dataset, 97% of them are essentially noise. Daily returns are dominated by random microstructure effects and macroeconomic shocks that no model can anticipate from price data alone. If you feed every bar into a classifier, the rare high-information moments get drowned out.

This is where event-based learning comes in. Instead of treating every bar as a training sample, we only sample the moments that geometrically matter, meaning bars where a recognised chart pattern has completed. Our detectors identify 137 such events out of 4,000 bars, which is a 3.4% event density.

The research question is: can a classifier trained on these geometry-conditioned events, labeled by a triple barrier, deliver out-of-sample profitability while avoiding the data leakage that typically haunts daily-bar ML?

---

## SLIDE 3 — Four detectors, one event book [50 sec]

We use four pattern detectors. Each one targets a different geometric structure.

Support and resistance produces 42 events. It identifies horizontal price levels from prior swing highs and lows, and fires a signal when the close price re-enters a band of plus or minus 0.3 times ATR around that level.

Channels produce 12 events by fitting parallel trendlines using least-squares regression, requiring at least 70% of bars to be contained within the channel.

Triangles produce 22 events from converging regression lines with a correlation threshold of 0.85 and 80% containment.

And multiple tops and bottoms, which is the dominant family, produce 63 events by detecting repeated swing extrema at the same price level.

In total we detect 137 events, but only 94 are used for training. Triangles and channels were excluded after supervisor review because their detection quality needed further work.

---

## SLIDE 4 — The signal bar [35 sec]

This slide shows the anti-leakage architecture, which is fundamental to the project.

The yellow diamond marks the signal bar. This is the only row that enters the feature matrix for each event. Features are computed from the backward window, looking only at data up to and including the signal bar. The label is determined from the forward window, starting at the very next bar. These two windows never overlap.

The pattern itself may have formed over 20 to 75 bars, but only the completion close enters the matrix. This strict separation is what prevents look-ahead bias.

---

## SLIDE 5 — Six feature families [40 sec]

Each event is described by about 45 features from six families.

Trend features like SMA distances tell the model where the price sits relative to its moving averages. Volatility features like ATR and Bollinger Band width describe how much the price is moving. Momentum features like RSI and MACD capture directional pressure. Volume features measure trading activity. Geometry features encode the shape of the detected pattern, things like slopes, touch counts, and containment ratios. And binary features provide discrete flags for conditions like a Bollinger Band breach or an SMA crossover.

The ANOVA ranking shows that volume standard deviation is the strongest individual feature, but overall the model relies on interactions between features rather than any single one.

We removed all features that could leak price level information, like raw ATR and absolute SMA values, keeping only normalised ratios.

---

## SLIDE 6 — Three barriers, one honest label [55 sec]

This is one of the most important slides.

For each event, three barriers race from the signal bar. The take-profit sits at entry price plus a multiple of ATR upward. The stop-loss sits at entry minus a multiple of ATR downward. And the time barrier fires after a set number of bars if neither price barrier is hit first.

The first barrier touched determines the label. Take-profit hit means long, stop-loss hit means short, time expiry means no-trade. This is the classical Lopez de Prado method from 2018.

Now, we do not fix these parameters. We treat them as hyperparameters and sweep 100 configurations. The best F1 configuration uses take-profit at 2.0, stop-loss at 1.5, and 10-bar holding. The best profitability configuration uses take-profit at 2.0, stop-loss at 3.0, and 20-bar holding.

The fact that these two optima disagree is the central finding. The best F1 reaches 0.654. The best cumulative return is 18.7% from Bagging. I will come back to this disagreement on slide 10.

---

## SLIDE 7 — Localise then trust [30 sec]

Before training any model, we audited every detector for geometric quality.

The containment ratios show that channels have a median of 0.99 and triangles have a median of 0.86. Every accepted channel meets at least 2 upper and 2 lower boundary touches. The quality gates table shows the specific thresholds for each detector type.

Starting from over 1,000 initial candidates, only 137 pass all quality gates, which is a 3.5% acceptance rate.

Several improvements were made on supervisor feedback: localising the signal at the event bar, removing trend-leaking features, adding binary technical signals, and replacing the single train-test split with walk-forward cross-validation.

---

## SLIDE 8 — Random Forest and Bagging [35 sec]

We train two tree-based ensembles: Random Forest and Bagging, both with 200 trees and max depth 8. We chose tree ensembles because they are robust on small tabular datasets and no single feature dominates, which makes ensemble methods appropriate.

The class distribution across the 94 labeled events is 38 long, 25 no-trade, and 31 short. We use inverse-frequency class weights to handle this imbalance rather than resampling.

As a baseline we use a stratified random classifier. On the small test fold, this baseline actually edges ahead on accuracy and F1 at 0.447, but as I will explain, that is a sign of small-sample variance, not real signal.

---

## SLIDE 9 — Three checks against self-deception [45 sec]

We use three validation strategies.

The primary evaluation is a chronological 60/20/20 split. The oldest events go to training, the most recent go to testing. This respects temporal ordering.

On the training block we also run 5-fold event cross-validation. The per-fold table shows F1 ranging from 0.371 to 0.488 with a mean of 0.425 and standard deviation of 0.046. Fold 4 is the hardest because it covers early-2020 COVID events. If we remove that fold, the mean F1 rises to 0.459.

F1 treats precision and recall equally. But in trading, these errors have different costs. A false positive means entering a losing trade, which costs real money. A false negative means missing a profitable trade, which is only an opportunity cost. F-beta lets us tilt this balance. I discuss this further on slide 11.

With 94 events, we do not claim individual fold differences are significant. We report distributions and let the test set adjudicate.

---

## SLIDE 10 — Classification, profitability, and the gap [65 sec]

This is the main results slide.

The best cross-validation F1 is 0.569. The best cumulative return from Bagging is 18.7%, with a win rate of 79%, 15 winning trades out of 19, and a Sharpe ratio of 0.39.

Looking at the model comparison table: Bagging beats Random Forest on validation F1, 0.425 versus 0.393. But on the 28-event test fold, the stratified baseline edges ahead at 0.447. This is not because the baseline is better. With only 28 test events, random fluctuations dominate. The Wilson 95% confidence interval on win rate is roughly 57% to 92%, which is very wide. The cross-validation distributions with their standard deviation of 0.046 are the more reliable signal.

The equity curve on the bottom right shows the profit-optimal configuration in action. Bagging achieves 18.7% return with a profit factor of 2.30, meaning the winning trades collectively earn 2.3 times what the losing trades give back.

For reference, SPY itself returned about 62% buy-and-hold over the same period, but that is a completely different risk profile with no defined stop-loss.

---

## SLIDE 11 — When F1 and return disagree [45 sec]

This scatter plot is the visual summary of the main finding. Each dot represents one of the 100 barrier configurations. F1 is on the horizontal axis, cumulative return on the vertical.

The key observation: these dots do not form a line. No single point maximises both objectives. The Pareto frontier runs along the upper-right edge.

Why does this happen? Maximising F1 picks moderate barriers that produce clean, easy-to-classify labels, but the resulting trades are small. Maximising return picks wider stops that let winning trades develop larger gains, even though the wider barriers produce noisier labels that are harder to classify.

The four key takeaways: event-based learning is viable, volume statistics dominate, the F1-to-profit gap demands reporting two separate optima, and Bagging outperforms Random Forest on validation.

---

## SLIDE 12 — What's solved, what's next [35 sec]

Four conclusions.

One: event-based labeling works. Geometric pattern completion plus triple-barrier labeling gives a clean supervised problem.

Two: Bagging is the right ensemble here, beating Random Forest on both validation F1 and cumulative return.

Three, the central finding: F1 and return are different objectives. Reporting only one is misleading.

Four: sample size dominates everything. 94 events is too few for stable test-set conclusions. The cross-validation distributions are the stronger signal.

Limitations include single-asset testing, small sample size, no transaction costs, and no regime-state conditioning. The roadmap includes multi-asset extension, an HMM regime layer, cost-aware F-beta tuning, and online retraining.

Thank you. I am happy to take questions.

---

## SLIDE 13 — Notation

[Do not present. Reference slide for Q&A.]

---

## TIMING

| Slide | Topic | Time | Running |
|-------|-------|------|---------|
| 1 | Title | 0:30 | 0:30 |
| 2 | Why ML is hard | 0:50 | 1:20 |
| 3 | Detectors | 0:50 | 2:10 |
| 4 | Signal bar | 0:35 | 2:45 |
| 5 | Features | 0:40 | 3:25 |
| 6 | Triple barrier | 0:55 | 4:20 |
| 7 | Quality audit | 0:30 | 4:50 |
| 8 | Models | 0:35 | 5:25 |
| 9 | Validation | 0:45 | 6:10 |
| 10 | **Results** | 1:05 | 7:15 |
| 11 | **F1 vs return** | 0:45 | 8:00 |
| 12 | Conclusion | 0:35 | 8:35 |
| 13 | (not presented) | 0:00 | 8:35 |
| | Buffer / pauses | 1:25 | **10:00** |

You will finish the spoken content in about 8.5 minutes, leaving over a minute of comfortable buffer for pointer transitions, taking a breath, and natural pauses. This is well within the 10-minute limit.

---

## IF RUNNING OVER

If you hit slide 9 at 7 minutes or later, compress slides 10-11 into one block:

"The best CV F1 is 0.569, the best Bagging return is 18.7% with a Sharpe of 0.39 and profit factor of 2.30. But the central finding is that the configurations that maximise F1 and the ones that maximise return are different. This scatter plot shows it: no single point maximises both. Tight stops classify well but produce small trades. Wide stops produce larger gains despite lower F1. The four takeaways are on slide 11."

Then jump to conclusion.

---

## Q&A QUESTIONS AND ANSWERS

---

### FUNDAMENTAL / "WHAT DOES THIS MEAN" QUESTIONS

**Q: What is SPY exactly?**

SPY is the SPDR S&P 500 ETF. It is an exchange-traded fund that tracks the 500 largest publicly traded companies in the United States. It is the most heavily traded equity security in the world, with daily volume exceeding 70 million shares. I used it because its high liquidity minimises noise from bid-ask spreads and slippage.

**Q: What is ATR?**

ATR stands for Average True Range. It measures volatility by taking the rolling 14-bar average of the daily price range, accounting for gaps between the close and next open. I use ATR as the unit for barrier distances because it adapts to the current volatility regime. A 2-ATR barrier is larger in absolute terms during high volatility and smaller during calm markets.

**Q: What is a triple barrier?**

It is a labeling method from Lopez de Prado's 2018 book. Instead of labeling each day as simply "up" or "down," you place three barriers around the entry price: a take-profit above, a stop-loss below, and a time limit. Whichever is hit first determines the label. This way, the labels represent realistic trade outcomes with built-in risk management, not arbitrary directional calls.

**Q: What does F1 score mean?**

F1 is the harmonic mean of precision and recall. Precision is the fraction of events the model labeled as "long" that were actually long. Recall is the fraction of actual long events the model correctly identified. F1 balances both. I use macro-F1, which averages F1 across all three classes equally.

**Q: What is the Sharpe ratio?**

The Sharpe ratio divides the average trade return by its standard deviation. It measures how much return you get per unit of risk. A Sharpe of 0.39 means the average return is about 0.39 standard deviations above zero, which is moderate.

**Q: What is a profit factor?**

Profit factor is the total dollar amount of winning trades divided by the total dollar amount of losing trades. A profit factor of 2.30 means the winning trades collectively made 2.3 times as much as the losing trades lost. Anything above 1.0 means the strategy is net profitable.

**Q: What do "long," "short," and "no-trade" mean?**

Long means the price rose to the take-profit level first, so you would profit by buying. Short means the price fell to the stop-loss level first, so you would profit by selling short. No-trade means neither barrier was hit within the time limit, so no strong directional move occurred.

**Q: What does "event-based learning" mean practically?**

Instead of using every daily bar as a training sample, which gives about 4,000 rows of mostly noise, we first run pattern detectors to find the roughly 100 bars where a recognisable chart structure has completed. Only these bars become training samples. The model is never asked to predict on a random Tuesday; it only predicts when a pattern has been identified.

**Q: What is a support/resistance level?**

Support is a price level where the price has historically bounced upward, suggesting buyers step in at that price. Resistance is a level where the price has historically been pushed back down, suggesting sellers dominate. Our detector identifies these from rolling highs and lows and fires a signal when the price re-approaches them.

**Q: What is containment ratio?**

Containment ratio measures what fraction of price bars in a pattern window sit inside the fitted boundaries. For a channel, 0.99 containment means 99% of the bars were between the upper and lower trendlines. Higher containment means the pattern is geometrically clean and not just noise.

---

### METHODOLOGY QUESTIONS

**Q: Why Random Forest and Bagging? Why not deep learning?**

Two reasons. First, with only 94 labeled events, a neural network would severely overfit. Tree ensembles handle small tabular datasets much better. Second, trees provide built-in feature importance, which gives us interpretability. We can see that volume statistics matter most, which is a meaningful insight.

**Q: Why not use more data? Can you get more events?**

The event count is driven by the detector sensitivity. We could lower thresholds to produce more events, but that would include lower-quality patterns with weaker signal. The tuning process started with over 1,000 candidates and narrowed to 137 through quality gates. We could also extend to multiple assets or lower timeframes, which is on the roadmap.

**Q: Why exclude triangles and channels from training?**

After reviewing the detection outputs with my supervisor, we identified that some triangle and channel detections were geometrically questionable, producing patterns that a human chartist would not recognise. Rather than train the classifier on unreliable labels, we excluded those 34 events. Their geometric features are still computed and used as input features, but they do not generate training labels.

**Q: How do you prevent data leakage?**

Three layers. First, the feature window looks only backward from the signal bar while the label window starts at the next bar, so they never overlap. Second, we removed features like raw ATR and absolute SMA values that encode absolute price level, which would let the model distinguish time periods. Third, walk-forward cross-validation ensures training data always precedes test data chronologically.

**Q: Why use ATR instead of fixed dollar amounts for barriers?**

Because fixed dollar barriers do not adapt to volatility. A $5 stop on SPY at $100 is a 5% move, but at $600 it is less than 1%. Using ATR multiples means the barriers automatically scale with current market conditions. A 2-ATR stop is always roughly the same number of "typical daily ranges" regardless of the price level.

**Q: What is walk-forward cross-validation and why is it better than standard k-fold?**

Standard k-fold randomly shuffles data, which means the model might train on 2023 data and test on 2018 data. In finance, that is cheating because you are letting the model see the future. Walk-forward CV always trains on older data and tests on newer data, which simulates how you would actually deploy the model: train on history, predict the future.

**Q: Why 200 trees and max depth 8?**

200 trees is a standard choice for Random Forest that provides a good balance between ensemble diversity and computational cost. Max depth 8 limits each tree's complexity, which acts as regularisation against overfitting. With only 94 training events, deeper trees would memorise the training set. These are common defaults in the scikit-learn documentation for small datasets.

**Q: What does class-balanced weighting do?**

Our label distribution is 38 long, 25 no-trade, 31 short, which is imbalanced. Without correction, the model would bias toward predicting the majority class. Class-balanced weighting assigns higher loss to errors on the minority class, forcing the model to pay equal attention to all three classes during training.

---

### RESULTS AND ANALYSIS QUESTIONS

**Q: The baseline beats your models on the test set. Doesn't that mean the models are useless?**

No. The test set has only 28 events. At that sample size, random fluctuation dominates. A stratified random classifier can luck into a good F1 on such a small sample. The 5-fold cross-validation, which averages over more events, consistently shows the real models outperforming the baseline: Bagging at 0.425, RF at 0.393, baseline at 0.361. The test-set anomaly is a statistical artefact of small sample size.

**Q: Is the 18.7% return realistic?**

It is the best result from one specific parameter configuration on the validation set. The cross-validation average return is about 10% with a standard deviation of 7.6%, so there is wide variance. The grid-search ceiling is even higher at 28.7%, which likely includes some overfitting. The honest estimate is the CV mean, not the peak. And SPY itself returned about 62% buy-and-hold over the same window, so the strategy is not beating the market in absolute terms, though it operates with defined risk.

**Q: Why does F1 disagree with profitability? Can you explain that simply?**

F1 counts how often the model gets the label right, treating every correct prediction equally. Profitability depends on how much money each prediction makes or loses. Tight stops produce labels that are easy to classify because the outcomes are small and clear. But those small outcomes also mean small profits. Wide stops produce noisier labels that are harder to classify, but when the model gets it right, the profit is much larger. The larger winners outweigh the classification errors.

**Q: What does the scatter plot on slide 11 actually show?**

Each of the 100 dots is one combination of take-profit, stop-loss, and holding period. I computed F1 and cumulative return for each. If accuracy equaled profitability, the dots would form a line from bottom-left to top-right. They do not. Some configurations have high F1 but low return. Others have lower F1 but higher return. This visual disconnect is the main finding.

**Q: Why does volume standard deviation rank as the most important feature?**

Volume standard deviation measures how erratic recent trading activity has been. High volume volatility often precedes breakouts or reversals, which are exactly the events our detectors are designed to catch. It is also naturally scale-invariant, meaning it works whether SPY is at 100 dollars or 600 dollars, unlike raw volume which trends upward over time.

**Q: What happens during fold 4, the COVID fold?**

Fold 4 covers early 2020, when the COVID crash caused extreme volatility and unprecedented market behaviour. The model, trained mostly on pre-COVID patterns, struggles because the statistical relationships it learned do not hold in a panic. Removing fold 4 lifts the mean F1 from 0.425 to 0.459, which shows how sensitive the results are to regime.

**Q: You mention a profit factor of 2.30. Is that good?**

In quantitative finance, a profit factor above 1.5 is considered decent and above 2.0 is considered good. Our 2.30 means winning trades earn more than twice what losing trades cost. However, this is on a very small sample of 19 trades, so the confidence interval is wide. It is promising but not conclusive.

---

### CRITICAL / CHALLENGING QUESTIONS

**Q: With only 94 events, can you really draw any conclusions?**

Honest answer: the individual point estimates are unreliable. What we can conclude is the structural finding that F1 and profitability optimise at different parameter settings, because this pattern appears consistently across multiple CV folds, not just on one test set. The specific numbers like "18.7% return" should be treated as indicative, not definitive. The framework is validated; scaling it to more data is the next step.

**Q: Isn't this just overfitting to 100 configurations on 94 events?**

It is a legitimate concern. Testing 100 barrier configurations on 94 events does risk finding configurations that look good by chance. This is why we rely on cross-validation distributions rather than single-split results. The mean CV F1 of 0.425 with standard deviation 0.046 is a more honest estimate than the best single-configuration F1 of 0.654. We also note that the F1-versus-return divergence is a structural property that appears across configurations, not an artefact of one lucky pick.

**Q: Why not model transaction costs?**

For SPY at daily frequency, the bid-ask spread is under one cent and daily volume exceeds 70 million shares, so transaction costs would be very small. But for a production system, slippage, commissions, and borrowing costs for short positions would need to be included. This is acknowledged as a limitation and is on the roadmap.

**Q: Would this work on other assets?**

We have not tested it, so we cannot claim it does. The pipeline is asset-agnostic by design, but the detector thresholds, the ATR-based barriers, and the feature engineering were tuned for SPY's characteristics. Other assets with different volatility profiles, trading hours, or microstructure would likely require recalibration. Multi-asset testing is the top item on the roadmap.

**Q: What would you do differently if you started over?**

Three things. First, I would start with multiple assets from the beginning to get more events and test generalization. Second, I would implement purged cross-validation from the start, which adds embargo periods between train and test folds to prevent any subtle temporal leakage. Third, I would add an HMM regime layer so the model could condition on whether the market is currently trending, ranging, or volatile.

**Q: Does technical analysis actually work? Isn't it just pseudoscience?**

The efficient market hypothesis suggests that publicly known patterns should not be profitable because they would be arbitraged away. Our results are consistent with this: the edge is thin and regime-dependent. However, the event-based filtering does concentrate the model on higher-information moments, and the CV F1 of 0.425 is above the 0.361 baseline. Whether this edge survives transaction costs and scales to live trading is an open question. We do not claim to have solved trading; we claim to have built a framework that honestly evaluates whether technical patterns contain measurable predictive information.
