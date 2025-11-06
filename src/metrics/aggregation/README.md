# Hierarchical Pooling Overview

Hierarchical pooling gives us a way to turn noisy lemma-level measurements into stable, language-level estimates. Instead of averaging raw scores, we fit a probabilistic model that lets lemmas “share” information: sparse lemmas borrow strength from richer ones, while languages keep their own personality. The end result is a shrinkage-adjusted difficulty score with uncertainty—perfect for comparing languages without overconfidence.

## How It Works

1. **Collect observations**  
   For each lemma in a language we gather its metric value (e.g., DDI), the number of samples behind it, and any auxiliary details (metric name, split, model ID). These become rows in a structured table.

2. **Define the model**  
   We assume each lemma score `y_ℓ` follows a normal distribution centered on its language mean `μ_L` plus a lemma-specific offset `β_ℓ`:  
   `y_ℓ ~ Normal(μ_L + β_ℓ, σ²)` with priors  
   `μ_L ~ Normal(μ_global, σ_L²)` and `β_ℓ ~ Normal(0, σ_β²)`.  
   Optional metric-wide effects slot in exactly the same way if we need them later.

3. **Fit with Bayesian inference**  
   Using a tool like PyMC, Stan, or NumPyro we run MCMC (exact, slower) or variational inference (approximate, faster) to obtain samples from the posterior distribution of every parameter: global mean, language means, lemma offsets, residual variance.

4. **Summarise posterior draws**  
   From the posterior samples we compute summary statistics for each language—posterior mean (our shrinkage-adjusted score), along with 5–95% credible intervals that quantify uncertainty. Lemmas inherit their own adjusted estimates via `μ_L + β_ℓ`.

5. **Report and reuse**  
   The language-level summaries feed downstream metrics such as the Context Burden Score or visualisations. Because everything is stored as draws, we can revisit the posterior to answer “what if” questions without rerunning probes.

Hierarchical pooling balances fidelity and robustness: it respects genuine language differences while smoothing away sampling noise from sparse lemmas, giving us trustworthy comparisons across the multilingual landscape.

## Worked Example

Input to the model is a tidy table where each row captures one lemma-level metric:

```
| language | lemma   | metric | value | support_size |
|----------|---------|--------|-------|--------------|
| en       | bank    | DDI    | 6.0   | 42           |
| en       | cell    | DDI    | 5.3   | 18           |
| en       | lead    | DDI    | 7.1   | 11           |
| tr       | düşmek  | DDI    | 4.2   | 25           |
| tr       | yazmak  | DDI    | 5.6   | 14           |
| tr       | yüz     | DDI    | 5.1   | 9            |
```

Each row becomes three arrays under the hood: the metric values, the integer-coded language IDs, and the integer-coded lemma IDs. Optionally, `support_size` informs weighting or observation noise if we model heteroskedastic variance.

After fitting the hierarchical model we sample from the posterior. A typical summary looks like:

```
| language | posterior_mean | hdi_5% | hdi_95% |
|----------|----------------|--------|--------|
| en       | 6.1            | 5.6    | 6.6    |
| tr       | 4.9            | 4.3    | 5.5    |
```

`posterior_mean` is the shrinkage-adjusted language estimate; the highest-density interval (HDI) columns show our uncertainty. Lemma-level posteriors are available too (`μ_language + β_lemma`) if we need smoothed per-lemma scores.

## Using the Aggregator

1. Gather lemma metrics into `LemmaMetricRecord` objects—usually by looping over probe outputs and logging the metric value, lemma, and language.
2. Call `aggregate_language_scores(records, draws=..., tune=...)` for a one-shot run, or construct `HierarchicalAggregator` directly if you want to reuse the fitted posterior.
3. The result is a list of `LanguageSummary` instances containing the posterior mean and highest-density interval for each language.

```python
from src.metrics.aggregation import LemmaMetricRecord, aggregate_language_scores

records = [
    LemmaMetricRecord(language="en", lemma="bank", metric="ddi", value=6.0),
    LemmaMetricRecord(language="en", lemma="cell", metric="ddi", value=5.3),
    LemmaMetricRecord(language="tr", lemma="düşmek", metric="ddi", value=4.2),
]

summaries = aggregate_language_scores(records, draws=1000, tune=500)
for summary in summaries:
    print(summary.language, summary.mean, summary.lower, summary.upper)
```

