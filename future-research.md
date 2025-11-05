## Mutual Information With Sense Labels

**Summary.** Estimate the mutual information between layer representations and gold sense labels to quantify how much each layer encodes sense-discriminative signal beyond linear separability. High mutual information indicates that knowing the representation substantially reduces uncertainty about the correct sense.

**Why it matters.**
- Captures non-linear relationships that may be invisible to linear probes or clustering.
- Complements the Disambiguation Depth Index by flagging the layer where sense information sharply increases.
- Provides an upper bound on the performance of lightweight probes; if MI plateaus, adding probe capacity is unlikely to help.

**Estimation sketch.**
1. Collect per-token hidden states for every layer (post layer norm). Optionally reduce dimensionality (e.g., PCA to 50–100 dimensions) to stabilize density estimates.
2. For each language ℓ and layer L, estimate `I(Z_L; Y_ℓ)` using a k-nearest-neighbor estimator (e.g., Kraskov–Stögbauer–Grassberger). Variational estimators such as MINE are an alternative when handling higher dimensions.
3. Normalize by the entropy of the sense labels `H(Y_ℓ)` and report both the raw MI and the fraction `I/H` to enable cross-language comparisons.

**Implementation cautions.**
- Ensure sufficient samples per sense to keep estimator variance manageable; subsample or smooth sparse senses.
- Bootstrap the metric to obtain confidence intervals compatible with the existing Bayesian aggregation pipeline.
- Document estimator hyperparameters (k for kNN, architecture and training split for MINE) so results remain reproducible.
