# Plan: principled fixes for the bridge cascade

> Follow-up to [bridge_cascade_experiments_report.md](bridge_cascade_experiments_report.md).
> Two interventions (N=512, MoG K=2) helped marginally but didn't close the gap
> to the 70% PASS criterion. This plan proposes the two principled solutions
> the user asked for: **normalising-flow bridge** and **longer span between
> windows**.

## Context recap

After the recent bug fixes (mu_0 sign + `extract_state_at_step` k/K), the
underlying model is fully identifiable from cold-start (W1 = 100% on
`fsa_high_res`). What remains is bridge-induced posterior drift across many
windows on a sharp 15-min likelihood. The Gaussian / MoG bridge is a
parametric fit to the previous posterior; with each hop the fit error
compounds and posterior coverage decays geometrically.

The two principled responses:

1. **Better bridge: normalising flow.** Replace the Gaussian/MoG fit with
   a flexible, expressive density that can match an arbitrary posterior
   shape (skew, banana correlations, multi-modality). A small Real-NVP
   or coupling-flow network fit per window.
2. **Less bridge: longer stride.** Increase stride/window ratio so each
   bridge spans less posterior change, lowering the per-hop approximation
   error. Costs more total compute because windows overlap less, but the
   per-window posterior should stay much closer to truth.

These are complementary — flows fix the QUALITY of each bridge; longer
strides reduce the NUMBER of bridges. They can be combined.

---

## Track 1 — normalising-flow bridge

### Goal

Replace `_fit_mog_bridge` with a per-window normalising flow `q_φ(u)` that
matches the previous posterior particles via maximum-likelihood / KL fit,
then use it as the base measure for the next window's tempering.

### Architecture choice

Keep it small (29 params, ~256 particles is not much training data per
window). Recommend **Real-NVP** with:
- 4-6 coupling layers
- Hidden width 64
- ReLU or GELU non-linearity
- Affine couplings (multiplicative scale + additive shift)

Input dim: `n_dim = 29` for high-res FSA. Mask split: alternating
even/odd → first half/second half → random permutations.

A simpler alternative is a **conditional Gaussian flow** (a.k.a. "deep
sigmoidal flow" or "AR flow"): scales monotonically. Fewer params,
easier training, less expressive. Try this first.

### Fitting recipe per bridge

For each new window w ≥ 2:

1. Take the previous posterior particles `prev_particles` (shape (N, d))
2. Initialise flow params from previous window's flow (warm start)
   — for w=1 init from a tight Gaussian fit
3. Train via maximum-likelihood: `min_φ -mean(log q_φ(prev_particles))`
   - Adam, lr=1e-3
   - 200-500 epochs (early stop on plateau)
   - Mini-batch full N (small)
4. Use trained flow as `logprior_fn` in the bridge SMC kernel
5. Sample `n_smc` initial particles from the flow

### Key implementation details

- **Differentiability**: BlackJAX HMC needs `logprior_fn` differentiable
  in `u`. Real-NVP gives this for free (the inverse map is smooth).
- **Numerical stability**: clip log-Jacobian terms; init flow weights
  small.
- **Wall-clock**: per-bridge flow training ~30-60s on GPU, negligible
  vs the per-window SMC (~150s for high-res). Can train on CPU since N=256
  is tiny.

### Library

Two options:
- **Distrax** (DeepMind): JAX-native, well-tested, used in many SBI papers.
  Recommended.
- **Hand-roll a Real-NVP in JAX**: ~150 lines. Avoids dependency. Useful
  if Distrax brings transitive deps we don't want.

### Risk register

| Risk | Mitigation |
|------|-----------|
| Flow over-fits to N=256 particles | KL-regularise toward fitted Gaussian: `loss = -mean(log q_φ) + α * KL(q_φ || N(μ̂, Σ̂))`, α ~ 0.1 |
| Flow training diverges (NaN) | Cap leapfrog gradients; use small init weights; add gradient clipping |
| Worse than Gaussian on early windows (fit-error variance > Gaussian-fit bias) | Fall back to Gaussian for w=1 cold-start; only use flow from w=2 |
| Per-window training time exceeds savings from better posterior | Time-box flow training to 30s; if not converged, use partial flow |

### Sequenced work

1. **Phase A** (~3h): Add Real-NVP module under `smc2bj/bridges/realnvp.py`. Distrax-based. Self-contained `fit_flow_to_particles(particles, ...)` returning a `tfp.distributions.Distribution`-like object with `log_prob` and `sample` methods.
2. **Phase B** (~2h): Wire into `smc2bj/estimation/smc_window.py` as `bridge_type='flow'`. Add `bridge_flow_layers`, `bridge_flow_hidden`, `bridge_flow_epochs` to `SMCConfig`.
3. **Phase C** (~1h + 1.3h GPU): Tests + smoke run on 2 windows. Verify `logprior_fn` is differentiable + produces sensible per-particle log-density.
4. **Phase D** (~1.3h GPU): Full 27-window run at seed=42, baseline N=256.
5. **Phase E** (~1h): Comparison report against the existing 3 runs.

**Pass criterion**: mean coverage ≥ 60% (vs 47% best-so-far for N=512). 70% would be a clear win and justify the added complexity.

---

## Track 2 — longer stride

### Goal

Increase the temporal overlap between consecutive windows so each bridge
spans less posterior change.

### Concrete experiments

The current high-res config is `window=96 bins (1 day)`, `stride=48 bins (12h)`. Three settings to try:

| Variant | window | stride | overlap | n_windows over 14d | wall-clock estimate |
|---------|--------|--------|---------|-------------------|---------------------|
| A — current | 96 | 48 | 50% | 27 | 1.3h |
| B — 75% overlap | 96 | 24 | 75% | 53 | 2.5h |
| C — 87.5% overlap | 96 | 12 | 87.5% | 105 | 5h |
| D — wider window | 192 | 48 | 75% | 24 | 2.5h (per-window cost ~2x) |

The intuition: at 87.5% overlap (Variant C), only 12 bins (=3h) of new data
arrive per window. The posterior at W_n is essentially the same as W_{n+1}
modulo 12 bins of evidence. Even a Gaussian fit should hold up over such
small steps.

**Recommended starting point**: Variant B (75% overlap, stride=24 bins).
That's a modest 2× of windows for an expected order-of-magnitude reduction in
per-bridge bias. If that works, push to Variant C.

### Trade-offs

- **More windows**: more checkpoints, more state extractions. Storage +
  bookkeeping cost is linear in n_windows.
- **More bridges**: but each bridge cheaper (fewer tempering levels because
  the prior is closer to the target).
- **Overall wall-clock**: roughly proportional to n_windows × (bridge cost).
  Bridge cost should drop ~30-50% with smaller stride, so net wall-clock
  scales sublinearly with n_windows.

### Sequenced work

1. **Phase A** (~5 min): config tweak — `STRIDE_BINS = BINS_PER_DAY // 4` in driver.
2. **Phase B** (~2.5h GPU): Variant B run.
3. **Phase C** (~30 min): comparison vs current (Variant A) — does mean coverage
   exceed 60%? Late-window collapse delayed?
4. **Phase D** (~1h): if Variant B helps, sketch Variant C run plan or jump to flow combination.

**Pass criterion**: mean coverage ≥ 55% on Variant B; ≥ 65% on Variant C.

---

## Recommended sequencing across both tracks

Given Track 2 is much cheaper (no new code, just config + GPU time) and
Track 1 is much more involved (new module + new tests + careful tuning):

1. **First: Track 2 Variant B** (~2.5h GPU + 30min analysis). If Variant B
   gives ≥55% mean and graceful late-window degradation, **the bridge
   cascade is partially explained by stride being too long** — cheap win.
2. **Second: Track 2 Variant C** if B helps. Confirms the trend.
3. **Third: Track 1 (flows)** if Track 2 stalls below 60% mean. The
   investment is justified once we know the limit of what stride alone
   can buy.
4. **Fourth (optional): combine flows + Variant B stride**. The two
   improvements should compound.

If Track 2 alone gets us to ≥70% mean coverage, declare victory and skip
Track 1. The flow infrastructure can sit in `smc2bj/bridges/` for SWAT or
later models that may need it more.

---

## Definition of done

- One of the variants achieves **mean coverage ≥ 70% data-informed** across
  the rolling rollout.
- AND the late-window coverage (last 3 windows) stays above ≥40% (no
  catastrophic collapse).
- AND the wall-clock is ≤4h per full rollout.

If all three are met, we have a high-res FSA pipeline ready to extend to SWAT.

---

## Open questions to resolve before starting

1. **Flow library choice**: Distrax (one new dep) vs hand-rolled (no new
   deps). Lean toward Distrax for time efficiency.
2. **Flow caching**: should the flow fit at W_n warm-start the W_{n+1}
   flow? Probably yes — saves training time.
3. **Variant C feasibility**: 105 windows × ~150s = 4.4h GPU. Tractable
   but at the edge.

These can be answered when starting Track 1; Track 2 has no open questions.
