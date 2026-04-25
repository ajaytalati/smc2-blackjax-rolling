# Bridge-cascade experiments — N_SMC and MoG bridge

**Date:** 2026-04-24/25
**Model:** `models/fsa_high_res/` (29 params, 4 obs channels, 15-min bins)
**Driver:** `drivers/fsa_high_res_rolling.py --seed 42`
**Rollout:** 14 days, 1-day window (96 bins), 12-h stride (48 bins) → 27 windows
**Commits:** baseline `41997bf`, MoG support `c87476a`

## Question

After confirming the high-res model is fully identifiable from cold-start
(W1: 100% raw / 100% data-informed), the rolling-window posterior degrades
across many bridges — by W25-W27 coverage drops to 3-10%. The user identified
this as **bias** (narrow CIs locked off-truth), not variance. Two candidate
mitigations were investigated:

1. **N_SMC=512** — double the particle count to preserve sample diversity through
   the Gaussian fit + tempering.
2. **Mixture-of-Gaussians bridge (K=2)** — replace the single-Gaussian fit of
   the previous posterior with a 2-component GMM (K-means clusters, per-component
   Ledoit-Wolf shrinkage, log-sum-exp prior).

## Results

| Run | mean raw | mean inf | PASS (≥70%) | Wall-clock |
|-----|---------:|---------:|:-----------:|-----------:|
| N=256 gaussian (baseline) | 37.5% | 33.6% | 1 / 27 | 1.27h |
| **N=512 gaussian** | **46.9%** | **40.4%** | 1 / 27 | 2.15h |
| N=256 MoG K=2 | 40.2% | 36.5% | 1 / 27 | 1.26h |

W1 cold-start was 100% in all three runs (confirming the underlying model is
identifiable; the cascade is purely a bridge-mechanism problem).

## Per-window breakdown

```
                W1    W2-W9 mean   W10-W20 mean    W25-W27
baseline       100   45.6           33.5           3, 7, 10
N=512          100   60.9           42.6          24, 10, 24
MoG K=2        100   53.2           29.6          24, 28, 38
```

### Two interventions help different parts of the cascade

**N=512 wins on early windows (W2-W9).** More particles → tighter Gaussian
fit → smaller bridge-induced bias in the first few hops. By W9 N=512 is at
65.5% vs MoG's 41.4% and baseline's 62.1%.

**MoG wins on late windows (W25-W27).** The mixture catches posterior
skewness that develops as the cascade compounds; coverage stays in the
24-38% band instead of collapsing to single digits. By W27 MoG is at 38%
while N=512 is at 24% and baseline is at 10%.

This split suggests the two failure modes are different:
- **Early-window bias** is dominated by sample-size limitations of the
  Gaussian fit (covariance estimate from N=256 particles in 29 dims is
  near-singular even with Ledoit-Wolf shrinkage). N=512 directly addresses
  this.
- **Late-window collapse** is dominated by accumulated approximation error
  from repeated Gaussian fits of progressively skewed posteriors. MoG's
  multi-modal flexibility resists this drift.

## Conclusions

1. **Both interventions help measurably**, but neither closes the gap to
   the ≥70% pass criterion on its own.
2. **N=512 is the better single intervention** for mean coverage (+9.4pp),
   but it costs 2× wall-clock and only marginally helps late-window
   collapse.
3. **MoG K=2 is the better single intervention** for late-window survival,
   at no wall-clock cost.
4. **They appear complementary** — combining N=512 + MoG would address
   both failure modes. Not yet measured (estimated cost ~2.5h, would
   plausibly hit 50-55% mean with late windows holding 30-40%, still
   short of the 70% target).
5. **The fundamental issue is that any parametric (Gaussian or low-K MoG)
   fit to a sharp 29-dim posterior is a first-order approximation that
   compounds across many bridges.** Closing the remaining gap requires
   either (a) a non-parametric bridge that captures the true posterior
   shape (normalising flows), or (b) lengthening the stride so each
   bridge spans less posterior change.

See `PLAN_principled_bridge_fixes.md` for the next-step proposal.

## Reproduction

```bash
# Baseline (already in this repo at C0_N256_s42_baseline)
python drivers/fsa_high_res_rolling.py --seed 42 --n-smc 256

# N=512
python drivers/fsa_high_res_rolling.py --seed 42 --n-smc 512

# MoG K=2
python drivers/fsa_high_res_rolling.py --seed 42 --n-smc 256 --bridge mog --bridge-K 2
```

Output directories:
- `outputs/fsa_high_res_rolling/C0_N256_s42_baseline/`
- `outputs/fsa_high_res_rolling/C0_N512_s42/`
- `outputs/fsa_high_res_rolling/C0_N256_s42_mog2/`
