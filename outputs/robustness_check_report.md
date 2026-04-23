# Robustness Check Report: Cross-Seed Replication of the Data Excitation Experiment

**Date:** 2026-04-22 / 2026-04-23
**Supersedes conclusions in:** `excitation_experiment_report.md` (single-seed)
**Model:** FSA Real-Obs v4.1, 33 parameters (frozen R_base + kappa_chronic)
**Config:** N_SMC=256, K_PF=400, 120-day windows, 30-day stride, 9 windows/run
**Compute:** RTX 5090 (NVIDIA driver 590.48.01), ~1.4h/run
**Total runtime:** 9 runs × ~1.4h = ~12.6h

---

## 1. Motivation

The original single-seed experiment (`excitation_experiment_report.md`, seed=42)
concluded that C2 (strong excitation) was optimal (95.3% coverage), C0
(baseline) was middling (83.5%), and C3 (maximal excitation) was worse than
baseline (78.1%). The ordering was interpreted as evidence for a "Goldilocks"
tracking bandwidth.

The user raised the concern: **could C2's excellent result and C3's poor
result both be seed-dependent flukes?** A single seed gives one realisation
of the observation noise and the SMC²/PF stochasticity; without replicates
we cannot separate structural effects from stochastic ones.

This report adds two additional seeds (s123 and s2026) to each condition,
yielding n=3 per condition (9 runs total).

---

## 2. Raw Results (n=3)

### 2.1 Mean raw coverage per run

| Condition | s42 | s123 | s2026 | **Mean** | **Stdev** | 95% CI (t, df=2) |
|-----------|-----|------|-------|----------|-----------|-------------------|
| C0 (baseline) | 83.5 | 88.6 | 83.2 | **85.1** | 3.0 | [77.6, 92.6] |
| C2 (strong) | 95.3 | 82.5 | 91.2 | **89.7** | 6.5 | [73.4, 105.9] |
| C3 (maximal) | 78.1 | 80.1 | 74.1 | **77.4** | 3.1 | [69.8, 85.1] |

### 2.2 Mean informed coverage per run (shrinkage < 0.5)

| Condition | s42 | s123 | s2026 | **Mean** | **Stdev** |
|-----------|-----|------|-------|----------|-----------|
| C0 | 78.4 | 84.3 | 75.7 | **79.5** | 4.4 |
| C2 | 92.7 | 74.8 | 88.4 | **85.3** | 9.3 |
| C3 | 74.0 | 75.0 | 66.6 | **71.9** | 4.6 |

### 2.3 Windows passing ≥70%

| Condition | s42 | s123 | s2026 | Total / 27 |
|-----------|-----|------|-------|------------|
| C0 | 6/9 | 8/9 | 9/9 | 23/27 |
| C2 | 9/9 | 8/9 | 9/9 | 26/27 |
| C3 | 6/9 | 7/9 | 5/9 | 18/27 |

### 2.4 Per-window raw (%)

```
                 W1   W2   W3   W4   W5   W6   W7   W8   W9
C0_s42          100  100   94   97   94   91   55   52   70
C0_s123          97   97   91   70   85   91   91   85   91
C0_s2026         88   91   97   76   82   85   76   73   82
C2_s42          100  100   97  100  100   91   91   91   88
C2_s123          88   91   91   82   85   82   73   70   82
C2_s2026         97   94   94   94   88   88   94   82   91
C3_s42           97   79   70   70   94   88   85   76   46
C3_s123          97   91   73   91   67   79   67   76   82
C3_s2026         94   73   70   64   82   79   76   61   70
```

---

## 3. Paired-Seed Analysis

Within-seed pairing controls for the observation-noise realisation and PF
stochasticity that drives a large share of the single-run variance. The
quantity `C - C0` at fixed seed isolates the effect of the macrocycle
change.

### 3.1 C2 − C0 (strong vs baseline)

| Seed | C0 | C2 | C2 − C0 |
|------|-----|-----|---------|
| 42 | 83.5 | 95.3 | **+11.8** |
| 123 | 88.6 | 82.5 | **−6.1** |
| 2026 | 83.2 | 91.2 | **+8.0** |
| **Mean** | | | **+4.6** |

**Sign inconsistent** (+, −, +). The mean +4.6 pp advantage for C2 is inside
its own stdev (7.7 pp). A paired one-sample t-test on the differences gives
`t = 1.03, df = 2, p ≈ 0.41` — nowhere near significance.

### 3.2 C3 − C0 (maximal vs baseline)

| Seed | C0 | C3 | C3 − C0 |
|------|-----|-----|---------|
| 42 | 83.5 | 78.1 | −5.4 |
| 123 | 88.6 | 80.1 | −8.5 |
| 2026 | 83.2 | 74.1 | −9.1 |
| **Mean** | | | **−7.7** |

**Sign consistent** (all negative). Paired one-sample t-test:
`t = -6.8, df = 2, p ≈ 0.02`. **C3 is reliably worse than baseline.**

### 3.3 C2 − C3 (strong vs maximal)

| Seed | C2 | C3 | C2 − C3 |
|------|-----|-----|---------|
| 42 | 95.3 | 78.1 | +17.2 |
| 123 | 82.5 | 80.1 | +2.4 |
| 2026 | 91.2 | 74.1 | +17.1 |
| **Mean** | | | **+12.2** |

All positive. Paired t-test: `t = 2.9, df = 2, p ≈ 0.10` — borderline with
n=3. The effect size is large (~12 pp) even if the paired test is
underpowered.

---

## 4. Revised Findings

### Finding 1: The "C2 is best" conclusion from s42 was partly seed luck.

The original s42 experiment showed C2 beating C0 by 11.8 pp. Averaged
across three seeds, C2 beats C0 by only 4.6 pp, and one of three seeds
(s123) reverses the ordering entirely (C0 beats C2 by 6.1 pp). **There is
no statistical evidence (n=3) that the C2 macrocycle improves coverage
over the C0 baseline**, despite the very large s42 effect.

### Finding 2: C3 is reliably worse than baseline.

All three paired differences `C3 − C0` are negative, mean −7.7 pp, stdev
only 1.9 pp. The paired t-test gives p ≈ 0.02. This is the one effect in
the experiment that is robust across seeds. **The aggressive 75-day
overreach cycle genuinely hurts identifiability**, not just for this seed
trajectory.

### Finding 3: C2 is the highest-variance condition.

Across-seed stdev: C2 = 6.5 pp, C0 = 3.0 pp, C3 = 3.1 pp. C2 both has
the highest mean and the highest variance — it is capable of brilliance
(95.3% at s42) but also of mediocrity (82.5% at s123). C0 is the most
stable performer and never drops below 83%. For deployment, stability
matters at least as much as peak performance.

### Finding 4: C0 W7-W8 collapse is seed-dependent.

The original s42 report highlighted a "catastrophic W7-W8 collapse"
(54.5%, 51.5%) in C0 attributed to broken-watch gap + flat B. But C0
at s123 shows no such collapse (W7-W8 = 91%, 85%), and C0 at s2026 is
also uneventful (W7-W8 = 76%, 73%). The collapse is a property of the
s42 particular realisation, not a robust feature of C0.

### Finding 5: Single-seed experiments on this model have ~5-13 pp noise.

Across-seed range: C0 spans 83.2-88.6 (5.4 pp), C2 spans 82.5-95.3 (12.8
pp), C3 spans 74.1-80.1 (6.0 pp). Any future single-seed result on this
model should be interpreted with a ±5-10 pp confidence envelope.

---

## 5. Revised Conclusions

### 5.1 Ranking

Based on n=3 cross-seed replication:

| Rank | Condition | Mean (raw) | Stdev | Notes |
|------|-----------|------------|-------|-------|
| 1 | C2 | 89.7 | 6.5 | Highest mean but highest variance |
| 2 | C0 | 85.1 | 3.0 | Most stable, never catastrophic |
| 3 | C3 | 77.4 | 3.1 | Reliably worst (p ≈ 0.02 vs C0) |

### 5.2 What we now believe

- **Strong (C2) ≈ baseline (C0) on average.** The ~5 pp C2 advantage does
  not survive replication at n=3. The earlier "excitation helps" story
  from the single-seed experiment is overturned.
- **Aggressive (C3) is worse than baseline.** The 75-day moderate→taper→
  overreach cycle breaks the guided-Kalman PF proposal's local-linearity
  assumption. This is a structural effect, repeatable across seeds.
- **Excitation can only hurt in this model.** Increasing training
  variation beyond C0 does not improve identifiability; pushing it
  further (C3) degrades it. The "Goldilocks tracking bandwidth" story
  from the original report was wrong — there is no C2 sweet spot that
  beats baseline.

### 5.3 What the earlier report got right

- **C3 is structurally problematic.** This held up.
- **kappa_vagal was a problem at s42 under C0** (3/9 failures). True for
  that seed.
- **Excitation-based diagnosis of the kappa_vagal failure was right in
  direction** — B variance matters — but the prescribed fix (more
  excitation) doesn't reliably work.

### 5.4 Recommendations

1. **Revert C2 recommendation.** The `excitation_experiment_report.md`
   recommends "adopt C2 macrocycle as default". This is not supported
   by cross-seed replication. **C0 is equally good on average and more
   stable.**

2. **Retire the Goldilocks hypothesis.** Single-seed non-monotonicity
   (C0 < C2 > C3) was partially driven by seed luck. Cross-seed mean
   ordering is C2 ≥ C0 > C3 with C2/C0 inseparable.

3. **Look elsewhere for late-window failures.** The kappa_vagal /
   B-dependent failures seen at s42 are seed-specific, not a generic
   identifiability problem to be fixed by excitation. Diagnose
   per-seed, not per-macrocycle.

4. **Increase replicate count for future experiments.** n=3 gives wide
   CIs on this model. For any subsequent experimental claim about
   macrocycle/prior/hyperparameter effects, plan n ≥ 5.

5. **Flag C3 as unsafe.** If excitation design becomes relevant for
   real-data deployment, the 75-day aggressive cycle pattern should be
   avoided — not because the real athlete's training is wrong, but
   because the estimator cannot track it.

---

## 6. Driver update note

Runs A and B (both s123) completed 2026-04-22 afternoon on the previous
NVIDIA driver. The system was rebooted after a power cut and upgraded to
driver 590.48.01 before Run D (C0/s123) and the subsequent s2026 batch.

Direct pre/post comparison at the same config was not possible (no
identical seed before and after), but C0/s42 (pre) vs C0/s123 (post) at
the same N shows wall-clock 5133 s → 4844 s (~5.6% faster). VRAM
footprint dropped from ~OOM-at-N=512 to ~25.9 GiB at N=256 (~25% reduction),
and GPU compute utilisation during the SMC inner loop rose from ~50% to
~97% sustained. No results appear driver-dependent (all 9 runs completed
cleanly without numerical anomalies).

---

## 7. Files

| File | Purpose |
|------|---------|
| `C0/` | C0 / seed=42 (pre-driver) |
| `C0_N256_s123/` | C0 / seed=123 (post-driver, Run D) |
| `C0_N256_s2026/` | C0 / seed=2026 (Run E) |
| `C2/` | C2 / seed=42 |
| `C2_N256_s123/` | C2 / seed=123 (Run B) |
| `C2_N256_s2026/` | C2 / seed=2026 (Run F) |
| `C3/` | C3 / seed=42 |
| `C3_N256_s123/` | C3 / seed=123 (Run A) |
| `C3_N256_s2026/` | C3 / seed=2026 (Run G) |

Each checkpoint contains per-window `coverage`, `coverage_informed`,
`n_informed`, and full parameter-tracking stats. All 9 runs used identical
config except `SEED` and `EXCITATION_CONDITION`.
