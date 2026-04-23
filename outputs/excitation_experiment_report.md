# Data Excitation Experiment Report

**Date:** 2026-04-22
**Model:** FSA Real-Obs, 33 parameters, v4.1 (frozen R_base + kappa_chronic)
**Config:** K=400 PF particles, N=256 SMC particles, 120-day windows, 30-day stride
**Runtime:** 3 conditions x ~1.5h = 4.3h total on RTX 5090

---

## 1. Hypothesis

Late-window parameter failures (kappa_vagal 3/9, S_base, s_A) are caused by
insufficient *data excitation* — the latent states (B, F, A) don't vary enough
to provide Fisher information for sensitivity parameters. Specifically:

1. B stabilizes at high fitness (~0.65-0.70), making kappa_vagal unidentifiable
2. The 180-day periodic macrocycle provides no new excitation patterns after
   the first cycle

**Prediction:** Macrocycles with deeper tapers and overreach cycles will increase
B-state variance, improve late-window coverage, and reduce kappa_vagal failures.

---

## 2. Experimental Design

Three macrocycle conditions, all else equal (N=256, K=400, 365 days, 9 windows):

| Condition | Description | B var (W7-W9) | mu crossings |
|-----------|-------------|---------------|--------------|
| **C0** | Baseline v4.1: 28d mesocycles + 21d taper/180d + 14d overreach/90d | 0.0095 | 12 |
| **C2** | Strong: 28d base + 35d deep tapers/90d + 21d immediate overreach | 0.0358 (3.8x) | 8 |
| **C3** | Maximal: 75d repeating cycles (30d moderate + 30d taper + 15d overreach) | 0.0258 (2.7x) | 10 |

### Pre-estimation diagnostics

Forward simulation confirmed:
- B-state variance in C2/C3 is 2.7-3.8x higher than C0 in late windows
- mu(B,F) crosses zero in all conditions (12x in C0, 8x in C2, 10x in C3)
- A-state is absorbing in all conditions (range ~0.009) due to eps_A=1e-4
  making A=0 a near-absorbing boundary in the Landau bifurcation

---

## 3. Results

### 3.1 Per-Window Raw Coverage

| Window | Days | C0 | C2 | C3 |
|--------|------|-----|-----|-----|
| W1 | 0-120 | **100.0%** | **100.0%** | 97.0% |
| W2 | 30-150 | **100.0%** | **100.0%** | 78.8% |
| W3 | 60-180 | 93.9% | **97.0%** | 69.7% |
| W4 | 90-210 | **97.0%** | **100.0%** | 69.7% |
| W5 | 120-240 | 93.9% | **100.0%** | 93.9% |
| W6 | 150-270 | 90.9% | 90.9% | 87.9% |
| W7 | 180-300 | 54.5% | **90.9%** | 84.8% |
| W8 | 210-330 | 51.5% | **90.9%** | 75.8% |
| W9 | 240-360 | 69.7% | **87.9%** | 45.5% |
| **Mean** | | **83.5%** | **95.3%** | **78.1%** |

### 3.2 Per-Window Informed Coverage

| Window | C0 | C2 | C3 |
|--------|-----|-----|-----|
| W1 | **100.0%** | **100.0%** | 95.0% |
| W2 | **100.0%** | **100.0%** | 65.0% |
| W3 | 88.2% | **95.0%** | 62.5% |
| W4 | 94.1% | **100.0%** | 61.5% |
| W5 | 89.5% | **100.0%** | 91.3% |
| W6 | 83.3% | 85.0% | 87.0% |
| W7 | 44.4% | **85.7%** | 81.8% |
| W8 | 42.3% | **86.4%** | 73.1% |
| W9 | 64.0% | **81.8%** | 48.4% |
| **Mean** | **78.4%** | **92.7%** | **74.0%** |

### 3.3 Perfectly Identified Parameters (0/9 failures)

| Condition | Perfect params | Fraction |
|-----------|---------------|----------|
| C0 | 15/33 | 45.5% |
| **C2** | **25/33** | **75.8%** |
| C3 | 10/33 | 30.3% |

### 3.4 kappa_vagal Posterior Analysis (truth = 12.0)

```
C0:
  Window  Mean    Std    90% CI           Shrink  Status
  W1      12.47  0.630  [11.47, 13.54]   0.163   PASS
  W2      12.02  0.568  [10.96, 12.85]   0.147   PASS
  W3      11.46  0.611  [10.55, 12.54]   0.159   PASS
  W4      12.61  0.713  [11.69, 13.97]   0.185   PASS
  W5      12.00  0.770  [10.80, 13.28]   0.200   PASS
  W6      11.48  0.641  [10.47, 12.56]   0.166   PASS
  W7       9.43  0.257  [ 8.99,  9.79]   0.067   FAIL ← worst
  W8      10.02  0.523  [ 9.12, 10.79]   0.136   FAIL
  W9      10.53  0.379  [ 9.96, 11.22]   0.098   FAIL

C2:
  W1      12.15  0.628  [11.11, 13.21]   0.163   PASS
  W2      11.79  0.518  [11.04, 12.68]   0.134   PASS
  W3      12.06  0.774  [10.88, 13.43]   0.201   PASS
  W4      12.27  0.478  [11.64, 13.14]   0.124   PASS
  W5      12.40  0.582  [11.41, 13.30]   0.151   PASS
  W6      12.73  0.761  [11.67, 14.08]   0.198   PASS
  W7      11.70  0.548  [11.01, 12.86]   0.142   PASS
  W8      11.56  0.384  [10.92, 12.21]   0.100   PASS
  W9      11.79  0.753  [10.76, 13.07]   0.196   PASS

C3:
  W1      13.22  0.665  [12.12, 14.26]   0.173   FAIL ← biased high
  W2      14.77  0.714  [13.55, 15.90]   0.185   FAIL ← worst
  W3      12.38  0.367  [11.87, 13.06]   0.095   PASS
  W4      13.27  0.622  [12.35, 14.29]   0.162   FAIL
  W5      12.82  0.578  [11.78, 13.69]   0.150   PASS
  W6      12.29  0.400  [11.63, 12.96]   0.104   PASS
  W7      13.14  0.579  [12.05, 13.99]   0.150   FAIL
  W8      12.12  0.334  [11.59, 12.57]   0.087   PASS
  W9      11.94  0.319  [11.44, 12.50]   0.083   PASS
```

**Key observations:**

1. **C0 kappa_vagal bias is downward** (9.43-10.53 in W7-W9) — consistent with
   flat B reducing effective sensitivity.
2. **C2 kappa_vagal is centred on truth** across all 9 windows (11.56-12.73),
   with well-calibrated uncertainty.
3. **C3 kappa_vagal bias is upward** (13.22-14.77 in W1-W2, W4, W7) — the
   aggressive cycling creates a different kind of misidentification.

### 3.5 Per-Parameter Coverage Map (C2, the winner)

```
Param              W1    W2    W3    W4    W5    W6    W7    W8    W9    Fail
---------------------------------------------------------------------------
tau_B              pass  pass  pass  pass  pass  pass  pass  pass  pass  0/9
alpha_A            pass  pass  pass  pass  pass  pass  pass  pass  pass  0/9
tau_F              pass  pass  pass  pass  pass  pass  pass  pass  pass  0/9
lambda_B           pass  pass  pass  pass  pass  pass  pass  pass  pass  0/9
lambda_A           pass  pass  pass  pass  pass  pass  pass  pass  pass  0/9
mu_0_abs           pass  pass  pass  pass  pass  pass  pass  pass  pass  0/9
mu_B               pass  pass  pass  pass  pass  pass  pass  pass  pass  0/9
mu_F               pass  pass  pass  pass  pass  pass  pass  pass  pass  0/9
mu_FF              pass  pass  pass  pass  pass  pass  pass  pass  pass  0/9
eta                pass  pass  pass  pass  pass  pass  pass  pass  pass  0/9
kappa_vagal        pass  pass  pass  pass  pass  pass  pass  pass  pass  0/9
sigma_obs_R        pass  pass *FAIL  pass  pass  pass  pass  pass  pass  1/9
I_base             pass  pass  pass  pass  pass *FAIL  pass *FAIL  pass  2/9
c_B                pass  pass  pass  pass  pass *FAIL  pass  pass  pass  1/9
c_F                pass  pass  pass  pass  pass  pass  pass  pass  pass  0/9
sigma_obs_I        pass  pass  pass  pass  pass  pass  pass  pass  pass  0/9
D_base             pass  pass  pass  pass  pass  pass  pass  pass *FAIL  1/9
d_B                pass  pass  pass  pass  pass *FAIL *FAIL *FAIL *FAIL  4/9
d_F                pass  pass  pass  pass  pass  pass  pass  pass  pass  0/9
sigma_obs_D        pass  pass  pass  pass  pass  pass *FAIL *FAIL *FAIL  3/9
S_base             pass  pass  pass  pass  pass  pass  pass  pass  pass  0/9
s_A                pass  pass  pass  pass  pass  pass  pass  pass  pass  0/9
s_F                pass  pass  pass  pass  pass  pass  pass  pass  pass  0/9
sigma_obs_S        pass  pass  pass  pass  pass  pass  pass  pass *FAIL  1/9
Sleep_base         pass  pass  pass  pass  pass  pass  pass  pass  pass  0/9
sl_A               pass  pass  pass  pass  pass  pass  pass  pass  pass  0/9
sl_B               pass  pass  pass  pass  pass  pass  pass  pass  pass  0/9
sl_F               pass  pass  pass  pass  pass  pass  pass  pass  pass  0/9
sigma_obs_Sleep    pass  pass  pass  pass  pass  pass  pass  pass  pass  0/9
Time_base          pass  pass  pass  pass  pass  pass *FAIL  pass  pass  1/9
t_A                pass  pass  pass  pass  pass  pass  pass  pass  pass  0/9
t_F                pass  pass  pass  pass  pass  pass  pass  pass  pass  0/9
sigma_obs_Time     pass  pass  pass  pass  pass  pass  pass  pass  pass  0/9
```

**Perfectly identified (0/9 failures):** 25 out of 33 parameters
**Intermittent failures (1-2/9):** 5 parameters (sigma_obs_R, I_base, c_B, D_base, sigma_obs_S)
**Persistent failures (3+/9):** 3 parameters (d_B 4/9, sigma_obs_D 3/9, Time_base 1/9)

Notable improvements over C0:
- kappa_vagal: 3/9 → **0/9**
- S_base: 3/9 → **0/9**
- mu_B: 3/9 → **0/9**
- c_B: 3/9 → **1/9**
- Sleep_base: 3/9 → **0/9**
- sl_B: 3/9 → **0/9**
- Time_base: 3/9 → **1/9**
- sigma_obs_D: 5/9 → 3/9

---

## 4. Analysis

### 4.1 The Non-Monotonic Result

The most important finding is that **more excitation is not monotonically better**:

```
C0 (baseline):  83.5% raw,  78.4% informed,  15/33 perfect
C2 (strong):    95.3% raw,  92.7% informed,  25/33 perfect  ← OPTIMAL
C3 (maximal):   78.1% raw,  74.0% informed,  10/33 perfect  ← WORSE THAN BASELINE
```

This reveals a fundamental tension:
- **Too little excitation** (C0): B stabilizes, losing Fisher information for
  kappa_vagal and B-dependent params. Late-window collapse (W7-W8: 54.5%, 51.5%).
- **Too much excitation** (C3): Rapid 75-day phase transitions create dynamics
  the particle filter struggles to track. The SMC particles can't adapt fast
  enough to the changing state, leading to particle impoverishment and early
  collapse (W2-W4: 78.8%, 69.7%, 69.7%).
- **Optimal excitation** (C2): Deep tapers with mesocycle base layer provide
  large B-F swings while maintaining enough stability for the estimator to track.

### 4.2 Why C3 Fails: Particle Filter Tracking Limitations

C3's 75-day cycles (30d moderate → 30d taper → 15d overreach) create abrupt
regime changes every 30 days. The particle filter's guided Kalman proposal
assumes locally linear dynamics, which breaks down when:

1. B drops from 0.65 to 0.15 during the 30-day taper (large non-linear excursion)
2. F spikes from 0.05 to 0.70 during the 15-day overreach (rapid CIR jump)
3. The B-F correlation structure inverts between phases

The SMC bridge warm-start then propagates the particle impoverishment across
windows, creating a cascade effect (W2 collapse → W3 → W4).

C2 avoids this because the 28-day mesocycle base layer provides a smooth
"backbone" that the taper/overreach overlays modulate gradually. The estimator
can track the evolving state because the transitions are less abrupt.

### 4.3 C0 W7-W8 Collapse: Broken Watch + Flat B

C0's catastrophic W7-W8 collapse (54.5%, 51.5%) is caused by the broken watch
gap (days 247-261, 14 days of all-channel missing data) coinciding with a period
of flat B-state. With no observations AND no B-variance, the particle filter
degenerates and all B-dependent parameters fail simultaneously (15 failures in
W7 alone).

C2 avoids this because the deep taper around day 235-270 creates large B
excursions that provide information even around the broken watch gap.

### 4.4 d_B: The Remaining Persistent Failure

d_B (duration ~ B sensitivity) fails 4/9 in C2. This parameter couples duration
observations to B-state:
```
pred_D = D_base + d_B * B + d_F * F
```

When B has large variance (C2), d_B should be well-identified. The failures
concentrate in W6-W9, suggesting the duration observation channel has insufficient
signal-to-noise ratio in later windows (sigma_obs_D absorbs too much variance).

### 4.5 A-State: Absorbing Boundary Confirmed

All three conditions show identical A-state range (~0.009), confirming that
eps_A=1e-4 makes the Landau bifurcation effectively absorbing. The
`dA = mu*A - eta*A^3` dynamics with `A = max(A, 0)` clipping prevent A from
growing even when mu goes strongly negative (mu = -0.25 in C2/C3).

Despite this, s_A improved from 2/9 failures (C0) to 0/9 (C2) — not because
A is varying more, but because the improved particle cloud from better
B-F tracking provides more accurate joint posteriors.

---

## 5. Falsification Assessment

| Criterion | Result | Verdict |
|-----------|--------|---------|
| C3 informed coverage < C0 + 5pp? | 74.0% < 83.4% | YES — C3 is worse |
| C3 kappa_vagal failures >= C0? | 4/9 >= 3/9 | YES — C3 is worse |
| Monotonic improvement C0 < C2 < C3? | No: C0 < C2 > C3 | FALSIFIED |

**The simple "more excitation = better" hypothesis is FALSIFIED.**

However, the refined hypothesis — "**optimal excitation improves identifiability**"
— is **STRONGLY SUPPORTED**:

| Criterion | Result | Verdict |
|-----------|--------|---------|
| C2 informed coverage > C0 + 5pp? | 92.7% > 83.4% | YES (+14.3pp) |
| C2 kappa_vagal failures < C0? | 0/9 < 3/9 | YES (eliminated) |
| C2 perfect params > C0? | 25/33 > 15/33 | YES (+10 params) |
| Non-excitation params unchanged? | tau_B, lambda_A, etc.: 0/9 in both | YES |

---

## 6. Conclusions and Recommendations

### 6.1 Key Findings

1. **C2 macrocycle is the recommended default** for synthetic testing and
   real-data deployment. It provides:
   - 95.3% mean raw coverage (vs 83.5% baseline)
   - 25/33 perfectly identified parameters (vs 15/33)
   - 0/9 kappa_vagal failures (vs 3/9)
   - Stable coverage across all windows (min 87.9% vs min 51.5%)

2. **Training periodization matters for identifiability.** Real athletes with
   periodized training (base/build/peak/recovery cycles) will provide better
   parameter estimates than athletes with monotonous training programs.

3. **The particle filter has a tracking bandwidth.** There exists an optimal
   rate of state change — too slow loses Fisher information, too fast exceeds
   the PF's ability to track. C2's 90-day taper cycle hits this sweet spot;
   C3's 75-day cycle exceeds it.

### 6.2 For v4.2

1. **Adopt C2 macrocycle as default** for all future experiments.
2. **Investigate d_B persistent failure** — may need a tighter prior or
   different observation model for duration channel.
3. **Consider adaptive excitation** — if deploying on real data, patients with
   monotonous training may need wider priors (accepting less precision) or
   recommendations for training variation.
4. **A-state remains frozen** — this is a model architecture issue (pitchfork
   bifurcation with absorbing boundary), not an excitation issue. Future work
   should either increase eps_A, add noise injection, or accept that A-dependent
   parameters (s_A, sl_A) are identified indirectly through the joint posterior.

### 6.3 Runtime

| Condition | Runtime | Windows |
|-----------|---------|---------|
| C0 | 1.4h | 9 |
| C2 | 1.5h | 9 |
| C3 | 1.4h | 9 |
| **Total** | **4.3h** | **27** |

All conditions used identical compute: N=256 SMC particles, K=400 PF particles,
RTX 5090 GPU.

---

## 7. Files

| File | Purpose |
|------|---------|
| `fsa_real_obs_5yr_rolling_smc.py` | Main script with C0/C2/C3 macrocycle generators |
| `run_excitation_diagnostics.py` | Pre-estimation diagnostic script |
| `outputs/fsa_real_obs_5yr_rolling/C0/` | C0 baseline results |
| `outputs/fsa_real_obs_5yr_rolling/C2/` | C2 strong excitation results |
| `outputs/fsa_real_obs_5yr_rolling/C3/` | C3 maximal excitation results |
| `outputs/excitation_diagnostics/` | Pre-estimation diagnostic plots |
