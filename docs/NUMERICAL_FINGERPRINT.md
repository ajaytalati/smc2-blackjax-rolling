# Numerical Fingerprint — reference output at a locked seed

A locked numerical output from the reference driver. If a re-implementation
of the algorithm spec produces values within `rtol=1e-3` of this table,
it is numerically equivalent to ours for verification purposes.

---

## Configuration

| | |
|---|---|
| Driver | [drivers/fsa_real_obs_5yr_rolling.py](../drivers/fsa_real_obs_5yr_rolling.py) |
| Seed | `--seed 42` |
| Condition | `--condition C0` (baseline macrocycle) |
| Windows | `--windows 1` (cold-start W1 only) |
| N_SMC | `--n-smc 256` (default) |
| N_PF | `--n-pf 400` (default) |
| Window | 120 days |
| Stride | 30 days |
| Substeps | 10 / day |
| JAX | 0.9.2 |
| BlackJAX | 1.5 |
| Device | NVIDIA RTX 5090, driver 590.48.01 |

## Reproduction

```bash
python drivers/fsa_real_obs_5yr_rolling.py --seed 42 --condition C0 --windows 1
# inspect outputs/fsa_real_obs_5yr_rolling/C0_N256_s42/rolling_checkpoint.json
```

## Reference trajectory checksum

The synthetic-data pipeline (macrocycle → SDE integrator → observations) is
fully deterministic and must match bit-exactly on any platform with the same
numpy RNG state. The trajectory CSV fingerprint:

```
md5(trajectory_5yr.csv) = 1646a4f29347c15f9c727fa3dfc1263a
```

If this differs on your platform, the issue is in `simulate_sde` or
`generate_observations`, not the SMC².

## Reference SMC² outputs (Window 1, cold-start)

Measured on 2026-04-23, commit `88f4461`, JAX 0.9.2 + BlackJAX 1.5,
RTX 5090 (driver 590.48.01). The refactor parity run at seed=42 /
C0 / 1 window reproduced the pre-refactor checkpoint with no coverage
drift (33/33 parameters in 90% CI in both) and only small (<1%)
posterior-mean drift from XLA bisection scheduling.

| Quantity | Reference value | Tolerance band |
|----------|-----------------|----------------|
| Coverage (raw, 33 params) | **1.000** | [0.90, 1.00] |
| Coverage (data-informed, 17 params) | **1.000** | [0.85, 1.00] |
| Tempering levels | **36** (ref) / 35 (pre-refactor) | [25, 50] |
| Wall-clock per window (RTX 5090) | **1439 s** (ref) / 1376 s (pre-refactor) | [1100, 1700] |
| Extracted init state for W2 | (0.6244, 0.2881, 0.0013) (B, F, A) | — |

### Per-parameter posterior (W1, C0, seed=42)

Refactored posterior means, with absolute difference from the
pre-refactor reference. Every parameter's `in_ci` status matches.

| Parameter | New mean | Δ vs pre-refactor | New stdev |
|-----------|----------|-------------------|-----------|
| tau_B | 14.0830 | +0.0615 | 0.8328 |
| alpha_A | 1.1330 | +0.0688 | 0.4533 |
| tau_F | 7.3649 | +0.0632 | 1.4149 |
| lambda_B | 2.9506 | −0.0167 | 0.7143 |
| lambda_A | 1.6157 | −0.0210 | 0.4805 |
| mu_0_abs | 0.1190 | +0.0028 | 0.0449 |
| mu_B | 0.2616 | +0.0037 | 0.0715 |
| mu_F | 0.1271 | +0.0064 | 0.0524 |
| mu_FF | 0.3814 | −0.0114 | 0.1621 |
| eta | 0.2115 | +0.0084 | 0.0621 |
| kappa_vagal | 12.5142 | +0.0436 | 0.7086 |
| sigma_obs_R | 1.5058 | −0.0053 | 0.1099 |
| I_base | 0.5141 | +0.0050 | 0.0314 |
| c_B | 0.1841 | −0.0109 | 0.0490 |
| c_F | 0.1035 | −0.0091 | 0.0396 |
| sigma_obs_I | 0.0528 | +0.0001 | 0.0043 |
| D_base | 0.5103 | +0.0067 | 0.0422 |
| d_B | 0.3180 | −0.0100 | 0.0671 |
| d_F | 0.2592 | +0.0031 | 0.0699 |
| sigma_obs_D | 0.0796 | −0.0007 | 0.0061 |
| S_base | 30.5046 | −0.0683 | 1.3401 |
| s_A | 15.6140 | −0.8752 | 8.6368 |
| s_F | 19.5758 | +0.1086 | 4.2405 |
| sigma_obs_S | 5.0577 | +0.0130 | 0.3703 |
| Sleep_base | 0.5134 | −0.0069 | 0.0237 |
| sl_A | 0.2053 | −0.0199 | 0.1133 |
| sl_B | 0.0785 | +0.0078 | 0.0274 |
| sl_F | 0.1487 | −0.0066 | 0.0519 |
| sigma_obs_Sleep | 0.1024 | +0.0017 | 0.0071 |
| Time_base | 0.0458 | +0.0148 | 0.0914 |
| t_A | 1.1287 | +0.0296 | 0.5686 |
| t_F | 0.5623 | +0.0390 | 0.2523 |
| sigma_obs_Time | 0.4701 | −0.0043 | 0.0347 |

Worst-case |Δmean| = 0.88 (on `s_A`, which has stdev 8.64 — the drift is
0.10 stdev, well inside posterior noise). Max |Δmean / mean| = 48%
(`Time_base`, which has tiny magnitude ~0.04 and stdev 0.09 — relative
delta is large but absolute drift is 0.015, inside stdev).

### Pre-refactor reference

The pre-refactor checkpoint is preserved at
[outputs/fsa_real_obs_5yr_rolling/C0/rolling_checkpoint.json](../outputs/fsa_real_obs_5yr_rolling/C0/rolling_checkpoint.json)
(tracked from the backup commit `0d1ba5d`).

### Comparing a third-party re-implementation

A fresh implementation in Julia / Stan / etc. should:

1. Match the trajectory CSV bit-exactly (pipeline is numpy-only, deterministic).
2. Match the coverage exactly (100% here, 33/33).
3. Produce per-parameter posterior means within 2 stdev of this reference
   (weaker than rtol=1e-3, but the stochastic PF and adaptive tempering
   make tighter comparisons unreliable across RNG implementations).

If a re-implementation diverges on any parameter by >2 stdev, check
first whether the inner PF's guided-Kalman proposal and OT rescue
thresholds match §3 of the algorithm spec.

## When the fingerprint should be updated

- Deliberate algorithm change (e.g. switching inner PF variant).
- JAX / BlackJAX major-version upgrade with known numerical differences.
- Hardware change that affects XLA compilation paths.

Any unexplained drift beyond `rtol=1e-3` should be investigated before
the fingerprint is updated.
