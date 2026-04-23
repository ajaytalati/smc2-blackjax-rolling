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

The reference run (2026-04-23, JAX 0.9.2, RTX 5090 driver 590.48.01)
produced the values in the checkpoint at
[outputs/fsa_real_obs_5yr_rolling/C0_N256_s42/rolling_checkpoint.json](../outputs/fsa_real_obs_5yr_rolling/C0_N256_s42/rolling_checkpoint.json)
(written during Phase D of the refactor).

Key regression thresholds:

| Quantity | Expected range (rtol=1e-3 around reference) |
|----------|---------------------------------------------|
| Window 1 coverage (raw) | 0.85 - 1.00 |
| Window 1 coverage (data-informed) | 0.80 - 1.00 |
| Number of tempering levels | 9 - 15 |
| Wall-clock per window (RTX 5090) | 1200 - 1600 s |

For per-parameter posterior mean / std, consult the reference
`rolling_checkpoint.json`; the `tests/test_smc2_fingerprint.py` regression
test loads both and compares.

## When the fingerprint should be updated

- Deliberate algorithm change (e.g. switching inner PF variant).
- JAX / BlackJAX major-version upgrade with known numerical differences.
- Hardware change that affects XLA compilation paths.

Any unexplained drift beyond `rtol=1e-3` should be investigated before
the fingerprint is updated.
