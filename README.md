# smc2-blackjax-rolling

**Rolling-window SMC² Bayesian estimation for SDE state-space models**, with a
guided-Kalman differentiable particle filter (GK-DPF v3-lite) as the inner
likelihood estimator. Built on JAX + BlackJAX, targeted at RTX-class GPUs.

This repo is the clean extraction of the framework originally developed at
`Python-BlackJAX-Centric-Estimation-Framework/version_4_1/`. The tangled 1538-line
driver was split into (a) a generic framework package (`smc2bj/`), (b)
per-model packages under `models/<name>/` following a 3-file convention, and
(c) thin scenario drivers under `drivers/`.

---

## Three audiences

| You are… | Start here |
|----------|------------|
| **Independent verifier** reimplementing the algorithm | [docs/SMC2_ALGORITHM_SPECIFICATION.md](docs/SMC2_ALGORITHM_SPECIFICATION.md) — algorithm spec. [docs/NUMERICAL_FINGERPRINT.md](docs/NUMERICAL_FINGERPRINT.md) — expected numerical output at a locked seed. |
| **Porting agent** swapping in a new model | [docs/PORTING_GUIDE.md](docs/PORTING_GUIDE.md) — the 3-file contract + worked sketch. [models/fsa_real_obs/](models/fsa_real_obs/) — reference example. |
| **Domain reader** interested in the FSA model | [docs/MODEL_SPECIFICATION.md](docs/MODEL_SPECIFICATION.md) — FSA v4.1 equations + priors. [docs/HIGH_RES_ADDENDUM.md](docs/HIGH_RES_ADDENDUM.md) — 15-min variant with SWAT-style mixed likelihood. [outputs/](outputs/) — experimental reports. |

## Repo layout

```
smc2bj/                  # generic framework (model-agnostic)
models/
  fsa_real_obs/          # reference model (daily obs, 6 Gaussian channels)
    simulation.py        #   SDE drift + diffusion + obs generators
    estimation.py        #   priors + PF hooks + EstimationModel instance
    sim_plots.py         #   per-model diagnostic plots
  fsa_high_res/          # 15-min variant (4 channels: HR + sleep Bernoulli + stress + steps)
    (same 3-file layout)
drivers/
  fsa_real_obs_5yr_rolling.py   # daily FSA driver (365d / 9 windows)
  fsa_high_res_rolling.py       # high-res FSA driver (14d / 12 windows / 15-min bins)
docs/                    # specifications (D1-D5) + addenda
outputs/                 # experimental results
tests/                   # regression fingerprints + protocol checks
```

## Quickstart

```bash
pip install -e .

# Daily FSA (reference)
python drivers/fsa_real_obs_5yr_rolling.py --seed 42 --condition C0

# High-res FSA (15-min bins, SWAT-style mixed likelihood)
python drivers/fsa_high_res_rolling.py --seed 42
```

## Porting to your model

Write three files under `models/<your_model>/`: `simulation.py`,
`estimation.py`, `sim_plots.py`. The framework in `smc2bj/` is unchanged.
See [docs/PORTING_GUIDE.md](docs/PORTING_GUIDE.md) for the contract.

## Provenance

- Pre-refactor snapshot: `pre-refactor-2026-04-23` tag in the upstream
  [Python-BlackJAX-Centric-Estimation-Framework](https://github.com/ajaytalati/Python-BlackJAX-Centric-Estimation-Framework) repo.
- Refactor tag here: `v0.1.0`.
