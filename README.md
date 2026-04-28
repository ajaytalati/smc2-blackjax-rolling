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

## Recommended bridge: Schrödinger-Föllmer Path B (decoupled)

**A single SF-bridge config strictly beats the Gaussian bridge on both
production models (no per-model tuning):**

```python
SMCConfig(
    bridge_type='schrodinger_follmer',
    sf_q1_mode='annealed',           # K-stage tempered SMC for q1 (issue #1 fix)
    sf_use_q0_cov=True,              # decoupled location/scale (issue #3 fix)
    sf_blend=0.7,                    # tuned vs 0.5 / 0.85 / 1.0
    sf_annealed_n_stages=3,
    sf_annealed_n_mh_steps=5,        # tuned vs 2 / 8
    sf_annealed_proposal_scale=0.4,  # Roberts-Gelman-Gilks for d ~ 30-35
)
```

Add to either driver via CLI:

```bash
PYTHONPATH=. python drivers/fsa_high_res_rolling.py --seed 42 \
    --bridge schrodinger_follmer --sf-q1-mode annealed \
    --sf-use-q0-cov --sf-blend 0.7 --sf-annealed-n-mh-steps 5

PYTHONPATH=. python -m drivers.swat.rolling --seed 42 \
    --bridge schrodinger_follmer --sf-q1-mode annealed \
    --sf-use-q0-cov --sf-blend 0.7 --sf-annealed-n-mh-steps 5
```

Results vs the Gaussian bridge:

| Model (dim) | Gauss | SF Path B-fixed |
|---|---:|---:|
| **fsa_high_res** C0 (29-D) | 96.8% / 27-of-27 PASS | **98.5% / 27-of-27** |
| **SWAT** Set A (35-D) | 49.8% / 4-of-27 PASS | **82.3% / 24-of-27** |

The SF impl lives in [`smc2bj/estimation/sf_bridge.py`](smc2bj/estimation/sf_bridge.py); the
debug-and-tune story across three failed designs (closing
[#1](https://github.com/ajaytalati/smc2-blackjax-rolling/issues/1) and
[#3](https://github.com/ajaytalati/smc2-blackjax-rolling/issues/3))
is documented in [`outputs/SF_BEST_PRACTICE_2_models.md`](outputs/SF_BEST_PRACTICE_2_models.md).

## Adding a new model

Models live canonically in the public dev repo
([Python-Model-Development-Simulation](https://github.com/ajaytalati/Python-Model-Development-Simulation))
following the 3-file convention; psim
([Python-Model-Scenario-Simulation](https://github.com/ajaytalati/Python-Model-Scenario-Simulation))
gates them through the §1.4 sim-est consistency discipline and
produces packaged scenario artifacts. The SMC² port is the third
leg: a per-model driver package under `drivers/<model>/` consuming
the validated artifact.

**The canonical guide for adding a new model to the SMC² repo is
[`how_to_add_a_new_model/`](how_to_add_a_new_model/)** — orientation,
prerequisites, 5-step checklist, and a line-by-line worked example
using SWAT.

⚠ **Before running any SMC**, complete the **Sim-Est Consistency
Validation** checks in [§1.4 of the porting guide](docs/PORTING_GUIDE.md).
A single phase-misalignment between simulator and estimator can
produce confidently-wrong posteriors that look like a bridge problem.
See the [postmortem](outputs/fsa_high_res_rolling/POSTMORTEM_three_bugs.md)
for the cost (~15h wasted GPU/analyst time across three bugs that the
checks would have caught in 30 minutes).

## Provenance

- Pre-refactor snapshot: `pre-refactor-2026-04-23` tag in the upstream
  [Python-BlackJAX-Centric-Estimation-Framework](https://github.com/ajaytalati/Python-BlackJAX-Centric-Estimation-Framework) repo.
- Refactor tag here: `v0.1.0`.
