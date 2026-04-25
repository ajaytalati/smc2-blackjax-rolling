# Plan: Sinkhorn-regularized normalising flow bridge

> Refines [PLAN_principled_bridge_fixes.md](PLAN_principled_bridge_fixes.md) Track 1
> by **regularising the flow training with the existing OT code**
> (`smc2bj/transport/sinkhorn.py`). This combination is expected to give
> the strongest particle-rejuvenation guarantee of any single bridge
> mechanism we can build on top of the current framework.

## Why an OT-regularised flow is the right architecture

The vanilla normalising-flow bridge from the earlier plan has a well-known
failure mode at small N (here N=256 particles, 29-dim posterior):

> **Flow overfits → narrow Gaussian-bump on every input particle → bridge
> samples have no more diversity than the input cloud.** Maximum-likelihood
> training drives the flow to put as much mass as possible on each observed
> point; with 256 points in 29 dim there's nothing to stop it collapsing
> mode-by-mode.

The OT machinery already in `smc2bj/transport/` exists precisely to **resist
this collapse** — the GK-DPF v3-lite uses it as a "rescue" when the inner
PF's ESS drops, smoothing the particle cloud back to a more-uniform
covering of the target. Same logic applies during flow training: penalising
the flow when it concentrates too tightly relative to its target keeps the
bridge density expressive *and* well-spread.

Conceptually:

| Loss term | What it enforces | What it punishes |
|-----------|------------------|------------------|
| `−E_p[log q_φ]` (negative log-likelihood) | flow density matches data empirically | flow density evaluated low at observed particles |
| `λ · W_ε(q_φ, p̂)` (Sinkhorn-regularised OT) | flow samples spread out to cover the data **as a measure** | mass-concentration / mode collapse |

The Sinkhorn term is **entropy-regularised**: the smoothness comes for
free. With λ tuned right, the flow learns the *shape* of the previous
posterior without painting it onto individual particles.

This is also the clean unification with the existing `gk_dpf_v3_lite.py`
OT rescue — same Sinkhorn module, same regularisation philosophy, just
applied at flow-training time instead of inside the inner PF.

---

## Architecture

### Flow

**Real-NVP**, 4–6 coupling layers, alternating split masks.

- 4 layers if N=256 (small data, avoid over-parameterisation)
- 6 layers if N=512+ or for SWAT later
- Hidden width 64
- GELU activations (smooth, good for HMC differentiability of the inverse map)
- **Affine** couplings: `s(x_a) → exp(scale) ⊙ x_b + shift(x_a)`
- Add **soft-clamp** on the log-scale output: `log_scale = c · tanh(raw / c)` with `c = 2`. Stops degenerate small/large determinants and keeps Jacobian numerically well-conditioned.

### Loss

$$
\mathcal{L}(\phi) = \underbrace{-\frac{1}{N}\sum_{i=1}^{N} \log q_\phi(x_i)}_{\text{NLL: density matches }p\text{}}
\;+\; \lambda \cdot \underbrace{\mathcal{S}_\varepsilon\!\left(q_\phi^{(M)}, \hat p_N\right)}_{\text{Sinkhorn(flow samples, particles)}}
$$

where:

- $\{x_i\}_{i=1}^N$ — previous posterior particles in unconstrained space
- $q_\phi^{(M)}$ — $M$ samples drawn from the flow each epoch (M = N is fine)
- $\hat p_N = \frac{1}{N}\sum \delta_{x_i}$ — empirical previous posterior
- $\mathcal{S}_\varepsilon$ — the **regularised Sinkhorn distance** from `smc2bj/transport/sinkhorn.py`, with the same low-rank/Sinkhorn iterations the OT-rescue code uses (rank ≈ 5, n_iter ≈ 2, ε ≈ 0.5, max_weight ≈ 0.01)
- $\lambda$ — regularisation strength, default **0.1** (start), grid 0.01 / 0.1 / 1.0

### Training

- Optimiser: Adam(lr=1e-3, weight_decay=1e-5)
- Epochs: 200–500 with early stop on plateau (patience 20)
- Each epoch: full batch (N=256) for both NLL and OT term
- Warm-start: weights from the previous window's flow (one less window = ~50 fewer epochs to converge)
- Time budget per bridge: ~30s flow training (small enough not to dominate the per-window 2.5min SMC cost)

### Initial particle sampling

Sample N_SMC = 256 fresh draws from $q_\phi$:

```python
z ~ N(0, I_d)                          # base noise
u_init = flow.inverse(z)                # → unconstrained-space samples
```

This is automatically diverse if the flow has been OT-regularised.

### `logprior_fn`

```python
@jax.jit
def logprior_fn(u):
    return flow.log_prob(u)             # change-of-variables built in
```

Differentiable in `u` by construction (Real-NVP gives smooth inverses), so
HMC works without modification.

---

## Why this should work where N=512 and MoG K=2 didn't

| Approach | Diversity preservation | Density expressiveness | Compute |
|----------|------------------------|------------------------|---------|
| Single Gaussian + LW | weak (rank-1 fit) | weak (unimodal Gaussian only) | cheap |
| MoG K=2 | medium (clusters) | medium (2 Gaussians) | cheap |
| Vanilla NF (MLE only) | **fails** (collapses on points) | strong | medium |
| **OT-regularised NF** | **strong** (Sinkhorn enforces spread) | strong | medium |
| Larger N alone | better empirical fit only | unchanged | 2× per particle count |

The two existing experiments (N=512 and MoG K=2) addressed sample-size and
shape-flexibility limitations of the **fit**. Neither addresses the
fundamental issue that **a tight fit to a sparse particle cloud is the
wrong target** — what we need is a smooth approximation of the *measure*
the cloud is sampled from. OT regularisation is the natural way to
formalise "smooth approximation of a measure" in 29 dimensions.

---

## Implementation phases

### Phase A — flow primitive (~3-4h)

1. New `smc2bj/bridges/__init__.py`
2. `smc2bj/bridges/realnvp.py`:
   - `class RealNVP`: stateless `init(rng, d, n_layers, hidden) → params`,
     pure functions `forward(params, z) → (u, log_det)`,
     `inverse(params, u) → (z, log_det)`, `log_prob(params, u) → ll`.
   - Hand-rolled JAX (avoid Distrax dependency for now; ~200 lines).
3. `smc2bj/bridges/training.py`:
   - `fit_flow_with_ot(prev_particles, n_layers, hidden, lambda_ot,
     n_epochs, sinkhorn_kwargs, key) → flow_params`
   - Inner Adam loop using `optax`. Hyperparams as defaults from this plan.
   - Logs final NLL, final OT distance, n_epochs to convergence.

### Phase B — wire into bridge (~1-2h)

1. Extend `SMCConfig`:
   ```python
   bridge_type: str = 'gaussian'      # also: 'mog', 'ot_flow'
   bridge_flow_n_layers: int = 4
   bridge_flow_hidden: int = 64
   bridge_flow_n_epochs: int = 300
   bridge_flow_lambda_ot: float = 0.1
   bridge_flow_warm_start: bool = True   # reuse params from previous window
   ```
2. `run_smc_window_bridge`:
   - New branch `cfg.bridge_type == 'ot_flow'`:
     - Call `fit_flow_with_ot(prev_particles, ...)` → `flow_params`
     - Define `logprior_fn(u) = realnvp.log_prob(flow_params, u)`
     - Sample initial particles via `realnvp.inverse(flow_params, z)`
   - Persist `flow_params` to `prev_particles_state` (dict) so next bridge
     can warm-start. Augment the rolling pipeline accordingly.
3. Driver: `--bridge ot_flow` CLI flag, `--bridge-flow-lambda` for tuning.

### Phase C — smoke test (~30 min)

1. Unit test: fit flow to 256 samples from a known 2-cluster mixture,
   verify the trained flow's `log_prob` is sensible at known cluster
   centres and that fresh samples cover both clusters with comparable
   weights.
2. Integration test: 2-window high-res run with `--bridge ot_flow`. Verify
   flow training completes in reasonable time and bridge tempering
   doesn't error out.

### Phase D — full 27-window run + comparisons (~1.5-2h GPU)

Run the proof-of-principle 14-day rollout at seed=42 with:
- `--bridge ot_flow --bridge-flow-lambda 0.1`
- (and ideally an ablation `--bridge ot_flow --bridge-flow-lambda 0.0`
  to confirm that the OT regulariser is what's helping, vs. just the
  flow-vs-Gaussian capacity)

Compare against the existing 4 runs (baseline, N=512, MoG K=2, C-fix).

### Phase E — document + commit (~30 min)

`outputs/fsa_high_res_rolling/ot_flow_bridge_report.md` with the
4-or-5-way comparison and recommendations.

**Tag** `v0.3.0-ot-flow-bridge` if it meaningfully beats the C-fix
result (mean coverage ≥ 65% data-informed, late-window survival ≥ 50%).

---

## Hyperparameter sensitivity

| Knob | Default | Plausible range | Effect of going up |
|------|---------|-----------------|---------------------|
| `lambda_ot` | 0.1 | 0.01–1.0 | More entropy / less density-fit |
| `n_layers` | 4 | 2–8 | More expressive, slower train |
| `hidden` | 64 | 32–128 | Same |
| `n_epochs` | 300 | 100–1000 | Better convergence; warm-start cuts this |
| Sinkhorn ε | 0.5 (match OT-rescue) | 0.1–2.0 | More smoothing |

The sensitivity scan would be one full extra run per `lambda_ot` value
(only if the default fails). Preferable: trust the default 0.1 and only
sweep if Phase D shows the result is on a knife-edge.

---

## Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| Flow training diverges (Adam → NaN) | Soft-clamp on log_scale (`tanh`); gradient clipping at 1.0; small initial weights |
| Flow's `log_prob` slow / non-differentiable enough for HMC | Real-NVP's analytic Jacobian is closed-form; stress-test on toy data first |
| OT term dominates, NLL term ignored | Anneal λ_ot from 1.0 → 0.1 over training |
| Per-bridge training time blows up | Time-box at 30s; if not converged, fall back to MoG bridge for that window only |
| Warm-start helps less than expected because the parameter scale changes | Re-init layer biases each window, keep weights |
| Flow gets stuck at the previous posterior with no real improvement | Worst case is "no better than vanilla flow"; we still have the OT regulariser as a safety net |

---

## Definition of done

- One full 27-window high-res run with `--bridge ot_flow`
- Mean data-informed coverage **≥ 60%** (vs. ~33% baseline, ~40% MoG/N=512)
- Late-window (W25-W27) coverage **≥ 50%** (vs. 3-38% in earlier runs)
- Wall-clock **≤ 3h** total
- Per-bridge flow training **≤ 60s** stably across all 26 bridges

If the OT regulariser does what the literature suggests (e.g. Onken et al.
"OT-Flow: Fast and Accurate Continuous Normalizing Flows via Optimal
Transport" 2021; or the more recent Sinkhorn-regularised flow papers), we
should clear all four bars on the first attempt with default hyperparams.

---

## Sequencing relative to other plans

This plan **supersedes** PLAN_principled_bridge_fixes.md Track 1 (the
vanilla-flow bridge), which would always need OT regularisation anyway
when applied to small particle counts.

**Sequencing recommendation:**

1. **Phase 0** (already in flight): wait for the C-phase fix run to
   complete. If it hits ≥ 70% mean coverage, **skip flow work entirely** —
   the bridge cascade was actually the C-bug, not the bridge per se.
2. **Phase 1**: if C-fix improves coverage but doesn't hit the target
   (say, 50–65% mean), implement this OT-regularised flow plan. Expected
   to clear 70%.
3. **Phase 2** (optional): combine OT-flow bridge with longer stride
   (Track 2 from the previous plan). Should compound.

---

## What we still won't have done after this

- A non-Gaussian inner PF (still GK-DPF v3-lite with guided Kalman). For
  models with truly non-Gaussian observations (counts, mixed-data) the
  inner PF would also need replacing — separate plan.
- A real-data ingestion path. All experiments still on synthetic data.
- The SWAT port. Once the high-res FSA pipeline closes its identifiability
  loop, SWAT becomes the next port.
