"""Schrödinger–Föllmer bridge between Gaussian endpoints.

Closed-form Path A: the Schrödinger bridge between Gaussian source
:math:`\\mu_0 = \\mathcal{N}(m_0, S_0)` and Gaussian target
:math:`\\mu_1 = \\mathcal{N}(m_1, S_1)` is the **Bures–Wasserstein
geodesic** (when entropic regularisation :math:`\\varepsilon \\to 0`):

.. math::

    m_t &= (1-t) m_0 + t m_1, \\\\
    S_t &= ((1-t) I + t T) S_0 ((1-t) I + t T)

where :math:`T = S_0^{-1/2} (S_0^{1/2} S_1 S_0^{1/2})^{1/2} S_0^{-1/2}`
is the Bures-Wasserstein optimal transport map from
:math:`\\mu_0` to :math:`\\mu_1`. With finite entropic regularisation,
the covariance gains an additional :math:`\\varepsilon t (1-t) I` term
(Schrödinger bridge between Gaussians; see e.g. Janati, Muzellec,
Peyré, Cuturi 2020).

This module provides:

  - ``bw_geodesic(m0, S0, m1, S1, t, entropy_reg)``: closed-form
    interpolation. JAX-jittable.
  - ``estimate_target_gaussian(prev_particles, log_w_to_target)``:
    importance-weighted moment-match of the new posterior, given prev
    particles weighted by ``new_ld(u) - log q_0(u)`` (one new_ld eval
    per prev particle — same cost as the existing bridge).
  - ``fit_sf_base(prev_particles, new_ld_fn, blend, entropy_reg, ...)``:
    full pipeline returning the SF base measure parameters at
    ``t = blend`` along the BW geodesic from prev-posterior Gaussian
    to importance-matched new-posterior Gaussian.

The SMC² rolling-window bridge dispatches on
``cfg.bridge_type == 'schrodinger_follmer'`` and uses ``fit_sf_base``
to produce the base measure for tempered SMC. ``blend=0`` recovers
the existing Gaussian bridge; ``blend=1`` puts the base at the
moment-matched new-posterior estimate; ``blend=0.5`` (default for
SF) sits at the BW midpoint — closer to π_new than the prev
posterior, so the tempering bridge has less distance to cover and
less prev-posterior bias to compound across windows.

Why this should help SWAT-class problems:
  The Gaussian bridge concentrates particles tightly (per-dim SD
  ~1e-2 for SWAT's 35-dim posterior) — it can't shed accumulated
  bias from previous windows. The SF base at the BW midpoint has
  higher variance than either endpoint when the endpoints differ,
  so the tempering bridge starts wider and is less biased.
"""

from __future__ import annotations

from typing import Callable, Tuple

import jax
import jax.numpy as jnp


# ─────────────────────────────────────────────────────────────────────
# Bures-Wasserstein geodesic between Gaussians
# ─────────────────────────────────────────────────────────────────────

def _matrix_sqrt_psd(A: jnp.ndarray) -> jnp.ndarray:
    """Symmetric matrix square root of a PSD matrix via eigendecomposition.

    A is assumed symmetric PSD up to numerical noise; eigenvalues are
    clamped to ``>= 0`` before sqrt.
    """
    A_sym = 0.5 * (A + A.T)
    eigvals, eigvecs = jnp.linalg.eigh(A_sym)
    eigvals_clipped = jnp.maximum(eigvals, 0.0)
    sqrt_eigvals = jnp.sqrt(eigvals_clipped)
    return (eigvecs * sqrt_eigvals[None, :]) @ eigvecs.T


def _matrix_invsqrt_psd(A: jnp.ndarray, eps: float = 1e-10) -> jnp.ndarray:
    """Symmetric inverse square root of a PSD matrix via eigendecomposition.

    Eigenvalues clamped to ``>= eps`` for numerical stability.
    """
    A_sym = 0.5 * (A + A.T)
    eigvals, eigvecs = jnp.linalg.eigh(A_sym)
    eigvals_clipped = jnp.maximum(eigvals, eps)
    invsqrt = 1.0 / jnp.sqrt(eigvals_clipped)
    return (eigvecs * invsqrt[None, :]) @ eigvecs.T


def bures_wasserstein_map(S0: jnp.ndarray, S1: jnp.ndarray) -> jnp.ndarray:
    """Bures-Wasserstein optimal transport map ``T`` from N(0, S0) to N(0, S1).

    Defined via :math:`T = S_0^{-1/2} (S_0^{1/2} S_1 S_0^{1/2})^{1/2} S_0^{-1/2}`.
    The push-forward of ``N(0, S0)`` by ``y → T y`` is ``N(0, S1)``.

    Pure JAX, jittable.
    """
    S0_half = _matrix_sqrt_psd(S0)
    inner = S0_half @ S1 @ S0_half
    inner_half = _matrix_sqrt_psd(inner)
    S0_invhalf = _matrix_invsqrt_psd(S0)
    return S0_invhalf @ inner_half @ S0_invhalf


def bw_geodesic(
    m0: jnp.ndarray, S0: jnp.ndarray,
    m1: jnp.ndarray, S1: jnp.ndarray,
    t: float,
    entropy_reg: float = 0.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Bures-Wasserstein geodesic between Gaussians at time ``t``.

    ::

        m_t = (1-t) m0 + t m1
        S_t = ((1-t) I + t T) S0 ((1-t) I + t T) + entropy_reg * t (1-t) I

    where ``T`` is the BW transport map from ``S0`` to ``S1``. The
    ``entropy_reg`` term is the Schrödinger-bridge entropic
    regularisation (Janati et al. 2020); ``0`` recovers exact OT.

    At ``t=0`` returns (m0, S0); at ``t=1`` returns (m1, S1).
    """
    d = m0.shape[0]
    m_t = (1.0 - t) * m0 + t * m1
    T_map = bures_wasserstein_map(S0, S1)
    M_t = (1.0 - t) * jnp.eye(d) + t * T_map
    S_t = M_t @ S0 @ M_t.T
    if entropy_reg > 0.0:
        S_t = S_t + entropy_reg * t * (1.0 - t) * jnp.eye(d)
    # Symmetrise to dampen numerical asymmetry.
    return m_t, 0.5 * (S_t + S_t.T)


# ─────────────────────────────────────────────────────────────────────
# Importance-weighted target moment-match
# ─────────────────────────────────────────────────────────────────────

def estimate_target_gaussian(
    prev_particles: jnp.ndarray,
    log_w_unnorm: jnp.ndarray,
    *,
    floor_eff_n: int = 5,
) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    """Importance-weighted moment-match: estimate ``N(m1, S1)`` of the
    new posterior using prev particles weighted by
    ``log_w_unnorm = new_ld(u) - log q_0(u)``.

    The weights are normalised via softmax (numerically stable). When
    the effective sample size is degenerate (``< floor_eff_n``), falls
    back to uniform weighting so the moment-match doesn't collapse to
    a few particles.

    Returns:
        m1   : (d,) weighted mean
        S1   : (d, d) weighted covariance + small reg
        n_eff: effective sample size (float)

    The raw covariance gets a small ``1e-4 I`` regulariser to ensure
    positive-definiteness for the BW-square-root step downstream.
    """
    log_w_norm = log_w_unnorm - jax.scipy.special.logsumexp(log_w_unnorm)
    w = jnp.exp(log_w_norm)                 # (N,)
    n_eff = float(1.0 / jnp.sum(w ** 2))

    # ESS floor: revert to uniform weighting if too degenerate.
    use_uniform = n_eff < floor_eff_n
    w_safe = jnp.where(use_uniform,
                        jnp.ones_like(w) / w.shape[0],
                        w)

    m1 = jnp.sum(w_safe[:, None] * prev_particles, axis=0)
    diffs = prev_particles - m1[None, :]
    S1_raw = (w_safe[:, None] * diffs).T @ diffs
    d = m1.shape[0]
    S1 = S1_raw + 1e-4 * jnp.eye(d)
    return m1, S1, n_eff


# ─────────────────────────────────────────────────────────────────────
# SF base measure for the rolling-window bridge
# ─────────────────────────────────────────────────────────────────────

def fit_sf_base(
    prev_particles: jnp.ndarray,
    new_ld_fn: Callable,
    *,
    blend: float = 0.5,
    entropy_reg: float = 0.0,
    lw_shrinkage: float = 1e-2,
):
    """Compute the SF bridge base measure for tempered SMC.

    Pipeline:

      1. Gaussian fit ``q_0 = N(m_0, S_0)`` of prev posterior with
         Ledoit-Wolf-style shrinkage at level ``lw_shrinkage``.
      2. Importance-weighted Gaussian estimate
         ``q_1 = N(m_1, S_1)`` of the new posterior, using prev
         particles weighted by ``new_ld(u) - log q_0(u)``.
      3. Bures-Wasserstein geodesic between ``q_0`` and ``q_1`` at
         time ``t = blend``, with optional entropic regularisation.
      4. Return ``(m_blend, S_blend)`` plus diagnostic info.

    ``blend = 0`` recovers the prev-posterior Gaussian (current
    bridge_type='gaussian'). ``blend = 1`` jumps directly to the
    importance-matched estimate of the new posterior. ``blend = 0.5``
    (default) sits at the BW midpoint — closer to π_new than the
    Gaussian bridge but still anchored on the prev-posterior shape.

    Args:
        prev_particles: (N, d) prev-window posterior particles
                        (in unconstrained space).
        new_ld_fn: callable u -> log π_new(u). Vectorised internally.
        blend: t parameter for BW geodesic, in [0, 1].
        entropy_reg: Schrödinger entropic regularisation. 0 = exact OT.
        lw_shrinkage: ridge factor for Gaussian fits (additive
                      ``lw_shrinkage * tr(S)/d * I``).

    Returns:
        dict with keys:
          'm', 'S':         the SF-base mean and covariance
          'L_chol':         Cholesky of S
          'L_inv':          inverse of L_chol (for log-density evals)
          'log_det':        log|S|
          'q0_mean', 'q0_cov': Gaussian fit of prev posterior
          'q1_mean', 'q1_cov': moment-matched new-posterior estimate
          'n_eff':          effective sample size of the IS weights
          'blend':          t value used
          'entropy_reg':    entropy_reg used
    """
    prev = jnp.asarray(prev_particles, dtype=jnp.float64)
    N, d = prev.shape

    # ── q_0: prev-posterior Gaussian fit (with LW-style shrinkage) ───
    m0 = jnp.mean(prev, axis=0)
    diffs0 = prev - m0[None, :]
    S0_raw = diffs0.T @ diffs0 / N
    tr_S0 = jnp.trace(S0_raw)
    S0 = (1.0 - lw_shrinkage) * S0_raw + lw_shrinkage * (tr_S0 / d) * jnp.eye(d)
    S0 = S0 + 1e-4 * jnp.eye(d)

    # log q_0 evaluated at each prev particle (for IS weights)
    L0 = jnp.linalg.cholesky(S0)
    L0_inv = jax.scipy.linalg.solve_triangular(L0, jnp.eye(d), lower=True)
    log_det_S0 = 2.0 * jnp.sum(jnp.log(jnp.diag(L0)))
    const0 = -0.5 * (d * jnp.log(2.0 * jnp.pi) + log_det_S0)

    def log_q0(u):
        diff = u - m0
        return const0 - 0.5 * jnp.sum((L0_inv @ diff) ** 2)

    # ── q_1: importance-weighted moment-match of π_new ───────────────
    log_q0_vec = jax.vmap(log_q0)(prev)
    log_p_vec = jax.vmap(new_ld_fn)(prev)
    log_w_unnorm = log_p_vec - log_q0_vec
    m1, S1, n_eff = estimate_target_gaussian(prev, log_w_unnorm)

    # ── BW geodesic at t = blend ──────────────────────────────────────
    m_blend, S_blend = bw_geodesic(
        m0, S0, m1, S1, t=float(blend),
        entropy_reg=float(entropy_reg),
    )
    S_blend = S_blend + 1e-4 * jnp.eye(d)
    L_blend = jnp.linalg.cholesky(S_blend)
    L_inv_blend = jax.scipy.linalg.solve_triangular(
        L_blend, jnp.eye(d), lower=True)
    log_det_blend = 2.0 * jnp.sum(jnp.log(jnp.diag(L_blend)))

    return {
        'm': m_blend,
        'S': S_blend,
        'L_chol': L_blend,
        'L_inv': L_inv_blend,
        'log_det': log_det_blend,
        'q0_mean': m0,
        'q0_cov': S0,
        'q1_mean': m1,
        'q1_cov': S1,
        'n_eff': n_eff,
        'blend': float(blend),
        'entropy_reg': float(entropy_reg),
    }


__all__ = [
    "bw_geodesic",
    "bures_wasserstein_map",
    "estimate_target_gaussian",
    "fit_sf_base",
]
