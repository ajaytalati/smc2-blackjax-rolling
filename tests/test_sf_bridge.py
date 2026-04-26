"""Unit tests for the Schrödinger-Föllmer bridge module.

Tests three properties:
  1. BW geodesic at t=0 returns (m0, S0); at t=1 returns (m1, S1).
  2. BW transport map T satisfies T S0 T = S1 (defining property).
  3. fit_sf_base reduces to the prev-posterior Gaussian when blend=0
     (so it's a continuous extension of the existing 'gaussian' bridge).
"""

from __future__ import annotations

import os
os.environ['JAX_ENABLE_X64'] = 'True'

import jax.numpy as jnp
import numpy as np
import pytest

from smc2bj.estimation.sf_bridge import (
    bw_geodesic,
    bures_wasserstein_map,
    estimate_target_gaussian,
    fit_sf_base,
)


# ─────────────────────────────────────────────────────────────────────
# BW geodesic correctness
# ─────────────────────────────────────────────────────────────────────

def test_bw_geodesic_endpoints():
    """At t=0 returns (m0, S0); at t=1 returns (m1, S1)."""
    d = 4
    rng = np.random.default_rng(42)
    m0 = jnp.asarray(rng.standard_normal(d))
    m1 = jnp.asarray(rng.standard_normal(d) + 2.0)
    A = rng.standard_normal((d, d))
    B = rng.standard_normal((d, d))
    S0 = jnp.asarray(A @ A.T + 0.5 * np.eye(d))
    S1 = jnp.asarray(B @ B.T + 1.0 * np.eye(d))

    m_t0, S_t0 = bw_geodesic(m0, S0, m1, S1, t=0.0)
    np.testing.assert_allclose(np.asarray(m_t0), np.asarray(m0), atol=1e-10)
    np.testing.assert_allclose(np.asarray(S_t0), np.asarray(S0), atol=1e-8)

    m_t1, S_t1 = bw_geodesic(m0, S0, m1, S1, t=1.0)
    np.testing.assert_allclose(np.asarray(m_t1), np.asarray(m1), atol=1e-10)
    np.testing.assert_allclose(np.asarray(S_t1), np.asarray(S1), atol=1e-6)


def test_bw_transport_map_pushes_S0_to_S1():
    """The BW map T satisfies T S0 T^T = S1 (defining property)."""
    d = 4
    rng = np.random.default_rng(7)
    A = rng.standard_normal((d, d))
    B = rng.standard_normal((d, d))
    S0 = jnp.asarray(A @ A.T + 1.0 * np.eye(d))
    S1 = jnp.asarray(B @ B.T + 0.7 * np.eye(d))

    T = bures_wasserstein_map(S0, S1)
    pushed = T @ S0 @ T.T
    np.testing.assert_allclose(np.asarray(pushed), np.asarray(S1), atol=1e-6)


def test_bw_geodesic_intermediate_psd():
    """Intermediate covariances stay PSD across t in [0, 1]."""
    d = 3
    rng = np.random.default_rng(11)
    m0 = jnp.zeros(d)
    m1 = jnp.zeros(d)
    A = rng.standard_normal((d, d))
    B = rng.standard_normal((d, d))
    S0 = jnp.asarray(A @ A.T + 0.5 * np.eye(d))
    S1 = jnp.asarray(B @ B.T + 0.5 * np.eye(d))
    for t in (0.1, 0.25, 0.5, 0.75, 0.9):
        m_t, S_t = bw_geodesic(m0, S0, m1, S1, t=t)
        eigvals = np.linalg.eigvalsh(np.asarray(S_t))
        assert eigvals.min() > -1e-8, f"S_t at t={t} not PSD: min eig {eigvals.min()}"


def test_bw_geodesic_entropy_reg_inflates_covariance():
    """Adding entropic regularisation strictly increases covariance trace
    at intermediate times (when t in (0, 1))."""
    d = 4
    rng = np.random.default_rng(3)
    m0 = jnp.zeros(d); m1 = jnp.zeros(d)
    A = rng.standard_normal((d, d))
    B = rng.standard_normal((d, d))
    S0 = jnp.asarray(A @ A.T + 1.0 * np.eye(d))
    S1 = jnp.asarray(B @ B.T + 0.5 * np.eye(d))

    _, S_no = bw_geodesic(m0, S0, m1, S1, t=0.5, entropy_reg=0.0)
    _, S_with = bw_geodesic(m0, S0, m1, S1, t=0.5, entropy_reg=0.5)
    assert float(jnp.trace(S_with)) > float(jnp.trace(S_no))


# ─────────────────────────────────────────────────────────────────────
# Importance-weighted moment-match
# ─────────────────────────────────────────────────────────────────────

def test_estimate_target_gaussian_uniform_weights_recovers_sample_moments():
    """When all log-weights are equal (uniform IS), the estimator
    recovers the sample mean and covariance."""
    d = 3
    rng = np.random.default_rng(0)
    X = jnp.asarray(rng.standard_normal((100, d)) * 2.0 + np.array([1.0, -1.0, 0.5]))
    log_w = jnp.zeros(100)   # uniform → softmax = 1/N
    m1, S1, n_eff = estimate_target_gaussian(X, log_w)
    # n_eff should be ≈ N for uniform weights
    assert n_eff > 99.0
    np.testing.assert_allclose(np.asarray(m1), np.asarray(jnp.mean(X, axis=0)), atol=1e-6)


def test_estimate_target_gaussian_skewed_weights_shifts_mean():
    """Skewed log-weights shift the estimated mean toward the
    high-weight subset."""
    d = 2
    rng = np.random.default_rng(1)
    X_low = rng.standard_normal((50, d))
    X_high = rng.standard_normal((50, d)) + np.array([10.0, 0.0])
    X = jnp.asarray(np.vstack([X_low, X_high]))
    log_w = jnp.concatenate([jnp.full(50, -10.0), jnp.full(50, 0.0)])
    m1, S1, n_eff = estimate_target_gaussian(X, log_w)
    assert float(m1[0]) > 5.0   # mean pulled toward X_high
    assert n_eff < 60.0          # ESS reduced by skewed weights


# ─────────────────────────────────────────────────────────────────────
# fit_sf_base end-to-end
# ─────────────────────────────────────────────────────────────────────

def test_fit_sf_base_blend0_recovers_q0():
    """blend=0 returns the prev-posterior Gaussian fit (LW-shrunk)."""
    d = 3
    rng = np.random.default_rng(5)
    prev = jnp.asarray(rng.standard_normal((100, d)) + np.array([1.0, 2.0, 3.0]))

    def silly_ld(u):
        return -0.5 * jnp.sum(u ** 2)

    sf = fit_sf_base(prev, silly_ld, blend=0.0)
    np.testing.assert_allclose(np.asarray(sf['m']),
                                np.asarray(sf['q0_mean']), atol=1e-8)
    np.testing.assert_allclose(np.asarray(sf['S']),
                                np.asarray(sf['q0_cov']) + 1e-4 * np.eye(d),
                                atol=1e-6)


def test_fit_sf_base_blend1_matches_q1():
    """blend=1 returns the moment-matched target Gaussian (q1)."""
    d = 3
    rng = np.random.default_rng(6)
    prev = jnp.asarray(rng.standard_normal((100, d)))

    def gaussian_ld(u):
        # Target: standard normal centred at (5, 0, 0)
        return -0.5 * jnp.sum((u - jnp.array([5.0, 0.0, 0.0])) ** 2)

    sf = fit_sf_base(prev, gaussian_ld, blend=1.0)
    np.testing.assert_allclose(np.asarray(sf['m']),
                                np.asarray(sf['q1_mean']), atol=1e-8)


def test_fit_sf_base_blend_half_is_geodesic_midpoint():
    """blend=0.5 lies on the BW geodesic between q0 and q1."""
    d = 2
    rng = np.random.default_rng(8)
    prev = jnp.asarray(rng.standard_normal((100, d)))

    def gaussian_ld(u):
        return -0.5 * jnp.sum((u - jnp.array([3.0, 1.0])) ** 2)

    sf = fit_sf_base(prev, gaussian_ld, blend=0.5)
    m_recomputed, S_recomputed = bw_geodesic(
        sf['q0_mean'], sf['q0_cov'],
        sf['q1_mean'], sf['q1_cov'],
        t=0.5,
    )
    np.testing.assert_allclose(np.asarray(sf['m']),
                                np.asarray(m_recomputed), atol=1e-8)
    # Covariance equality up to the +1e-4 reg added in fit_sf_base
    np.testing.assert_allclose(np.asarray(sf['S']),
                                np.asarray(S_recomputed) + 1e-4 * np.eye(d),
                                atol=1e-6)
