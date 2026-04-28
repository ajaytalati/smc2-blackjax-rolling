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
    estimate_target_gaussian_annealed,
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


# ─────────────────────────────────────────────────────────────────────
# Path B (annealed q1) tests
# ─────────────────────────────────────────────────────────────────────

def test_annealed_q1_recovers_target_mean_in_low_d():
    """In 4-D with a moderately-shifted Gaussian target, the annealed
    K-stage estimator should recover the target mean within ~1 prior SD.
    Pure-IS in the same setting collapses; this is the regime Path B
    is designed for."""
    d = 4
    rng = np.random.default_rng(123)
    prev = jnp.asarray(rng.standard_normal((300, d)))   # q0 ~ N(0, I)
    target_mean = jnp.array([2.5, -1.5, 0.5, 1.0])

    def gaussian_ld(u):
        return -0.5 * jnp.sum((u - target_mean) ** 2)

    rng_key = jnp.array([0, 42], dtype=jnp.uint32)
    m1, S1, n_eff_min, acc = estimate_target_gaussian_annealed(
        prev, gaussian_ld,
        n_stages=4, n_mh_steps=3, proposal_scale=0.6,
        rng_key=rng_key,
    )
    # Annealed should move meaningfully toward target — at least 60% of
    # the way from q0 mean (origin) to target. Full convergence isn't
    # the goal here; the outer tempering takes the bridge the rest of
    # the way.
    q0_mean = jnp.zeros_like(target_mean)
    init_dist = float(jnp.linalg.norm(target_mean - q0_mean))
    final_dist = float(jnp.linalg.norm(m1 - target_mean))
    assert final_dist < 0.4 * init_dist, \
        f"Annealed insufficient: started {init_dist:.2f} away, ended {final_dist:.2f} away"
    # Covariance shouldn't blow up
    eigvals = np.linalg.eigvalsh(np.asarray(S1))
    assert eigvals.max() < 5.0
    # Acceptance rate should be in a reasonable RW-MH range
    assert 0.05 < acc < 0.95


def test_annealed_q1_outperforms_is_when_likelihood_is_sharp():
    """When the new posterior is far from q0 *and* sharp, single-step
    IS produces n_eff ≈ 1 and m1 ≈ q0 mean. Annealed should produce
    m1 meaningfully closer to target."""
    d = 6
    rng = np.random.default_rng(7)
    prev = jnp.asarray(rng.standard_normal((400, d)))
    target_mean = jnp.array([3.0, 3.0, 0.0, 0.0, -2.0, 1.0])
    sharp_var = 0.05   # sharp likelihood

    def sharp_ld(u):
        return -0.5 * jnp.sum((u - target_mean) ** 2) / sharp_var

    # Path A: single-step IS (using fit_sf_base with q1_mode='is')
    sf_is = fit_sf_base(prev, sharp_ld, blend=1.0, q1_mode='is')
    # IS should degenerate
    assert sf_is['n_eff'] < 5.0
    # m1 from IS effectively returns q0 (uniform fallback)
    is_dist_to_target = float(jnp.linalg.norm(sf_is['q1_mean'] - target_mean))

    # Path B: annealed
    sf_ann = fit_sf_base(prev, sharp_ld, blend=1.0, q1_mode='annealed',
                          annealed_n_stages=4, annealed_n_mh_steps=3,
                          annealed_proposal_scale=0.5,
                          rng_key=jnp.array([0, 99], dtype=jnp.uint32))
    ann_dist_to_target = float(jnp.linalg.norm(sf_ann['q1_mean'] - target_mean))

    # Annealed should be substantially closer to target than IS
    assert ann_dist_to_target < 0.6 * is_dist_to_target, \
        f"Annealed dist {ann_dist_to_target:.3f} vs IS dist {is_dist_to_target:.3f}"


def test_fit_sf_base_q1_mode_annealed_serialises():
    """Returns dict has q1_mode='annealed' and accept_mean field."""
    d = 3
    rng = np.random.default_rng(4)
    prev = jnp.asarray(rng.standard_normal((100, d)))

    def silly_ld(u):
        return -0.5 * jnp.sum((u - 1.0) ** 2)

    sf = fit_sf_base(prev, silly_ld, blend=0.5, q1_mode='annealed',
                      annealed_n_stages=2, annealed_n_mh_steps=1,
                      rng_key=jnp.array([0, 1], dtype=jnp.uint32))
    assert sf['q1_mode'] == 'annealed'
    assert sf['accept_mean'] == sf['accept_mean']   # not NaN
    assert 'm' in sf and 'S' in sf and 'L_chol' in sf


def test_fit_sf_base_q1_mode_is_keeps_accept_nan():
    """q1_mode='is' (Path A) returns accept_mean=NaN (not applicable)."""
    d = 3
    rng = np.random.default_rng(5)
    prev = jnp.asarray(rng.standard_normal((50, d)))

    def silly_ld(u):
        return -0.5 * jnp.sum(u ** 2)

    sf = fit_sf_base(prev, silly_ld, blend=0.0, q1_mode='is')
    assert sf['q1_mode'] == 'is'
    import math
    assert math.isnan(sf['accept_mean'])


def test_fit_sf_base_decoupled_uses_q0_cov():
    """use_q0_cov=True: bridge cov is q0's cov (not BW-interpolated).
    Mean is still linearly interpolated."""
    d = 4
    rng = np.random.default_rng(11)
    prev = jnp.asarray(rng.standard_normal((150, d)))   # q0 around origin

    target_mean = jnp.array([5.0, 5.0, 0.0, 0.0])
    def gaussian_ld(u):
        return -0.5 * jnp.sum((u - target_mean) ** 2)

    sf_dec = fit_sf_base(prev, gaussian_ld, blend=1.0, q1_mode='annealed',
                          annealed_n_stages=2, annealed_n_mh_steps=2,
                          use_q0_cov=True,
                          rng_key=jnp.array([0, 7], dtype=jnp.uint32))

    # Bridge cov == q0_cov (up to the +1e-4 reg added in both fits).
    np.testing.assert_allclose(np.asarray(sf_dec['S']),
                                np.asarray(sf_dec['q0_cov']) + 1e-4 * np.eye(d),
                                atol=1e-6)
    # Bridge mean shifted toward q1 (blend=1 → m = m1)
    assert sf_dec['use_q0_cov'] is True
    np.testing.assert_allclose(np.asarray(sf_dec['m']),
                                np.asarray(sf_dec['q1_mean']), atol=1e-8)


def test_fit_sf_base_decoupled_log_det_pinned_to_q0():
    """Decoupled mode pins bridge log_det to q0's, by construction —
    the safety property that motivates the mode.

    BW mode in low-D / well-mixed cases may give log_det smaller OR
    larger than q0 depending on q1's spread. Decoupled mode removes
    that variability and guarantees the bridge cov equals q0's cov."""
    d = 6
    rng = np.random.default_rng(13)
    prev = jnp.asarray(rng.standard_normal((200, d)))

    target_mean = jnp.array([4.0, 4.0, 4.0, 0.0, 0.0, 0.0])
    def sharp_ld(u):
        return -0.5 * jnp.sum((u - target_mean) ** 2) / 0.05

    sf_dec = fit_sf_base(prev, sharp_ld, blend=0.5, q1_mode='annealed',
                         annealed_n_stages=3, annealed_n_mh_steps=2,
                         use_q0_cov=True,
                         rng_key=jnp.array([0, 17], dtype=jnp.uint32))
    # Decoupled bridge cov == q0_cov exactly (modulo +1e-4 reg).
    log_det_q0_only = float(2.0 * jnp.sum(jnp.log(jnp.diag(
        jnp.linalg.cholesky(sf_dec['q0_cov'] + 1e-4 * jnp.eye(d))
    ))))
    assert abs(float(sf_dec['log_det']) - log_det_q0_only) < 1e-6


def test_fit_sf_base_q1_mode_invalid_raises():
    d = 2
    prev = jnp.zeros((10, d))
    def ld(u): return 0.0
    with pytest.raises(ValueError, match="q1_mode"):
        fit_sf_base(prev, ld, q1_mode='nonsense')


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
