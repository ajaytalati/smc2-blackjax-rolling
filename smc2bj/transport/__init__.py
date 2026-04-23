"""transport — Low-rank optimal transport resampling for particle filters.

This package provides differentiable OT resampling using Nyström-
approximated Sinkhorn, designed for use inside jax.lax.scan.

Public API:
    ot_resample_lr(particles, log_weights, rng_key, ...) -> new_particles

Submodules:
    kernel    — Nyström kernel factorisation
    sinkhorn  — Low-rank Sinkhorn iterations (scaling form)
    project   — Barycentric projection (convex combination)
    resample  — Main API combining kernel + sinkhorn + project
"""

from smc2bj.transport.resample import ot_resample_lr

__all__ = ["ot_resample_lr"]
