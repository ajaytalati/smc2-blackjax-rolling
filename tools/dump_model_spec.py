#!/usr/bin/env python3
"""Emit a Markdown priors table from a model's PARAM_PRIOR_CONFIG.

Usage:
    python tools/dump_model_spec.py [model_package]

    Default model is models.fsa_real_obs. Other option: models.<name>
    where <name>/estimation.py exports a PARAM_PRIOR_CONFIG.
"""

import importlib
import math
import sys


def render(param_prior_config):
    print("| Parameter | Distribution | Prior location (mean) | Prior scale |")
    print("|-----------|--------------|-----------------------|-------------|")
    for name, (dist, args) in param_prior_config.items():
        if dist == 'lognormal':
            mu_ln, sigma_ln = args
            mean = math.exp(float(mu_ln))
            print(f"| ${name}$ | LogNormal | {mean:.4g} | {sigma_ln:.2f} (log-space) |")
        elif dist == 'normal':
            mu, sigma = args
            print(f"| ${name}$ | Normal | {mu:.4g} | {sigma:.4g} |")
        else:
            print(f"| ${name}$ | {dist} | {args} | |")


def main():
    mod = sys.argv[1] if len(sys.argv) > 1 else 'models.fsa_real_obs.estimation'
    pkg = importlib.import_module(mod)
    if not hasattr(pkg, 'PARAM_PRIOR_CONFIG'):
        print(f"error: {mod} has no PARAM_PRIOR_CONFIG", file=sys.stderr)
        sys.exit(1)
    render(pkg.PARAM_PRIOR_CONFIG)


if __name__ == '__main__':
    main()
