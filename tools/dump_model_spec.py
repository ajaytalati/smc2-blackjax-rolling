#!/usr/bin/env python3
"""Dump a model's PARAM_PRIOR_CONFIG as Markdown.

Two modes:

    python tools/dump_model_spec.py
        → prints the prior table to stdout

    python tools/dump_model_spec.py --update-docs
        → rewrites the auto-generated block inside
          docs/MODEL_SPECIFICATION.md (between the
          <!-- AUTO-GENERATED-PRIORS-START --> and -END markers).

Invoked via CI or by hand whenever a model's PARAM_PRIOR_CONFIG
changes, so the doc table stays in sync with the code.
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC_PATH = os.path.join(REPO_ROOT, 'docs', 'MODEL_SPECIFICATION.md')

MARKER_START = '<!-- AUTO-GENERATED-PRIORS-START -->'
MARKER_END = '<!-- AUTO-GENERATED-PRIORS-END -->'


def render_table(param_prior_config) -> str:
    lines = [
        '| # | Parameter | Distribution | Location (constrained mean) | Scale |',
        '|---|-----------|--------------|------------------------------|-------|',
    ]
    for i, (name, (dist, args)) in enumerate(param_prior_config.items(), 1):
        if dist == 'lognormal':
            mu_ln, sigma_ln = args
            mean = math.exp(float(mu_ln))
            lines.append(
                f'| {i} | `{name}` | LogNormal | {mean:.4g} | '
                f'{sigma_ln:.3g} (log-space) |')
        elif dist == 'normal':
            mu, sigma = args
            lines.append(
                f'| {i} | `{name}` | Normal | {mu:.4g} | {sigma:.4g} |')
        else:
            lines.append(f'| {i} | `{name}` | {dist} | {args} | — |')
    return '\n'.join(lines)


def update_docs(model_module: str, table: str):
    if not os.path.exists(DOC_PATH):
        print(f"error: {DOC_PATH} not found", file=sys.stderr)
        sys.exit(1)
    with open(DOC_PATH) as f:
        content = f.read()
    if MARKER_START not in content or MARKER_END not in content:
        print(f"error: markers not found in {DOC_PATH}. "
              f"Add this block manually between the prior sections:\n"
              f"{MARKER_START}\n{MARKER_END}", file=sys.stderr)
        sys.exit(1)
    new_block = (
        f'{MARKER_START}\n'
        f'<!-- regenerate via: python tools/dump_model_spec.py {model_module} --update-docs -->\n\n'
        f'{table}\n\n'
        f'{MARKER_END}'
    )
    new_content = re.sub(
        rf'{re.escape(MARKER_START)}.*?{re.escape(MARKER_END)}',
        new_block, content, flags=re.DOTALL,
    )
    with open(DOC_PATH, 'w') as f:
        f.write(new_content)
    print(f"updated: {DOC_PATH}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('module', nargs='?',
                   default='models.fsa_real_obs.estimation',
                   help='Python module with PARAM_PRIOR_CONFIG')
    p.add_argument('--update-docs', action='store_true',
                   help='Rewrite the auto-generated block inside '
                        'docs/MODEL_SPECIFICATION.md')
    args = p.parse_args()

    sys.path.insert(0, REPO_ROOT)
    pkg = importlib.import_module(args.module)
    if not hasattr(pkg, 'PARAM_PRIOR_CONFIG'):
        print(f"error: {args.module} has no PARAM_PRIOR_CONFIG", file=sys.stderr)
        sys.exit(1)
    table = render_table(pkg.PARAM_PRIOR_CONFIG)

    if args.update_docs:
        update_docs(args.module, table)
    else:
        print(table)


if __name__ == '__main__':
    main()
