"""Pytest setup: inject the public dev repo's version_1/ into sys.path.

The fsa_high_res model lives canonically in
Python-Model-Development-Simulation. With both repos' ``models/``
directories as PEP-420 namespace packages (no ``__init__.py``), Python
merges their ``__path__`` and ``models.fsa_high_res`` resolves to the
public dev copy while ``models.fsa_real_obs`` (with SMC²-specific
edits) resolves locally — provided the SMC² root is searched first.
"""

from __future__ import annotations

import os
import sys


_PUBLIC_DEV_V1 = os.path.expanduser(
    "~/Repos/Python-Model-Development-Simulation/version_1"
)


def _inject_public_dev_path():
    if not os.path.isdir(_PUBLIC_DEV_V1):
        # Tests touching fsa_high_res will fail with a helpful import
        # error; non-fsa_high_res tests still run.
        return
    if _PUBLIC_DEV_V1 not in sys.path:
        sys.path.append(_PUBLIC_DEV_V1)


_inject_public_dev_path()
