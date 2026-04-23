"""Generic prior mean computation for any EstimationModel.

Date:    15 April 2026
Version: 5.0 (model-agnostic)
"""

import math
from typing import Dict, Tuple

from smc2bj.estimation_model import EstimationModel


def prior_mean(prior_type: str, prior_args: Tuple[float, ...]) -> float:
    """Compute the mean of a prior distribution.

    Args:
        prior_type: One of 'lognormal', 'normal', 'vonmises', 'beta'.
        prior_args: Distribution parameters.

    Returns:
        The expected value of the distribution.
    """
    if prior_type == 'lognormal':
        mu, sigma = prior_args
        return math.exp(mu + sigma ** 2 / 2.0)
    elif prior_type == 'normal':
        return prior_args[0]
    elif prior_type == 'vonmises':
        return prior_args[0]
    elif prior_type == 'beta':
        a, b = prior_args
        return a / (a + b)
    return 0.0


def get_prior_means_dict(model: EstimationModel) -> Dict[str, float]:
    """Compute prior means for all parameters and init states.

    Args:
        model: EstimationModel.

    Returns:
        Dict mapping name -> prior mean value.
    """
    d: Dict[str, float] = {}
    for name, (ptype, pargs) in model.param_prior_config.items():
        d[name] = prior_mean(ptype, pargs)
    for name, (ptype, pargs) in model.init_state_prior_config.items():
        d[name] = prior_mean(ptype, pargs)
    return d
