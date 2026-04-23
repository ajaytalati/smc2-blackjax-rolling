"""3-State FSA Model with Real Observation Channels.

Same latent SDE dynamics as fitness_strain_amplitude (B, F, A), but
replaces the direct-state observations (obs_B, obs_F, obs_A) with
6 physiologically-motivated observation channels:

    Ch1: RHR          = R_base - kappa_vagal*B + kappa_chronic*F + noise
    Ch2: I_norm       = I_base + c_B*B - c_F*F + noise
    Ch3: D_norm       = D_base + d_B*B - d_F*F + noise
    Ch4: Stress       = S_base - s_A*A + s_F*F + noise
    Ch5: Sleep_norm   = Sleep_base + sl_A*A + sl_B*B - sl_F*F + noise
    Ch6: Time_logit   = Time_base + t_A*A - t_F*F + noise

Each channel has independent Gaussian noise with its own sigma.
Total estimated parameters: 38 (10 dynamical + 25 obs + 3 init states).

Date:    19 April 2026
Version: 1.0
"""

from models.fsa_real_obs.estimation import FSA_REAL_OBS_ESTIMATION
from models.fsa_real_obs.simulation import FSA_REAL_OBS_MODEL
