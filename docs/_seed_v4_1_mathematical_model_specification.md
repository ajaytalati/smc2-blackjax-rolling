
# Mathematical Specification: 3-State FSA Model v4.1

**Model Version:** 4.1 (Real-Obs, Rolling SMC²)

**State Dimension:** 3 Latent States ($B, F, A$)

**Observation Dimension:** 6 Physiological Channels

**Parameter Dimension:** 34 Estimated Parameters

This document defines the complete mathematical structure of the Fitness-Strain-Amplitude (FSA) state-space model, incorporating all identifiability fixes from v4.0 and v4.1 (mean-centering RHR, removing initial states, and orthogonal $\kappa$-ratio reparameterization).

## 1. Latent Dynamical System (SDEs)

The core biology is governed by a 3-dimensional system of Itô stochastic differential equations. Time $t$ is measured in **days**.

### 1.1 Exogenous Inputs

- $T_B(t) \in [0, 1]$: Adaptation Target (Training Load)
    
- $\Phi(t) \in \mathbb{R}_{\geq 0}$: Strain Production (Training Volume/Intensity)
    

### 1.2 The State Equations

**1. Fitness (**$B$**) — Jacobi Diffusion:**

  

$$dB_t = \frac{1 + \alpha_A A_t}{\tau_B} \Big(T_B(t) - B_t\Big) dt + \sigma_B \sqrt{B_t(1-B_t)} dW_{B,t}$$

**2. Strain (**$F$**) — CIR Diffusion:**

  

$$dF_t = \left[ \Phi(t) - \frac{1 + \lambda_B B_t + \lambda_A A_t}{\tau_F} F_t \right] dt + \sigma_F \sqrt{F_t} dW_{F,t}$$

**3. Endocrine Amplitude (**$A$**) — Regularized Landau:**

  

$$dA_t = \Big(\mu(B_t, F_t) A_t - \eta A_t^3\Big) dt + \sigma_A \sqrt{A_t + \epsilon_A} dW_{A,t}$$

Where the bifurcation parameter $\mu$ drives the phase transition:

  

$$\mu(B_t, F_t) = \mu_0 + \mu_B B_t - \mu_F F_t - \mu_{FF} F_t^2$$

## 2. Fixed Constants & Initial States (Not Estimated)

To guarantee structural identifiability, the following variables are strictly frozen and removed from the SMC² parameter block.

**Process Noise Scales:**

- $\sigma_B = 0.01$  
    
- $\sigma_F = 0.005$  
    
- $\sigma_A = 0.02$  
    
- $\epsilon_A = 10^{-4}$ (Non-absorbing boundary regularization)
    

**Latent Initial States (**$B_0, F_0, A_0$**):**

- _Window 1 (Cold Start):_ Fixed to $[0.05, 0.10, 0.01]$.
    
- _Windows 2+ (Warm Bridge):_ Hardcoded to the PF-extracted, smoothed posterior state from the previous window at $t = \text{STRIDE\_DAYS}$.
    

## 3. Observation Model (6 Independent Gaussian Channels)

### 3.1 The Orthogonalized RHR Channel (Channel 1)

To break the $(B, F)$ collinearity, RHR is mean-centered (removing $R_{base}$ from estimation) and the sensitivities are reparameterized into a ratio.

**Internal Transformation:**

- $\kappa_{chronic} = \frac{\kappa_{total}}{1 + \kappa_{ratio}}$  
    
- $\kappa_{vagal} = \kappa_{ratio} \cdot \kappa_{chronic}$  
    

**Observation Equation:**

  

$$RHR_{centered}(t) \sim \mathcal{N}\Big( - \kappa_{vagal} B(t) + \kappa_{chronic} F(t), \, \sigma_{obs,R}^2 \Big)$$

_(Note:_ $RHR_{centered}$ _is computed dynamically per-window during preprocessing by subtracting the rolling mean)._

### 3.2 Performance & Behavioral Channels (Channels 2-6)

**Channel 2: Global Performance Intensity**

  

$$I_{norm}(t) \sim \mathcal{N}\Big(I_{base} + c_B B(t) - c_F F(t), \, \sigma_{obs,I}^2\Big)$$

**Channel 3: Global Performance Duration**

  

$$D_{norm}(t) \sim \mathcal{N}\Big(D_{base} + d_B B(t) - d_F F(t), \, \sigma_{obs,D}^2\Big)$$

**Channel 4: Daily Stress**

  

$$S_{obs}(t) \sim \mathcal{N}\Big(S_{base} - s_A A(t) + s_F F(t), \, \sigma_{obs,S}^2\Big)$$

**Channel 5: Sleep Quality**

  

$$Sleep_{norm}(t) \sim \mathcal{N}\Big(Sleep_{base} + sl_B B(t) - sl_F F(t) + sl_A A(t), \, \sigma_{obs,Sleep}^2\Big)$$

**Channel 6: Circadian Exercise Timing**

_(Note: Bounded circadian scores are transformed via logit_ $\ln(\frac{y}{1-y})$ _to map to_ $(-\infty, \infty)$ _support)._

  

$$Time_{logit}(t) \sim \mathcal{N}\Big(Time_{base} + t_A A(t) - t_F F(t), \, \sigma_{obs,Time}^2\Big)$$

## 4. The Parameter Vector & Concrete Priors ($\theta \in \mathbb{R}^{34}$)

The SMC² algorithm tracks exactly 34 parameters. Lognormal distributions are parameterized by the mean and standard deviation of the underlying normal: $\text{Lognormal}(\ln(\mu), \sigma)$.

### 4.1 Dynamical Parameters (10)

|   |   |   |
|---|---|---|
|**Parameter**|**Description**|**Concrete Prior**|
|$\tau_B$|Fitness time constant|$\text{Lognormal}(\ln(14.0), 0.08)$ _(Tightened)_|
|$\alpha_A$|Amplitude enhancement of fitness|$\text{Lognormal}(\ln(1.0), 0.4)$|
|$\tau_F$|Strain time constant|$\text{Lognormal}(\ln(7.0), 0.3)$|
|$\lambda_B$|Fitness enhancement of recovery|$\text{Lognormal}(\ln(3.0), 0.3)$|
|$\lambda_A$|Amplitude enhancement of recovery|$\text{Lognormal}(\ln(1.5), 0.3)$|
|$\mu_0$ (abs)|Absolute baseline bifurcation|$\text{Lognormal}(\ln(0.10), 0.4)$|
|$\mu_B$|Fitness protection of bifurcation|$\text{Lognormal}(\ln(0.30), 0.4)$|
|$\mu_F$|Linear strain penalty|$\text{Lognormal}(\ln(0.10), 0.4)$|
|$\mu_{FF}$|Quadratic strain penalty|$\text{Lognormal}(\ln(0.40), 0.4)$|
|$\eta$|Landau restoring force|$\text{Lognormal}(\ln(0.20), 0.3)$|

### 4.2 Observation Parameters (24)

**RHR (3 Params):**

|   |   |   |
|---|---|---|
|**Parameter**|**Description**|**Concrete Prior**|
|$\kappa_{ratio}$|Ratio of vagal vs. chronic tone|$\text{Lognormal}(\ln(1.2), 0.20)$ _(New)_|
|$\kappa_{total}$|Total RHR sensitivity scale|$\text{Lognormal}(\ln(22.0), 0.30)$ _(New)_|
|$\sigma_{obs,R}$|Measurement noise|$\text{Lognormal}(\ln(1.5), 0.4)$|

**Intensity (4 Params):**

|   |   |   |
|---|---|---|
|**Parameter**|**Description**|**Concrete Prior**|
|$I_{base}$|Baseline capacity|$\text{Normal}(0.5, 0.1)$|
|$c_B$|Fitness sensitivity|$\text{Lognormal}(\ln(0.2), 0.5)$|
|$c_F$|Strain penalty|$\text{Lognormal}(\ln(0.1), 0.5)$|
|$\sigma_{obs,I}$|Measurement noise|$\text{Lognormal}(\ln(0.05), 0.4)$|

**Duration (4 Params):**

|   |   |   |
|---|---|---|
|**Parameter**|**Description**|**Concrete Prior**|
|$D_{base}$|Baseline capacity|$\text{Normal}(0.5, 0.1)$|
|$d_B$|Fitness sensitivity|$\text{Lognormal}(\ln(0.3), 0.5)$|
|$d_F$|Strain penalty|$\text{Lognormal}(\ln(0.2), 0.5)$|
|$\sigma_{obs,D}$|Measurement noise|$\text{Lognormal}(\ln(0.08), 0.4)$|

**Stress (4 Params):**

|   |   |   |
|---|---|---|
|**Parameter**|**Description**|**Concrete Prior**|
|$S_{base}$|Baseline stress|$\text{Normal}(30.0, 10.0)$|
|$s_A$|Vitality suppression|$\text{Lognormal}(\ln(15.0), 0.5)$|
|$s_F$|Strain increase|$\text{Lognormal}(\ln(20.0), 0.5)$|
|$\sigma_{obs,S}$|Measurement noise|$\text{Lognormal}(\ln(5.0), 0.4)$|

**Sleep (5 Params):**

|   |   |   |
|---|---|---|
|**Parameter**|**Description**|**Concrete Prior**|
|$Sleep_{base}$|Baseline quality|$\text{Normal}(0.5, 0.1)$|
|$sl_A$|Amplitude enhancement|$\text{Lognormal}(\ln(0.2), 0.5)$|
|$sl_B$|Fitness enhancement|$\text{Lognormal}(\ln(0.1), 0.5)$|
|$sl_F$|Strain disruption|$\text{Lognormal}(\ln(0.2), 0.5)$|
|$\sigma_{obs,Sleep}$|Measurement noise|$\text{Lognormal}(\ln(0.1), 0.4)$|

**Timing (4 Params):**

|   |   |   |
|---|---|---|
|**Parameter**|**Description**|**Concrete Prior**|
|$Time_{base}$|Baseline tendency|$\text{Normal}(0.0, 1.0)$|
|$t_A$|Amplitude morning pull|$\text{Lognormal}(\ln(1.0), 0.5)$|
|$t_F$|Strain disruption|$\text{Lognormal}(\ln(0.5), 0.5)$|
|$\sigma_{obs,Time}$|Measurement noise|$\text{Lognormal}(\ln(0.5), 0.4)$|