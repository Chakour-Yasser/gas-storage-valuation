"""Two-factor Schwartz-Smith (2000) model with deterministic seasonality.

Log spot price:
    ln S_t = f(t) + chi_t + xi_t

Short-term factor (Ornstein-Uhlenbeck, mean zero):
    d chi_t = -kappa * chi_t dt + sigma_chi dW1_t

Long-term factor (arithmetic Brownian motion):
    d xi_t = mu_xi dt + sigma_xi dW2_t,      d<W1, W2>_t = rho dt

All dynamics are under the pricing measure Q. In the original paper chi has a
constant risk premium lambda_chi and xi a risk-neutral drift mu_xi*. Both only
add a deterministic function of time to ln S_t, so they are absorbed by the
seasonal term f(t), which is fitted to the input forward curve:

    F(0, T) = E[S_T] = F_input(T)   exactly, for every delivery day T.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SchwartzSmithParams:
    kappa: float
    sigma_chi: float
    sigma_xi: float
    rho: float
    mu_xi: float
    chi0: float
    xi0: float


def validate_params(params):
    """Raise ValueError for parameters outside their admissible range."""
    if params.kappa <= 0.0:
        raise ValueError("kappa must be > 0")
    if params.sigma_chi < 0.0 or params.sigma_xi < 0.0:
        raise ValueError("volatilities must be >= 0")
    if params.rho < -1.0 or params.rho > 1.0:
        raise ValueError("rho must be in [-1, 1]")


def params_from_config(config):
    """Build the model parameters from the [model] section."""
    model_cfg = config["model"]
    params = SchwartzSmithParams(
        kappa=float(model_cfg["kappa"]),
        sigma_chi=float(model_cfg["sigma_chi"]),
        sigma_xi=float(model_cfg["sigma_xi"]),
        rho=float(model_cfg["rho"]),
        mu_xi=float(model_cfg["mu_xi"]),
        chi0=float(model_cfg["chi0"]),
        xi0=float(model_cfg["xi0"]),
    )
    validate_params(params)
    return params


# ---------------------------------------------------------------------------
# Moments of X_t = chi_t + xi_t
# ---------------------------------------------------------------------------

def mean_log_factors(params, t):
    """E[chi_t + xi_t] given chi_0, xi_0.

    chi_t = chi_0 e^{-kappa t} + Gaussian noise with mean 0,
    xi_t  = xi_0 + mu_xi t     + Gaussian noise with mean 0.
    """
    t = np.asarray(t, dtype=float)
    return params.chi0 * np.exp(-params.kappa * t) + params.xi0 + params.mu_xi * t


def variance_log_factors(params, tau):
    """Var[chi_{t+tau} + xi_{t+tau} | F_t]. It depends only on the horizon tau.

        Var(chi) = sigma_chi^2 (1 - e^{-2 kappa tau}) / (2 kappa)
        Var(xi)  = sigma_xi^2 tau
        Cov      = rho sigma_chi sigma_xi (1 - e^{-kappa tau}) / kappa
        Var(chi + xi) = Var(chi) + Var(xi) + 2 Cov
    """
    var_chi, var_xi, cov = factor_covariance(params, tau)
    return var_chi + var_xi + 2.0 * cov


def factor_covariance(params, tau):
    """Covariance of the Gaussian increments of (chi, xi) over a horizon tau.

    Exact formulas from Ito isometry:
        chi_{t+tau} - chi_t e^{-kappa tau} = sigma_chi int_t^{t+tau} e^{-kappa (t+tau-s)} dW1_s
        xi_{t+tau}  - xi_t - mu_xi tau     = sigma_xi  (W2_{t+tau} - W2_t)
    """
    tau = np.asarray(tau, dtype=float)
    kappa = params.kappa
    var_chi = params.sigma_chi ** 2 * (1.0 - np.exp(-2.0 * kappa * tau)) / (2.0 * kappa)
    var_xi = params.sigma_xi ** 2 * tau
    cov = params.rho * params.sigma_chi * params.sigma_xi * (1.0 - np.exp(-kappa * tau)) / kappa
    return var_chi, var_xi, cov


# ---------------------------------------------------------------------------
# Seasonal term and futures prices
# ---------------------------------------------------------------------------

def seasonal_term(params, times, forward_prices):
    """Deterministic term f(T) such that model forwards match the input curve.

    Since ln S_T is Gaussian, F(0, T) = E[S_T] = exp(f(T) + E[X_T] + Var[X_T] / 2).
    Setting F(0, T) = F_input(T) gives
        f(T) = ln F_input(T) - E[X_T] - Var[X_T] / 2.
    """
    times = np.asarray(times, dtype=float)
    forward_prices = np.asarray(forward_prices, dtype=float)
    mean_x = mean_log_factors(params, times)
    var_x = variance_log_factors(params, times)
    return np.log(forward_prices) - mean_x - 0.5 * var_x


def futures_price(params, seasonal_at_maturity, tau, chi_t, xi_t):
    """Closed-form futures price F(t, T) = E_t[S_T], with tau = T - t >= 0.

        E_t[ln S_T]   = f(T) + chi_t e^{-kappa tau} + xi_t + mu_xi tau
        Var_t[ln S_T] = variance_log_factors(tau)
        F(t, T)       = exp(E_t[ln S_T] + Var_t[ln S_T] / 2)

    Arguments broadcast with NumPy rules, so chi_t and xi_t can be arrays of
    paths with shape (n_paths, 1) and tau, seasonal_at_maturity arrays of
    maturities with shape (n_maturities,).
    """
    tau = np.asarray(tau, dtype=float)
    if np.any(tau < 0.0):
        raise ValueError("futures_price needs maturities T >= t")
    mean_log = (seasonal_at_maturity
                + np.exp(-params.kappa * tau) * chi_t
                + xi_t
                + params.mu_xi * tau)
    return np.exp(mean_log + 0.5 * variance_log_factors(params, tau))


# ---------------------------------------------------------------------------
# Exact simulation
# ---------------------------------------------------------------------------

def step_cholesky(params, dt):
    """Lower Cholesky factor (l11, l21, l22) of the 2x2 increment covariance."""
    var_chi, var_xi, cov = factor_covariance(params, dt)
    l11 = np.sqrt(var_chi)
    if l11 > 0.0:
        l21 = cov / l11
    else:
        l21 = 0.0
    # max(.., 0) only protects against tiny negative rounding errors (|rho| = 1).
    l22 = np.sqrt(max(var_xi - l21 ** 2, 0.0))
    return l11, l21, l22


def simulate_factors(params, times, n_paths, rng):
    """Exact simulation of (chi, xi) at the given times, vectorised over paths.

    The transition of (chi, xi) over a step dt is Gaussian with known mean and
    covariance (see factor_covariance), so there is no discretisation error,
    whatever the step size.

    Parameters
    ----------
    times : increasing array of times (years) > 0, measured from the valuation date
    n_paths : number of Monte Carlo paths
    rng : np.random.Generator

    Returns
    -------
    chi, xi : arrays of shape (n_paths, n_times)
    """
    times = np.asarray(times, dtype=float)
    n_times = len(times)
    chi = np.empty((n_paths, n_times))
    xi = np.empty((n_paths, n_times))

    chi_current = np.full(n_paths, params.chi0)
    xi_current = np.full(n_paths, params.xi0)
    decay_previous_dt = None
    t_previous = 0.0

    for i in range(n_times):
        dt = times[i] - t_previous
        if dt <= 0.0:
            raise ValueError("times must be strictly increasing and > 0")
        # The time grid is usually uniform: recompute the Cholesky factor only
        # when the step size changes.
        if decay_previous_dt is None or dt != decay_previous_dt:
            l11, l21, l22 = step_cholesky(params, dt)
            decay = np.exp(-params.kappa * dt)
            decay_previous_dt = dt

        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        chi_current = chi_current * decay + l11 * z1
        xi_current = xi_current + params.mu_xi * dt + l21 * z1 + l22 * z2

        chi[:, i] = chi_current
        xi[:, i] = xi_current
        t_previous = times[i]

    return chi, xi


def spot_prices(seasonal, chi, xi):
    """Spot prices S_t = exp(f(t) + chi_t + xi_t); seasonal has shape (n_times,)."""
    return np.exp(seasonal + chi + xi)
