"""Tests of the Schwartz-Smith model: closed forms and exact simulation."""

import numpy as np
import pytest
from scipy import integrate, stats

from src.config_loader import load_config
from src.evaluation import mean_confidence_interval
from src.forward_curve import (daily_forward_curve, delivery_dates_from_config,
                               month_index_of_days, monthly_averages,
                               monthly_curve_from_config, year_fractions)
from src.model import (SchwartzSmithParams, factor_covariance, futures_price,
                       mean_log_factors, params_from_config, seasonal_term,
                       simulate_factors, spot_prices, variance_log_factors)

# Tolerance in standard errors for Monte Carlo tests. With a fixed seed the
# tests are deterministic; 4 standard errors keeps them meaningful but robust.
Z_TOL = 4.0


@pytest.fixture(scope="module")
def setup():
    config = load_config()
    params = params_from_config(config)
    months, prices = monthly_curve_from_config(config)
    dates, valuation_date = delivery_dates_from_config(config)
    times = year_fractions(dates, valuation_date)
    month_index = month_index_of_days(dates, months)
    forward = daily_forward_curve(month_index, prices)
    seasonal = seasonal_term(params, times, forward)
    return {
        "params": params, "months": months, "prices": prices, "dates": dates,
        "times": times, "month_index": month_index, "forward": forward,
        "seasonal": seasonal,
    }


@pytest.fixture(scope="module")
def paths(setup):
    rng = np.random.default_rng(123)
    chi, xi = simulate_factors(setup["params"], setup["times"], 20000, rng)
    return chi, xi


def test_model_forwards_match_input_curve_exactly(setup):
    """F(0, T) from the closed form equals the input forward for every day."""
    p = setup["params"]
    model_forward = futures_price(p, setup["seasonal"], setup["times"], p.chi0, p.xi0)
    np.testing.assert_allclose(model_forward, setup["forward"], rtol=1e-12)


def test_exact_fit_holds_with_nonzero_initial_factors_and_drift(setup):
    """The seasonal term absorbs chi0, xi0 and mu_xi: the fit stays exact."""
    p = SchwartzSmithParams(kappa=2.0, sigma_chi=0.5, sigma_xi=0.3, rho=0.4,
                            mu_xi=0.1, chi0=0.2, xi0=-0.3)
    seasonal = seasonal_term(p, setup["times"], setup["forward"])
    model_forward = futures_price(p, seasonal, setup["times"], p.chi0, p.xi0)
    np.testing.assert_allclose(model_forward, setup["forward"], rtol=1e-12)


def test_variance_formula_matches_numerical_integration(setup):
    """Var[X_tau] = int_0^tau (sigma_chi e^{-kappa s} + ...)^2 ... computed by quad.

    X_tau - E[X_tau] = int_0^tau sigma_chi e^{-kappa (tau-s)} dW1 + sigma_xi dW2, so
    Var = int_0^tau [a(s)^2 + sigma_xi^2 + 2 rho a(s) sigma_xi] ds,
    a(s) = sigma_chi e^{-kappa (tau - s)}.
    """
    p = setup["params"]
    for tau in [0.01, 0.25, 1.0, 3.0]:
        def integrand(s):
            a = p.sigma_chi * np.exp(-p.kappa * (tau - s))
            return a ** 2 + p.sigma_xi ** 2 + 2.0 * p.rho * a * p.sigma_xi
        numeric, _ = integrate.quad(integrand, 0.0, tau)
        assert variance_log_factors(p, tau) == pytest.approx(numeric, rel=1e-10)


def test_step_covariance_is_positive_definite(setup):
    p = setup["params"]
    for dt in [1.0 / 365.0, 0.1, 1.0]:
        var_chi, var_xi, cov = factor_covariance(p, dt)
        eigenvalues = np.linalg.eigvalsh(np.array([[var_chi, cov], [cov, var_xi]]))
        assert np.all(eigenvalues > 0.0)


def test_simulated_mean_of_spot_matches_forward_per_month(setup, paths):
    """MC mean of the monthly average spot price equals the monthly forward."""
    chi, xi = paths
    spot = spot_prices(setup["seasonal"], chi, xi)
    monthly = monthly_averages(spot, setup["month_index"], len(setup["months"]))
    for m in range(len(setup["months"])):
        ci = mean_confidence_interval(monthly[:, m])
        z = (ci["mean"] - setup["prices"][m]) / ci["std_error"]
        assert abs(z) < Z_TOL, "month {} z-score {:.2f}".format(m, z)


def test_simulated_mean_of_log_factors(setup, paths):
    chi, xi = paths
    x = chi + xi
    theory = mean_log_factors(setup["params"], setup["times"])
    for i in [0, 100, 200, 365]:
        ci = mean_confidence_interval(x[:, i])
        assert abs(ci["mean"] - theory[i]) < Z_TOL * ci["std_error"]


def test_simulated_variance_of_log_factors(setup, paths):
    """Var of chi + xi: the normalised sample variance is chi-square(n-1)."""
    chi, xi = paths
    x = chi + xi
    n = x.shape[0]
    theory = variance_log_factors(setup["params"], setup["times"])
    for i in [0, 60, 180, 365]:
        statistic = (n - 1) * x[:, i].var(ddof=1) / theory[i]
        p_value = 2.0 * min(stats.chi2.cdf(statistic, n - 1), stats.chi2.sf(statistic, n - 1))
        assert p_value > 1e-4


def test_simulation_is_exact_for_any_step_size(setup):
    """One step of length T and 365 daily steps give the same distribution at T.

    With an Euler scheme the variance would differ; with exact simulation both
    must match the closed form.
    """
    p = setup["params"]
    n = 40000
    one_step_chi, one_step_xi = simulate_factors(p, np.array([1.0]), n, np.random.default_rng(1))
    daily_times = np.arange(1, 366) / 365.0
    daily_chi, daily_xi = simulate_factors(p, daily_times, n, np.random.default_rng(2))

    var_chi, var_xi, cov = factor_covariance(p, 1.0)
    for chi_end, xi_end in [(one_step_chi[:, -1], one_step_xi[:, -1]),
                            (daily_chi[:, -1], daily_xi[:, -1])]:
        # Standard error of a sample variance of Gaussians: sigma^2 sqrt(2 / (n - 1)).
        assert abs(chi_end.var(ddof=1) - var_chi) < Z_TOL * var_chi * np.sqrt(2.0 / (n - 1))
        assert abs(xi_end.var(ddof=1) - var_xi) < Z_TOL * var_xi * np.sqrt(2.0 / (n - 1))
        # Standard error of the sample covariance: sqrt((var_chi var_xi + cov^2) / n).
        cov_se = np.sqrt((var_chi * var_xi + cov ** 2) / n)
        assert abs(np.cov(chi_end, xi_end)[0, 1] - cov) < Z_TOL * cov_se


def test_futures_are_martingales(setup, paths):
    """E[F(t, T)] = F(0, T) for a mid-year observation date t."""
    chi, xi = paths
    p = setup["params"]
    obs = 182
    t_obs = setup["times"][obs]
    for maturity in [200, 280, 365]:
        tau = setup["times"][maturity] - t_obs
        f_t = futures_price(p, setup["seasonal"][maturity], tau, chi[:, obs], xi[:, obs])
        ci = mean_confidence_interval(f_t)
        assert abs(ci["mean"] - setup["forward"][maturity]) < Z_TOL * ci["std_error"]


def test_futures_converge_to_spot_at_maturity(setup, paths):
    """F(T, T) = S_T."""
    chi, xi = paths
    i = 250
    spot = spot_prices(setup["seasonal"][i], chi[:, i], xi[:, i])
    f_tt = futures_price(setup["params"], setup["seasonal"][i], 0.0, chi[:, i], xi[:, i])
    np.testing.assert_allclose(f_tt, spot, rtol=1e-12)


def test_simulation_is_reproducible_with_seed(setup):
    a = simulate_factors(setup["params"], setup["times"][:10], 100, np.random.default_rng(5))
    b = simulate_factors(setup["params"], setup["times"][:10], 100, np.random.default_rng(5))
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_invalid_times_raise(setup):
    with pytest.raises(ValueError):
        simulate_factors(setup["params"], np.array([0.1, 0.1]), 10, np.random.default_rng(0))
