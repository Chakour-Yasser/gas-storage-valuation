"""Tests of the LSMC valuation."""

import copy

import numpy as np
import pytest

from src.config_loader import load_config
from src.contract import days_to_empty, max_injection, max_withdrawal
from src.evaluation import (control_variate_confidence_interval, difference_confidence_interval,
                            mean_confidence_interval)
from src.intrinsic import schedule_discounted_cash_flows, schedule_violations, solve_intrinsic
from src.lsmc import (action_candidates, grid_position, lsmc_backward, lsmc_forward, polynomial_basis,
                      reachable_band, regression_basis)
from src.scenario import build_scenario, simulate_paths


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def scenario(config):
    return build_scenario(config)


@pytest.fixture(scope="module")
def base_run(config, scenario):
    """Small base-case run (config defaults), shared by several tests."""
    contract = scenario["contract"]
    regression = simulate_paths(scenario, 2000, 11)
    policy, in_sample = lsmc_backward(contract, regression["spot"], scenario["discount"],
                                      config["lsmc"], factors=regression)
    valuation = simulate_paths(scenario, 4000, 12)
    values, volumes, actions = lsmc_forward(contract, valuation["spot"], scenario["discount"],
                                            policy, factors=valuation)
    return {"policy": policy, "values": values, "volumes": volumes, "spot": valuation["spot"],
            "regression": regression, "valuation": valuation}


# ---------------------------------------------------------------------------
# Reachable band
# ---------------------------------------------------------------------------

def test_band_end_points(scenario):
    contract = scenario["contract"]
    lower, upper = reachable_band(contract, len(scenario["dates"]))
    assert upper[-1] == contract.final_inventory
    assert lower[-1] == contract.final_inventory
    assert upper[0] == contract.working_volume
    np.testing.assert_allclose(lower, 0.0)  # final inventory 0: no lower constraint


def test_band_matches_days_to_empty(scenario):
    """U_d < WGV exactly on the last days_to_empty days (emptying needs 104 days)."""
    contract = scenario["contract"]
    lower, upper = reachable_band(contract, len(scenario["dates"]))
    n_restricted = np.sum(upper[:-1] < contract.working_volume - 1e-6)
    assert abs(n_restricted - days_to_empty(contract)) <= 1


def test_band_is_tight(scenario):
    """From U_d, withdrawing at max lands exactly on U_{d+1} (when U_d < WGV)."""
    contract = scenario["contract"]
    lower, upper = reachable_band(contract, len(scenario["dates"]))
    for d in range(len(upper) - 1):
        after = upper[d] - float(max_withdrawal(contract, upper[d]))
        if upper[d] < contract.working_volume:
            assert after == pytest.approx(upper[d + 1], abs=1e-6)
        else:
            assert after <= upper[d + 1] + 1e-6


def test_band_with_positive_final_inventory(config):
    """With V_final = 50% WGV, the lower bound must be reachable by injecting at max."""
    cfg = copy.deepcopy(config)
    cfg["contract"]["final_inventory"] = 500000.0
    scenario = build_scenario(cfg)
    contract = scenario["contract"]
    lower, upper = reachable_band(contract, len(scenario["dates"]))
    assert lower[-1] == 500000.0 and upper[-1] == 500000.0
    for d in range(len(lower) - 1):
        after = lower[d] + float(max_injection(contract, lower[d]))
        assert after >= lower[d + 1] - 1e-6


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def test_linear_interpolation_is_exact_for_linear_functions():
    grid = np.linspace(100.0, 1100.0, 11)
    values = 3.0 * grid + 7.0
    v = np.array([100.0, 155.0, 640.0, 1100.0])
    j, w = grid_position(grid, v)
    interpolated = (1.0 - w) * values[j] + w * values[j + 1]
    np.testing.assert_allclose(interpolated, 3.0 * v + 7.0)


def test_degenerate_grid_position():
    j, w = grid_position(np.zeros(5), np.array([0.0, 0.0]))
    np.testing.assert_array_equal(j, [0, 0])
    np.testing.assert_array_equal(w, [0.0, 0.0])


def test_action_candidates_inside_band(scenario):
    contract = scenario["contract"]
    wgv = contract.working_volume
    inventory = np.array([0.0, 0.5 * wgv])
    q, feasible = action_candidates(contract, inventory, 0.0, wgv, [0.0, 1.0])
    # Empty storage: "withdraw at max" is clipped to 0, "inject at max" is the full rate.
    assert q[0, 0] == 0.0
    assert q[1, 0] == pytest.approx(contract.max_injection_rate)
    # Half full: withdraw and inject at max rates; doing nothing is feasible.
    assert q[0, 1] == pytest.approx(-contract.max_withdrawal_rate)
    assert q[1, 1] == pytest.approx(contract.max_injection_rate)
    assert np.all(feasible[-1])


def test_action_candidates_clip_to_band(scenario):
    """Near maturity (U_{d+1} small) injection is cut and doing nothing can be infeasible."""
    contract = scenario["contract"]
    upper_next = 10000.0
    # 14000 - W(14000) = 14000 - 4755.6 <= 10000: inside the band, but above U_{d+1}.
    inventory = np.array([5000.0, 14000.0])
    q, feasible = action_candidates(contract, inventory, 0.0, upper_next, [0.0, 1.0])
    assert q[1, 0] == pytest.approx(5000.0)        # inject only up to U_{d+1}
    assert feasible[-1, 0]                          # 5000 <= U_{d+1}: nothing is allowed
    assert not feasible[-1, 1]                      # 14000 > U_{d+1}: must withdraw
    assert q[1, 1] == pytest.approx(-4000.0)        # "inject" is clipped to the smallest withdrawal
    assert np.all(inventory + q[:2] <= upper_next + 1e-9)


def test_standardised_basis_has_same_fit_as_raw_polynomial():
    rng = np.random.default_rng(0)
    prices = 30.0 + 5.0 * rng.standard_normal(500)
    target = np.sin(prices / 5.0) + 0.1 * rng.standard_normal(500)
    raw = np.column_stack([np.ones(500), prices, prices ** 2, prices ** 3])
    standard = regression_basis(prices, prices.mean(), prices.std(), 3)
    fit_raw = raw @ np.linalg.lstsq(raw, target, rcond=None)[0]
    fit_standard = standard @ np.linalg.lstsq(standard, target, rcond=None)[0]
    np.testing.assert_allclose(fit_raw, fit_standard, atol=1e-8)


# ---------------------------------------------------------------------------
# Valuation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("grid_type", ["fixed", "adaptive"])
def test_near_zero_volatility_recovers_intrinsic(config, grid_type):
    """Without uncertainty the best policy is the intrinsic one: LSMC must be
    very close to the LP value and cannot exceed it (it is a feasible schedule)."""
    cfg = copy.deepcopy(config)
    cfg["model"]["sigma_chi"] = 1e-4
    cfg["model"]["sigma_xi"] = 1e-4
    scenario = build_scenario(cfg)
    contract = scenario["contract"]
    settings = copy.deepcopy(cfg["lsmc"])
    settings["grid_type"] = grid_type
    lp_value = solve_intrinsic(contract, scenario["daily_forward"], scenario["discount"])["value"]
    regression = simulate_paths(scenario, 300, 1)
    valuation = simulate_paths(scenario, 300, 2)
    policy, _ = lsmc_backward(contract, regression["spot"], scenario["discount"], settings,
                              factors=regression)
    values, _, _ = lsmc_forward(contract, valuation["spot"], scenario["discount"], policy,
                                factors=valuation)
    assert values.mean() <= lp_value * (1.0 + 1e-4)
    assert values.mean() >= lp_value * (1.0 - 0.002)


def test_out_of_sample_schedules_are_feasible(scenario, base_run):
    violations = schedule_violations(scenario["contract"], base_run["volumes"])
    for name in violations:
        assert violations[name] < 1e-6, name


def test_value_equals_recomputed_cash_flows(scenario, base_run):
    recomputed = schedule_discounted_cash_flows(scenario["contract"], base_run["spot"],
                                                base_run["volumes"], scenario["discount"])
    np.testing.assert_allclose(recomputed.sum(axis=1), base_run["values"], rtol=1e-10)


def test_lsmc_beats_intrinsic(scenario, base_run):
    """With volatility, the dynamic policy is worth more than the intrinsic value."""
    contract = scenario["contract"]
    intrinsic = solve_intrinsic(contract, scenario["daily_forward"], scenario["discount"])
    control = schedule_discounted_cash_flows(contract, base_run["spot"], intrinsic["net_volume"],
                                             scenario["discount"]).sum(axis=1)
    ci = control_variate_confidence_interval(base_run["values"], control, intrinsic["value"])
    assert ci["low"] > intrinsic["value"]


def test_lsmc_is_reproducible(config, scenario):
    contract = scenario["contract"]
    results = []
    for repeat in range(2):
        regression = simulate_paths(scenario, 300, 5)
        valuation = simulate_paths(scenario, 300, 6)
        policy, _ = lsmc_backward(contract, regression["spot"], scenario["discount"],
                                  config["lsmc"], factors=regression)
        values, _, _ = lsmc_forward(contract, valuation["spot"], scenario["discount"], policy,
                                    factors=valuation)
        results.append(values)
    np.testing.assert_array_equal(results[0], results[1])


def test_polynomial_basis_terms():
    """Two variables, degree 3: 1 + 2 + 3 + 4 = 10 monomials; one variable: 4."""
    x = np.array([[2.0, 3.0]])
    basis = polynomial_basis(x, 3)
    assert basis.shape == (1, 10)
    expected = [1, 2, 3, 4, 6, 9, 8, 12, 18, 27]
    np.testing.assert_allclose(basis[0], expected)
    assert polynomial_basis(np.array([[2.0]]), 3).shape == (1, 4)


def test_factor_basis_needs_factors(config, scenario):
    with pytest.raises(ValueError):
        lsmc_backward(scenario["contract"], simulate_paths(scenario, 50, 1)["spot"],
                      scenario["discount"], config["lsmc"])


def test_factor_basis_beats_spot_basis(config, scenario, base_run):
    """The spot only reveals chi + xi; the factor basis uses the full state and
    must give a higher out-of-sample value on the same paths."""
    contract = scenario["contract"]
    settings = copy.deepcopy(config["lsmc"])
    settings["basis"] = "spot"
    policy, _ = lsmc_backward(contract, base_run["regression"]["spot"], scenario["discount"],
                              settings)
    spot_values, _, _ = lsmc_forward(contract, base_run["spot"], scenario["discount"], policy)
    diff = difference_confidence_interval(base_run["values"], spot_values)
    assert diff["low"] > 0.0


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def test_control_variate_keeps_mean_and_reduces_variance():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(100000)
    y = 2.0 + 3.0 * x + 0.5 * rng.standard_normal(100000)
    plain = mean_confidence_interval(y)
    with_cv = control_variate_confidence_interval(y, x, 0.0)
    assert abs(with_cv["mean"] - 2.0) < 4.0 * with_cv["std_error"]
    assert with_cv["std_error"] < 0.2 * plain["std_error"]
    assert with_cv["beta"] == pytest.approx(3.0, rel=0.01)


def test_paired_difference_is_tighter_than_independent():
    rng = np.random.default_rng(4)
    common = rng.standard_normal(10000)
    a = common + 0.1 * rng.standard_normal(10000)
    b = common + 0.1 * rng.standard_normal(10000)
    paired = difference_confidence_interval(a, b, paired=True)
    independent = difference_confidence_interval(a, b, paired=False)
    assert paired["std_error"] < 0.2 * independent["std_error"]
