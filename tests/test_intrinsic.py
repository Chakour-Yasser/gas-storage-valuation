"""Tests of the intrinsic LP."""

import copy

import numpy as np
import pytest

from src.config_loader import load_config
from src.contract import contract_from_config
from src.intrinsic import (intrinsic_value_dp, schedule_discounted_cash_flows,
                           schedule_violations, solve_intrinsic)
from src.scenario import build_scenario


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def scenario(config):
    return build_scenario(config)


@pytest.fixture(scope="module")
def base_solution(scenario):
    return solve_intrinsic(scenario["contract"], scenario["daily_forward"], scenario["discount"])


def small_contract(config, wgv=1000.0, injection_days=10.0, withdrawal_days=10.0,
                   flat=True, costs=(0.0, 0.0)):
    """A small contract for hand-solvable cases."""
    cfg = copy.deepcopy(config)
    cfg["contract"]["working_volume"] = wgv
    cfg["contract"]["full_injection_days"] = injection_days
    cfg["contract"]["full_withdrawal_days"] = withdrawal_days
    cfg["contract"]["injection_cost"] = costs[0]
    cfg["contract"]["withdrawal_cost"] = costs[1]
    if flat:
        cfg["contract"]["injection_ratchet"] = {"fill_levels": [0.0, 1.0], "rate_fractions": [1.0, 1.0]}
        cfg["contract"]["withdrawal_ratchet"] = {"fill_levels": [0.0, 1.0], "rate_fractions": [1.0, 1.0]}
    return contract_from_config(cfg)


def test_hand_solvable_case(config):
    """4 days, prices 10, 10, 20, 20, rate WGV/2 per day, no costs, no discounting:
    fill on days 0-1, empty on days 2-3, value = WGV * (20 - 10)."""
    contract = small_contract(config, wgv=1000.0, injection_days=2.0, withdrawal_days=2.0)
    prices = np.array([10.0, 10.0, 20.0, 20.0])
    solution = solve_intrinsic(contract, prices, np.ones(4))
    assert solution["value"] == pytest.approx(10000.0, rel=1e-9)
    np.testing.assert_allclose(solution["net_volume"], [500.0, 500.0, -500.0, -500.0], atol=1e-6)


def test_hand_solvable_case_with_costs(config):
    """Same case with costs 0.2 / 0.1: value = WGV * (20 - 0.1 - 10 - 0.2)."""
    contract = small_contract(config, wgv=1000.0, injection_days=2.0, withdrawal_days=2.0,
                              costs=(0.2, 0.1))
    prices = np.array([10.0, 10.0, 20.0, 20.0])
    solution = solve_intrinsic(contract, prices, np.ones(4))
    assert solution["value"] == pytest.approx(1000.0 * 9.7, rel=1e-9)


def test_spread_below_costs_gives_zero(config):
    """If the spread does not cover the costs, the optimum is to do nothing."""
    contract = small_contract(config, costs=(0.2, 0.1))
    prices = np.array([10.0] * 20 + [10.25] * 20)
    solution = solve_intrinsic(contract, prices, np.ones(40))
    assert solution["value"] == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(solution["net_volume"], 0.0, atol=1e-6)


def test_flat_curve_gives_zero(scenario):
    flat = np.full(len(scenario["dates"]), 30.0)
    solution = solve_intrinsic(scenario["contract"], flat, scenario["discount"])
    assert solution["value"] == pytest.approx(0.0, abs=1e-3)


def test_base_solution_is_feasible(scenario, base_solution):
    violations = schedule_violations(scenario["contract"], base_solution["net_volume"])
    for name in violations:
        assert violations[name] < 1e-3, name  # MWh, on a 1e6 MWh storage


def test_no_simultaneous_injection_and_withdrawal(base_solution):
    both = np.minimum(base_solution["injection"], base_solution["withdrawal"])
    assert both.max() < 1e-3


def test_value_equals_recomputed_cash_flows(scenario, base_solution):
    cash = schedule_discounted_cash_flows(scenario["contract"], scenario["daily_forward"],
                                          base_solution["net_volume"], scenario["discount"])
    assert cash.sum() == pytest.approx(base_solution["value"], rel=1e-9)


def test_scale_invariance(config, scenario, base_solution):
    """Doubling WGV (rates scale with WGV) doubles the value."""
    cfg = copy.deepcopy(config)
    cfg["contract"]["working_volume"] = 2.0 * cfg["contract"]["working_volume"]
    contract = contract_from_config(cfg)
    solution = solve_intrinsic(contract, scenario["daily_forward"], scenario["discount"])
    assert solution["value"] == pytest.approx(2.0 * base_solution["value"], rel=1e-7)


def test_value_increases_with_spread(scenario, base_solution):
    """Scaling the curve around its mean by 1.5 increases the spread and the value."""
    forward = scenario["daily_forward"]
    wider = forward.mean() + 1.5 * (forward - forward.mean())
    solution = solve_intrinsic(scenario["contract"], wider, scenario["discount"])
    assert solution["value"] > base_solution["value"]


def test_value_decreases_with_costs(config, scenario, base_solution):
    cfg = copy.deepcopy(config)
    cfg["contract"]["injection_cost"] = 1.0
    contract = contract_from_config(cfg)
    solution = solve_intrinsic(contract, scenario["daily_forward"], scenario["discount"])
    assert solution["value"] < base_solution["value"]


def test_ratchets_reduce_value(config, scenario, base_solution):
    """Removing the ratchets (flat max rates) relaxes the problem: value goes up."""
    cfg = copy.deepcopy(config)
    cfg["contract"]["injection_ratchet"] = {"fill_levels": [0.0, 1.0], "rate_fractions": [1.0, 1.0]}
    cfg["contract"]["withdrawal_ratchet"] = {"fill_levels": [0.0, 1.0], "rate_fractions": [1.0, 1.0]}
    contract = contract_from_config(cfg)
    solution = solve_intrinsic(contract, scenario["daily_forward"], scenario["discount"])
    assert solution["value"] > base_solution["value"]


def test_dp_equals_lp_when_rates_are_on_the_grid(config, scenario):
    """Flat ratchets with rates that are multiples of the grid step: the LP
    optimum is bang-bang on the grid, so DP and LP must agree exactly."""
    contract = small_contract(config, wgv=1000.0, injection_days=100.0, withdrawal_days=50.0,
                              costs=(0.2, 0.1))
    lp_value = solve_intrinsic(contract, scenario["daily_forward"], scenario["discount"])["value"]
    dp_value = intrinsic_value_dp(contract, scenario["daily_forward"], scenario["discount"], 10.0)
    assert dp_value == pytest.approx(lp_value, rel=1e-9)


def test_dp_is_lower_bound_and_converges(scenario, base_solution):
    """With ratchets the DP restricted to grid schedules is below the LP and
    gets closer as the grid is refined."""
    lp_value = base_solution["value"]
    coarse = intrinsic_value_dp(scenario["contract"], scenario["daily_forward"],
                                scenario["discount"], 500.0)
    fine = intrinsic_value_dp(scenario["contract"], scenario["daily_forward"],
                              scenario["discount"], 100.0)
    assert coarse <= lp_value + 1e-6
    assert fine <= lp_value + 1e-6
    assert fine > coarse
    assert (lp_value - fine) / lp_value < 0.005


def test_non_concave_ratchet_is_rejected(config, scenario):
    cfg = copy.deepcopy(config)
    cfg["contract"]["injection_ratchet"] = {"fill_levels": [0.0, 0.5, 1.0],
                                            "rate_fractions": [1.0, 0.5, 1.0]}
    contract = contract_from_config(cfg)
    with pytest.raises(ValueError):
        solve_intrinsic(contract, scenario["daily_forward"], scenario["discount"])


def test_start_from_given_inventory(scenario):
    """Starting half full with a final inventory of zero must be feasible and
    worth more than starting empty (the gas in storage has value)."""
    contract = scenario["contract"]
    empty = solve_intrinsic(contract, scenario["daily_forward"], scenario["discount"])
    half = solve_intrinsic(contract, scenario["daily_forward"], scenario["discount"],
                           initial_inventory=0.5 * contract.working_volume)
    assert half["value"] > empty["value"]
    violations = schedule_violations(contract, half["net_volume"],
                                     initial_inventory=0.5 * contract.working_volume)
    assert max(violations.values()) < 1e-3
