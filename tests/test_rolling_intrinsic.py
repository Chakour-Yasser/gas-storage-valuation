"""Tests of the rolling intrinsic strategy."""

import copy

import numpy as np
import pytest

from src.config_loader import load_config
from src.evaluation import control_variate_confidence_interval, ratio_confidence_interval
from src.intrinsic import schedule_discounted_cash_flows, schedule_violations, solve_intrinsic
from src.rolling_intrinsic import model_forward_curve, reoptimisation_days, run_rolling_intrinsic
from src.scenario import build_scenario, simulate_paths


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def scenario(config):
    return build_scenario(config)


@pytest.fixture(scope="module")
def small_run(scenario):
    paths = simulate_paths(scenario, 30, 21)
    volumes = run_rolling_intrinsic(scenario, paths["chi"], paths["xi"], 7)
    return {"paths": paths, "volumes": volumes}


def test_reoptimisation_days():
    assert reoptimisation_days(15, 7) == [0, 7, 14]
    assert reoptimisation_days(3, 1) == [0, 1, 2]


def test_forward_curve_starts_at_spot_and_matches_initial_curve(scenario):
    """F(t_d, t_d) = S_d on each path, and at t = 0 the curve is the input curve."""
    paths = simulate_paths(scenario, 5, 3)
    d = 100
    for i in range(5):
        forwards = model_forward_curve(scenario["params"], scenario["seasonal"], scenario["times"],
                                       d, paths["chi"][i, d], paths["xi"][i, d])
        assert forwards[0] == pytest.approx(paths["spot"][i, d], rel=1e-12)
        assert len(forwards) == len(scenario["times"]) - d
    # At the valuation date (chi0, xi0, tau = T) the model reproduces the input curve.
    p = scenario["params"]
    from src.model import futures_price
    np.testing.assert_allclose(
        futures_price(p, scenario["seasonal"], scenario["times"], p.chi0, p.xi0),
        scenario["daily_forward"], rtol=1e-12)


def test_rolling_schedules_are_feasible(scenario, small_run):
    violations = schedule_violations(scenario["contract"], small_run["volumes"])
    for name in violations:
        assert violations[name] < 1e-4, name


def test_serial_and_parallel_runs_agree(scenario, small_run):
    paths = small_run["paths"]
    parallel = run_rolling_intrinsic(scenario, paths["chi"], paths["xi"], 7, n_workers=2,
                                     n_chunks=3)
    np.testing.assert_allclose(parallel, small_run["volumes"], atol=1e-6)


def test_near_zero_volatility_recovers_intrinsic(config):
    """Without uncertainty the forward curve never moves, every re-optimisation
    returns the same plan, and the value is the intrinsic value."""
    cfg = copy.deepcopy(config)
    cfg["model"]["sigma_chi"] = 1e-6
    cfg["model"]["sigma_xi"] = 1e-6
    scenario = build_scenario(cfg)
    contract = scenario["contract"]
    lp_value = solve_intrinsic(contract, scenario["daily_forward"], scenario["discount"])["value"]
    paths = simulate_paths(scenario, 3, 1)
    volumes = run_rolling_intrinsic(scenario, paths["chi"], paths["xi"], 30)
    values = schedule_discounted_cash_flows(contract, paths["spot"], volumes,
                                            scenario["discount"]).sum(axis=1)
    np.testing.assert_allclose(values, lp_value, rtol=1e-4)


def test_rolling_beats_intrinsic(scenario):
    """In expectation rolling intrinsic is worth at least the intrinsic value."""
    contract = scenario["contract"]
    intrinsic = solve_intrinsic(contract, scenario["daily_forward"], scenario["discount"])
    paths = simulate_paths(scenario, 200, 22)
    volumes = run_rolling_intrinsic(scenario, paths["chi"], paths["xi"], 14, n_workers=4)
    values = schedule_discounted_cash_flows(contract, paths["spot"], volumes,
                                            scenario["discount"]).sum(axis=1)
    control = schedule_discounted_cash_flows(contract, paths["spot"], intrinsic["net_volume"],
                                             scenario["discount"]).sum(axis=1)
    ci = control_variate_confidence_interval(values, control, intrinsic["value"])
    assert ci["low"] > intrinsic["value"]


def test_ratio_confidence_interval():
    rng = np.random.default_rng(0)
    b = 2.0 + rng.standard_normal(100000)
    a = 0.5 * b + 0.1 * rng.standard_normal(100000)
    ci = ratio_confidence_interval(a, b)
    assert ci["low"] < 0.5 < ci["high"]
    assert ci["std_error"] < 0.01
