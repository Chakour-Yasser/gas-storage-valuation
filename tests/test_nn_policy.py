"""Tests of the neural network policy."""

import copy

import numpy as np
import pytest
import torch

from src.config_loader import load_config
from src.contract import max_injection, max_withdrawal
from src.intrinsic import schedule_violations
from src.nn_policy import (N_FEATURES, PolicyNetwork, build_environment, evaluate_policy,
                           ratchet_limit, rollout, train_policy)
from src.scenario import build_scenario, simulate_paths


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def scenario(config):
    return build_scenario(config)


def to_torch(paths):
    return (torch.tensor(paths["spot"], dtype=torch.float64),
            torch.tensor(paths["chi"], dtype=torch.float64),
            torch.tensor(paths["xi"], dtype=torch.float64))


def test_torch_ratchets_match_numpy(scenario):
    contract = scenario["contract"]
    environment = build_environment(scenario, torch.float64)
    inventory = np.linspace(0.0, contract.working_volume, 501)
    inventory_t = torch.tensor(inventory, dtype=torch.float64)
    np.testing.assert_allclose(ratchet_limit(environment["injection_pieces"], inventory_t).numpy(),
                               max_injection(contract, inventory), rtol=1e-12)
    np.testing.assert_allclose(ratchet_limit(environment["withdrawal_pieces"], inventory_t).numpy(),
                               max_withdrawal(contract, inventory), rtol=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_any_network_gives_feasible_schedules(scenario, seed):
    """Feasibility holds by construction, whatever the weights."""
    torch.manual_seed(seed)
    network = PolicyNetwork(N_FEATURES, [16, 16])
    # Large random weights push the outputs to 0 or 1 (the hardest cases).
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.mul_(5.0)
    _, volumes = evaluate_policy(network, scenario, simulate_paths(scenario, 200, seed))
    violations = schedule_violations(scenario["contract"], volumes)
    for name in violations:
        assert violations[name] < 1e-6, name


def test_rollout_value_matches_numpy_cash_flows(scenario):
    """The differentiable torch value equals the NumPy recomputation from the volumes."""
    torch.manual_seed(0)
    network = PolicyNetwork(N_FEATURES, [16]).double()
    paths = simulate_paths(scenario, 100, 5)
    environment = build_environment(scenario, torch.float64)
    with torch.no_grad():
        torch_values, _ = rollout(network, environment, *to_torch(paths))
    numpy_values, _ = evaluate_policy(network, scenario, paths)
    np.testing.assert_allclose(torch_values.numpy(), numpy_values, rtol=1e-10)


def test_gradient_reaches_all_parameters(scenario):
    torch.manual_seed(0)
    network = PolicyNetwork(N_FEATURES, [16, 16])
    environment = build_environment(scenario, torch.float32)
    paths = simulate_paths(scenario, 64, 6)
    spot, chi, xi = to_torch(paths)
    values, _ = rollout(network, environment, spot.float(), chi.float(), xi.float())
    (-values.mean()).backward()
    for parameter in network.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0.0


def test_short_training_improves_value(config, scenario):
    """A short training run must raise the value on fixed paths well above the start
    (same paths before and after, so the path noise cancels in the comparison)."""
    settings = copy.deepcopy(config["nn_policy"])
    settings["n_iterations"] = 80
    settings["batch_size"] = 512
    settings["validate_every"] = 40
    settings["hidden_sizes"] = [32, 32]
    validation = simulate_paths(scenario, 500, 7)
    torch.manual_seed(int(settings["seed_torch"]))
    untrained = PolicyNetwork(N_FEATURES, settings["hidden_sizes"])
    start_value = evaluate_policy(untrained, scenario, validation)[0].mean()
    network, history = train_policy(scenario, settings, validation, verbose=False)
    trained_value = evaluate_policy(network, scenario, validation)[0].mean()
    assert trained_value > start_value + 5e5
