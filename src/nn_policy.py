"""Neural network operating policy trained by Monte Carlo gradient (deep-hedging style).

Policy
------
One network shared by all days maps the state of day d to a fraction a in [0, 1]:
    inputs: time left (N - d) / N, fill level V / WGV, position of V inside the
            reachable band [L_d, U_d], and the factors chi_d, xi_d standardised
            with their theoretical mean and standard deviation.
    action: q = q_lo + a (q_hi - q_lo), with the feasible interval
            q_lo = max(-W(V), L_{d+1} - V),   q_hi = min(I(V), U_{d+1} - V).
Every action is feasible by construction (ratchets, capacity and the hard
terminal condition V_N = V_final), so no penalty is needed.

Training
--------
Maximise E[ sum_d DF_d cash(S_d, q_d) ] over the network weights, estimated on
fresh simulated paths at every iteration. The cash flows are differentiable
in the actions and the inventory dynamics V_{d+1} = V_d + q_d are linear, so
the gradient is obtained by backpropagation through the 366 daily steps.

Ratchets in torch: both ratchets are concave, so each is the minimum of its
affine pieces, rate(V) = min_k (a_k + b_k V), which is exact and differentiable.
"""

import copy
import time

import numpy as np
import torch
from torch import nn

from src.contract import ratchet_affine_pieces
from src.intrinsic import schedule_discounted_cash_flows
from src.lsmc import reachable_band
from src.model import factor_covariance, simulate_factors, spot_prices


class PolicyNetwork(nn.Module):
    """Multilayer perceptron with a sigmoid output (fraction of the feasible interval)."""

    def __init__(self, n_inputs, hidden_sizes):
        super().__init__()
        layers = []
        size = n_inputs
        for hidden in hidden_sizes:
            layers.append(nn.Linear(size, hidden))
            layers.append(nn.SiLU())
            size = hidden
        layers.append(nn.Linear(size, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, features):
        return torch.sigmoid(self.layers(features)).squeeze(-1)


N_FEATURES = 5


def build_environment(scenario, dtype):
    """Constant tensors and numbers needed to roll out the policy."""
    contract = scenario["contract"]
    params = scenario["params"]
    times = scenario["times"]
    n_days = len(times)
    lower, upper = reachable_band(contract, n_days)

    # Theoretical moments of the factors, used to standardise the inputs.
    var_chi, var_xi, _ = factor_covariance(params, times)
    chi_mean = params.chi0 * np.exp(-params.kappa * times)
    xi_mean = params.xi0 + params.mu_xi * times

    environment = {
        "contract": contract,
        "n_days": n_days,
        "dtype": dtype,
        "lower": lower,
        "upper": upper,
        "discount": torch.tensor(scenario["discount"], dtype=dtype),
        "chi_mean": torch.tensor(chi_mean, dtype=dtype),
        "chi_std": torch.tensor(np.sqrt(var_chi), dtype=dtype),
        "xi_mean": torch.tensor(xi_mean, dtype=dtype),
        "xi_std": torch.tensor(np.sqrt(var_xi), dtype=dtype),
        "injection_pieces": ratchet_affine_pieces(
            contract.injection_fill_levels, contract.injection_fractions,
            contract.max_injection_rate, contract.working_volume),
        "withdrawal_pieces": ratchet_affine_pieces(
            contract.withdrawal_fill_levels, contract.withdrawal_fractions,
            contract.max_withdrawal_rate, contract.working_volume),
    }
    return environment


def ratchet_limit(pieces, inventory):
    """Concave piecewise-linear rate = minimum of its affine pieces (torch)."""
    limit = None
    for intercept, slope in pieces:
        value = intercept + slope * inventory
        if limit is None:
            limit = value
        else:
            limit = torch.minimum(limit, value)
    return limit


def day_features(environment, d, inventory, chi_d, xi_d):
    """Network inputs for day d, shape (n_paths, N_FEATURES)."""
    contract = environment["contract"]
    n_days = environment["n_days"]
    time_left = torch.full_like(inventory, (n_days - d) / n_days)
    fill = inventory / contract.working_volume
    width = environment["upper"][d] - environment["lower"][d]
    if width > 0.0:
        band_position = (inventory - environment["lower"][d]) / width
    else:
        band_position = torch.zeros_like(inventory)
    chi_standard = (chi_d - environment["chi_mean"][d]) / environment["chi_std"][d]
    xi_standard = (xi_d - environment["xi_mean"][d]) / environment["xi_std"][d]
    return torch.stack([time_left, fill, band_position, chi_standard, xi_standard], dim=1)


def rollout(network, environment, spot, chi, xi):
    """Apply the policy along the paths (torch tensors of shape (n_paths, n_days)).

    Returns the discounted cash flow of each path and the daily volumes.
    """
    contract = environment["contract"]
    n_paths = spot.shape[0]
    inventory = torch.full((n_paths,), contract.initial_inventory, dtype=environment["dtype"])
    path_values = torch.zeros(n_paths, dtype=environment["dtype"])
    volumes = []

    for d in range(environment["n_days"]):
        # Feasible interval for today's trade.
        inject_limit = ratchet_limit(environment["injection_pieces"], inventory)
        withdraw_limit = ratchet_limit(environment["withdrawal_pieces"], inventory)
        q_lo = torch.maximum(-withdraw_limit, environment["lower"][d + 1] - inventory)
        q_hi = torch.minimum(inject_limit, environment["upper"][d + 1] - inventory)
        # Guard against rounding (the interval is never empty inside the band).
        q_hi = torch.maximum(q_hi, q_lo)

        fraction = network(day_features(environment, d, inventory, chi[:, d], xi[:, d]))
        q = q_lo + fraction * (q_hi - q_lo)

        cash = (-spot[:, d] * q
                - contract.injection_cost * torch.relu(q)
                - contract.withdrawal_cost * torch.relu(-q))
        path_values = path_values + environment["discount"][d] * cash
        inventory = inventory + q
        volumes.append(q)

    return path_values, torch.stack(volumes, dim=1)


def simulate_batch(scenario, n_paths, rng, dtype):
    """Fresh paths as torch tensors (spot, chi, xi)."""
    chi, xi = simulate_factors(scenario["params"], scenario["times"], n_paths, rng)
    spot = spot_prices(scenario["seasonal"], chi, xi)
    return (torch.tensor(spot, dtype=dtype), torch.tensor(chi, dtype=dtype),
            torch.tensor(xi, dtype=dtype))


def evaluate_policy(network, scenario, paths):
    """Out-of-sample evaluation in float64.

    The volumes come from the network; the value is recomputed in NumPy from
    those volumes with the same cash-flow function as every other method.
    Returns (path_values, volumes) as NumPy arrays.
    """
    network64 = copy.deepcopy(network).double()
    environment = build_environment(scenario, torch.float64)
    with torch.no_grad():
        _, volumes = rollout(network64, environment,
                             torch.tensor(paths["spot"], dtype=torch.float64),
                             torch.tensor(paths["chi"], dtype=torch.float64),
                             torch.tensor(paths["xi"], dtype=torch.float64))
    volumes = volumes.numpy()
    path_values = schedule_discounted_cash_flows(scenario["contract"], paths["spot"], volumes,
                                                 scenario["discount"]).sum(axis=1)
    return path_values, volumes


def train_policy(scenario, settings, validation_paths, verbose=True):
    """Train the policy network by stochastic gradient ascent on the mean value.

    validation_paths (independent seed) are used only to monitor training and to
    keep the best weights; the final value is always measured on separate
    valuation paths.
    """
    torch.manual_seed(int(settings["seed_torch"]))
    rng = np.random.default_rng(int(settings["seed_training"]))
    dtype = torch.float32
    environment = build_environment(scenario, dtype)
    wgv = scenario["contract"].working_volume

    network = PolicyNetwork(N_FEATURES, list(settings["hidden_sizes"]))
    optimiser = torch.optim.Adam(network.parameters(), lr=float(settings["learning_rate"]))
    n_iterations = int(settings["n_iterations"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=n_iterations, eta_min=float(settings["learning_rate"]) / 20.0)

    history = []
    best_value = -np.inf
    best_state = None
    start = time.perf_counter()

    for iteration in range(1, n_iterations + 1):
        spot, chi, xi = simulate_batch(scenario, int(settings["batch_size"]), rng, dtype)
        path_values, _ = rollout(network, environment, spot, chi, xi)
        # Loss in EUR per MWh of WGV (order 10) for a well-scaled gradient.
        loss = -path_values.mean() / wgv
        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), float(settings["gradient_clip"]))
        optimiser.step()
        scheduler.step()

        if iteration % int(settings["validate_every"]) == 0 or iteration == n_iterations:
            validation_values, _ = evaluate_policy(network, scenario, validation_paths)
            validation_mean = validation_values.mean()
            history.append({"iteration": iteration, "train_value": -loss.item() * wgv,
                            "validation_value": validation_mean,
                            "elapsed_s": time.perf_counter() - start})
            if validation_mean > best_value:
                best_value = validation_mean
                best_state = copy.deepcopy(network.state_dict())
            if verbose:
                print("  iteration {:5d}  batch value {:12,.0f}  validation {:12,.0f}  ({:.0f} s)".format(
                    iteration, -loss.item() * wgv, validation_mean, time.perf_counter() - start),
                    flush=True)

    network.load_state_dict(best_state)
    return network, history
