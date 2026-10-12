"""Least-Squares Monte Carlo valuation of gas storage (Boogert and de Jong, 2008).

State: (day d, inventory V, factors chi_d, xi_d). The regression basis is
polynomials of either the spot price S_d only (basis = "spot") or of the two
factors (basis = "factors"). The spot price only reveals chi + xi, so the spot
basis is an approximation of the conditional expectation; the factor basis
uses the full Markov state of the price model.

Backward induction on in-sample (regression) paths
--------------------------------------------------
All values are present values at the valuation date (cash flows are
multiplied by their discount factors), so no extra discounting is needed.

For each day d = N-1, ..., 0 and each inventory grid level v_g:
  1. Continuation value C_d(v', S_d) = E[ Y_{d+1}(v') | S_d ] is estimated by
     regressing the realised future value Y_{d+1}(v_g') of every path on a
     polynomial basis of S_d, one regression per grid level.
  2. For each candidate action q (inject at max, do nothing, withdraw at max)
     the next inventory v_g + q is usually off-grid; the continuation value is
     interpolated linearly between the two neighbouring grid levels.
  3. The action maximising  DF_d cash(S_d, q) + C_d(v_g + q, S_d)  is chosen
     path by path, and the realised value is updated in the Longstaff-Schwartz
     way:  Y_d(v_g) = DF_d cash(S_d, q*) + Y_{d+1}(v_g + q*)  (interpolated).

Out-of-sample valuation
-----------------------
The regression coefficients define a policy. It is applied to fresh,
independent paths; the mean discounted cash flow is an unbiased estimate of
the value of a feasible policy, i.e. of a lower bound of the true value.
The in-sample estimate is biased upward by foresight and is only reported.

Constraint handling
-------------------
The terminal condition V_N = V_final is hard. We compute the reachable band
[L_d, U_d]: the inventories at the start of day d from which V_final can still
be reached with the ratchets. Every action keeps V_{d+1} inside the band, so the
out-of-sample terminal inventory is exactly V_final (forced withdrawal near
the end). Two grid types:
  - "fixed": levels equally spaced on [0, WGV] every day. Grid levels outside
    the band cannot be brought back; they take the forced action and pay a
    terminal penalty per MWh missed, which keeps their values finite.
  - "adaptive": levels equally spaced on [L_d, U_d] every day, so every grid
    level is feasible and no penalty is needed.
"""

import time
from itertools import combinations_with_replacement

import numpy as np

from src.contract import daily_cash_flow, max_injection, max_withdrawal, ratchet_affine_pieces


# ---------------------------------------------------------------------------
# Reachable inventory band
# ---------------------------------------------------------------------------

def check_rate_slopes(contract):
    """The band computation needs v - W(v) and v + I(v) to be increasing in v,
    i.e. ratchet slopes (MWh/day per MWh of inventory) strictly between -1 and 1."""
    pieces = ratchet_affine_pieces(contract.injection_fill_levels, contract.injection_fractions,
                                   contract.max_injection_rate, contract.working_volume)
    pieces += ratchet_affine_pieces(contract.withdrawal_fill_levels, contract.withdrawal_fractions,
                                    contract.max_withdrawal_rate, contract.working_volume)
    for intercept, slope in pieces:
        if abs(slope) >= 1.0:
            raise ValueError("Ratchet slope too steep for the reachability band")


def largest_start_inventory(contract, target):
    """Largest v in [0, WGV] with v - W(v) <= target (can reach `target` or below in one day)."""
    wgv = contract.working_volume
    if wgv - float(max_withdrawal(contract, wgv)) <= target:
        return wgv
    low = target      # v = target is always feasible since W(v) >= 0
    high = wgv
    for iteration in range(100):
        middle = 0.5 * (low + high)
        if middle - float(max_withdrawal(contract, middle)) <= target:
            low = middle
        else:
            high = middle
    return low


def smallest_start_inventory(contract, target):
    """Smallest v in [0, WGV] with v + I(v) >= target (can reach `target` or above in one day)."""
    if float(max_injection(contract, 0.0)) >= target:
        return 0.0
    low = 0.0
    high = target     # v = target is always feasible since I(v) >= 0
    for iteration in range(100):
        middle = 0.5 * (low + high)
        if middle + float(max_injection(contract, middle)) >= target:
            high = middle
        else:
            low = middle
    return high


def reachable_band(contract, n_days):
    """Bounds [L_d, U_d], d = 0..N, of start-of-day inventories from which the
    final inventory can still be reached by the end of the last day."""
    check_rate_slopes(contract)
    lower = np.empty(n_days + 1)
    upper = np.empty(n_days + 1)
    lower[n_days] = contract.final_inventory
    upper[n_days] = contract.final_inventory
    for d in range(n_days - 1, -1, -1):
        lower[d] = smallest_start_inventory(contract, lower[d + 1])
        upper[d] = largest_start_inventory(contract, upper[d + 1])
    return lower, upper


# ---------------------------------------------------------------------------
# Inventory grid and interpolation
# ---------------------------------------------------------------------------

def build_grid(contract, lower, upper, n_levels, grid_type):
    """Inventory levels of shape (N + 1, n_levels), equally spaced within each day."""
    n_times = len(lower)
    levels = np.empty((n_times, n_levels))
    for d in range(n_times):
        if grid_type == "fixed":
            levels[d] = np.linspace(0.0, contract.working_volume, n_levels)
        elif grid_type == "adaptive":
            levels[d] = np.linspace(lower[d], upper[d], n_levels)
        else:
            raise ValueError("grid_type must be 'fixed' or 'adaptive'")
    return levels


def grid_position(grid_row, inventory):
    """Left neighbour index j and weight w for linear interpolation on a uniform grid:
        value(v) = (1 - w) value[j] + w value[j + 1].
    Inventories slightly outside the grid (rounding) are clipped to its ends.
    """
    inventory = np.asarray(inventory, dtype=float)
    n_levels = len(grid_row)
    step = grid_row[1] - grid_row[0]
    if step <= 0.0:
        # Degenerate grid (all levels equal, e.g. the terminal day of the adaptive grid).
        zeros = np.zeros(inventory.shape)
        return zeros.astype(int), zeros
    x = (inventory - grid_row[0]) / step
    x = np.clip(x, 0.0, n_levels - 1.0)
    j = np.minimum(np.floor(x).astype(int), n_levels - 2)
    w = x - j
    return j, w


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def action_candidates(contract, inventory, lower_next, upper_next, action_fractions):
    """Candidate volume changes q for start-of-day inventories (vectorised).

    The feasible interval is [q_lo, q_hi] with
        q_lo = max(-W(v), L_{d+1} - v),   q_hi = min(I(v), U_{d+1} - v).
    Candidates are q_lo + f (q_hi - q_lo) for f in action_fractions (f = 0 is
    withdraw at max, f = 1 is inject at max, both clipped to the band), plus q = 0
    when it is feasible. If the interval is empty (only for fixed-grid levels
    outside the band) the forced action towards the band is used.

    Returns q of shape (n_actions,) + inventory.shape and a feasibility mask.
    """
    inventory = np.asarray(inventory, dtype=float)
    withdraw_limit = max_withdrawal(contract, inventory)
    inject_limit = max_injection(contract, inventory)
    q_lo = np.maximum(-withdraw_limit, lower_next - inventory)
    q_hi = np.minimum(inject_limit, upper_next - inventory)

    # Outside the band: too high -> withdraw at max, too low -> inject at max.
    too_high = inventory - withdraw_limit > upper_next
    too_low = inventory + inject_limit < lower_next
    q_lo = np.where(too_high, -withdraw_limit, np.where(too_low, inject_limit, q_lo))
    q_hi = np.where(too_high, -withdraw_limit, np.where(too_low, inject_limit, q_hi))

    n_actions = len(action_fractions) + 1
    q = np.empty((n_actions,) + inventory.shape)
    feasible = np.ones((n_actions,) + inventory.shape, dtype=bool)
    for k in range(len(action_fractions)):
        q[k] = q_lo + action_fractions[k] * (q_hi - q_lo)
    # "Do nothing" is the last action.
    q[-1] = 0.0
    feasible[-1] = (q_lo <= 1e-9) & (q_hi >= -1e-9)
    return q, feasible


# ---------------------------------------------------------------------------
# Regression basis
# ---------------------------------------------------------------------------

def polynomial_basis(x, degree):
    """All monomials of total degree <= degree in the columns of x (n_paths, k).

    k = 1: 1, x, x^2, x^3.  k = 2 (degree 2): 1, x1, x2, x1^2, x1 x2, x2^2.
    """
    n_paths, n_variables = x.shape
    columns = [np.ones(n_paths)]
    for total in range(1, degree + 1):
        for combination in combinations_with_replacement(range(n_variables), total):
            column = np.ones(n_paths)
            for variable in combination:
                column = column * x[:, variable]
            columns.append(column)
    return np.column_stack(columns)


def regression_basis(prices, mean, std, degree):
    """Columns 1, x, x^2, ..., x^degree with x = (S - mean) / std.

    Standardising S does not change the span of the basis (same fitted values
    as 1, S, S^2, S^3) but makes the least-squares problem well conditioned.
    """
    x = (prices - mean) / std
    return polynomial_basis(x[:, None], degree)


def state_variables(spot, factors, basis_type):
    """Regression state variables of shape (n_paths, n_days, k).

    basis_type "spot": k = 1, the spot price.
    basis_type "factors": k = 2, the factors chi and xi (factors = dict of arrays).
    """
    if basis_type == "spot":
        return spot[:, :, None]
    if basis_type == "factors":
        if factors is None:
            raise ValueError("basis 'factors' needs the simulated factors chi and xi")
        return np.stack([factors["chi"], factors["xi"]], axis=2)
    raise ValueError("basis must be 'spot' or 'factors'")


def standardised_basis(states_d, mean_d, std_d, degree):
    """Polynomial basis of the standardised state variables of one day."""
    return polynomial_basis((states_d - mean_d) / std_d, degree)


# ---------------------------------------------------------------------------
# Backward induction (in-sample)
# ---------------------------------------------------------------------------

def lsmc_backward(contract, spot, discount, settings, factors=None):
    """Fit the LSMC policy on regression paths.

    settings: dict with n_levels, grid_type, action_fractions, polynomial_degree,
    terminal_penalty, and optionally basis ("spot" by default, or "factors").
    factors: dict with the simulated chi and xi, needed for basis "factors".

    Returns the policy (dict) and the in-sample value estimate.
    """
    start = time.perf_counter()
    n_paths, n_days = spot.shape
    degree = int(settings["polynomial_degree"])
    fractions = list(settings["action_fractions"])
    lower, upper = reachable_band(contract, n_days)
    levels = build_grid(contract, lower, upper, int(settings["n_levels"]), settings["grid_type"])
    n_levels = levels.shape[1]

    # State standardisation for the regression basis, stored for out-of-sample use.
    basis_type = settings.get("basis", "spot")
    states = state_variables(spot, factors, basis_type)
    state_mean = states.mean(axis=0)              # (n_days, k)
    state_std = states.std(axis=0)
    state_std[state_std < 1e-12] = 1.0
    n_basis = polynomial_basis(np.zeros((1, states.shape[2])), degree).shape[1]

    # Terminal values: penalty per MWh away from the required final inventory.
    terminal = -settings["terminal_penalty"] * np.abs(levels[n_days] - contract.final_inventory)
    realised = np.tile(terminal, (n_paths, 1))     # Y_{d+1}(path, level)

    coefficients = np.empty((n_days, n_levels, n_basis))
    for d in range(n_days - 1, -1, -1):
        prices = spot[:, d]
        basis = standardised_basis(states[:, d], state_mean[d], state_std[d], degree)
        # One least-squares fit with n_levels right-hand sides (one per grid level).
        coef, _, _, _ = np.linalg.lstsq(basis, realised, rcond=None)
        coefficients[d] = coef.T
        continuation = basis @ coef               # (n_paths, n_levels), at day d+1 levels

        q, feasible = action_candidates(contract, levels[d], lower[d + 1], upper[d + 1], fractions)
        best_score = np.full((n_paths, n_levels), -np.inf)
        new_realised = np.zeros((n_paths, n_levels))
        for k in range(q.shape[0]):
            # Next inventory for each grid level does not depend on the path.
            j, w = grid_position(levels[d + 1], levels[d] + q[k])
            cont_k = (1.0 - w) * continuation[:, j] + w * continuation[:, j + 1]
            realised_k = (1.0 - w) * realised[:, j] + w * realised[:, j + 1]
            cash_k = discount[d] * daily_cash_flow(contract, prices[:, None], q[k][None, :])
            score = cash_k + cont_k
            score[:, ~feasible[k]] = -np.inf
            better = score > best_score
            best_score = np.where(better, score, best_score)
            new_realised = np.where(better, cash_k + realised_k, new_realised)
        realised = new_realised

    # In-sample value at the initial inventory (interpolated on the day-0 grid).
    j, w = grid_position(levels[0], np.array([contract.initial_inventory]))
    in_sample_paths = (1.0 - w[0]) * realised[:, j[0]] + w[0] * realised[:, j[0] + 1]

    policy = {
        "coefficients": coefficients,
        "basis_type": basis_type,
        "state_mean": state_mean,
        "state_std": state_std,
        "degree": degree,
        "levels": levels,
        "lower": lower,
        "upper": upper,
        "action_fractions": fractions,
        "fit_time": time.perf_counter() - start,
    }
    return policy, in_sample_paths


# ---------------------------------------------------------------------------
# Forward simulation (out-of-sample)
# ---------------------------------------------------------------------------

def lsmc_forward(contract, spot, discount, policy, factors=None):
    """Apply the LSMC policy to (fresh) paths (factors needed for basis "factors").

    Returns the discounted cash flow of each path (n_paths,), the daily
    volume changes (n_paths, n_days) and the index of the chosen candidate
    action (n_paths, n_days); the last index is "do nothing".
    """
    n_paths, n_days = spot.shape
    rows = np.arange(n_paths)
    inventory = np.full(n_paths, contract.initial_inventory)
    path_values = np.zeros(n_paths)
    volumes = np.empty((n_paths, n_days))
    chosen_actions = np.empty((n_paths, n_days), dtype=int)
    states = state_variables(spot, factors, policy["basis_type"])

    for d in range(n_days):
        prices = spot[:, d]
        basis = standardised_basis(states[:, d], policy["state_mean"][d], policy["state_std"][d],
                                   policy["degree"])
        continuation = basis @ policy["coefficients"][d].T     # (n_paths, n_levels)

        q, feasible = action_candidates(contract, inventory, policy["lower"][d + 1],
                                        policy["upper"][d + 1], policy["action_fractions"])
        best_score = np.full(n_paths, -np.inf)
        best_q = np.zeros(n_paths)
        best_k = np.zeros(n_paths, dtype=int)
        for k in range(q.shape[0]):
            j, w = grid_position(policy["levels"][d + 1], inventory + q[k])
            cont_k = (1.0 - w) * continuation[rows, j] + w * continuation[rows, j + 1]
            cash_k = discount[d] * daily_cash_flow(contract, prices, q[k])
            score = cash_k + cont_k
            score[~feasible[k]] = -np.inf
            better = score > best_score
            best_score = np.where(better, score, best_score)
            best_q = np.where(better, q[k], best_q)
            best_k = np.where(better, k, best_k)

        path_values += discount[d] * daily_cash_flow(contract, prices, best_q)
        inventory = inventory + best_q
        volumes[:, d] = best_q
        chosen_actions[:, d] = best_k

    return path_values, volumes, chosen_actions
