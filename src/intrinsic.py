"""Intrinsic value of a gas storage contract by linear programming.

The intrinsic value is the value of the best schedule fixed today, given the
forward curve. Such a schedule can be locked in by trading forwards, so it is
a lower bound for the storage value. The extra value of reacting to price
moves (the extrinsic value) is captured later by LSMC and the neural policy.

LP formulation (volumes in units of WGV for numerical conditioning)
-------------------------------------------------------------------
Variables, for days d = 0..N-1:
    i_d >= 0   injection on day d
    w_d >= 0   withdrawal on day d
    V_d        inventory at the start of day d, d = 0..N (V_N = end of last day)

Maximise   sum_d DF_d [ (F_d - c_wd) w_d - (F_d + c_inj) i_d ]
subject to V_{d+1} = V_d + i_d - w_d                  (inventory balance)
           0 <= V_d <= 1                              (capacity)
           V_0 = V_init,  V_N = V_final               (hard boundary conditions)
           i_d <= a_k + b_k V_d   for every affine piece k of the injection ratchet
           w_d <= a_k + b_k V_d   for every affine piece k of the withdrawal ratchet

The ratchet constraints are exact only for concave ratchets (then the ratchet
is the minimum of its affine pieces); this is checked before solving.

Injecting and withdrawing on the same day is never strictly better: it pays
both costs and the ratchet limits apply to V_d, not to the trades. The solver
output is checked for it anyway.
"""

import time

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from src.contract import (daily_cash_flow, is_concave, max_injection, max_withdrawal,
                          ratchet_affine_pieces)


def check_ratchets_are_concave(contract):
    """The LP formulation is only exact for concave ratchets."""
    if not is_concave(contract.injection_fill_levels, contract.injection_fractions):
        raise ValueError("Injection ratchet is not concave: the LP would be wrong")
    if not is_concave(contract.withdrawal_fill_levels, contract.withdrawal_fractions):
        raise ValueError("Withdrawal ratchet is not concave: the LP would be wrong")


def build_intrinsic_lp(contract, forward_prices, discount, initial_inventory, final_inventory):
    """Build the LP matrices in the format expected by scipy.optimize.linprog.

    Variable vector x = [i_0..i_{N-1}, w_0..w_{N-1}, V_0..V_N], in units of WGV.
    Returns a dict with c, A_ub, b_ub, A_eq, b_eq, bounds.
    """
    n_days = len(forward_prices)
    wgv = contract.working_volume
    n_vars = 3 * n_days + 1

    def inj(d):
        return d

    def wd(d):
        return n_days + d

    def inv(d):
        return 2 * n_days + d

    # Objective: linprog minimises, so we minimise minus the discounted cash flows.
    # Coefficients are in EUR/MWh (objective = value per MWh of WGV): keeping them
    # of order 10-100 avoids numerical difficulties in the simplex solver.
    c = np.zeros(n_vars)
    for d in range(n_days):
        c[inj(d)] = discount[d] * (forward_prices[d] + contract.injection_cost)
        c[wd(d)] = -discount[d] * (forward_prices[d] - contract.withdrawal_cost)

    # Equality constraints: V_{d+1} - V_d - i_d + w_d = 0.
    eq_rows = []
    eq_cols = []
    eq_vals = []
    for d in range(n_days):
        eq_rows += [d, d, d, d]
        eq_cols += [inv(d + 1), inv(d), inj(d), wd(d)]
        eq_vals += [1.0, -1.0, -1.0, 1.0]
    A_eq = sparse.csr_matrix((eq_vals, (eq_rows, eq_cols)), shape=(n_days, n_vars))
    b_eq = np.zeros(n_days)

    # Inequality constraints from the ratchets, in WGV units:
    # rate(V) = a_k + b_k V (MWh), so  i_d / wgv - (b_k) V_d <= a_k / wgv.
    injection_pieces = ratchet_affine_pieces(contract.injection_fill_levels,
                                             contract.injection_fractions,
                                             contract.max_injection_rate, wgv)
    withdrawal_pieces = ratchet_affine_pieces(contract.withdrawal_fill_levels,
                                              contract.withdrawal_fractions,
                                              contract.max_withdrawal_rate, wgv)
    ub_rows = []
    ub_cols = []
    ub_vals = []
    b_ub = []
    row = 0
    for d in range(n_days):
        for intercept, slope in injection_pieces:
            # slope is in MWh/day per MWh of inventory, so it is unit-free.
            ub_rows += [row, row]
            ub_cols += [inj(d), inv(d)]
            ub_vals += [1.0, -slope]
            b_ub.append(intercept / wgv)
            row += 1
        for intercept, slope in withdrawal_pieces:
            ub_rows += [row, row]
            ub_cols += [wd(d), inv(d)]
            ub_vals += [1.0, -slope]
            b_ub.append(intercept / wgv)
            row += 1
    A_ub = sparse.csr_matrix((ub_vals, (ub_rows, ub_cols)), shape=(row, n_vars))
    b_ub = np.array(b_ub)

    # Bounds: trades >= 0, inventory in [0, 1], fixed start and end inventory.
    bounds = []
    for d in range(2 * n_days):
        bounds.append((0.0, None))
    for d in range(n_days + 1):
        bounds.append((0.0, 1.0))
    bounds[inv(0)] = (initial_inventory / wgv, initial_inventory / wgv)
    bounds[inv(n_days)] = (final_inventory / wgv, final_inventory / wgv)

    return {"c": c, "A_ub": A_ub, "b_ub": b_ub, "A_eq": A_eq, "b_eq": b_eq, "bounds": bounds}


def solve_intrinsic(contract, forward_prices, discount, initial_inventory=None,
                    final_inventory=None):
    """Solve the intrinsic LP.

    Parameters
    ----------
    forward_prices : forward price of each remaining delivery day (EUR/MWh)
    discount : discount factor of each remaining delivery day
    initial_inventory, final_inventory : MWh; default to the contract values.
        initial_inventory is an argument so that rolling intrinsic can re-solve
        from the current inventory later.

    Returns
    -------
    dict with value (EUR), injection, withdrawal, net_volume (MWh per day,
    > 0 injection), inventory (MWh, length N + 1), and solver information.
    """
    check_ratchets_are_concave(contract)
    if initial_inventory is None:
        initial_inventory = contract.initial_inventory
    if final_inventory is None:
        final_inventory = contract.final_inventory

    forward_prices = np.asarray(forward_prices, dtype=float)
    discount = np.asarray(discount, dtype=float)
    n_days = len(forward_prices)
    lp = build_intrinsic_lp(contract, forward_prices, discount, initial_inventory,
                            final_inventory)

    start = time.perf_counter()
    result = linprog(lp["c"], A_ub=lp["A_ub"], b_ub=lp["b_ub"], A_eq=lp["A_eq"],
                     b_eq=lp["b_eq"], bounds=lp["bounds"], method="highs")
    elapsed = time.perf_counter() - start
    if result.status != 0:
        raise RuntimeError("Intrinsic LP failed: " + result.message)

    wgv = contract.working_volume
    injection = result.x[0:n_days] * wgv
    withdrawal = result.x[n_days:2 * n_days] * wgv
    inventory = result.x[2 * n_days:3 * n_days + 1] * wgv

    solution = {
        "value": -result.fun * wgv,   # objective is per MWh of WGV
        "injection": injection,
        "withdrawal": withdrawal,
        "net_volume": injection - withdrawal,
        "inventory": inventory,
        "solve_time": elapsed,
        "n_variables": len(lp["c"]),
        "n_constraints": lp["A_ub"].shape[0] + lp["A_eq"].shape[0],
    }
    return solution


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def schedule_violations(contract, net_volume, initial_inventory=None, final_inventory=None):
    """Largest violation (MWh) of each contract constraint for a daily schedule.

    The inventory path is rebuilt from the net volumes, independently of the
    LP inventory variables. All entries should be ~0 for a feasible schedule.
    net_volume can be one schedule (n_days,) or one schedule per path
    (n_paths, n_days); the maximum violation over all paths is returned.
    """
    if initial_inventory is None:
        initial_inventory = contract.initial_inventory
    if final_inventory is None:
        final_inventory = contract.final_inventory

    net_volume = np.asarray(net_volume, dtype=float)
    # Inventory at the end of each day: V_{d+1} = V_init + sum_{k <= d} q_k.
    end_of_day = initial_inventory + np.cumsum(net_volume, axis=-1)
    start_of_day = np.concatenate(
        [np.full(net_volume.shape[:-1] + (1,), initial_inventory), end_of_day[..., :-1]], axis=-1)
    injection = np.maximum(net_volume, 0.0)
    withdrawal = np.maximum(-net_volume, 0.0)

    violations = {
        "below_zero": max(0.0, -end_of_day.min()),
        "above_wgv": max(0.0, end_of_day.max() - contract.working_volume),
        "injection_ratchet": max(0.0, (injection - max_injection(contract, start_of_day)).max()),
        "withdrawal_ratchet": max(0.0, (withdrawal - max_withdrawal(contract, start_of_day)).max()),
        "final_inventory": np.abs(end_of_day[..., -1] - final_inventory).max(),
    }
    return violations


def schedule_discounted_cash_flows(contract, prices, net_volume, discount):
    """Discounted cash flows of a fixed schedule.

    prices can be the forward curve (n_days,) or simulated spot paths
    (n_paths, n_days); the result has the same shape.
    """
    return discount * daily_cash_flow(contract, prices, net_volume)


# ---------------------------------------------------------------------------
# Independent check: deterministic dynamic programming on an inventory grid
# ---------------------------------------------------------------------------

def intrinsic_value_dp(contract, forward_prices, discount, grid_step):
    """Intrinsic value by backward induction on an inventory grid.

    Inventory takes values j * grid_step, j = 0..M, and daily trades are whole
    multiples of grid_step. Every such schedule is feasible for the LP, so this
    value is a lower bound of the LP value that converges to it as grid_step -> 0.

    Bellman equation (deterministic prices):
        U_N(v) = 0 if v = V_final, -inf otherwise
        U_d(v) = max_q { DF_d cash(F_d, q) + U_{d+1}(v + q) }
        with -W(v) <= q <= I(v) and 0 <= v + q <= WGV.
    """
    wgv = contract.working_volume
    n_levels = int(round(wgv / grid_step))
    if abs(n_levels * grid_step - wgv) > 1e-6:
        raise ValueError("grid_step must divide the working volume")
    levels = np.arange(n_levels + 1) * grid_step

    j_initial = int(round(contract.initial_inventory / grid_step))
    j_final = int(round(contract.final_inventory / grid_step))

    # Largest number of grid steps that can be injected / withdrawn per day.
    max_up = int(np.floor(contract.max_injection_rate / grid_step + 1e-9))
    max_down = int(np.floor(contract.max_withdrawal_rate / grid_step + 1e-9))
    injection_limit = max_injection(contract, levels)
    withdrawal_limit = max_withdrawal(contract, levels)

    value_next = np.full(n_levels + 1, -np.inf)
    value_next[j_final] = 0.0

    for d in range(len(forward_prices) - 1, -1, -1):
        value_today = np.full(n_levels + 1, -np.inf)
        for move in range(-max_down, max_up + 1):
            q = move * grid_step
            # Ratchet feasibility for each start-of-day level (small tolerance
            # so that a trade exactly at the ratchet limit is allowed).
            if q > 0:
                allowed = q <= injection_limit + 1e-6
            elif q < 0:
                allowed = -q <= withdrawal_limit + 1e-6
            else:
                allowed = np.ones(n_levels + 1, dtype=bool)

            # Target level j + move must stay on the grid [0, M].
            target = np.arange(n_levels + 1) + move
            on_grid = (target >= 0) & (target <= n_levels)
            allowed = allowed & on_grid

            candidate = np.full(n_levels + 1, -np.inf)
            cash = discount[d] * float(daily_cash_flow(contract, forward_prices[d], q))
            candidate[allowed] = cash + value_next[target[allowed]]
            value_today = np.maximum(value_today, candidate)
        value_next = value_today

    return value_next[j_initial]
