"""Rolling intrinsic valuation of gas storage.

At each re-optimisation day d (e.g. every 7 days):
  1. read the forward curve implied by the model at t_d,
         F(t_d, T_e) for every remaining delivery day e >= d,
     which for e = d is F(t_d, t_d) = S_d, the spot price known on day d;
  2. re-solve the intrinsic LP from the current inventory V_d to V_final;
  3. execute the volumes of that plan until the next re-optimisation day.

Each new plan can always keep the previous one (it stays feasible), so in
expectation rolling intrinsic is worth at least the intrinsic value.

In practice the desk locks in each plan with forwards and re-hedges at every
re-optimisation. The re-hedging P&L is a sum of forward price changes, which
has zero expectation under Q (futures are martingales), so the expected value
of the strategy equals the expected discounted spot cash flows of the executed
volumes. That is what we compute, path by path.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from src.intrinsic import solve_intrinsic
from src.model import futures_price


def reoptimisation_days(n_days, interval_days):
    """Days on which the LP is re-solved: 0, interval, 2 * interval, ..."""
    days = []
    d = 0
    while d < n_days:
        days.append(d)
        d += interval_days
    return days


def model_forward_curve(params, seasonal, times, d, chi_d, xi_d):
    """Daily forward curve F(t_d, T_e), e = d..N-1, for one path."""
    tau = times[d:] - times[d]
    return futures_price(params, seasonal[d:], tau, chi_d, xi_d)


def rolling_intrinsic_path(contract, params, seasonal, times, discount, chi_path, xi_path,
                           reopt_days):
    """Executed daily volumes of the rolling intrinsic strategy on one path."""
    n_days = len(times)
    reopt_set = set(reopt_days)
    volumes = np.zeros(n_days)
    inventory = contract.initial_inventory
    plan = None
    plan_start = 0

    for d in range(n_days):
        if d in reopt_set:
            forwards = model_forward_curve(params, seasonal, times, d, chi_path[d], xi_path[d])
            # Remove tiny solver rounding (1e-9 MWh) before using the inventory as a bound.
            start_inventory = min(max(inventory, 0.0), contract.working_volume)
            solution = solve_intrinsic(contract, forwards, discount[d:],
                                       initial_inventory=start_inventory)
            plan = solution["net_volume"]
            plan_start = d
        q = plan[d - plan_start]
        volumes[d] = q
        inventory = inventory + q

    return volumes


def rolling_intrinsic_chunk(contract, params, seasonal, times, discount, chi, xi, reopt_days):
    """Volumes for a block of paths (run in one worker process)."""
    n_paths = chi.shape[0]
    volumes = np.empty((n_paths, len(times)))
    for i in range(n_paths):
        volumes[i] = rolling_intrinsic_path(contract, params, seasonal, times, discount,
                                            chi[i], xi[i], reopt_days)
    return volumes


def run_rolling_intrinsic(scenario, chi, xi, interval_days, n_workers=1, n_chunks=None,
                          verbose=False):
    """Rolling intrinsic volumes on all paths, shape (n_paths, n_days).

    Each path needs its own sequence of LPs, so paths are split into chunks and
    solved in parallel processes (n_workers > 1). n_workers = 1 runs serially.
    """
    n_paths = chi.shape[0]
    reopt_days = reoptimisation_days(len(scenario["times"]), interval_days)
    common = (scenario["contract"], scenario["params"], scenario["seasonal"], scenario["times"],
              scenario["discount"])

    if n_workers <= 1:
        return rolling_intrinsic_chunk(*common, chi, xi, reopt_days)

    if n_chunks is None:
        n_chunks = 4 * n_workers
    boundaries = np.linspace(0, n_paths, n_chunks + 1).astype(int)
    volumes = np.empty((n_paths, len(scenario["times"])))

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for c in range(n_chunks):
            start = boundaries[c]
            stop = boundaries[c + 1]
            future = executor.submit(rolling_intrinsic_chunk, *common, chi[start:stop],
                                     xi[start:stop], reopt_days)
            futures[future] = (start, stop)
        n_done = 0
        for future in as_completed(futures):
            start, stop = futures[future]
            volumes[start:stop] = future.result()
            n_done += 1
            if verbose:
                print("  rolling intrinsic: chunk {}/{} done".format(n_done, n_chunks), flush=True)
    return volumes
