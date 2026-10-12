"""Step 4: rolling intrinsic valuation, compared with intrinsic and LSMC.

Run from the repository root:
    python -m scripts.run_step4_rolling_intrinsic

All methods are evaluated on the same valuation paths (config [valuation]).
Reported with 95% CIs:
    - rolling intrinsic value (weekly re-optimisation, all valuation paths);
    - paired differences vs intrinsic and vs LSMC (same paths);
    - share of the LSMC extrinsic value captured by rolling intrinsic;
    - weekly vs daily re-optimisation on a subset of paths.
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config_loader import REPO_ROOT, load_config
from src.evaluation import (control_variate_confidence_interval, difference_confidence_interval,
                            print_table, ratio_confidence_interval, save_table_csv)
from src.intrinsic import schedule_discounted_cash_flows, schedule_violations, solve_intrinsic
from src.lsmc import lsmc_backward, lsmc_forward
from src.rolling_intrinsic import run_rolling_intrinsic
from src.scenario import build_scenario, simulate_paths

FIGURE_DIR = REPO_ROOT / "results" / "figures"
TABLE_DIR = REPO_ROOT / "results" / "tables"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"


def style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY)


def plot_inventory_comparison(scenario, lsmc_volumes, rolling_volumes, intrinsic_solution, path):
    contract = scenario["contract"]
    dates = scenario["dates"]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    series = [("LSMC", lsmc_volumes, BLUE), ("Rolling intrinsic (weekly)", rolling_volumes, ORANGE)]
    for name, volumes, colour in series:
        fill = 100.0 * (contract.initial_inventory + np.cumsum(volumes, axis=1)) / contract.working_volume
        # Quantile lines instead of filled bands: two overlapping fills mix colours.
        ax.plot(dates, np.median(fill, axis=0), color=colour, linewidth=2.0,
                label=name + ": median")
        ax.plot(dates, np.quantile(fill, 0.05, axis=0), color=colour, linewidth=1.0,
                linestyle=":", label=name + ": 5% and 95% quantiles")
        ax.plot(dates, np.quantile(fill, 0.95, axis=0), color=colour, linewidth=1.0,
                linestyle=":")
    ax.plot(dates, 100.0 * intrinsic_solution["inventory"][1:] / contract.working_volume,
            color=INK, linewidth=2.0, linestyle="--", label="Intrinsic schedule")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Inventory, end of day (% WGV)")
    ax.set_xlabel("Delivery date")
    ax.set_title("Inventory under LSMC and rolling intrinsic (out-of-sample)", color=INK, loc="left")
    style_axes(ax)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_values(value_rows, path):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(value_rows))
    labels = []
    for i in range(len(value_rows)):
        row = value_rows[i]
        labels.append(row["method"])
        mean = row["value_EUR"] / 1e6
        low = row["ci_low"] / 1e6
        high = row["ci_high"] / 1e6
        ax.errorbar([mean], [y[i]], xerr=[[mean - low], [high - mean]], fmt="o", color=BLUE,
                    markersize=7, capsize=4, linewidth=1.5)
        ax.text(high + 0.05, y[i], "{:.2f}".format(mean), color=INK, fontsize=9, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    x_min = min(row["ci_low"] for row in value_rows) / 1e6
    x_max = max(row["ci_high"] for row in value_rows) / 1e6
    ax.set_xlim(x_min - 0.2, x_max + 0.4)
    ax.set_ylim(len(value_rows) - 0.5, -0.5)
    ax.set_xlabel("Value (million EUR), 95% CI, same valuation paths")
    ax.set_title("Storage value by method", color=INK, loc="left")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config()
    scenario = build_scenario(config)
    contract = scenario["contract"]
    level = float(config["simulation"]["confidence_level"])
    ri_cfg = config["rolling_intrinsic"]
    interval = int(ri_cfg["reoptimisation_interval_days"])
    n_workers = int(ri_cfg["n_workers"])

    # Same valuation paths as Step 3 (same seed); factors are needed for the forward curve.
    paths = simulate_paths(scenario, int(config["valuation"]["n_paths"]),
                           int(config["valuation"]["seed"]))
    spot = paths["spot"]
    n_paths = spot.shape[0]

    # Intrinsic benchmark and control variate (expectation known exactly).
    intrinsic_solution = solve_intrinsic(contract, scenario["daily_forward"], scenario["discount"])
    intrinsic_value = intrinsic_solution["value"]
    intrinsic_on_paths = schedule_discounted_cash_flows(
        contract, spot, intrinsic_solution["net_volume"], scenario["discount"]).sum(axis=1)

    # LSMC base case (config defaults) on the same valuation paths.
    lsmc_settings = config["lsmc"]
    regression_paths = simulate_paths(scenario, int(lsmc_settings["n_paths_regression"]),
                                      int(lsmc_settings["seed_regression"]))
    policy, _ = lsmc_backward(contract, regression_paths["spot"], scenario["discount"],
                              lsmc_settings, factors=regression_paths)
    lsmc_values, lsmc_volumes, _ = lsmc_forward(contract, spot, scenario["discount"], policy,
                                                factors=paths)

    # --- Rolling intrinsic, all paths ------------------------------------
    print("=== Rolling intrinsic: re-optimisation every {} days, {} paths, {} workers ===".format(
        interval, n_paths, n_workers))
    start = time.perf_counter()
    rolling_volumes = run_rolling_intrinsic(scenario, paths["chi"], paths["xi"], interval,
                                            n_workers=n_workers, verbose=False)
    elapsed = time.perf_counter() - start
    print("Run time: {:.1f} min".format(elapsed / 60.0))
    rolling_values = schedule_discounted_cash_flows(contract, spot, rolling_volumes,
                                                    scenario["discount"]).sum(axis=1)
    violations = schedule_violations(contract, rolling_volumes)
    print("Max constraint violation over all paths: {:.2e} MWh".format(max(violations.values())))
    print()

    # --- Values and paired comparisons -----------------------------------
    ci_ri = control_variate_confidence_interval(rolling_values, intrinsic_on_paths,
                                                intrinsic_value, level)
    ci_lsmc = control_variate_confidence_interval(lsmc_values, intrinsic_on_paths,
                                                  intrinsic_value, level)
    ri_minus_intrinsic = control_variate_confidence_interval(rolling_values - intrinsic_value,
                                                             intrinsic_on_paths, intrinsic_value,
                                                             level)
    lsmc_minus_ri = difference_confidence_interval(lsmc_values, rolling_values, level)
    # Share of LSMC extrinsic value captured: E[RI - I] / E[LSMC - I]. Using the
    # intrinsic schedule on the paths (expectation = I) instead of the constant I
    # gives the same ratio of expectations with much less noise.
    captured = ratio_confidence_interval(rolling_values - intrinsic_on_paths,
                                         lsmc_values - intrinsic_on_paths, level)

    fmt = "{:42s} {:>14,.0f}   [{:>12,.0f}, {:>12,.0f}]"
    print("=== Values on the same {} valuation paths (95% CI, control variate) ===".format(n_paths))
    print(fmt.format("Intrinsic (LP, exact)", intrinsic_value, intrinsic_value, intrinsic_value))
    print(fmt.format("Rolling intrinsic, weekly", ci_ri["mean"], ci_ri["low"], ci_ri["high"]))
    print(fmt.format("LSMC ({} basis)".format(lsmc_settings["basis"]), ci_lsmc["mean"],
                     ci_lsmc["low"], ci_lsmc["high"]))
    print(fmt.format("Rolling intrinsic - intrinsic", ri_minus_intrinsic["mean"],
                     ri_minus_intrinsic["low"], ri_minus_intrinsic["high"]))
    print(fmt.format("LSMC - rolling intrinsic (paired)", lsmc_minus_ri["mean"], lsmc_minus_ri["low"],
                     lsmc_minus_ri["high"]))
    print("{:42s} {:>13.1%}   [{:>11.1%}, {:>11.1%}]".format(
        "Share of LSMC extrinsic captured by RI", captured["ratio"], captured["low"],
        captured["high"]))
    print("Pathwise: RI > LSMC on {:.1%} of paths".format(np.mean(rolling_values > lsmc_values)))
    print()

    # --- Weekly vs daily re-optimisation on a subset ----------------------
    n_daily = int(ri_cfg["n_paths_daily_check"])
    start = time.perf_counter()
    daily_volumes = run_rolling_intrinsic(scenario, paths["chi"][:n_daily], paths["xi"][:n_daily],
                                          1, n_workers=n_workers)
    elapsed = time.perf_counter() - start
    daily_values = schedule_discounted_cash_flows(contract, spot[:n_daily], daily_volumes,
                                                  scenario["discount"]).sum(axis=1)
    daily_minus_weekly = difference_confidence_interval(daily_values, rolling_values[:n_daily], level)
    lsmc_minus_daily = difference_confidence_interval(lsmc_values[:n_daily], daily_values, level)
    print("=== Daily vs weekly re-optimisation (first {} paths, {:.1f} min) ===".format(
        n_daily, elapsed / 60.0))
    print(fmt.format("Daily RI - weekly RI (paired)", daily_minus_weekly["mean"],
                     daily_minus_weekly["low"], daily_minus_weekly["high"]))
    print(fmt.format("LSMC - daily RI (paired)", lsmc_minus_daily["mean"], lsmc_minus_daily["low"],
                     lsmc_minus_daily["high"]))
    print()

    # --- Tables and figures -------------------------------------------------
    value_rows = [
        {"method": "Intrinsic (LP)", "value_EUR": intrinsic_value, "ci_low": intrinsic_value,
         "ci_high": intrinsic_value},
        {"method": "Rolling intrinsic (weekly)", "value_EUR": ci_ri["mean"], "ci_low": ci_ri["low"],
         "ci_high": ci_ri["high"]},
        {"method": "LSMC", "value_EUR": ci_lsmc["mean"], "ci_low": ci_lsmc["low"],
         "ci_high": ci_lsmc["high"]},
    ]
    comparison_rows = [
        {"comparison": "rolling intrinsic - intrinsic", "mean_EUR": ri_minus_intrinsic["mean"],
         "ci_low": ri_minus_intrinsic["low"], "ci_high": ri_minus_intrinsic["high"]},
        {"comparison": "LSMC - rolling intrinsic", "mean_EUR": lsmc_minus_ri["mean"],
         "ci_low": lsmc_minus_ri["low"], "ci_high": lsmc_minus_ri["high"]},
        {"comparison": "daily RI - weekly RI ({} paths)".format(n_daily),
         "mean_EUR": daily_minus_weekly["mean"], "ci_low": daily_minus_weekly["low"],
         "ci_high": daily_minus_weekly["high"]},
    ]
    print_table(comparison_rows, ["comparison", "mean_EUR", "ci_low", "ci_high"])
    save_table_csv(value_rows, ["method", "value_EUR", "ci_low", "ci_high"],
                   TABLE_DIR / "step4_values.csv")
    save_table_csv(comparison_rows, ["comparison", "mean_EUR", "ci_low", "ci_high"],
                   TABLE_DIR / "step4_comparisons.csv")
    # Per-path values, so later methods can be compared with paired differences.
    np.savez_compressed(TABLE_DIR / "step4_path_values.npz", rolling_weekly=rolling_values,
                        lsmc=lsmc_values, intrinsic_schedule=intrinsic_on_paths)
    plot_values(value_rows, FIGURE_DIR / "step4_values.png")
    plot_inventory_comparison(scenario, lsmc_volumes, rolling_volumes, intrinsic_solution,
                              FIGURE_DIR / "step4_inventory_comparison.png")
    print("Figures saved to", FIGURE_DIR)
    print("Tables saved to", TABLE_DIR)


if __name__ == "__main__":
    main()
