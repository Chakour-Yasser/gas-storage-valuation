"""Step 3: Least-Squares Monte Carlo valuation (Boogert and de Jong, 2008).

Run from the repository root:
    python -m scripts.run_step3_lsmc

The policy is fitted on regression paths (seed_regression) and always valued
on fresh, independent paths (valuation seed). Reported:
    - in-sample and out-of-sample values with 95% CIs (in-sample is biased upward);
    - out-of-sample value with the intrinsic schedule as a control variate;
    - extrinsic value = LSMC - intrinsic;
    - variants (grid, actions, basis, paths) as paired differences vs the base case;
    - perfect-foresight upper bound on a subset of the valuation paths;
    - feasibility of every out-of-sample schedule.
"""

import copy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config_loader import REPO_ROOT, load_config
from src.evaluation import (control_variate_confidence_interval, difference_confidence_interval,
                            mean_confidence_interval, print_table, save_table_csv)
from src.intrinsic import schedule_discounted_cash_flows, schedule_violations, solve_intrinsic
from src.lsmc import lsmc_backward, lsmc_forward
from src.scenario import build_scenario, simulate_paths

FIGURE_DIR = REPO_ROOT / "results" / "figures"
TABLE_DIR = REPO_ROOT / "results" / "tables"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
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


def run_lsmc(scenario, settings, regression_paths, valuation_paths):
    """Fit on the regression paths, value on the valuation paths.

    Paths are dicts with spot, chi and xi (the factors feed the "factors" basis).
    """
    contract = scenario["contract"]
    policy, in_sample = lsmc_backward(contract, regression_paths["spot"], scenario["discount"],
                                      settings, factors=regression_paths)
    out_of_sample, volumes, actions = lsmc_forward(contract, valuation_paths["spot"],
                                                   scenario["discount"], policy,
                                                   factors=valuation_paths)
    return {"policy": policy, "in_sample": in_sample, "out_of_sample": out_of_sample,
            "volumes": volumes, "actions": actions}


def interior_action_share(actions, n_fractions):
    """Share of injection/withdrawal decisions that are partial volumes.

    Candidate k = 0 is q_lo (withdraw at max), k = n_fractions - 1 is q_hi
    (inject at max), the last index is "do nothing"; anything else is partial.
    """
    is_trade = actions != n_fractions
    is_interior = (actions > 0) & (actions < n_fractions - 1)
    return is_interior.sum() / max(is_trade.sum(), 1)


def plot_inventory(scenario, base, intrinsic_solution, path):
    contract = scenario["contract"]
    dates = scenario["dates"]
    end_inventory = contract.initial_inventory + np.cumsum(base["volumes"], axis=1)
    fill = 100.0 * end_inventory / contract.working_volume
    fig, ax = plt.subplots(figsize=(10, 5.0))
    ax.fill_between(dates, np.quantile(fill, 0.05, axis=0), np.quantile(fill, 0.95, axis=0),
                    color=BLUE, alpha=0.15, linewidth=0, label="LSMC 5%-95% band")
    for i in range(6):
        if i == 0:
            label = "LSMC inventory, sample paths"
        else:
            label = None
        ax.plot(dates, fill[i], color=BLUE, linewidth=0.8, alpha=0.6, label=label)
    ax.plot(dates, np.median(fill, axis=0), color=BLUE, linewidth=2.0, label="LSMC median")
    ax.plot(dates, 100.0 * intrinsic_solution["inventory"][1:] / contract.working_volume,
            color=INK, linewidth=2.0, linestyle="--", label="Intrinsic schedule")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Inventory, end of day (% WGV)")
    ax.set_xlabel("Delivery date")
    ax.set_title("LSMC inventory paths (out-of-sample) vs intrinsic schedule",
                 color=INK, loc="left")
    style_axes(ax)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_values(value_rows, path):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    y = np.arange(len(value_rows))
    labels = []
    for i in range(len(value_rows)):
        row = value_rows[i]
        labels.append(row["estimate"])
        mean = row["value_EUR"] / 1e6
        low = row["ci_low"] / 1e6
        high = row["ci_high"] / 1e6
        ax.errorbar([mean], [y[i]], xerr=[[mean - low], [high - mean]], fmt="o", color=BLUE,
                    markersize=7, capsize=4, linewidth=1.5)
        ax.text(high + 0.03, y[i] + 0.18, "{:.2f}".format(mean), color=INK, fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Value (million EUR), 95% CI")
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
    base_settings = config["lsmc"]

    # Intrinsic benchmark (Step 2).
    intrinsic_solution = solve_intrinsic(contract, scenario["daily_forward"], scenario["discount"])
    intrinsic_value = intrinsic_solution["value"]

    # Independent path sets.
    regression_paths = simulate_paths(scenario, int(base_settings["n_paths_regression"]),
                                      int(base_settings["seed_regression"]))
    valuation_paths = simulate_paths(scenario, int(config["valuation"]["n_paths"]),
                                     int(config["valuation"]["seed"]))
    valuation_spot = valuation_paths["spot"]
    # Control variate: the fixed intrinsic schedule on the valuation paths.
    # Its expectation is exactly the intrinsic value since E[S_d] = F(0, d).
    intrinsic_on_paths = schedule_discounted_cash_flows(
        contract, valuation_spot, intrinsic_solution["net_volume"], scenario["discount"]).sum(axis=1)

    # --- Base case -----------------------------------------------------------
    base = run_lsmc(scenario, base_settings, regression_paths, valuation_paths)
    ci_in = mean_confidence_interval(base["in_sample"], level)
    ci_out = mean_confidence_interval(base["out_of_sample"], level)
    ci_cv = control_variate_confidence_interval(base["out_of_sample"], intrinsic_on_paths,
                                                intrinsic_value, level)
    bias = difference_confidence_interval(base["in_sample"], base["out_of_sample"], level,
                                          paired=False)
    extrinsic = control_variate_confidence_interval(base["out_of_sample"] - intrinsic_value,
                                                    intrinsic_on_paths, intrinsic_value, level)

    print("=== LSMC base case ===")
    print("Grid: {} levels ({}), actions {}, basis {} degree {}, regression paths {}, valuation paths {}".format(
        base_settings["n_levels"], base_settings["grid_type"], base_settings["action_fractions"],
        base_settings["basis"], base_settings["polynomial_degree"], base_settings["n_paths_regression"],
        config["valuation"]["n_paths"]))
    print("Fit time: {:.1f} s".format(base["policy"]["fit_time"]))
    fmt = "{:40s} {:>14,.0f}   [{:>12,.0f}, {:>12,.0f}]"
    print(fmt.format("In-sample value (biased up)", ci_in["mean"], ci_in["low"], ci_in["high"]))
    print(fmt.format("Out-of-sample value", ci_out["mean"], ci_out["low"], ci_out["high"]))
    print(fmt.format("Out-of-sample, control variate", ci_cv["mean"], ci_cv["low"], ci_cv["high"]))
    print("  (control variate: corr = {:.3f}, beta = {:.3f}, CI width / {:.1f})".format(
        ci_cv["correlation"], ci_cv["beta"], ci_out["std_error"] / ci_cv["std_error"]))
    print(fmt.format("Intrinsic value (LP, exact)", intrinsic_value, intrinsic_value, intrinsic_value))
    print(fmt.format("Extrinsic = LSMC OOS - intrinsic", extrinsic["mean"], extrinsic["low"],
                     extrinsic["high"]))
    print("  extrinsic / intrinsic = {:.1f}%".format(100.0 * extrinsic["mean"] / intrinsic_value))
    print(fmt.format("In-sample - out-of-sample", bias["mean"], bias["low"], bias["high"]))
    print()

    violations = schedule_violations(contract, base["volumes"])
    print("=== Feasibility of the {} out-of-sample schedules (max violation, MWh) ===".format(
        base["volumes"].shape[0]))
    for name in violations:
        print("  {:20s} {:.2e}".format(name, violations[name]))
    print()

    # --- Perfect-foresight upper bound --------------------------------------
    # Solving the LP on each realised spot path gives the value with perfect
    # knowledge of the future: an upper bound for any non-anticipative policy.
    n_pf = int(config["valuation"]["n_paths_perfect_foresight"])
    pf_values = np.empty(n_pf)
    for i in range(n_pf):
        pf_values[i] = solve_intrinsic(contract, valuation_spot[i], scenario["discount"])["value"]
    ci_pf = mean_confidence_interval(pf_values, level)
    pf_gap = difference_confidence_interval(pf_values, base["out_of_sample"][:n_pf], level)
    print("=== Perfect-foresight upper bound ({} valuation paths) ===".format(n_pf))
    print(fmt.format("Perfect foresight value", ci_pf["mean"], ci_pf["low"], ci_pf["high"]))
    print(fmt.format("Perfect foresight - LSMC (paired)", pf_gap["mean"], pf_gap["low"],
                     pf_gap["high"]))
    print()

    # --- Variants, paired against the base case on the same valuation paths --
    variants = [
        ("fixed grid [0, WGV] + penalty", {"grid_type": "fixed"}),
        ("5 actions (0, 1/4, 1/2, 3/4, 1)", {"action_fractions": [0.0, 0.25, 0.5, 0.75, 1.0]}),
        ("51 levels", {"n_levels": 51}),
        ("201 levels", {"n_levels": 201}),
        ("spot basis, degree 3", {"basis": "spot"}),
        ("degree 2", {"polynomial_degree": 2}),
        ("degree 4", {"polynomial_degree": 4}),
        ("2,500 regression paths", {"n_paths_regression": 2500}),
        ("5,000 regression paths", {"n_paths_regression": 5000}),
        ("20,000 regression paths", {"n_paths_regression": 20000}),
    ]
    variant_rows = []
    variant_results = {}
    for name, changes in variants:
        settings = copy.deepcopy(base_settings)
        for key in changes:
            settings[key] = changes[key]
        if settings["n_paths_regression"] != base_settings["n_paths_regression"]:
            paths_fit = simulate_paths(scenario, int(settings["n_paths_regression"]),
                                       int(settings["seed_regression"]))
        else:
            paths_fit = regression_paths
        result = run_lsmc(scenario, settings, paths_fit, valuation_paths)
        variant_results[name] = result
        ci_v = control_variate_confidence_interval(result["out_of_sample"], intrinsic_on_paths,
                                                   intrinsic_value, level)
        diff = difference_confidence_interval(result["out_of_sample"], base["out_of_sample"], level)
        max_violation = max(schedule_violations(contract, result["volumes"]).values())
        variant_rows.append({
            "variant": name,
            "oos_value": ci_v["mean"],
            "ci_low": ci_v["low"],
            "ci_high": ci_v["high"],
            "diff_vs_base": diff["mean"],
            "diff_low": diff["low"],
            "diff_high": diff["high"],
            "max_violation_MWh": max_violation,
            "fit_time_s": result["policy"]["fit_time"],
        })
    variant_cols = ["variant", "oos_value", "ci_low", "ci_high", "diff_vs_base", "diff_low",
                    "diff_high", "max_violation_MWh", "fit_time_s"]
    print("=== Variants (out-of-sample, paired difference vs base, 95% CI) ===")
    print_table(variant_rows, variant_cols)
    save_table_csv(variant_rows, variant_cols, TABLE_DIR / "step3_lsmc_variants.csv")
    five = variant_results["5 actions (0, 1/4, 1/2, 3/4, 1)"]
    print("Partial volumes chosen (5-action variant): {:.2%} of trading decisions".format(
        interior_action_share(five["actions"], 5)))
    print()

    # --- Summary table and figures ------------------------------------------
    value_rows = [
        {"estimate": "Intrinsic (LP)", "value_EUR": intrinsic_value,
         "ci_low": intrinsic_value, "ci_high": intrinsic_value},
        {"estimate": "LSMC out-of-sample", "value_EUR": ci_cv["mean"],
         "ci_low": ci_cv["low"], "ci_high": ci_cv["high"]},
        {"estimate": "LSMC in-sample", "value_EUR": ci_in["mean"],
         "ci_low": ci_in["low"], "ci_high": ci_in["high"]},
        {"estimate": "Perfect foresight", "value_EUR": ci_pf["mean"],
         "ci_low": ci_pf["low"], "ci_high": ci_pf["high"]},
    ]
    save_table_csv(value_rows, ["estimate", "value_EUR", "ci_low", "ci_high"],
                   TABLE_DIR / "step3_values.csv")
    plot_values(value_rows, FIGURE_DIR / "step3_values.png")
    plot_inventory(scenario, base, intrinsic_solution, FIGURE_DIR / "step3_lsmc_inventory.png")
    print("Figures saved to", FIGURE_DIR)
    print("Tables saved to", TABLE_DIR)


if __name__ == "__main__":
    main()
