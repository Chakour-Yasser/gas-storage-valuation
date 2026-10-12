"""Step 2: intrinsic value by linear programming, with validation.

Run from the repository root:
    python -m scripts.run_step2_intrinsic

Checks:
    1. feasibility of the optimal schedule, rebuilt independently of the LP;
    2. LP value = cash flows of the schedule recomputed outside the LP;
    3. dynamic programming on an inventory grid converges to the LP value from below;
    4. the fixed schedule applied to simulated spot paths has MC value equal to
       the LP value (E[S_d] = F(0, d), so the expectation is linear), with 95% CI.
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config_loader import REPO_ROOT, load_config
from src.evaluation import mean_confidence_interval, print_table, save_table_csv
from src.intrinsic import (intrinsic_value_dp, schedule_discounted_cash_flows,
                           schedule_violations, solve_intrinsic)
from src.scenario import build_scenario, simulate_spot

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


def monthly_positions(scenario, solution):
    """Monthly volumes of the intrinsic schedule: the forward hedge a desk would trade."""
    contract = scenario["contract"]
    month_index = scenario["month_index"]
    cash = schedule_discounted_cash_flows(contract, scenario["daily_forward"],
                                          solution["net_volume"], scenario["discount"])
    rows = []
    for m in range(len(scenario["months"])):
        in_month = month_index == m
        injected = solution["injection"][in_month].sum()
        withdrawn = solution["withdrawal"][in_month].sum()
        end_inventory = solution["inventory"][1:][in_month][-1]
        rows.append({
            "month": scenario["month_labels"][m],
            "forward": scenario["monthly_prices"][m],
            "injected_MWh": injected,
            "withdrawn_MWh": withdrawn,
            "net_MWh": injected - withdrawn,
            "end_fill_pct": 100.0 * end_inventory / contract.working_volume,
            "pv_cash_EUR": cash[in_month].sum(),
        })
    return rows


def plot_schedule(scenario, solution, path):
    contract = scenario["contract"]
    dates = scenario["dates"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.0, 1.2]})

    ax = axes[0]
    ax.step(dates, scenario["daily_forward"], where="post", color=INK, linewidth=2.0)
    ax.set_ylabel("Forward (EUR/MWh)")
    ax.set_title("Intrinsic schedule on the forward curve", color=INK, loc="left")
    style_axes(ax)

    ax = axes[1]
    ax.bar(dates, solution["injection"] / 1000.0, width=1.0, color=BLUE, label="Injection")
    ax.bar(dates, -solution["withdrawal"] / 1000.0, width=1.0, color=ORANGE, label="Withdrawal")
    ax.axhline(0.0, color=INK_SECONDARY, linewidth=0.8)
    ax.set_ylabel("Daily volume (GWh/day)")
    style_axes(ax)
    ax.legend(frameon=False, loc="upper right")

    ax = axes[2]
    fill_pct = 100.0 * solution["inventory"][1:] / contract.working_volume
    ax.plot(dates, fill_pct, color=BLUE, linewidth=2.0)
    # Ratchet breakpoints: injection slows above 70%, withdrawal slows below 30%.
    ax.axhline(70.0, color=INK_SECONDARY, linewidth=0.8, linestyle="--")
    ax.axhline(30.0, color=INK_SECONDARY, linewidth=0.8, linestyle="--")
    ax.text(dates[3], 72.0, "injection ratchet starts (70%)", color=INK_SECONDARY, fontsize=9)
    ax.text(dates[3], 32.0, "withdrawal ratchet starts (30%)", color=INK_SECONDARY, fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Inventory, end of day (% WGV)")
    ax.set_xlabel("Delivery date")
    style_axes(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_dp_convergence(dp_rows, path):
    steps = []
    gaps = []
    for row in dp_rows:
        steps.append(row["grid_step_MWh"])
        gaps.append(row["gap_pct"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(steps, gaps, color=BLUE, linewidth=2.0, marker="o", markersize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    # Plain tick labels at the grid steps used, instead of log-scale exponents.
    ax.minorticks_off()
    ax.set_xticks(steps)
    step_labels = []
    for step in steps:
        step_labels.append("{:g}".format(step))
    ax.set_xticklabels(step_labels)
    ax.set_yticks([0.1, 0.2, 0.5, 1.0])
    ax.set_yticklabels(["0.1", "0.2", "0.5", "1.0"])
    ax.set_xlabel("Inventory grid step (MWh), finer to the right")
    ax.set_ylabel("(LP - DP) / LP  (%)")
    ax.set_title("DP on an inventory grid converges to the LP value", color=INK, loc="left")
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

    # --- Solve the LP --------------------------------------------------------
    solution = solve_intrinsic(contract, scenario["daily_forward"], scenario["discount"])
    lp_value = solution["value"]
    print("=== Intrinsic LP ===")
    print("Variables / constraints : {} / {}".format(solution["n_variables"],
                                                     solution["n_constraints"]))
    print("Solve time (HiGHS)      : {:.3f} s".format(solution["solve_time"]))
    print("Intrinsic value         : {:,.0f} EUR".format(lp_value))
    print("Per MWh of WGV          : {:.4f} EUR/MWh".format(lp_value / contract.working_volume))
    print("Total injected          : {:,.0f} MWh".format(solution["injection"].sum()))
    print("Total withdrawn         : {:,.0f} MWh".format(solution["withdrawal"].sum()))
    print("Max fill level          : {:.2f}% of WGV".format(
        100.0 * solution["inventory"].max() / contract.working_volume))
    print()

    # --- Check 1: feasibility ----------------------------------------------
    violations = schedule_violations(contract, solution["net_volume"])
    both = np.minimum(solution["injection"], solution["withdrawal"]).max()
    print("=== Check 1: constraint violations of the schedule (MWh) ===")
    for name in violations:
        print("  {:20s} {:.2e}".format(name, violations[name]))
    print("  {:20s} {:.2e}".format("inject_and_withdraw", both))
    print()

    # --- Check 2: value from recomputed cash flows -------------------------
    cash = schedule_discounted_cash_flows(contract, scenario["daily_forward"],
                                          solution["net_volume"], scenario["discount"])
    print("=== Check 2: value recomputed from cash flows ===")
    print("Sum of discounted cash flows : {:,.2f} EUR".format(cash.sum()))
    print("Difference with LP objective : {:.2e} EUR".format(cash.sum() - lp_value))
    print()

    # --- Check 3: DP convergence --------------------------------------------
    dp_rows = []
    for step in config["intrinsic"]["dp_grid_steps"]:
        start = time.perf_counter()
        dp_value = intrinsic_value_dp(contract, scenario["daily_forward"], scenario["discount"],
                                      float(step))
        elapsed = time.perf_counter() - start
        dp_rows.append({
            "grid_step_MWh": float(step),
            "dp_value_EUR": dp_value,
            "gap_EUR": lp_value - dp_value,
            "gap_pct": 100.0 * (lp_value - dp_value) / lp_value,
            "time_s": elapsed,
        })
    dp_cols = ["grid_step_MWh", "dp_value_EUR", "gap_EUR", "gap_pct", "time_s"]
    print("=== Check 3: DP on an inventory grid (lower bound of the LP) ===")
    print_table(dp_rows, dp_cols)
    save_table_csv(dp_rows, dp_cols, TABLE_DIR / "step2_dp_convergence.csv")
    print()

    # --- Check 4: MC value of the fixed schedule on simulated paths ---------
    n_paths = int(config["simulation"]["n_paths_check"])
    seed = int(config["simulation"]["seed"])
    spot = simulate_spot(scenario, n_paths, seed)
    path_values = schedule_discounted_cash_flows(contract, spot, solution["net_volume"],
                                                 scenario["discount"]).sum(axis=1)
    ci = mean_confidence_interval(path_values, level)
    z = (ci["mean"] - lp_value) / ci["std_error"]
    print("=== Check 4: fixed intrinsic schedule on {} simulated spot paths ===".format(n_paths))
    print("MC mean value  : {:,.0f} EUR   {:.0%} CI [{:,.0f}, {:,.0f}]".format(
        ci["mean"], level, ci["low"], ci["high"]))
    print("LP value       : {:,.0f} EUR   z = {:.2f}   inside CI: {}".format(
        lp_value, z, ci["low"] <= lp_value <= ci["high"]))
    print("Std of path PV : {:,.0f} EUR  (spot risk of the schedule if left unhedged)".format(
        path_values.std(ddof=1)))
    print()

    # --- Monthly positions ---------------------------------------------------
    pos_rows = monthly_positions(scenario, solution)
    pos_cols = ["month", "forward", "injected_MWh", "withdrawn_MWh", "net_MWh",
                "end_fill_pct", "pv_cash_EUR"]
    print("=== Intrinsic schedule by month (forward hedge) ===")
    print_table(pos_rows, pos_cols)
    save_table_csv(pos_rows, pos_cols, TABLE_DIR / "step2_intrinsic_monthly.csv")
    print()

    plot_schedule(scenario, solution, FIGURE_DIR / "step2_intrinsic_schedule.png")
    plot_dp_convergence(dp_rows, FIGURE_DIR / "step2_dp_convergence.png")
    print("Figures saved to", FIGURE_DIR)
    print("Tables saved to", TABLE_DIR)


if __name__ == "__main__":
    main()
