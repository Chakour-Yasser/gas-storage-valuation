"""Step 1 validation: contract, forward curve and Schwartz-Smith simulation.

Run from the repository root:
    python -m scripts.run_step1_checks

Outputs:
    results/tables/step1_*.csv
    results/figures/step1_*.png
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config_loader import REPO_ROOT, load_config, parse_date
from src.contract import (contract_from_config, days_to_empty, days_to_fill, is_concave,
                          max_injection, max_withdrawal)
from src.evaluation import (check_log_variance, check_monthly_means, mean_confidence_interval,
                            print_table, save_table_csv)
from src.forward_curve import (daily_forward_curve, delivery_dates_from_config,
                               month_index_of_days, month_label, monthly_averages,
                               monthly_curve_from_config, year_fractions)
from src.model import (futures_price, params_from_config, seasonal_term, simulate_factors,
                       spot_prices, variance_log_factors)

FIGURE_DIR = REPO_ROOT / "results" / "figures"
TABLE_DIR = REPO_ROOT / "results" / "tables"

# Colours: validated categorical palette (slots 1-3) and neutral inks.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"


def style_axes(ax):
    """Recessive grid and spines."""
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def report_contract(contract):
    print("=== Contract ===")
    print("Working volume (MWh)          : {:,.0f}".format(contract.working_volume))
    print("Max injection rate (MWh/day)  : {:,.1f}".format(contract.max_injection_rate))
    print("Max withdrawal rate (MWh/day) : {:,.1f}".format(contract.max_withdrawal_rate))
    print("Days to fill from empty       : {}".format(days_to_fill(contract)))
    print("Days to empty from full       : {}".format(days_to_empty(contract)))
    inj_concave = is_concave(contract.injection_fill_levels, contract.injection_fractions)
    wd_concave = is_concave(contract.withdrawal_fill_levels, contract.withdrawal_fractions)
    print("Injection ratchet concave     : {}".format(inj_concave))
    print("Withdrawal ratchet concave    : {}".format(wd_concave))
    print()


def run_martingale_check(params, seasonal, times, dates, chi, xi, month_index,
                         months, monthly_prices, check_date, level):
    """Futures prices are Q-martingales: E[F(t, T)] = F(0, T) for t < T.

    At the check date we compute, path by path, the model monthly futures of
    every month that starts after it, and compare their MC mean to the input.
    """
    obs = dates.index(check_date)
    t_obs = times[obs]
    chi_t = chi[:, obs:obs + 1]
    xi_t = xi[:, obs:obs + 1]

    later_days = np.arange(obs + 1, len(dates))
    tau = times[later_days] - t_obs
    daily_futures = futures_price(params, seasonal[later_days], tau, chi_t, xi_t)
    monthly_futures = monthly_averages(daily_futures, month_index[later_days], len(months))

    rows = []
    for m in range(len(months)):
        first_day_of_month = np.argmax(month_index == m)
        if first_day_of_month <= obs:
            continue  # month already (partly) delivered
        ci = mean_confidence_interval(monthly_futures[:, m], level)
        rows.append({
            "month": month_label(months[m]),
            "forward_t0": monthly_prices[m],
            "mc_mean_F_t": ci["mean"],
            "ci_low": ci["low"],
            "ci_high": ci["high"],
            "forward_in_ci": ci["low"] <= monthly_prices[m] <= ci["high"],
        })
    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_paths(dates, daily_forward, spot, n_plot, path):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    q05 = np.quantile(spot, 0.05, axis=0)
    q95 = np.quantile(spot, 0.95, axis=0)
    ax.fill_between(dates, q05, q95, color=BLUE, alpha=0.12, linewidth=0,
                    label="5%-95% quantile band")
    for i in range(n_plot):
        if i == 0:
            label = "Simulated spot paths"
        else:
            label = None
        ax.plot(dates, spot[i], color=BLUE, linewidth=0.8, alpha=0.55, label=label)
    # Forward curve drawn thick first, MC mean thin on top so both stay visible.
    ax.step(dates, daily_forward, where="post", color=INK, linewidth=3.0,
            label="Input forward curve")
    ax.plot(dates, spot.mean(axis=0), color=ORANGE, linewidth=1.2,
            label="Monte Carlo mean of spot")
    ax.set_title("Schwartz-Smith spot price paths vs forward curve", color=INK, loc="left")
    ax.set_xlabel("Delivery date")
    ax.set_ylabel("Price (EUR/MWh)")
    style_axes(ax)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_monthly_means(rows, path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(rows))
    labels = []
    forwards = []
    means = []
    err_low = []
    err_high = []
    for row in rows:
        labels.append(row["month"])
        forwards.append(row["forward"])
        means.append(row["mc_mean"])
        err_low.append(row["mc_mean"] - row["ci_low"])
        err_high.append(row["ci_high"] - row["mc_mean"])
    ax.scatter(x, forwards, marker="_", s=400, color=INK, linewidths=2.0,
               label="Input monthly forward", zorder=3)
    ax.errorbar(x, means, yerr=[err_low, err_high], fmt="o", color=BLUE, markersize=5,
                capsize=4, linewidth=1.5, label="MC mean of monthly average spot (95% CI)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Price (EUR/MWh)")
    ax.set_title("Monthly check: E[average spot over month] = monthly forward",
                 color=INK, loc="left")
    style_axes(ax)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_log_variance(dates, theory, rows, path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    mc = []
    low = []
    high = []
    for row in rows:
        mc.append(row["mc_variance"])
        low.append(row["ci_low"])
        high.append(row["ci_high"])
    ax.fill_between(dates, low, high, color=BLUE, alpha=0.18, linewidth=0,
                    label="95% CI of empirical variance")
    ax.plot(dates, mc, color=BLUE, linewidth=1.2, label="Empirical Var[ln S_t]")
    ax.plot(dates, theory, color=INK, linewidth=2.0, linestyle="--",
            label="Theoretical Var[ln S_t]")
    ax.set_xlabel("Delivery date")
    ax.set_ylabel("Variance of log price")
    ax.set_title("Variance of log spot price: simulation vs closed form", color=INK, loc="left")
    style_axes(ax)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ratchets(contract, path):
    fill = np.linspace(0.0, 1.0, 201)
    inventory = fill * contract.working_volume
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(100 * fill, max_injection(contract, inventory), color=BLUE, linewidth=2.0)
    ax.plot(100 * fill, max_withdrawal(contract, inventory), color=ORANGE, linewidth=2.0)
    # Direct labels (two series) instead of colour alone.
    ax.text(35, contract.max_injection_rate * 1.04, "Max injection", color=INK)
    ax.text(55, contract.max_withdrawal_rate * 1.03, "Max withdrawal", color=INK)
    ax.set_xlabel("Fill level at start of day (% of working volume)")
    ax.set_ylabel("Rate (MWh/day)")
    ax.set_ylim(0, contract.max_withdrawal_rate * 1.15)
    ax.set_title("Inventory-dependent ratchets", color=INK, loc="left")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config()
    sim_cfg = config["simulation"]
    level = float(sim_cfg["confidence_level"])
    n_paths = int(sim_cfg["n_paths_check"])

    contract = contract_from_config(config)
    params = params_from_config(config)
    months, monthly_prices = monthly_curve_from_config(config)
    dates, valuation_date = delivery_dates_from_config(config)
    times = year_fractions(dates, valuation_date)
    month_index = month_index_of_days(dates, months)
    daily_forward = daily_forward_curve(month_index, monthly_prices)
    seasonal = seasonal_term(params, times, daily_forward)

    report_contract(contract)
    print("=== Simulation ===")
    print("Delivery days : {} ({} to {})".format(len(dates), dates[0], dates[-1]))
    print("Paths         : {}, seed {}".format(n_paths, sim_cfg["seed"]))
    print()

    rng = np.random.default_rng(int(sim_cfg["seed"]))
    chi, xi = simulate_factors(params, times, n_paths, rng)
    spot = spot_prices(seasonal, chi, xi)
    log_spot = seasonal + chi + xi

    # Check 1: monthly means vs forward curve.
    month_labels = []
    for m in months:
        month_labels.append(month_label(m))
    monthly_spot = monthly_averages(spot, month_index, len(months))
    mean_rows = check_monthly_means(monthly_spot, monthly_prices, month_labels, level)
    mean_cols = ["month", "forward", "mc_mean", "ci_low", "ci_high", "rel_error_pct",
                 "z_score", "forward_in_ci"]
    print("=== Check 1: monthly average spot vs input forward ({:.0%} CI) ===".format(level))
    print_table(mean_rows, mean_cols)
    save_table_csv(mean_rows, mean_cols, TABLE_DIR / "step1_monthly_means.csv")
    print()

    # Check 2: variance of log prices vs closed form, every delivery day.
    theory_var = variance_log_factors(params, times)
    date_labels = []
    for d in dates:
        date_labels.append(d.isoformat())
    var_rows = check_log_variance(log_spot, theory_var, date_labels, level)
    var_cols = ["date", "theory", "mc_variance", "ci_low", "ci_high", "rel_error_pct",
                "theory_in_ci"]
    save_table_csv(var_rows, var_cols, TABLE_DIR / "step1_log_variance.csv")
    first_of_month_rows = []
    n_inside = 0
    max_abs_rel_error = 0.0
    for row in var_rows:
        if row["date"].endswith("-01"):
            first_of_month_rows.append(row)
        if row["theory_in_ci"]:
            n_inside += 1
        max_abs_rel_error = max(max_abs_rel_error, abs(row["rel_error_pct"]))
    print("=== Check 2: Var[ln S_t], simulated vs theoretical (first day of each month) ===")
    print_table(first_of_month_rows, var_cols)
    print("Days with theoretical variance inside the {:.0%} CI: {} / {} ({:.1f}%)".format(
        level, n_inside, len(var_rows), 100.0 * n_inside / len(var_rows)))
    print("Max |relative error| over all days: {:.2f}%".format(max_abs_rel_error))
    print("(Days are strongly correlated along a path, so the coverage is noisy.)")
    print()

    # Check 3: futures martingale property at the end of the injection season.
    check_date = parse_date(sim_cfg["martingale_check_date"])
    mart_rows = run_martingale_check(params, seasonal, times, dates, chi, xi, month_index,
                                     months, monthly_prices, check_date, level)
    mart_cols = ["month", "forward_t0", "mc_mean_F_t", "ci_low", "ci_high", "forward_in_ci"]
    print("=== Check 3: E[F({}, T)] = F(0, T) for winter months ===".format(check_date))
    print_table(mart_rows, mart_cols)
    save_table_csv(mart_rows, mart_cols, TABLE_DIR / "step1_futures_martingale.csv")
    print()

    # Figures.
    plot_paths(dates, daily_forward, spot, int(sim_cfg["n_paths_plot"]),
               FIGURE_DIR / "step1_paths_vs_forward.png")
    plot_monthly_means(mean_rows, FIGURE_DIR / "step1_monthly_means.png")
    plot_log_variance(dates, theory_var, var_rows, FIGURE_DIR / "step1_log_variance.png")
    plot_ratchets(contract, FIGURE_DIR / "step1_ratchets.png")
    print("Figures saved to", FIGURE_DIR)
    print("Tables saved to", TABLE_DIR)


if __name__ == "__main__":
    main()
