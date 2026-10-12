"""Step 5: neural network policy trained by Monte Carlo gradient.

Run from the repository root:
    python -m scripts.run_step5_nn_policy

Paths:
    training   : fresh paths every iteration (seed_training)
    validation : monitoring and best-weights selection (seed_validation)
    valuation  : the same 20,000 paths as every other method (config [valuation])
The reported value is measured on the valuation paths only, and compared with
LSMC and rolling intrinsic through paired differences on the same paths.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config_loader import REPO_ROOT, load_config
from src.evaluation import (control_variate_confidence_interval, difference_confidence_interval,
                            print_table, ratio_confidence_interval, save_table_csv)
from src.intrinsic import schedule_discounted_cash_flows, schedule_violations, solve_intrinsic
from src.lsmc import lsmc_backward, lsmc_forward
from src.nn_policy import evaluate_policy, train_policy
from src.scenario import build_scenario, simulate_paths

FIGURE_DIR = REPO_ROOT / "results" / "figures"
TABLE_DIR = REPO_ROOT / "results" / "tables"
MODEL_DIR = REPO_ROOT / "results" / "models"

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


def plot_training(history, lsmc_validation_value, path):
    iterations = []
    batch_values = []
    validation_values = []
    for row in history:
        iterations.append(row["iteration"])
        batch_values.append(row["train_value"] / 1e6)
        validation_values.append(row["validation_value"] / 1e6)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(iterations, batch_values, color=BLUE, linewidth=1.0, alpha=0.5,
            label="Training batch value (2,048 fresh paths)")
    ax.plot(iterations, validation_values, color=BLUE, linewidth=2.0,
            label="Validation value (5,000 fixed paths)")
    ax.axhline(lsmc_validation_value / 1e6, color=INK, linewidth=1.5, linestyle="--",
               label="LSMC on the same validation paths")
    ax.set_ylim(min(validation_values[2:] + batch_values[2:]) - 0.3,
                max(validation_values + [lsmc_validation_value / 1e6]) + 0.3)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Value (million EUR)")
    ax.set_title("Neural network policy: training curve", color=INK, loc="left")
    style_axes(ax)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_values(value_rows, path):
    fig, ax = plt.subplots(figsize=(8, 4.0))
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
    ax.set_xlabel("Value (million EUR), 95% CI, same 20,000 valuation paths")
    ax.set_title("Storage value by method", color=INK, loc="left")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_inventory(scenario, lsmc_volumes, nn_volumes, path):
    contract = scenario["contract"]
    dates = scenario["dates"]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    series = [("LSMC", lsmc_volumes, BLUE), ("Neural network", nn_volumes, AQUA)]
    for name, volumes, colour in series:
        fill = 100.0 * (contract.initial_inventory + np.cumsum(volumes, axis=1)) / contract.working_volume
        ax.plot(dates, np.median(fill, axis=0), color=colour, linewidth=2.0, label=name + ": median")
        ax.plot(dates, np.quantile(fill, 0.05, axis=0), color=colour, linewidth=1.0, linestyle=":",
                label=name + ": 5% and 95% quantiles")
        ax.plot(dates, np.quantile(fill, 0.95, axis=0), color=colour, linewidth=1.0, linestyle=":")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Inventory, end of day (% WGV)")
    ax.set_xlabel("Delivery date")
    ax.set_title("Inventory under LSMC and the neural network policy (out-of-sample)",
                 color=INK, loc="left")
    style_axes(ax)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def bang_bang_share(scenario, volumes, tolerance=0.02):
    """Share of trading decisions within `tolerance` of a full-rate trade.

    Compares |q| with the maximum rate allowed by the ratchet at the current
    inventory (the network may also trade less because of the band; those
    days count as partial here, so this is a conservative measure)."""
    from src.contract import max_injection, max_withdrawal
    contract = scenario["contract"]
    start = contract.initial_inventory + np.concatenate(
        [np.zeros((volumes.shape[0], 1)), np.cumsum(volumes, axis=1)[:, :-1]], axis=1)
    limit = np.where(volumes > 0, max_injection(contract, start), max_withdrawal(contract, start))
    trading = np.abs(volumes) > 1.0
    full = np.abs(np.abs(volumes) - limit) <= tolerance * limit
    return (full & trading).sum() / max(trading.sum(), 1), trading.mean()


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config()
    scenario = build_scenario(config)
    contract = scenario["contract"]
    level = float(config["simulation"]["confidence_level"])
    settings = config["nn_policy"]

    valuation = simulate_paths(scenario, int(config["valuation"]["n_paths"]),
                               int(config["valuation"]["seed"]))
    validation = simulate_paths(scenario, int(settings["n_paths_validation"]),
                                int(settings["seed_validation"]))

    intrinsic_solution = solve_intrinsic(contract, scenario["daily_forward"], scenario["discount"])
    intrinsic_value = intrinsic_solution["value"]
    intrinsic_on_paths = schedule_discounted_cash_flows(
        contract, valuation["spot"], intrinsic_solution["net_volume"], scenario["discount"]).sum(axis=1)

    # Per-path values of earlier methods on the same valuation paths (Step 4).
    saved = np.load(TABLE_DIR / "step4_path_values.npz")
    if not np.allclose(saved["intrinsic_schedule"], intrinsic_on_paths):
        raise RuntimeError("Saved Step 4 values are not on the same valuation paths: rerun Step 4")
    lsmc_values = saved["lsmc"]
    rolling_values = saved["rolling_weekly"]

    # LSMC policy (config defaults) for reference on the validation paths and inventory plots.
    lsmc_cfg = config["lsmc"]
    regression = simulate_paths(scenario, int(lsmc_cfg["n_paths_regression"]),
                                int(lsmc_cfg["seed_regression"]))
    lsmc_policy, _ = lsmc_backward(contract, regression["spot"], scenario["discount"], lsmc_cfg,
                                   factors=regression)
    lsmc_validation, _, _ = lsmc_forward(contract, validation["spot"], scenario["discount"],
                                         lsmc_policy, factors=validation)
    lsmc_check, lsmc_volumes, _ = lsmc_forward(contract, valuation["spot"], scenario["discount"],
                                               lsmc_policy, factors=valuation)
    if not np.allclose(lsmc_check, lsmc_values):
        raise RuntimeError("LSMC values differ from the saved Step 4 values: rerun Step 4")

    # --- Training --------------------------------------------------------
    print("=== Training: {} iterations x {} paths, network {} ===".format(
        settings["n_iterations"], settings["batch_size"], settings["hidden_sizes"]))
    network, history = train_policy(scenario, settings, validation)
    torch.save(network.state_dict(), MODEL_DIR / "nn_policy.pt")
    history_cols = ["iteration", "train_value", "validation_value", "elapsed_s"]
    save_table_csv(history, history_cols, TABLE_DIR / "step5_training_history.csv")
    best = max(history, key=lambda row: row["validation_value"])
    print("Best validation value {:,.0f} at iteration {} (LSMC on validation paths: {:,.0f})".format(
        best["validation_value"], best["iteration"], lsmc_validation.mean()))
    print("Training time: {:.1f} min".format(history[-1]["elapsed_s"] / 60.0))
    print()

    # --- Out-of-sample evaluation on the valuation paths -------------------
    nn_values, nn_volumes = evaluate_policy(network, scenario, valuation)
    violations = schedule_violations(contract, nn_volumes)
    ci_nn = control_variate_confidence_interval(nn_values, intrinsic_on_paths, intrinsic_value, level)
    ci_lsmc = control_variate_confidence_interval(lsmc_values, intrinsic_on_paths, intrinsic_value, level)
    ci_ri = control_variate_confidence_interval(rolling_values, intrinsic_on_paths, intrinsic_value,
                                                level)
    nn_minus_lsmc = difference_confidence_interval(nn_values, lsmc_values, level)
    nn_minus_ri = difference_confidence_interval(nn_values, rolling_values, level)
    captured = ratio_confidence_interval(nn_values - intrinsic_on_paths,
                                         lsmc_values - intrinsic_on_paths, level)
    full_share, trading_share = bang_bang_share(scenario, nn_volumes)

    fmt = "{:40s} {:>14,.0f}   [{:>12,.0f}, {:>12,.0f}]"
    print("=== Out-of-sample values on the same {} valuation paths (95% CI) ===".format(
        valuation["spot"].shape[0]))
    print(fmt.format("Intrinsic (LP, exact)", intrinsic_value, intrinsic_value, intrinsic_value))
    print(fmt.format("Rolling intrinsic, weekly", ci_ri["mean"], ci_ri["low"], ci_ri["high"]))
    print(fmt.format("LSMC (factor basis)", ci_lsmc["mean"], ci_lsmc["low"], ci_lsmc["high"]))
    print(fmt.format("Neural network policy", ci_nn["mean"], ci_nn["low"], ci_nn["high"]))
    print(fmt.format("NN - LSMC (paired)", nn_minus_lsmc["mean"], nn_minus_lsmc["low"],
                     nn_minus_lsmc["high"]))
    print(fmt.format("NN - rolling intrinsic (paired)", nn_minus_ri["mean"], nn_minus_ri["low"],
                     nn_minus_ri["high"]))
    print("{:40s} {:>13.1%}   [{:>11.1%}, {:>11.1%}]".format(
        "NN extrinsic / LSMC extrinsic", captured["ratio"], captured["low"], captured["high"]))
    print("Pathwise: NN > LSMC on {:.1%} of paths".format(np.mean(nn_values > lsmc_values)))
    print("Max constraint violation: {:.2e} MWh".format(max(violations.values())))
    print("Days with a trade: {:.1%}; trades at full ratchet rate (+/- 2%): {:.1%}".format(
        trading_share, full_share))
    print()

    value_rows = [
        {"method": "Intrinsic (LP)", "value_EUR": intrinsic_value, "ci_low": intrinsic_value,
         "ci_high": intrinsic_value},
        {"method": "Rolling intrinsic (weekly)", "value_EUR": ci_ri["mean"], "ci_low": ci_ri["low"],
         "ci_high": ci_ri["high"]},
        {"method": "LSMC (factor basis)", "value_EUR": ci_lsmc["mean"], "ci_low": ci_lsmc["low"],
         "ci_high": ci_lsmc["high"]},
        {"method": "Neural network policy", "value_EUR": ci_nn["mean"], "ci_low": ci_nn["low"],
         "ci_high": ci_nn["high"]},
    ]
    comparison_rows = [
        {"comparison": "NN - LSMC", "mean_EUR": nn_minus_lsmc["mean"], "ci_low": nn_minus_lsmc["low"],
         "ci_high": nn_minus_lsmc["high"]},
        {"comparison": "NN - rolling intrinsic", "mean_EUR": nn_minus_ri["mean"],
         "ci_low": nn_minus_ri["low"], "ci_high": nn_minus_ri["high"]},
    ]
    print_table(comparison_rows, ["comparison", "mean_EUR", "ci_low", "ci_high"])
    save_table_csv(value_rows, ["method", "value_EUR", "ci_low", "ci_high"],
                   TABLE_DIR / "step5_values.csv")
    save_table_csv(comparison_rows, ["comparison", "mean_EUR", "ci_low", "ci_high"],
                   TABLE_DIR / "step5_comparisons.csv")
    np.savez_compressed(TABLE_DIR / "step5_path_values.npz", nn=nn_values, lsmc=lsmc_values,
                        rolling_weekly=rolling_values, intrinsic_schedule=intrinsic_on_paths)

    plot_training(history, lsmc_validation.mean(), FIGURE_DIR / "step5_training.png")
    plot_values(value_rows, FIGURE_DIR / "step5_values.png")
    plot_inventory(scenario, lsmc_volumes, nn_volumes, FIGURE_DIR / "step5_inventory.png")
    print("Figures saved to", FIGURE_DIR)
    print("Tables saved to", TABLE_DIR)


if __name__ == "__main__":
    main()
