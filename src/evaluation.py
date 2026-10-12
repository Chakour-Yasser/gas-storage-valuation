"""Monte Carlo statistics, validation checks and table output.

Every Monte Carlo estimate in the project goes through mean_confidence_interval
so it is always reported with a confidence interval.
"""

import csv

import numpy as np
from scipy import stats


def mean_confidence_interval(samples, level=0.95):
    """Sample mean with a CLT confidence interval.

    CI = mean +/- z_{(1+level)/2} * std / sqrt(n)
    """
    samples = np.asarray(samples, dtype=float)
    n = samples.size
    if n < 2:
        raise ValueError("need at least two samples")
    mean = samples.mean()
    std_error = samples.std(ddof=1) / np.sqrt(n)
    z = stats.norm.ppf(0.5 + level / 2.0)
    result = {
        "mean": mean,
        "std_error": std_error,
        "low": mean - z * std_error,
        "high": mean + z * std_error,
        "n": n,
    }
    return result


def control_variate_confidence_interval(samples, control, control_mean, level=0.95):
    """Mean of `samples` with a control variate whose expectation is known exactly.

        Y_cv = Y - beta (X - E[X]),   beta = Cov(Y, X) / Var(X)

    Y_cv has the same mean as Y and a smaller variance (by a factor 1 - corr^2).
    Estimating beta on the same sample adds a bias of order 1/n, negligible here.
    """
    samples = np.asarray(samples, dtype=float)
    control = np.asarray(control, dtype=float)
    covariance = np.cov(samples, control)
    beta = covariance[0, 1] / covariance[1, 1]
    adjusted = samples - beta * (control - control_mean)
    result = mean_confidence_interval(adjusted, level)
    result["beta"] = beta
    result["correlation"] = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])
    return result


def difference_confidence_interval(samples_a, samples_b, level=0.95, paired=True):
    """Mean of a - b with a confidence interval.

    paired=True: a and b are evaluated on the same paths, so the CI uses the
    variance of the pathwise differences (common randomness cancels).
    paired=False: independent samples, standard errors add in quadrature.
    """
    a = np.asarray(samples_a, dtype=float)
    b = np.asarray(samples_b, dtype=float)
    if paired:
        return mean_confidence_interval(a - b, level)
    ci_a = mean_confidence_interval(a, level)
    ci_b = mean_confidence_interval(b, level)
    mean = ci_a["mean"] - ci_b["mean"]
    std_error = np.sqrt(ci_a["std_error"] ** 2 + ci_b["std_error"] ** 2)
    z = stats.norm.ppf(0.5 + level / 2.0)
    return {"mean": mean, "std_error": std_error, "low": mean - z * std_error,
            "high": mean + z * std_error, "n": min(a.size, b.size)}


def ratio_confidence_interval(samples_a, samples_b, level=0.95):
    """Ratio of means E[a] / E[b] with a delta-method confidence interval.

    R = mean(a) / mean(b),
    Var(R) ~ [Var(a) - 2 R Cov(a, b) + R^2 Var(b)] / (n mean(b)^2).
    """
    a = np.asarray(samples_a, dtype=float)
    b = np.asarray(samples_b, dtype=float)
    n = a.size
    ratio = a.mean() / b.mean()
    covariance = np.cov(a, b)
    variance = (covariance[0, 0] - 2.0 * ratio * covariance[0, 1]
                + ratio ** 2 * covariance[1, 1]) / (n * b.mean() ** 2)
    std_error = np.sqrt(variance)
    z = stats.norm.ppf(0.5 + level / 2.0)
    return {"ratio": ratio, "std_error": std_error, "low": ratio - z * std_error,
            "high": ratio + z * std_error, "n": n}


def variance_confidence_interval(samples, level=0.95):
    """Sample variance with the exact chi-square confidence interval.

    Valid for Gaussian samples (as log prices are in this model):
        (n - 1) s^2 / sigma^2 ~ chi2(n - 1)
    """
    samples = np.asarray(samples, dtype=float)
    n = samples.size
    variance = samples.var(ddof=1)
    alpha = 1.0 - level
    low = (n - 1) * variance / stats.chi2.ppf(1.0 - alpha / 2.0, n - 1)
    high = (n - 1) * variance / stats.chi2.ppf(alpha / 2.0, n - 1)
    return {"variance": variance, "low": low, "high": high, "n": n}


# ---------------------------------------------------------------------------
# Step 1 checks
# ---------------------------------------------------------------------------

def check_monthly_means(monthly_average_prices, monthly_forwards, month_labels, level=0.95):
    """Compare the MC mean of the monthly average spot price with the forward.

    monthly_average_prices : (n_paths, n_months), average spot price over the
        delivery days of each month, path by path.
    Under Q, E[average of S over the month] = monthly forward exactly.
    """
    rows = []
    for m in range(len(month_labels)):
        ci = mean_confidence_interval(monthly_average_prices[:, m], level)
        forward = monthly_forwards[m]
        row = {
            "month": month_labels[m],
            "forward": forward,
            "mc_mean": ci["mean"],
            "ci_low": ci["low"],
            "ci_high": ci["high"],
            "rel_error_pct": 100.0 * (ci["mean"] - forward) / forward,
            "z_score": (ci["mean"] - forward) / ci["std_error"],
            "forward_in_ci": ci["low"] <= forward <= ci["high"],
        }
        rows.append(row)
    return rows


def check_log_variance(log_prices, theoretical_variance, labels, level=0.95):
    """Compare the empirical variance of ln S_t with the theoretical one, per date."""
    rows = []
    for i in range(len(labels)):
        ci = variance_confidence_interval(log_prices[:, i], level)
        theory = theoretical_variance[i]
        row = {
            "date": labels[i],
            "theory": theory,
            "mc_variance": ci["variance"],
            "ci_low": ci["low"],
            "ci_high": ci["high"],
            "rel_error_pct": 100.0 * (ci["variance"] - theory) / theory,
            "theory_in_ci": ci["low"] <= theory <= ci["high"],
        }
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def format_value(value):
    """Format numbers with 4 significant decimals, keep other values as text."""
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "NO"
    if isinstance(value, (float, np.floating)):
        return "{:.4f}".format(value)
    return str(value)


def print_table(rows, columns):
    """Print a list of dicts as an aligned text table."""
    widths = {}
    for col in columns:
        width = len(col)
        for row in rows:
            width = max(width, len(format_value(row[col])))
        widths[col] = width

    header = ""
    for col in columns:
        header += col.rjust(widths[col] + 2)
    print(header)
    for row in rows:
        line = ""
        for col in columns:
            line += format_value(row[col]).rjust(widths[col] + 2)
        print(line)


def save_table_csv(rows, columns, path):
    """Write a list of dicts to a CSV file."""
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for row in rows:
            values = []
            for col in columns:
                values.append(format_value(row[col]))
            writer.writerow(values)
