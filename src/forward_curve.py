"""Monthly forward curve and the daily delivery calendar.

The input curve gives one price per delivery month. On a daily grid we use a
step curve: every delivery day of a month has the forward price of that month.
The average of the daily forwards over a month is then exactly the monthly
forward, which is what the monthly futures contract delivers.
"""

from datetime import timedelta

import numpy as np

from src.config_loader import parse_date

DAYS_PER_YEAR = 365.0  # ACT/365 day count


def parse_month(text):
    """Parse 'YYYY-MM' into a (year, month) tuple."""
    parts = text.split("-")
    if len(parts) != 2:
        raise ValueError("Month must be 'YYYY-MM', got " + repr(text))
    year = int(parts[0])
    month = int(parts[1])
    if month < 1 or month > 12:
        raise ValueError("Invalid month in " + repr(text))
    return (year, month)


def next_month(year_month):
    """Return the calendar month that follows (year, month)."""
    year, month = year_month
    if month == 12:
        return (year + 1, 1)
    return (year, month + 1)


def month_label(year_month):
    """Return 'YYYY-MM' for a (year, month) tuple."""
    year, month = year_month
    return str(year) + "-" + str(month).zfill(2)


def monthly_curve_from_config(config):
    """Read the monthly forward curve from the config.

    Returns
    -------
    months : list of (year, month) tuples, consecutive calendar months
    prices : np.ndarray of monthly forward prices (EUR/MWh), all > 0
    """
    curve_cfg = config["forward_curve"]
    months = []
    for text in curve_cfg["months"]:
        months.append(parse_month(text))
    prices = np.array(curve_cfg["prices"], dtype=float)

    if len(months) != len(prices):
        raise ValueError("forward_curve: months and prices have different lengths")
    if np.any(prices <= 0.0):
        raise ValueError("forward_curve: prices must be strictly positive")
    for i in range(1, len(months)):
        if months[i] != next_month(months[i - 1]):
            raise ValueError("forward_curve: months must be consecutive")
    return months, prices


def delivery_dates(start_date, end_date):
    """List of all delivery days from start_date to end_date (both included)."""
    if end_date < start_date:
        raise ValueError("end_date is before start_date")
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current = current + timedelta(days=1)
    return dates


def delivery_dates_from_config(config):
    """Delivery days and valuation date from the [dates] section."""
    start_date = parse_date(config["dates"]["start_date"])
    end_date = parse_date(config["dates"]["end_date"])
    valuation_date = parse_date(config["dates"]["valuation_date"])
    if valuation_date >= start_date:
        raise ValueError("valuation_date must be strictly before start_date")
    return delivery_dates(start_date, end_date), valuation_date


def year_fractions(dates, valuation_date):
    """Time in years from the valuation date to each date (ACT/365)."""
    times = np.empty(len(dates))
    for i in range(len(dates)):
        times[i] = (dates[i] - valuation_date).days / DAYS_PER_YEAR
    return times


def month_index_of_days(dates, months):
    """For each delivery day, the index of its month in the list `months`.

    Raises an error if a day falls in a month not covered by the curve.
    """
    position = {}
    for i in range(len(months)):
        position[months[i]] = i

    index = np.empty(len(dates), dtype=int)
    for d in range(len(dates)):
        key = (dates[d].year, dates[d].month)
        if key not in position:
            raise ValueError("No forward price for delivery month " + month_label(key))
        index[d] = position[key]
    return index


def daily_forward_curve(month_index, monthly_prices):
    """Step curve: the forward of each delivery day is its month's forward."""
    return monthly_prices[month_index]


def monthly_averages(daily_values, month_index, n_months):
    """Average daily values over the days of each month.

    daily_values has shape (..., n_days); the result has shape (..., n_months).
    Months with no delivery day get NaN.
    """
    daily_values = np.asarray(daily_values, dtype=float)
    out_shape = daily_values.shape[:-1] + (n_months,)
    averages = np.full(out_shape, np.nan)
    for m in range(n_months):
        in_month = month_index == m
        if np.any(in_month):
            averages[..., m] = daily_values[..., in_month].mean(axis=-1)
    return averages
