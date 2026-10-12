"""Tests of the forward curve and the delivery calendar."""

import copy
from datetime import date

import numpy as np
import pytest

from src.config_loader import load_config
from src.forward_curve import (daily_forward_curve, delivery_dates, delivery_dates_from_config,
                               month_index_of_days, monthly_averages,
                               monthly_curve_from_config, year_fractions)


@pytest.fixture(scope="module")
def config():
    return load_config()


def test_storage_year_calendar(config):
    dates, valuation_date = delivery_dates_from_config(config)
    assert dates[0] == date(2027, 4, 1)
    assert dates[-1] == date(2028, 3, 31)
    assert len(dates) == 366  # leap year: contains 29 February 2028
    times = year_fractions(dates, valuation_date)
    assert times[0] == pytest.approx(1.0 / 365.0)
    assert np.all(np.diff(times) > 0.0)


def test_curve_is_summer_cheaper_than_winter(config):
    months, prices = monthly_curve_from_config(config)
    summer = prices[:6].mean()   # April to September
    winter = prices[6:].mean()   # October to March
    assert winter > summer


def test_daily_step_curve_averages_to_monthly_forward(config):
    months, prices = monthly_curve_from_config(config)
    dates, _ = delivery_dates_from_config(config)
    month_index = month_index_of_days(dates, months)
    daily = daily_forward_curve(month_index, prices)
    np.testing.assert_allclose(monthly_averages(daily, month_index, len(months)), prices)
    # Flat within each month.
    for m in range(len(months)):
        assert np.all(daily[month_index == m] == prices[m])


def test_days_per_month(config):
    months, _ = monthly_curve_from_config(config)
    dates, _ = delivery_dates_from_config(config)
    month_index = month_index_of_days(dates, months)
    counts = np.bincount(month_index, minlength=len(months))
    np.testing.assert_array_equal(counts, [30, 31, 30, 31, 31, 30, 31, 30, 31, 31, 29, 31])


def test_monthly_averages_on_paths():
    values = np.array([[1.0, 3.0, 10.0], [2.0, 4.0, 20.0]])
    month_index = np.array([0, 0, 1])
    np.testing.assert_allclose(monthly_averages(values, month_index, 2), [[2.0, 10.0], [3.0, 20.0]])


def test_missing_month_raises():
    dates = delivery_dates(date(2027, 4, 30), date(2027, 5, 1))
    with pytest.raises(ValueError):
        month_index_of_days(dates, [(2027, 4)])


def test_non_consecutive_months_raise(config):
    bad = copy.deepcopy(config)
    bad["forward_curve"]["months"][3] = "2027-12"
    with pytest.raises(ValueError):
        monthly_curve_from_config(bad)
