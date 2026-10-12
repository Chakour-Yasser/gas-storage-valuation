"""Build the common market and contract setup used by every step.

All methods (intrinsic, LSMC, neural policy) start from the same scenario and
the same simulated paths, so their results are directly comparable.
"""

import numpy as np

from src.contract import contract_from_config, discount_factors
from src.forward_curve import (daily_forward_curve, delivery_dates_from_config,
                               month_index_of_days, month_label, monthly_curve_from_config,
                               year_fractions)
from src.model import params_from_config, seasonal_term, simulate_factors, spot_prices


def build_scenario(config):
    """Contract, model, calendar, forward curve and discount factors."""
    contract = contract_from_config(config)
    params = params_from_config(config)
    months, monthly_prices = monthly_curve_from_config(config)
    dates, valuation_date = delivery_dates_from_config(config)
    times = year_fractions(dates, valuation_date)
    month_index = month_index_of_days(dates, months)
    daily_forward = daily_forward_curve(month_index, monthly_prices)

    month_labels = []
    for m in months:
        month_labels.append(month_label(m))

    scenario = {
        "contract": contract,
        "params": params,
        "months": months,
        "month_labels": month_labels,
        "monthly_prices": monthly_prices,
        "dates": dates,
        "valuation_date": valuation_date,
        "times": times,
        "month_index": month_index,
        "daily_forward": daily_forward,
        "seasonal": seasonal_term(params, times, daily_forward),
        "discount": discount_factors(contract, times),
    }
    return scenario


def simulate_paths(scenario, n_paths, seed):
    """Simulate the factors and spot prices, each of shape (n_paths, n_days).

    The factors are needed by methods that read the forward curve along the
    path (rolling intrinsic); the same seed gives the same spot as simulate_spot.
    """
    rng = np.random.default_rng(seed)
    chi, xi = simulate_factors(scenario["params"], scenario["times"], n_paths, rng)
    return {"chi": chi, "xi": xi, "spot": spot_prices(scenario["seasonal"], chi, xi)}


def simulate_spot(scenario, n_paths, seed):
    """Simulate spot price paths of shape (n_paths, n_days) with a fixed seed."""
    return simulate_paths(scenario, n_paths, seed)["spot"]
