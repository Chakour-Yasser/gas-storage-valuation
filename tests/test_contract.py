"""Tests of the storage contract: ratchets, cash flows, discounting."""

import copy

import numpy as np
import pytest

from src.config_loader import load_config
from src.contract import (contract_from_config, daily_cash_flow, days_to_empty, days_to_fill,
                          discount_factors, is_concave, max_injection, max_withdrawal,
                          ratchet_affine_pieces)


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def contract(config):
    return contract_from_config(config)


def test_max_rates(contract):
    assert contract.max_injection_rate == pytest.approx(1e6 / 150.0)
    assert contract.max_withdrawal_rate == pytest.approx(1e6 / 90.0)


def test_injection_ratchet_values(contract):
    wgv = contract.working_volume
    rate = contract.max_injection_rate
    assert max_injection(contract, 0.0) == pytest.approx(rate)
    assert max_injection(contract, 0.7 * wgv) == pytest.approx(rate)
    assert max_injection(contract, 0.85 * wgv) == pytest.approx(0.75 * rate)
    assert max_injection(contract, wgv) == pytest.approx(0.5 * rate)


def test_withdrawal_ratchet_values(contract):
    wgv = contract.working_volume
    rate = contract.max_withdrawal_rate
    assert max_withdrawal(contract, 0.0) == pytest.approx(0.4 * rate)
    assert max_withdrawal(contract, 0.15 * wgv) == pytest.approx(0.7 * rate)
    assert max_withdrawal(contract, 0.3 * wgv) == pytest.approx(rate)
    assert max_withdrawal(contract, wgv) == pytest.approx(rate)


def test_ratchets_are_vectorised(contract):
    inventory = np.linspace(0.0, contract.working_volume, 11)
    assert max_injection(contract, inventory).shape == (11,)
    assert max_withdrawal(contract, inventory).shape == (11,)


def test_concave_ratchets_equal_min_of_affine_pieces(contract):
    """Needed for the LP: rate(V) = min_k (a_k + b_k V) when the ratchet is concave."""
    inventory = np.linspace(0.0, contract.working_volume, 1001)
    cases = [
        (contract.injection_fill_levels, contract.injection_fractions,
         contract.max_injection_rate, max_injection),
        (contract.withdrawal_fill_levels, contract.withdrawal_fractions,
         contract.max_withdrawal_rate, max_withdrawal),
    ]
    for levels, fractions, max_rate, rate_function in cases:
        assert is_concave(levels, fractions)
        pieces = ratchet_affine_pieces(levels, fractions, max_rate, contract.working_volume)
        lower_envelope = np.full(inventory.shape, np.inf)
        for intercept, slope in pieces:
            lower_envelope = np.minimum(lower_envelope, intercept + slope * inventory)
        np.testing.assert_allclose(lower_envelope, rate_function(contract, inventory), rtol=1e-12)


def test_is_concave_detects_convex_shape():
    assert not is_concave(np.array([0.0, 0.5, 1.0]), np.array([1.0, 0.5, 1.0]))


def test_days_to_fill_and_empty_close_to_continuous_time(contract):
    """Continuous-time fill time: 0.7*150 + 150*(0.3/0.5) ln 2 = 167.4 days.
    Empty time: 0.7*90 + 90*(0.3/0.6) ln(1/0.4) = 104.2 days.
    The daily discretisation (rate fixed at start of day) must be close."""
    fill_continuous = 0.7 * 150 + 150 * (0.3 / 0.5) * np.log(2.0)
    empty_continuous = 0.7 * 90 + 90 * (0.3 / 0.6) * np.log(1.0 / 0.4)
    assert abs(days_to_fill(contract) - fill_continuous) <= 2
    assert abs(days_to_empty(contract) - empty_continuous) <= 2


def test_storage_year_allows_full_cycle(contract):
    """Summer (Apr-Sep, 183 days) is long enough to fill, winter (182 days) to empty."""
    assert days_to_fill(contract) < 183
    assert days_to_empty(contract) < 182


def test_cash_flow_signs(contract):
    price = 30.0
    inject = daily_cash_flow(contract, price, 100.0)
    withdraw = daily_cash_flow(contract, price, -100.0)
    assert inject == pytest.approx(-(30.0 + 0.20) * 100.0)
    assert withdraw == pytest.approx((30.0 - 0.10) * 100.0)
    assert daily_cash_flow(contract, price, 0.0) == 0.0


def test_discount_factors(contract):
    df = discount_factors(contract, np.array([0.0, 1.0]))
    np.testing.assert_allclose(df, [1.0, np.exp(-0.03)])


def test_invalid_contract_raises(config):
    bad = copy.deepcopy(config)
    bad["contract"]["initial_inventory"] = 2e6
    with pytest.raises(ValueError):
        contract_from_config(bad)
    bad = copy.deepcopy(config)
    bad["contract"]["injection_ratchet"]["fill_levels"] = [0.0, 0.8, 0.7]
    with pytest.raises(ValueError):
        contract_from_config(bad)
