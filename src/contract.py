"""Gas storage contract: volumes, ratchets, costs and discounting.

Conventions
-----------
- Daily decisions. On day d the holder chooses a volume change q_d (MWh):
  q_d > 0 is injection (buy gas), q_d < 0 is withdrawal (sell gas).
- Inventory dynamics: V_{d+1} = V_d + q_d, with 0 <= V_d <= WGV.
- Ratchets: the maximum injection and withdrawal on day d depend on the
  inventory at the START of the day, V_d, through piecewise-linear functions
  of the fill level V_d / WGV.
- Cash flow on day d, paid on the delivery day:
      injection  (q > 0): -(S_d + c_inj) * q
      withdrawal (q < 0): (S_d - c_wd) * (-q)
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class StorageContract:
    working_volume: float
    initial_inventory: float
    final_inventory: float
    max_injection_rate: float       # MWh/day at ratchet fraction 1
    max_withdrawal_rate: float      # MWh/day at ratchet fraction 1
    injection_fill_levels: np.ndarray
    injection_fractions: np.ndarray
    withdrawal_fill_levels: np.ndarray
    withdrawal_fractions: np.ndarray
    injection_cost: float
    withdrawal_cost: float
    discount_rate: float


def validate_ratchet(fill_levels, fractions, name):
    """Check that a ratchet table is a valid piecewise-linear function on [0, 1]."""
    if len(fill_levels) != len(fractions) or len(fill_levels) < 2:
        raise ValueError(name + ": need at least two points, same length")
    if fill_levels[0] != 0.0 or fill_levels[-1] != 1.0:
        raise ValueError(name + ": fill levels must start at 0 and end at 1")
    for i in range(1, len(fill_levels)):
        if fill_levels[i] <= fill_levels[i - 1]:
            raise ValueError(name + ": fill levels must be strictly increasing")
    for value in fractions:
        if value <= 0.0 or value > 1.0:
            raise ValueError(name + ": rate fractions must be in (0, 1]")


def contract_from_config(config):
    """Build the contract from the [contract] section and validate it."""
    c = config["contract"]
    wgv = float(c["working_volume"])

    inj_levels = np.array(c["injection_ratchet"]["fill_levels"], dtype=float)
    inj_fractions = np.array(c["injection_ratchet"]["rate_fractions"], dtype=float)
    wd_levels = np.array(c["withdrawal_ratchet"]["fill_levels"], dtype=float)
    wd_fractions = np.array(c["withdrawal_ratchet"]["rate_fractions"], dtype=float)
    validate_ratchet(inj_levels, inj_fractions, "injection_ratchet")
    validate_ratchet(wd_levels, wd_fractions, "withdrawal_ratchet")

    contract = StorageContract(
        working_volume=wgv,
        initial_inventory=float(c["initial_inventory"]),
        final_inventory=float(c["final_inventory"]),
        max_injection_rate=wgv / float(c["full_injection_days"]),
        max_withdrawal_rate=wgv / float(c["full_withdrawal_days"]),
        injection_fill_levels=inj_levels,
        injection_fractions=inj_fractions,
        withdrawal_fill_levels=wd_levels,
        withdrawal_fractions=wd_fractions,
        injection_cost=float(c["injection_cost"]),
        withdrawal_cost=float(c["withdrawal_cost"]),
        discount_rate=float(c["discount_rate"]),
    )

    if wgv <= 0.0:
        raise ValueError("working_volume must be > 0")
    for name in ["initial_inventory", "final_inventory"]:
        value = getattr(contract, name)
        if value < 0.0 or value > wgv:
            raise ValueError(name + " must be in [0, working_volume]")
    if contract.injection_cost < 0.0 or contract.withdrawal_cost < 0.0:
        raise ValueError("costs must be >= 0")
    return contract


# ---------------------------------------------------------------------------
# Ratchets
# ---------------------------------------------------------------------------

def max_injection(contract, inventory):
    """Maximum injection (MWh/day) for a start-of-day inventory (vectorised)."""
    fill = np.asarray(inventory, dtype=float) / contract.working_volume
    fraction = np.interp(fill, contract.injection_fill_levels, contract.injection_fractions)
    return fraction * contract.max_injection_rate


def max_withdrawal(contract, inventory):
    """Maximum withdrawal (MWh/day, positive number) for a start-of-day inventory."""
    fill = np.asarray(inventory, dtype=float) / contract.working_volume
    fraction = np.interp(fill, contract.withdrawal_fill_levels, contract.withdrawal_fractions)
    return fraction * contract.max_withdrawal_rate


def ratchet_affine_pieces(fill_levels, fractions, max_rate, working_volume):
    """Affine pieces a_k + b_k * V (V in MWh) of a piecewise-linear ratchet.

    If the ratchet is concave, it equals the minimum of its affine pieces,
    so the constraint q <= rate(V) becomes the linear constraints
    q <= a_k + b_k V for every k. This keeps the intrinsic problem an LP.
    """
    pieces = []
    for k in range(len(fill_levels) - 1):
        slope_per_fill = (fractions[k + 1] - fractions[k]) / (fill_levels[k + 1] - fill_levels[k])
        intercept_fraction = fractions[k] - slope_per_fill * fill_levels[k]
        intercept = intercept_fraction * max_rate
        slope = slope_per_fill * max_rate / working_volume
        pieces.append((intercept, slope))
    return pieces


def is_concave(fill_levels, fractions):
    """True if the piecewise-linear function has non-increasing slopes."""
    previous_slope = np.inf
    for k in range(len(fill_levels) - 1):
        slope = (fractions[k + 1] - fractions[k]) / (fill_levels[k + 1] - fill_levels[k])
        if slope > previous_slope + 1e-12:
            return False
        previous_slope = slope
    return True


def days_to_fill(contract, start_inventory=0.0):
    """Number of days to reach full WGV by injecting at the maximum rate daily."""
    inventory = start_inventory
    days = 0
    while inventory < contract.working_volume - 1e-9:
        rate = float(max_injection(contract, inventory))
        inventory = min(inventory + rate, contract.working_volume)
        days += 1
    return days


def days_to_empty(contract, start_inventory=None):
    """Number of days to empty the storage by withdrawing at the maximum rate daily."""
    if start_inventory is None:
        start_inventory = contract.working_volume
    inventory = start_inventory
    days = 0
    while inventory > 1e-9:
        rate = float(max_withdrawal(contract, inventory))
        inventory = max(inventory - rate, 0.0)
        days += 1
    return days


# ---------------------------------------------------------------------------
# Cash flows and discounting
# ---------------------------------------------------------------------------

def daily_cash_flow(contract, price, volume_change):
    """Cash flow of one day for a volume change q (q > 0 inject, q < 0 withdraw).

        cash = -S q - c_inj max(q, 0) - c_wd max(-q, 0)
    """
    price = np.asarray(price, dtype=float)
    q = np.asarray(volume_change, dtype=float)
    injected = np.maximum(q, 0.0)
    withdrawn = np.maximum(-q, 0.0)
    return -price * q - contract.injection_cost * injected - contract.withdrawal_cost * withdrawn


def discount_factors(contract, times):
    """Discount factors exp(-r t) for payment times t in years."""
    return np.exp(-contract.discount_rate * np.asarray(times, dtype=float))
