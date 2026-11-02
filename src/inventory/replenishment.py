from __future__ import annotations

import math


def apply_order_constraints(qty: float, moq: float = 0, pack_size: float = 1) -> float:
    """Round up to pack multiples, then enforce the minimum order quantity (also pack-aligned).
    A zero requirement stays zero."""
    if qty <= 0:
        return 0.0
    pack = max(float(pack_size), 1.0)
    q = math.ceil(qty / pack) * pack
    if moq and q < moq:
        q = math.ceil(moq / pack) * pack
    return float(q)


def cap_by_shelf_life(qty: float, inventory_position: float, demand_within_shelf_life: float,
                      pack_size: float = 1) -> tuple[float, bool]:
    """Never order more than can plausibly be sold before expiry. Returns (qty, was_capped)."""
    room = max(0.0, demand_within_shelf_life - inventory_position)
    if qty <= room:
        return qty, False
    pack = max(float(pack_size), 1.0)
    return float(math.floor(room / pack) * pack), True
