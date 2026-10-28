"""Fixed vocabularies for currencies and payment formats.

Nothing here is fitted from the data being processed, so a cut in the data
cannot change a category code. A value outside a vocabulary raises. A new
Variant adds its values here.
"""

from __future__ import annotations

import hashlib
import json


# USD per one unit of currency. Fixed constants taken once from the HI-Small
# transactions that convert into US Dollar. They do not vary with time.
USD_PER_UNIT: dict[str, float] = {
    "Australian Dollar": 0.707814,
    "Bitcoin": 11881.623,
    "Brazil Real": 0.177101,
    "Canadian Dollar": 0.757978,
    "Euro": 1.171783,
    "Mexican Peso": 0.047297,
    "Ruble": 0.012853,
    "Rupee": 0.013616,
    "Saudi Riyal": 0.266588,
    "Shekel": 0.296121,
    "Swiss Franc": 1.092896,
    "UK Pound": 1.291656,
    "US Dollar": 1.0,
    "Yen": 0.009488,
    "Yuan": 0.149307,
}

CURRENCIES: tuple[str, ...] = tuple(sorted(USD_PER_UNIT))

PAYMENT_FORMATS: tuple[str, ...] = (
    "ACH",
    "Bitcoin",
    "Cash",
    "Cheque",
    "Credit Card",
    "Reinvestment",
    "Wire",
)


class VocabularyError(ValueError):
    """Error for a value outside a fixed vocabulary."""


def vocabulary_hash() -> str:
    """Return a short hash of the vocabularies and rates, stored in bundles."""
    payload = json.dumps(
        {"rates": USD_PER_UNIT, "formats": PAYMENT_FORMATS}, sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
