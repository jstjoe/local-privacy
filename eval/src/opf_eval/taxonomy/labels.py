"""The canonical label hierarchy.

Two levels:

- **Coarse** — the 15 categories the harness has always scored on. Every
  report defaults to this level, the API's `CanonicalLabel` enum is this set,
  and published results in RESULTS.md are at this level.
- **Fine** — optional sub-types under a coarse parent (e.g. `GOV_ID` and
  `MEDICAL_ID` under `ACCOUNT`, `GIVEN_NAME` under `PERSON`). Newer
  benchmarks annotate at this granularity; scoring at `level="fine"` keeps
  the distinction instead of folding it away.

A coarse label is also a valid leaf: `ACCOUNT` on its own means "some
account-like identifier, sub-type unspecified". Sources map each raw label to
the *most specific* canonical label it unambiguously denotes — OPF's
`account_number` stays coarse `ACCOUNT`; Presidio's `US_SSN` becomes `GOV_ID`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LabelInfo:
    name: str
    parent: str  # coarse label; equals `name` for coarse labels
    description: str

    @property
    def is_coarse(self) -> bool:
        return self.name == self.parent


# Order matters: CANONICAL_LABELS (and therefore report tables, the API enum,
# and prompt/request-type ordering) follow this sequence.
_COARSE: tuple[tuple[str, str], ...] = (
    ("PERSON", "Names — full, given, family, titles"),
    ("EMAIL", "Email addresses"),
    ("PHONE", "Phone and fax numbers, IMEI"),
    ("ADDRESS", "Physical locations — street, city, region, postcode, country, coordinates"),
    ("URL", "URLs and network identifiers (IP / MAC addresses)"),
    ("DATE", "Dates, times, dates of birth"),
    ("ACCOUNT", "Identifiers — financial accounts, cards, government, medical, internal IDs"),
    ("SECRET", "Passwords, API keys and other credentials"),
    ("USERNAME", "Logins and handles"),
    ("DEMOGRAPHIC", "Age, gender, sexuality, nationality, religion, politics"),
    ("ORGANIZATION", "Companies, institutions, organizations"),
    ("OCCUPATION", "Job titles, roles, professions"),
    ("MONEY", "Monetary amounts and currencies"),
    ("VEHICLE", "Vehicle identifiers — VIN, licence plate"),
    ("PHYSICAL", "Physical attributes — height, eye colour"),
)

_FINE: tuple[tuple[str, str, str], ...] = (
    # name, parent, description
    ("GIVEN_NAME", "PERSON", "First / given / middle name"),
    ("FAMILY_NAME", "PERSON", "Last / family name"),
    ("STREET_ADDRESS", "ADDRESS", "Street line, building number, unit"),
    ("CITY", "ADDRESS", "City or town"),
    ("STATE", "ADDRESS", "State, province, region"),
    ("POSTCODE", "ADDRESS", "Postal / ZIP code"),
    ("COUNTRY", "ADDRESS", "Country"),
    ("GEO_COORDINATE", "ADDRESS", "Latitude / longitude"),
    ("IP_ADDRESS", "URL", "IPv4 / IPv6 address"),
    ("MAC_ADDRESS", "URL", "MAC address"),
    ("DATE_OF_BIRTH", "DATE", "Date of birth"),
    ("TIME", "DATE", "Time of day"),
    ("BANK_ACCOUNT", "ACCOUNT", "Bank account, IBAN, routing, SWIFT/BIC"),
    ("CREDIT_CARD", "ACCOUNT", "Payment card number, CVV"),
    ("CRYPTO_WALLET", "ACCOUNT", "Cryptocurrency address"),
    ("GOV_ID", "ACCOUNT", "Government ID — SSN, passport, driver licence, national / tax ID"),
    ("MEDICAL_ID", "ACCOUNT", "Healthcare ID — MRN, health-plan number, provider ID"),
    ("EMPLOYEE_ID", "ACCOUNT", "Employee / staff identifier"),
    ("CUSTOMER_ID", "ACCOUNT", "Customer / member identifier"),
    ("DEVICE_ID", "ACCOUNT", "Device or hardware identifier"),
    ("PASSWORD", "SECRET", "Password or passphrase"),
    ("API_KEY", "SECRET", "API key or access token"),
    ("AGE", "DEMOGRAPHIC", "Age"),
    ("GENDER", "DEMOGRAPHIC", "Gender or sex"),
)

LABELS: dict[str, LabelInfo] = {
    **{name: LabelInfo(name, name, desc) for name, desc in _COARSE},
    **{name: LabelInfo(name, parent, desc) for name, parent, desc in _FINE},
}

COARSE_LABELS: tuple[str, ...] = tuple(name for name, _ in _COARSE)
FINE_LABELS: tuple[str, ...] = tuple(name for name, _, _ in _FINE)
ALL_LABELS: tuple[str, ...] = COARSE_LABELS + FINE_LABELS

LEVELS: tuple[str, ...] = ("coarse", "fine")


def parent(label: str) -> str:
    """Coarse parent of a canonical label. Unknown labels map to themselves
    (they won't be in any scored label set, so they fall out of scoring)."""
    info = LABELS.get(label)
    return info.parent if info else label


def children(coarse: str) -> tuple[str, ...]:
    """Fine labels under a coarse label (empty for categories with none)."""
    return tuple(n for n, p, _ in _FINE if p == coarse)


def is_known(label: str) -> bool:
    return label in LABELS


def check_level(level: str) -> str:
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}; expected one of {LEVELS}")
    return level
