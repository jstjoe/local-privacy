"""Canonical label taxonomy bridging OPF, Skyflow, and PII-Masking-300k.

Refine on first benchmark run — the Skyflow entity_type strings below match the
public Detect API but vendor sometimes adds new ones.
"""

from __future__ import annotations

from typing import Iterable

# canonical -> (OPF labels, Skyflow entity_types, PII-Masking-300k labels)
CANONICAL_MAP: dict[str, dict[str, tuple[str, ...]]] = {
    "PERSON": {
        "opf": ("private_person",),
        "skyflow": ("NAME", "NAME_GIVEN", "NAME_FAMILY", "NAME_MEDICAL_PROFESSIONAL"),
        "pii300k": (
            "GIVENNAME1",
            "GIVENNAME2",
            "LASTNAME1",
            "LASTNAME2",
            "LASTNAME3",
            "TITLE",
        ),
        "pii200k": (
            "FIRSTNAME",
            "MIDDLENAME",
            "LASTNAME",
            "PREFIX",
            "SUFFIX",
        ),
        "openpii": (
            "GIVENNAME",
            "SURNAME",
            "TITLE",
        ),
        "presidio": ("PERSON",),
        "gliner": ("person",),
    },
    "EMAIL": {
        "opf": ("private_email",),
        "skyflow": ("EMAIL_ADDRESS",),
        "pii300k": ("EMAIL",),
        "pii200k": ("EMAIL",),
        "openpii": ("EMAIL",),
        "presidio": ("EMAIL_ADDRESS",),
        "gliner": ("email",),
    },
    "PHONE": {
        "opf": ("private_phone",),
        "skyflow": ("PHONE_NUMBER",),
        "pii300k": ("TEL",),
        "pii200k": ("PHONENUMBER", "PHONEIMEI"),
        "openpii": ("TELEPHONENUM",),
        "presidio": ("PHONE_NUMBER",),
        "gliner": ("phone number",),
    },
    "ADDRESS": {
        "opf": ("private_address",),
        "skyflow": (
            "LOCATION",
            "LOCATION_ADDRESS",
            "LOCATION_ADDRESS_STREET",
            "LOCATION_CITY",
            "LOCATION_STATE",
            "LOCATION_ZIP",
            "LOCATION_COUNTRY",
            "LOCATION_COORDINATE",
        ),
        "pii300k": (
            "STREET",
            "CITY",
            "STATE",
            "COUNTRY",
            "POSTCODE",
            "BUILDING",
            "SECADDRESS",
            "GEOCOORD",
        ),
        "pii200k": (
            "STREET",
            "CITY",
            "COUNTY",
            "STATE",
            "ZIPCODE",
            "BUILDINGNUMBER",
            "SECONDARYADDRESS",
            "NEARBYGPSCOORDINATE",
        ),
        "openpii": (
            "STREET",
            "CITY",
            "STATE",
            "ZIPCODE",
            "BUILDINGNUM",
            "SECONDARYADDRESS",
        ),
        "presidio": ("LOCATION",),
        "gliner": ("address", "location", "city", "country", "postal code"),
    },
    "URL": {
        "opf": ("private_url",),
        "skyflow": ("URL", "IP_ADDRESS"),
        "pii300k": ("IP",),
        "pii200k": ("URL", "IP", "IPV4", "IPV6"),
        "openpii": ("URL", "IP", "IPV4", "IPV6"),
        "presidio": ("URL", "IP_ADDRESS"),
        "gliner": ("url", "ip address"),
    },
    "DATE": {
        "opf": ("private_date",),
        "skyflow": ("DATE", "DATE_INTERVAL", "DOB", "TIME", "DAY", "MONTH", "YEAR"),
        "pii300k": ("DATE", "TIME", "BOD"),
        "pii200k": ("DATE", "TIME", "DOB"),
        "openpii": ("DATE", "TIME", "DATEOFBIRTH"),
        "presidio": ("DATE_TIME",),
        "gliner": ("date", "date of birth", "time"),
    },
    "ACCOUNT": {
        "opf": ("account_number",),
        "skyflow": (
            "ACCOUNT_NUMBER",
            "BANK_ACCOUNT",
            "CREDIT_CARD",
            "ROUTING_NUMBER",
            "NUMERICAL_PII",
            "SSN",
            "PASSPORT_NUMBER",
            "DRIVER_LICENSE",
            "HEALTHCARE_NUMBER",
        ),
        "pii300k": ("SOCIALNUMBER", "IDCARD", "PASSPORT", "DRIVERLICENSE"),
        "pii200k": (
            "ACCOUNTNUMBER",
            "ACCOUNTNAME",
            "CREDITCARDNUMBER",
            "CREDITCARDISSUER",
            "CREDITCARDCVV",
            "BITCOINADDRESS",
            "ETHEREUMADDRESS",
            "LITECOINADDRESS",
            "IBAN",
            "BIC",
            "PIN",
        ),
        "openpii": (
            "ACCOUNTNUM",
            "CREDITCARDNUMBER",
            "IDCARDNUM",
            "SOCIALNUM",
            "PASSPORTNUM",
            "DRIVERLICENSENUM",
            "TAXNUM",
        ),
        "presidio": (
            "CREDIT_CARD",
            "IBAN_CODE",
            "US_SSN",
            "US_PASSPORT",
            "US_DRIVER_LICENSE",
            "US_BANK_NUMBER",
            "US_ITIN",
            "UK_NHS",
            "UK_NINO",
            "ES_NIE",
            "ES_NIF",
            "IT_DRIVER_LICENSE",
            "IT_FISCAL_CODE",
            "IT_IDENTITY_CARD",
            "IT_PASSPORT",
            "IT_VAT_CODE",
            "AU_ABN",
            "AU_ACN",
            "AU_MEDICARE",
            "AU_TFN",
            "IN_AADHAAR",
            "IN_PAN",
            "IN_VEHICLE_REGISTRATION",
            "MEDICAL_LICENSE",
            "CRYPTO",
        ),
        "gliner": (
            "social security number",
            "passport",
            "passport number",
            "driver license",
            "driver's license",
            "credit card",
            "credit card number",
            "account number",
            "bank account",
            "national id",
            "id number",
            "tax id",
        ),
    },
    "SECRET": {
        "opf": ("secret",),
        "skyflow": ("PASSWORD",),
        "pii300k": ("PASS",),
        "pii200k": ("PASSWORD",),
        "openpii": ("PASSWORD",),
        "presidio": (),  # No default password recognizer in Presidio
        "gliner": ("password",),
    },
    "USERNAME": {
        "opf": (),  # OPF has no native USERNAME label
        "skyflow": ("USERNAME",),
        "pii300k": ("USERNAME",),
        "pii200k": ("USERNAME",),
        "openpii": ("USERNAME",),
        "presidio": (),
        "gliner": ("username",),
    },
    "DEMOGRAPHIC": {
        "opf": (),  # OPF has no native gender/age label
        "skyflow": ("GENDER", "AGE", "GENDER_SEXUALITY", "MARITAL_STATUS"),
        "pii300k": ("SEX",),
        "pii200k": ("GENDER", "SEX", "AGE"),
        "openpii": ("GENDER", "SEX", "AGE"),
        "presidio": ("NRP",),  # Nationality, Religion, Political affiliation
        "gliner": (),
    },
}


def _build_reverse(source: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for canonical, by_source in CANONICAL_MAP.items():
        for raw in by_source.get(source, ()):
            out[raw] = canonical
    return out


_OPF_TO_CANONICAL = _build_reverse("opf")
_SKYFLOW_TO_CANONICAL = _build_reverse("skyflow")
_PII300K_TO_CANONICAL = _build_reverse("pii300k")
_PII200K_TO_CANONICAL = _build_reverse("pii200k")
_OPENPII_TO_CANONICAL = _build_reverse("openpii")
_PRESIDIO_TO_CANONICAL = _build_reverse("presidio")
_GLINER_TO_CANONICAL = _build_reverse("gliner")


_DATASET_REVERSE = {
    "pii300k": _PII300K_TO_CANONICAL,
    "pii200k": _PII200K_TO_CANONICAL,
    "openpii": _OPENPII_TO_CANONICAL,
}


# Detector name -> CANONICAL_MAP source key. Variants (skyflow_minimal,
# presidio_multilang) collapse to their parent vocabulary.
_DETECTOR_VOCAB_KEY: dict[str, str] = {
    "opf": "opf",
    "opf_calibrated": "opf",
    "skyflow": "skyflow",
    "skyflow_minimal": "skyflow",
    "presidio": "presidio",
    "presidio_multilang": "presidio",
    "gliner": "gliner",
}


def opf_to_canonical(label: str) -> str | None:
    return _OPF_TO_CANONICAL.get(label)


def skyflow_to_canonical(label: str) -> str | None:
    return _SKYFLOW_TO_CANONICAL.get(label)


def presidio_to_canonical(label: str) -> str | None:
    return _PRESIDIO_TO_CANONICAL.get(label)


def gliner_to_canonical(label: str) -> str | None:
    """Reverse-map a GLiNER prompt string back to a canonical label.
    Case-insensitive since GLiNER may echo prompts with varied casing."""
    return _GLINER_TO_CANONICAL.get(label) or _GLINER_TO_CANONICAL.get(label.lower())


def gliner_prompts(canonicals: "Iterable[str] | None" = None) -> list[str]:
    """Flat list of GLiNER prompts to feed the model in one call.

    canonicals: when provided, only emit prompts for these canonical labels
        (used for dataset-aware detector configuration). When None, emit
        the full union — backward-compatible behavior.
    """
    targets = set(canonicals) if canonicals is not None else None
    out: list[str] = []
    seen: set[str] = set()
    for canonical, by_source in CANONICAL_MAP.items():
        if targets is not None and canonical not in targets:
            continue
        for p in by_source.get("gliner", ()):
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


def pii300k_to_canonical(label: str) -> str | None:
    # accept either raw form or NER-tag form like "B-FIRSTNAME"
    bare = label.split("-", 1)[1] if "-" in label and label[1:2] == "-" else label
    return _PII300K_TO_CANONICAL.get(bare.upper())


def dataset_to_canonical(vocab_key: str, label: str) -> str | None:
    """Generic dataset-vocab lookup. Strips BIO prefix if present."""
    table = _DATASET_REVERSE.get(vocab_key)
    if table is None:
        raise KeyError(f"unknown dataset vocab: {vocab_key!r}")
    bare = label.split("-", 1)[1] if "-" in label and label[1:2] == "-" else label
    return table.get(bare.upper())


def dataset_canonicals(vocab_key: str) -> set[str]:
    """Canonical labels actually annotated by this dataset's vocabulary."""
    return {
        canonical for canonical, by_source in CANONICAL_MAP.items()
        if by_source.get(vocab_key)
    }


def detector_supported_canonicals(detector: str) -> set[str]:
    """Canonical labels this detector can produce (handles variants like
    skyflow_minimal -> skyflow vocabulary)."""
    source = _DETECTOR_VOCAB_KEY.get(detector, detector)
    return {
        canonical for canonical, by_source in CANONICAL_MAP.items()
        if by_source.get(source)
    }


def fair_labels(detector: str, vocab_key: str) -> set[str]:
    """Labels for the fair-view headline: intersection of (dataset annotates,
    detector supports). Apples-to-apples within each detector's claims."""
    return detector_supported_canonicals(detector) & dataset_canonicals(vocab_key)


CANONICAL_LABELS: tuple[str, ...] = tuple(CANONICAL_MAP.keys())

# The 8 categories OPF actually supports natively. Used as the default
# allow-list for apples-to-apples comparison against Skyflow.
OPF_CANONICAL_LABELS: tuple[str, ...] = tuple(
    canonical for canonical, by_source in CANONICAL_MAP.items() if by_source.get("opf")
)


# Empirically-tuned minimal Skyflow entity_types: drops bare general types
# (NAME, LOCATION, LOCATION_ADDRESS) and individual labels with <50% gold hit
# rate on PII-Masking-300k 1k sample. See eval/scripts/analyze_skyflow_hitrate.py.
SKYFLOW_MINIMAL_ENTITY_TYPES: tuple[str, ...] = (
    # Names — drop bare 'name' (38% hit), keep components
    "name_given",
    "name_family",
    # Email / phone / contact
    "email_address",
    "phone_number",
    # Location — drop bare 'location' (46%) and 'location_address' (20%);
    # keep components which all hit >=80%
    "location_address_street",
    "location_city",
    "location_state",
    "location_zip",
    "location_country",
    "location_coordinate",
    # URL / IP
    "url",
    "ip_address",
    # Dates — keep date, date_interval, dob, time; drop year/month/day (low hit)
    "date",
    "date_interval",
    "dob",
    "time",
    # IDs / accounts
    "account_number",
    "ssn",
    "passport_number",
    "driver_license",
    "healthcare_number",
    "bank_account",
    "numerical_pii",
    # Secrets
    "password",
)


def canonical_to_skyflow_request_types(canonicals: tuple[str, ...] | list[str]) -> list[str]:
    """Return Skyflow's request-side entity_type strings (lowercase) for the
    given canonical labels.

    The Skyflow Detect API enum is the lowercase form of the response
    entity_type values, e.g. response 'EMAIL_ADDRESS' -> request 'email_address'.
    """
    out: list[str] = []
    seen: set[str] = set()
    for canonical in canonicals:
        for raw in CANONICAL_MAP.get(canonical, {}).get("skyflow", ()):
            lower = raw.lower()
            if lower not in seen:
                seen.add(lower)
                out.append(lower)
    return out
