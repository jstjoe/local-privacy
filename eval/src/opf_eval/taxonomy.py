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
        # Hierarchy: NAME (generic) > NAME_GIVEN/NAME_FAMILY (components).
        # Every dataset we currently support annotates split components, so
        # requesting NAME causes Skyflow to emit broad spans like "John Smith"
        # that don't match component-level gold (counted as INC under strict).
        # Drop NAME + the specialized NAME_MEDICAL_PROFESSIONAL (subsumed by
        # the components) — see docstring at top of file.
        "skyflow": ("NAME_GIVEN", "NAME_FAMILY"),
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
        "gretel": ("first_name", "last_name", "name"),
        "openmed": ("first_name", "last_name"),
    },
    "EMAIL": {
        "opf": ("private_email",),
        "skyflow": ("EMAIL_ADDRESS",),
        "pii300k": ("EMAIL",),
        "pii200k": ("EMAIL",),
        "openpii": ("EMAIL",),
        "presidio": ("EMAIL_ADDRESS",),
        "gliner": ("email",),
        "gretel": ("email",),
        "openmed": ("email",),
    },
    "PHONE": {
        "opf": ("private_phone",),
        "skyflow": ("PHONE_NUMBER",),
        "pii300k": ("TEL",),
        "pii200k": ("PHONENUMBER", "PHONEIMEI"),
        "openpii": ("TELEPHONENUM",),
        "presidio": ("PHONE_NUMBER",),
        "gliner": ("phone number",),
        "gretel": ("phone_number",),
        "openmed": ("phone_number", "fax_number"),
    },
    "ADDRESS": {
        "opf": ("private_address",),
        # Hierarchy: LOCATION (any place) > LOCATION_ADDRESS (full addr) >
        # LOCATION_ADDRESS_STREET (street part); LOCATION_CITY/STATE/ZIP/
        # COUNTRY/COORDINATE are sibling components. Drop the two parents so
        # Skyflow consistently emits components — matches the granularity
        # every dataset we support actually annotates.
        "skyflow": (
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
        "presidio": ("LOCATION", "UK_POSTCODE"),
        "gliner": ("address", "location", "city", "country", "postal code"),
        "gretel": ("address", "street_address", "city", "state", "country", "postcode", "coordinate"),
        "openmed": ("street_address", "city", "state", "county", "country", "postcode", "coordinate"),
    },
    "URL": {
        "opf": ("private_url",),
        "skyflow": ("URL", "IP_ADDRESS"),
        "pii300k": ("IP",),
        "pii200k": ("URL", "IP", "IPV4", "IPV6"),
        "openpii": ("URL", "IP", "IPV4", "IPV6"),
        "presidio": ("URL", "IP_ADDRESS", "MAC_ADDRESS"),
        "gliner": ("url", "ip address"),
        "gretel": ("url", "ipv4", "ipv6"),
        "openmed": ("url", "ipv4", "ipv6", "mac_address"),
    },
    "DATE": {
        "opf": ("private_date",),
        # Drop DAY/MONTH/YEAR — these are sub-units below DATE/DATE_INTERVAL/
        # DOB. Datasets annotate full dates ("2040-06-02"), not bare years or
        # months, so requesting them produces orphan spans Skyflow then has to
        # disambiguate against the higher-confidence DATE classification.
        "skyflow": ("DATE", "DATE_INTERVAL", "DOB", "TIME"),
        "pii300k": ("DATE", "TIME", "BOD"),
        "pii200k": ("DATE", "TIME", "DOB"),
        "openpii": ("DATE", "TIME", "DATEOFBIRTH"),
        "presidio": ("DATE_TIME",),
        "gliner": ("date", "date of birth", "time"),
        "gretel": ("date", "date_of_birth", "date_time", "time"),
        "openmed": ("date", "date_of_birth", "date_time", "time"),
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
            "SSN",
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
            "ABA_ROUTING_NUMBER",
            "US_SSN",
            "US_PASSPORT",
            "US_DRIVER_LICENSE",
            "US_BANK_NUMBER",
            "US_ITIN",
            "US_NPI",                       # National Provider Identifier (medical)
            "US_MBI",                       # Medicare Beneficiary Identifier
            "UK_NHS",
            "UK_NINO",
            "UK_PASSPORT",
            "UK_VEHICLE_REGISTRATION",      # see comment under VEHICLE
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
            "IN_GSTIN",                     # India Goods + Services Tax ID
            "IN_PASSPORT",
            "IN_VEHICLE_REGISTRATION",
            "IN_VOTER",
            "KR_BRN",                       # Korea Business Registration Number
            "KR_DRIVER_LICENSE",
            "KR_FRN",                       # Korea Foreign Registration Number
            "KR_PASSPORT",
            "KR_RRN",                       # Korea Resident Registration Number
            "NG_NIN",                       # Nigeria National Identification Number
            "NG_VEHICLE_REGISTRATION",
            "PL_PESEL",                     # Poland personal identification number
            "SG_NRIC_FIN",                  # Singapore NRIC / FIN
            "SG_UEN",                       # Singapore Unique Entity Number (business)
            "TH_TNIN",                      # Thailand Tax ID
            "FI_PERSONAL_IDENTITY_CODE",    # Finland personal identity code
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
        "gretel": (
            "ssn",
            "credit_card_number",
            "bank_routing_number",
            "account_number",
            "national_id",
            "tax_id",
            "swift_bic",
            "cvv",
            "pin",
            "medical_record_number",
            "health_plan_beneficiary_number",
            "unique_identifier",
            "customer_id",
            "employee_id",
            "device_identifier",
            "biometric_identifier",
            "certificate_license_number",
        ),
        "openmed": (
            "ssn",
            "credit_debit_card",
            "bank_routing_number",
            "account_number",
            "tax_id",
            "swift_bic",
            "cvv",
            "pin",
            "medical_record_number",
            "health_plan_beneficiary_number",
            "customer_id",
            "employee_id",
            "device_identifier",
            "biometric_identifier",
            "certificate_license_number",
            "unique_id",
            "npi",  # National Provider Identifier
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
        "gretel": ("password", "api_key"),
        "openmed": ("password", "api_key", "http_cookie"),
    },
    "USERNAME": {
        "opf": (),  # OPF has no native USERNAME label
        "skyflow": ("USERNAME",),
        "pii300k": ("USERNAME",),
        "pii200k": ("USERNAME",),
        "openpii": ("USERNAME",),
        "presidio": (),
        "gliner": ("username",),
        "gretel": ("user_name",),
        "openmed": ("user_name",),
    },
    "DEMOGRAPHIC": {
        "opf": (),  # OPF has no native gender/age label
        "skyflow": ("GENDER", "AGE", "SEXUALITY", "MARITAL_STATUS"),
        "pii300k": ("SEX",),
        "pii200k": ("GENDER", "SEX", "AGE"),
        "openpii": ("GENDER", "SEX", "AGE"),
        "presidio": ("NRP",),  # Nationality, Religion, Political affiliation
        "gliner": (),
        "gretel": (),
        "openmed": (
            "age",
            "gender",
            "sexuality",
            "race_ethnicity",
            "religious_belief",
            "political_view",
            "blood_type",
            "education_level",
            "employment_status",
            "language",
        ),
    },
    "ORGANIZATION": {
        "opf": (),  # OPF has no native ORG label
        "skyflow": ("ORGANIZATION", "ORGANIZATION_MEDICAL_FACILITY"),
        "pii300k": (),  # Not annotated
        "pii200k": ("COMPANYNAME",),
        "openpii": (),  # Not annotated
        "presidio": (),  # No native ORG recognizer in default Presidio
        "gliner": ("organization", "company"),
        "gretel": ("company_name",),
        "openmed": ("company_name",),
    },
    "OCCUPATION": {
        "opf": (),  # OPF has no native job label
        "skyflow": ("OCCUPATION",),
        "pii300k": (),  # Not annotated
        "pii200k": ("JOBTITLE", "JOBAREA", "JOBTYPE"),
        "openpii": (),  # Not annotated
        "presidio": (),
        "gliner": ("occupation", "job title"),
        "gretel": (),
        "openmed": ("occupation",),
    },
    "MONEY": {
        "opf": (),
        "skyflow": ("MONEY", "FINANCIAL_METRIC"),
        "pii300k": (),
        "pii200k": ("AMOUNT", "CURRENCYSYMBOL", "CURRENCY", "CURRENCYCODE", "CURRENCYNAME"),
        "openpii": (),
        "presidio": (),
        "gliner": ("monetary amount", "currency", "price"),
        "gretel": (),
        "openmed": (),
    },
    "VEHICLE": {
        "opf": (),
        "skyflow": ("VEHICLE_ID",),
        "pii300k": (),
        "pii200k": ("VEHICLEVIN", "VEHICLEVRM"),
        "openpii": (),
        # Presidio's *_VEHICLE_REGISTRATION entities (IN_, UK_, NG_) all stay
        # under ACCOUNT — they're country-specific ID strings, not generic
        # vehicle identifiers like a VIN. Our canonical VEHICLE is reserved
        # for VIN-shaped + license-plate-shaped tokens that the other
        # detectors emit. Presidio has no recognizer in that shape.
        "presidio": (),
        "gliner": ("vehicle id", "license plate", "vin"),
        "gretel": ("license_plate", "vehicle_identifier"),
        "openmed": ("license_plate", "vehicle_identifier"),
    },
    "PHYSICAL": {
        "opf": (),
        "skyflow": ("PHYSICAL_ATTRIBUTE",),
        "pii300k": (),
        "pii200k": ("HEIGHT", "EYECOLOR"),
        "openpii": (),
        "presidio": (),
        "gliner": ("height", "eye color", "physical attribute"),
        "gretel": (),
        "openmed": (),
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


_GRETEL_TO_CANONICAL = _build_reverse("gretel")
_OPENMED_TO_CANONICAL = _build_reverse("openmed")

_DATASET_REVERSE = {
    "pii300k": _PII300K_TO_CANONICAL,
    "pii200k": _PII200K_TO_CANONICAL,
    "openpii": _OPENPII_TO_CANONICAL,
}


# Detector name -> CANONICAL_MAP source key. Variants (skyflow_full,
# presidio_multilang) collapse to their parent vocabulary.
_DETECTOR_VOCAB_KEY: dict[str, str] = {
    "opf": "opf",
    "opf_calibrated": "opf",
    "skyflow": "skyflow",
    "skyflow_full": "skyflow",
    "presidio": "presidio",
    "presidio_multilang": "presidio",
    "gliner": "gliner",
    "gliner_nvidia": "gliner",  # uses our default gliner_prompts
    "gliner_gretel_small": "gretel",
    "gliner_gretel_large": "gretel",
    "ai4privacy_modernbert": "openpii",
    "openmed": "openmed",
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


def gretel_to_canonical(label: str) -> str | None:
    """Reverse-map a Gretel snake_case label to a canonical."""
    return _GRETEL_TO_CANONICAL.get(label) or _GRETEL_TO_CANONICAL.get(label.lower())


def openmed_to_canonical(label: str) -> str | None:
    """Reverse-map an OpenMed snake_case label to a canonical."""
    return _OPENMED_TO_CANONICAL.get(label) or _OPENMED_TO_CANONICAL.get(label.lower())


def gretel_prompts(canonicals: "Iterable[str] | None" = None) -> list[str]:
    """All Gretel snake_case labels (or restricted to a canonical subset)."""
    targets = set(canonicals) if canonicals is not None else None
    out: list[str] = []
    seen: set[str] = set()
    for canonical, by_source in CANONICAL_MAP.items():
        if targets is not None and canonical not in targets:
            continue
        for p in by_source.get("gretel", ()):
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


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
    skyflow_full / presidio_multilang -> their parent vocabulary)."""
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
