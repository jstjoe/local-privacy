"""Regression tests for the Presidio -> canonical label mapping.

Asserts every Presidio default recognizer's `supported_entity` resolves to
some canonical label via `presidio_to_canonical`. Catches the failure mode
where Presidio ships a new recognizer (or our install pins a newer
version) and our taxonomy silently drops its predictions because the
entity name has no entry in CANONICAL_MAP.
"""

from __future__ import annotations

import pytest

from opf_eval.taxonomy import presidio_to_canonical


# Default Presidio v2 entities present in
# presidio_analyzer/predefined_recognizers/. Updated 2026-05-14. If
# Presidio adds a recognizer the test below will fail loudly — at which
# point either map it here or add a deliberate skip entry.
PRESIDIO_DEFAULT_ENTITIES = {
    # Global
    "CREDIT_CARD", "CRYPTO", "DATE_TIME", "EMAIL_ADDRESS",
    "IBAN_CODE", "IP_ADDRESS", "MAC_ADDRESS", "MEDICAL_LICENSE",
    "PERSON", "PHONE_NUMBER", "URL", "LOCATION", "NRP",
    # US
    "ABA_ROUTING_NUMBER", "US_BANK_NUMBER", "US_DRIVER_LICENSE",
    "US_ITIN", "US_MBI", "US_NPI", "US_PASSPORT", "US_SSN",
    # UK
    "UK_NHS", "UK_NINO", "UK_PASSPORT", "UK_POSTCODE",
    "UK_VEHICLE_REGISTRATION",
    # Spain
    "ES_NIE", "ES_NIF",
    # Italy
    "IT_DRIVER_LICENSE", "IT_FISCAL_CODE", "IT_IDENTITY_CARD",
    "IT_PASSPORT", "IT_VAT_CODE",
    # Australia
    "AU_ABN", "AU_ACN", "AU_MEDICARE", "AU_TFN",
    # India
    "IN_AADHAAR", "IN_GSTIN", "IN_PAN", "IN_PASSPORT",
    "IN_VEHICLE_REGISTRATION", "IN_VOTER",
    # Korea
    "KR_BRN", "KR_DRIVER_LICENSE", "KR_FRN", "KR_PASSPORT", "KR_RRN",
    # Nigeria
    "NG_NIN", "NG_VEHICLE_REGISTRATION",
    # Poland
    "PL_PESEL",
    # Singapore
    "SG_NRIC_FIN", "SG_UEN",
    # Thailand
    "TH_TNIN",
    # Finland
    "FI_PERSONAL_IDENTITY_CODE",
}


def test_every_presidio_default_entity_maps_to_a_canonical():
    """Every Presidio default recognizer must resolve via presidio_to_canonical
    — silent None means PresidioDetector's spans get dropped at score time."""
    unmapped = sorted(
        e for e in PRESIDIO_DEFAULT_ENTITIES if presidio_to_canonical(e) is None
    )
    assert unmapped == [], (
        f"Presidio entities with no canonical mapping: {unmapped}. "
        "Add them to CANONICAL_MAP[<canonical>]['presidio'] in taxonomy.py."
    )


@pytest.mark.parametrize(
    "entity,canonical",
    [
        # Global identifiers
        ("PERSON", "PERSON"),
        ("EMAIL_ADDRESS", "EMAIL"),
        ("PHONE_NUMBER", "PHONE"),
        ("LOCATION", "ADDRESS"),
        ("URL", "URL"),
        ("IP_ADDRESS", "URL"),
        ("MAC_ADDRESS", "URL"),
        ("DATE_TIME", "DATE"),
        ("NRP", "DEMOGRAPHIC"),
        # ACCOUNT (national IDs, bank/medical numbers, passports, plates)
        ("CREDIT_CARD", "ACCOUNT"),
        ("IBAN_CODE", "ACCOUNT"),
        ("ABA_ROUTING_NUMBER", "ACCOUNT"),
        ("US_SSN", "ACCOUNT"),
        ("US_NPI", "ACCOUNT"),
        ("US_MBI", "ACCOUNT"),
        ("UK_PASSPORT", "ACCOUNT"),
        ("UK_VEHICLE_REGISTRATION", "ACCOUNT"),
        ("IN_AADHAAR", "ACCOUNT"),
        ("IN_GSTIN", "ACCOUNT"),
        ("IN_PASSPORT", "ACCOUNT"),
        ("IN_VOTER", "ACCOUNT"),
        ("KR_RRN", "ACCOUNT"),
        ("KR_BRN", "ACCOUNT"),
        ("KR_FRN", "ACCOUNT"),
        ("NG_NIN", "ACCOUNT"),
        ("NG_VEHICLE_REGISTRATION", "ACCOUNT"),
        ("PL_PESEL", "ACCOUNT"),
        ("SG_NRIC_FIN", "ACCOUNT"),
        ("SG_UEN", "ACCOUNT"),
        ("TH_TNIN", "ACCOUNT"),
        ("FI_PERSONAL_IDENTITY_CODE", "ACCOUNT"),
        ("MEDICAL_LICENSE", "ACCOUNT"),
        ("CRYPTO", "ACCOUNT"),
        # ADDRESS extras
        ("UK_POSTCODE", "ADDRESS"),
    ],
)
def test_specific_presidio_entity_resolves(entity: str, canonical: str):
    """Spot-check the new mappings explicitly, beyond the universal-coverage
    assertion above. Documents intent so a future reviewer reading the test
    file can see what each new entity was meant to be."""
    assert presidio_to_canonical(entity) == canonical


def test_unknown_presidio_entity_returns_none():
    """presidio_to_canonical must continue returning None for genuinely
    unknown labels — that's the signal the report uses to drop spans
    rather than scoring them as a wrong canonical."""
    assert presidio_to_canonical("OBVIOUSLY_FAKE_ENTITY_XYZ") is None
