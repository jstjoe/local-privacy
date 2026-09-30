"""Taxonomy hierarchy + vocabulary registry.

`data/taxonomy_v1_snapshot.json` is the flat taxonomy as it was before the
coarse/fine hierarchy existed (every source's raw label -> canonical label,
plus the derived prompt / request-type lists). The coarse view of the new
taxonomy must reproduce it exactly, so historical results stay comparable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opf_eval import taxonomy
from opf_eval.taxonomy import (
    ALL_LABELS,
    CANONICAL_LABELS,
    COARSE_LABELS,
    FINE_LABELS,
    canonical_to_skyflow_request_types,
    dataset_canonicals,
    detector_supported_canonicals,
    fair_labels,
    gliner_prompts,
    gretel_prompts,
    parent,
    register_vocab,
    shared_projector,
    to_canonical,
)

SNAPSHOT = json.loads((Path(__file__).parent / "data" / "taxonomy_v1_snapshot.json").read_text())


# ------------------------------------------------------ back-compat snapshot


@pytest.mark.parametrize("source", sorted(SNAPSHOT["sources"]))
def test_coarse_mapping_matches_v1(source):
    expected = SNAPSHOT["sources"][source]
    got = {raw: to_canonical(source, raw, level="coarse") for raw in expected}
    assert got == expected
    # ...and nothing new was added to a built-in source silently.
    assert set(taxonomy.get_vocab(source).mapping) == set(expected)


def test_canonical_map_view_matches_v1():
    for coarse, by_source in taxonomy.CANONICAL_MAP.items():
        for source, raws in by_source.items():
            for raw in raws:
                assert SNAPSHOT["sources"][source][raw] == coarse


@pytest.mark.parametrize("vk", ["pii300k", "pii200k", "openpii"])
def test_dataset_canonicals_match_v1(vk):
    assert sorted(dataset_canonicals(vk)) == SNAPSHOT["dataset_canonicals"][vk]


def test_detector_support_matches_v1():
    for det, expected in SNAPSHOT["detector_supported"].items():
        assert sorted(detector_supported_canonicals(det)) == expected, det


def test_prompt_and_request_lists_match_v1():
    assert gliner_prompts() == SNAPSHOT["gliner_prompts_all"]
    assert gretel_prompts() == SNAPSHOT["gretel_prompts_all"]
    for vk in ("pii300k", "pii200k", "openpii"):
        coarse = dataset_canonicals(vk)
        assert gliner_prompts(coarse) == SNAPSHOT["gliner_prompts_by_ds"][vk]
        # Fine label sets select the same prompts as their coarse parents.
        assert gliner_prompts(dataset_canonicals(vk, "fine")) == SNAPSHOT["gliner_prompts_by_ds"][vk]
        assert (
            canonical_to_skyflow_request_types(sorted(coarse))
            == SNAPSHOT["skyflow_request_by_ds"][vk]
        )


def test_coarse_labels_unchanged():
    assert CANONICAL_LABELS == COARSE_LABELS
    assert len(COARSE_LABELS) == 15


# ------------------------------------------------------------- hierarchy


def test_every_fine_label_has_a_coarse_parent():
    for lbl in FINE_LABELS:
        assert parent(lbl) in COARSE_LABELS
        assert parent(lbl) != lbl
    for lbl in COARSE_LABELS:
        assert parent(lbl) == lbl


def test_unknown_label_is_its_own_parent():
    assert parent("NOT_A_LABEL") == "NOT_A_LABEL"


@pytest.mark.parametrize(
    "vocab, raw, fine, coarse",
    [
        ("presidio", "US_SSN", "GOV_ID", "ACCOUNT"),
        ("presidio", "UK_NHS", "MEDICAL_ID", "ACCOUNT"),
        ("skyflow", "NAME_GIVEN", "GIVEN_NAME", "PERSON"),
        ("skyflow", "ACCOUNT_NUMBER", "ACCOUNT", "ACCOUNT"),
        ("gretel", "employee_id", "EMPLOYEE_ID", "ACCOUNT"),
        ("opf", "account_number", "ACCOUNT", "ACCOUNT"),
        ("pii200k", "IBAN", "BANK_ACCOUNT", "ACCOUNT"),
    ],
)
def test_fine_and_coarse_lookup(vocab, raw, fine, coarse):
    assert to_canonical(vocab, raw) == fine
    assert to_canonical(vocab, raw, level="coarse") == coarse


def test_lookup_is_case_and_bio_tolerant():
    assert to_canonical("pii300k", "B-GIVENNAME1") == "GIVEN_NAME"
    assert to_canonical("pii300k", "i-email") == "EMAIL"
    assert to_canonical("gliner", "Phone Number") == "PHONE"
    assert to_canonical("opf", "nope") is None


# -------------------------------------------------------- registration


def test_register_vocab_rejects_unknown_targets():
    with pytest.raises(ValueError, match="unknown canonical"):
        register_vocab("bad_vocab_test", {"x": "NOT_A_LABEL"})


def test_register_vocab_and_duplicate_guard():
    register_vocab("tmp_vocab_test", {"emp": "EMPLOYEE_ID"}, overwrite=True)
    assert to_canonical("tmp_vocab_test", "EMP") == "EMPLOYEE_ID"
    with pytest.raises(ValueError, match="already registered"):
        register_vocab("tmp_vocab_test", {"emp": "EMPLOYEE_ID"})


def test_identity_vocab_covers_every_label():
    assert taxonomy.vocab_labels("canonical") == set(ALL_LABELS)


# ------------------------------------------------------ fine-level fairness


def test_shared_projector_rolls_up_when_detector_is_coarse():
    # OPF only knows coarse PERSON; the dataset splits names. Scoring OPF at
    # fine level must not demand GIVEN_NAME/FAMILY_NAME from it.
    proj = shared_projector({"PERSON"}, {"GIVEN_NAME", "FAMILY_NAME", "PERSON"})
    assert proj("GIVEN_NAME") == "PERSON"
    # Both sides split ACCOUNT -> keep the fine label.
    proj = shared_projector({"GOV_ID", "ACCOUNT"}, {"GOV_ID", "BANK_ACCOUNT"})
    assert proj("GOV_ID") == "GOV_ID"
    assert proj("BANK_ACCOUNT") == "BANK_ACCOUNT"


def test_fair_labels_fine_level():
    fine = fair_labels("opf", "pii300k", level="fine")
    assert "PERSON" in fine and "GIVEN_NAME" not in fine
    fine = fair_labels("gliner_gretel_small", "pii300k", level="fine")
    assert {"GIVEN_NAME", "FAMILY_NAME", "GOV_ID"} <= fine
    assert fair_labels("opf", "pii300k") == set(SNAPSHOT["detector_supported"]["opf"]) & set(
        SNAPSHOT["dataset_canonicals"]["pii300k"]
    )
