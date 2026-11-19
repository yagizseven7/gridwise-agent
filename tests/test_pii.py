from __future__ import annotations

import pytest

from gridwise import pii


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("mail jan.jansen@example.com please", "EMAIL"),
        ("IBAN NL91 ABNA 0417 1643 00", "IBAN"),
        ("IBAN NL91ABNA0417164300", "IBAN"),
        ("BSN 111222333", "BSN"),
        ("BSN 111.22.23.33", "BSN"),
        ("card 4111 1111 1111 1111", "CARD"),
        ("bel 06-12345678", "PHONE"),
        ("call +31 6 1234 5678", "PHONE"),
    ],
)
def test_identifiers_are_redacted(text: str, kind: str) -> None:
    redacted, kinds = pii.redact(text)
    assert kinds == [kind]
    assert f"[{kind}]" in redacted


@pytest.mark.parametrize(
    "text",
    [
        "n_readings = 19200",              # ordinary counts
        "peak 426.1 kW at 2026-08-31 13:00",
        "123456789 rows",                  # 9 digits that fail the BSN 11-proef
        "NL91 ABNA 0417 1643 0X",          # IBAN-shaped, wrong checksum
        "4111 1111 1111 1112",             # card number failing Luhn
        "site NL-UTR-01",
    ],
)
def test_ordinary_numbers_are_left_alone(text: str) -> None:
    assert pii.redact(text) == (text, [])


def test_redact_obj_walks_nested_structures() -> None:
    data = {"a": ["mail x@y.nl"], "b": {"c": "BSN 111222333"}, "n": 5}
    assert pii.redact_obj(data) == {"a": ["mail [EMAIL]"], "b": {"c": "BSN [BSN]"}, "n": 5}
