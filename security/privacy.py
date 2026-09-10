"""Privacy guardrails for biomedical research data."""

from __future__ import annotations

import re


IDENTIFIER_NAMES = {
    "name", "patientname", "email", "phone", "mobile", "address",
    "aadhaar", "ssn", "dob", "dateofbirth",
}


def is_direct_identifier(column_name: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", str(column_name).lower())
    return normalized in IDENTIFIER_NAMES


def find_direct_identifiers(columns) -> list[str]:
    return [str(column) for column in columns if is_direct_identifier(column)]


def research_notice() -> str:
    return (
        "Do not upload identifiable patient information. QureNova is a research prototype; "
        "use de-identified or synthetic data for demonstrations."
    )
