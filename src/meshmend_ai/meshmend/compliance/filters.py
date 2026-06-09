from __future__ import annotations

import re
from dataclasses import dataclass


FORBIDDEN_TERMS = {
    "warhammer": "grimdark tabletop",
    "space marine": "armored star knight",
    "ultramarine": "blue armored knight",
    "tyranid": "alien chitin creature",
    "termagant": "small alien bioform",
    "stormcast": "celestial armored warrior",
    "d&d": "fantasy roleplaying",
}

FORBIDDEN_SYMBOLS = {
    "aquila",
    "chaos star",
    "imperial eagle",
    "ultramarines logo",
    "blood angels logo",
}


@dataclass(slots=True)
class ComplianceResult:
    prompt: str
    sanitized_prompt: str
    warnings: list[str]


def sanitize_prompt(prompt: str) -> tuple[str, list[str]]:
    """Remove direct-copy cues and protected symbols before local generation."""
    sanitized = prompt
    warnings: list[str] = []
    for term, replacement in FORBIDDEN_TERMS.items():
        pattern = re.compile(rf"\b{re.escape(term)}\b", flags=re.IGNORECASE)
        if pattern.search(sanitized):
            sanitized = pattern.sub(replacement, sanitized)
            warnings.append(f"replaced protected/source-specific term '{term}'")
    for symbol in FORBIDDEN_SYMBOLS:
        pattern = re.compile(rf"\b{re.escape(symbol)}\b", flags=re.IGNORECASE)
        if pattern.search(sanitized):
            sanitized = pattern.sub("original non-branded emblem", sanitized)
            warnings.append(f"removed protected symbol '{symbol}'")
    return sanitized.strip(), warnings


def check_prompt(prompt: str) -> ComplianceResult:
    sanitized, warnings = sanitize_prompt(prompt)
    return ComplianceResult(prompt=prompt, sanitized_prompt=sanitized, warnings=warnings)
