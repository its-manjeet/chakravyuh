"""
Chakravyuh · Ring III — PII Tokenization Shield
================================================
A zero-trust data layer for banking AI agents.

Detects Indian PII in untrusted text and replaces each value with a reversible
placeholder token BEFORE the text reaches the LLM or any external API, then
restores the original values on the way back out.

Design principles
------------------
1. Fail closed, never crash: the regex/checksum core has zero heavy deps and
   always runs. spaCy NER for names is layered on ONLY if the model is present
   (graceful degradation
2. Reversible, scoped vault: each session gets its own token<->value map so the
   real PII never lives in the LLM context. Detokenize only on trusted output.
3. High recall on structured IDs: a missed Aadhaar is a DPDP breach. Aadhaar is
   validated with the Verhoeff checksum to cut false positives without dropping
   recall on genuine IDs.

Usage
-----
    shield = PIIShield()
    safe_text, vault = shield.tokenize(user_message)   # -> send safe_text to LLM
    final = shield.detokenize(llm_response, vault)      # -> restore for trusted output
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ----------------------------------------------------------------------------
# Verhoeff checksum — the algorithm Aadhaar uses for its 12th digit.
# Validating it lets us accept real Aadhaar numbers while rejecting random
# 12-digit strings (phone-pairs, order ids, etc.) -> fewer false positives.
# ----------------------------------------------------------------------------
_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],    
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],       
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],   
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],    
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def _verhoeff_valid(number: str) -> bool:
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) != 12:
        return False
    c = 0
    for i, d in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][d]]
    return c == 0


# ----------------------------------------------------------------------------
# Detector registry. Order matters: more specific / higher-risk patterns first
# so they claim their spans before looser ones can.
# ----------------------------------------------------------------------------
@dataclass
class Detector:
    label: str                 # token category, e.g. "AADHAAR"
    pattern: re.Pattern
    validator: callable = None  # optional extra check on the matched string
    priority: int = 50          # lower = matched earlier


def _luhn_ok(_s: str) -> bool:  # placeholder hook for card numbers if needed
    return True


_DETECTORS: List[Detector] = [
    # Aadhaar: 12 digits, optionally space/hyphen grouped 4-4-4, checksum-verified.
    Detector(
        "AADHAAR",
        re.compile(r"\b[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}\b"),
        validator=_verhoeff_valid,
        priority=10,
    ),
    # PAN: 5 letters, 4 digits, 1 letter (e.g. ABCDE1234F).
    Detector(
        "PAN",
        re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
        priority=20,
    ),
    # IFSC: 4 letters + 0 + 6 alphanumerics (e.g. HDFC0001234).
    Detector(
        "IFSC",
        re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
        priority=25,
    ),
    # Email.
    Detector(
        "EMAIL",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        priority=30,
    ),
    # Indian mobile: optional +91 / 0 prefix, then a 6-9 leading 10-digit
    # number that may be split by one space/hyphen (e.g. "98765 43210").
    Detector(
        "PHONE",
        re.compile(r"(?<!\d)(?:\+?91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)"),
        priority=40,
    ),
    # Bank account: 9-18 digit run (kept lower priority so IDs above win first).
    Detector(
        "ACCOUNT",
        re.compile(r"(?<!\d)\d{9,18}(?!\d)"),
        priority=60,
    ),
]


@dataclass
class Match:
    start: int
    end: int
    label: str
    value: str


@dataclass
class Vault:
    """Per-session reversible map of token -> original value."""
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    mapping: Dict[str, str] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)

    def mint(self, label: str, value: str) -> str:
        # Reuse a token if we've already seen this exact value this session,
        # so "my account 123... the account 123..." maps to one stable token.
        for tok, val in self.mapping.items():
            if val == value and tok.startswith(f"[{label}_"):
                return tok
        self.counts[label] = self.counts.get(label, 0) + 1
        token = f"[{label}_{self.counts[label]}]"
        self.mapping[token] = value
        return token


class PIIShield:
    """Tokenize PII out of untrusted text; detokenize it back into trusted output."""

    def __init__(self, use_ner: bool = True):
        self._nlp = None
        self.ner_available = False
        if use_ner:
            self._try_load_ner()

    def _try_load_ner(self) -> None:
        """Load spaCy NER for person names — optional, degrades silently."""
        try:
            import spacy  # noqa
            self._nlp = spacy.load("en_core_web_sm")
            self.ner_available = True
        except Exception:
            # No spaCy or no model? Structured-ID detection still fully works.
            self._nlp = None
            self.ner_available = False

    # ----- detection -------------------------------------------------------
    def _regex_matches(self, text: str) -> List[Match]:
        found: List[Match] = []
        for det in sorted(_DETECTORS, key=lambda d: d.priority):
            for m in det.pattern.finditer(text):
                s, e = m.span()
                raw = m.group()
                if det.validator and not det.validator(raw):
                    continue
                # Skip if this span overlaps one already claimed by a
                # higher-priority detector.
                if any(not (e <= f.start or s >= f.end) for f in found):
                    continue
                found.append(Match(s, e, det.label, raw))
        return found

    def _ner_matches(self, text: str, taken: List[Match]) -> List[Match]:
        if not self.ner_available:
            return []
        out: List[Match] = []
        doc = self._nlp(text)
        for ent in doc.ents:
            if ent.label_ != "PERSON":
                continue
            s, e = ent.start_char, ent.end_char
            if any(not (e <= f.start or s >= f.end) for f in taken + out):
                continue
            out.append(Match(s, e, "NAME", ent.text))
        return out

    # ----- public API ------------------------------------------------------
    def tokenize(self, text: str, vault: Vault | None = None) -> Tuple[str, Vault]:
        """Return (safe_text, vault). Send safe_text to the LLM; keep vault private."""
        vault = vault or Vault()
        matches = self._regex_matches(text)
        matches += self._ner_matches(text, matches)
        # Replace right-to-left so earlier indices stay valid.
        matches.sort(key=lambda m: m.start, reverse=True)
        safe = text
        for m in matches:
            token = vault.mint(m.label, m.value)
            safe = safe[:m.start] + token + safe[m.end:]
        return safe, vault

    def detokenize(self, text: str, vault: Vault) -> str:
        """Restore original PII from a trusted output string."""
        restored = text
        for token, value in vault.mapping.items():
            restored = restored.replace(token, value)
        return restored

    def report(self, vault: Vault) -> List[dict]:
        """Audit view: what was shielded this session (values masked)."""
        rows = []
        for token, value in vault.mapping.items():
            label = token.strip("[]").rsplit("_", 1)[0]
            masked = (value[:2] + "*" * max(0, len(value) - 4) + value[-2:]) if len(value) > 4 else "****"
            rows.append({"token": token, "type": label, "masked": masked})
        return rows


# ----------------------------------------------------------------------------
# Self-test — run `python pii_shield.py` to see the shield in action.
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    shield = PIIShield()

    sample = (
        "Hi, I'm Rajesh Kumar. My Aadhaar is 2994 1234 5678 and PAN ABCDE1234F. "
        "Please refund to account 123456789012 at IFSC HDFC0001234. "
        "Reach me on +91 98765 43210 or rajesh.k@example.com."
    )

    print("NER (names) available:", shield.ner_available)
    print("\n--- ORIGINAL (untrusted input) ---\n" + sample)

    safe, vault = shield.tokenize(sample)
    print("\n--- TOKENIZED (this is what the LLM sees) ---\n" + safe)

    # Simulate an LLM reply that echoes some tokens back.
    llm_reply = ("I've initiated the refund to [ACCOUNT_1] for the customer "
                 "and sent a confirmation to [EMAIL_1].")
    print("\n--- LLM REPLY (still tokenized) ---\n" + llm_reply)

    final = shield.detokenize(llm_reply, vault)
    print("\n--- DETOKENIZED (trusted output to customer) ---\n" + final)

    print("\n--- AUDIT REPORT (values masked) ---")
    for row in shield.report(vault):
        print(f"  {row['token']:<14} {row['type']:<9} {row['masked']}")