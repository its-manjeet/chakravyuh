# Chakravyuh — Project Intelligence

## What This Is

Chakravyuh is a **zero-trust security gateway for banking AI agents**. The name refers to a concentric defensive military formation — fitting, because the architecture is a series of deterministic security rings wrapping an untrusted LLM core.

The central design principle: **the LLM is never trusted alone**. Hard rules live in code, not in prompts. The prompt guides cooperative behaviour; the rings enforce correctness.

---

## Architecture

```
Untrusted user input
        |
   [ Ring IV ]  — Input Firewall / Prompt Injection Detection
        |
   [ Ring III ] — PII Tokenization (replace real values with reversible tokens)
        |
   [ LLM CORE ] — The "untrusted brain": reasons, picks tools, produces tool calls
        |
   [ Ring II ]  — Identity & Authorization (is this agent allowed to call this tool with these args for this user?)
        |
   [ Ring I ]   — Policy Enforcement + Output Scanner + Audit Log
        |
Trusted output to customer
```

The rings are **outside-in on input** and **inside-out on output**. Every token of user text passes through IV → III before the model sees it; every word of model output passes through I before the customer sees it. Ring II intercepts at the tool-call boundary — between the model deciding to act and the code actually executing.

### Why This Layering

| Ring | What it stops |
|------|--------------|
| IV   | Prompt injection, jailbreaks, instruction smuggling in user text |
| III  | PII leaking into LLM context / external APIs / logs |
| Core | (Untrusted) Reasoning and tool selection |
| II   | Privilege escalation, cross-user data access, unauthorised tool calls |
| I    | Policy violations in output, data exfil via response text, audit gaps |

No single ring is sufficient. A bypassed IV still hits III. A jailbroken core still hits II. Defense in depth.

---

## Current Files

### `agent_core.py`
The LLM agent core (Groq / LLaMA 3.3 70B). Deliberately **unguarded by design** — the rings add safety, not this file.

- Implements a Reason → Act → Observe loop capped at `MAX_STEPS = 5`.
- Three mock tools backed by an in-memory fake DB: `check_balance`, `approve_refund`, `get_user_info`.
- Tool schemas are sent to the model; the model names a tool + args; Python executes it. The model never runs code.
- Enforces one business rule in code: refunds over ₹5,000 are auto-rejected (`REFUND_LIMIT = 5000`).
- Hook comments mark every future ring insertion point explicitly (`# ---- RING IV/III: ... ----`, etc.).
- Provider-specific code is isolated in `_call_llm()` — swap that function to change LLM providers.

### `pii_shield.py`
Ring III implementation. Tokenizes PII before text reaches the LLM; detokenizes on trusted output.

- Detects: Aadhaar (Verhoeff checksum validated), PAN, IFSC, email, Indian mobile numbers, bank account numbers.
- Optional spaCy NER layer for person names — degrades silently if model not installed.
- Per-session `Vault` holds the token ↔ value map. Vault never leaves the trusted boundary.
- Stable tokens within a session: the same value always maps to the same token.
- `PIIShield.tokenize()` → send to LLM. `PIIShield.detokenize()` → restore for output.
- `PIIShield.report()` → audit view with values partially masked.

### `.env`
Holds `GROQ_API_KEY`. Never committed (in `.gitignore`).

---

## Future Ring Conventions

When implementing the remaining rings, follow these patterns so all rings stay consistent.

### General Rules
- Each ring lives in its own file: `ring_iv_firewall.py`, `ring_ii_authz.py`, `ring_i_policy.py`.
- Rings must **fail closed**: on any error or uncertainty, block — never pass through.
- Rings are **stateless where possible**; state (vault, audit log, session) is passed in, not stored on the ring object.
- No ring should call the LLM. Rings are deterministic; the LLM core is the only model call.
- Hooks in `agent_core.py` mark exactly where each ring plugs in — do not move the hook sites.

### Ring IV — Input Firewall (`ring_iv_firewall.py`)
- Runs before Ring III. Input: raw user text. Output: allow / block + reason.
- Detect: instruction injection (`ignore previous`, `you are now`, `system:`), role-switching attempts, encoded payloads (base64, unicode escapes).
- Should be regex + heuristic first; LLM-based classifiers are a secondary option only if latency allows.
- Returns a structured verdict: `{"allow": bool, "risk_score": float, "flags": [...]}`.

### Ring II — Authorization (`ring_ii_authz.py`)
- Runs at the tool-call boundary, before `TOOLS[name](**args)` executes.
- Receives: `user_id`, `tool_name`, `args`.
- Enforces: a user can only access their own `user_id`; no tool call may reference a different user's data.
- Policy table should be declarative (dict or config file), not imperative if-chains.
- Returns: `{"allow": bool, "reason": str}`. On block, the agent loop gets a synthetic tool error, not a crash.

### Ring I — Policy + Output Scanner + Audit (`ring_i_policy.py`)
- Runs on the final answer string before it is returned to the caller.
- Checks: no raw PII tokens leaked through (any `[LABEL_N]` still present is a detokenization miss), no account numbers or card numbers in plaintext output, no off-topic or harmful content.
- Writes the canonical audit record: timestamp, user_id, all tool calls + results (from `agent.audit_log`), final answer hash.
- Audit records must be append-only. Never modify or delete.

### Audit Log Schema
Every entry written by Ring I should follow this shape:
```json
{
  "ts": "<ISO-8601>",
  "session_id": "<vault.session_id>",
  "user_id": "<str>",
  "steps": [
    {"step": 1, "tool": "check_balance", "args": {...}, "result": {...}}
  ],
  "final_answer_hash": "<sha256 of detokenized answer>",
  "pii_shielded": ["AADHAAR_1", "EMAIL_1"],
  "policy_flags": []
}
```

---

## Development Notes

- **Provider swap**: change `MODEL` and `_call_llm()` in `agent_core.py` only. No other file should import `groq` directly.
- **Adding a tool**: add the Python function to `TOOLS`, add the schema to `TOOL_SCHEMA`. Ring II policy table must be updated at the same time — a new tool with no authz rule should default to **deny**.
- **Testing rings in isolation**: each ring file should be runnable as `python ring_*.py` with its own `if __name__ == "__main__"` self-test, following the pattern in `pii_shield.py`.
- **No secrets in prompts**: `SYSTEM_PROMPT` must never contain API keys, internal user data, or bypass phrases. The prompt is logged; treat it as public.
- **Fake DB**: `_FAKE_DB` in `agent_core.py` is for demo only. Real integration will replace `check_balance`, `approve_refund`, `get_user_info` with actual API calls — the tool signatures stay identical so the agent loop and Ring II do not change.

---

## Non-Goals

- This is not a general-purpose agent framework. Every design decision is optimised for the banking + PII + compliance context.
- Rings are not ML models. Deterministic code is the point. Accuracy of an ML-based ring is never a substitute for hard code enforcement in another ring.
- The `SYSTEM_PROMPT` is not a security control. It makes the agent cooperative; the rings make it safe.
