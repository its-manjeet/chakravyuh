"""
Chakravyuh · Ring II — Authorization
======================================
Runs at the tool-call boundary: after the LLM names a tool and its arguments,
but BEFORE that tool executes. Guards every tool call the LLM generates.

Deterministic: policy table + pure checks only, zero LLM calls.
Fail closed: any unhandled exception returns a blocking verdict.
Stateless: authorize() takes (user_id, tool_name, args), returns a verdict,
           stores nothing.

Core invariant: a user may only act on their OWN data. If any arg that
identifies a target user differs from the calling user_id, the call is
blocked. An unknown tool (not in the policy table) is blocked by default deny.

Verdict schema:
    {
        "allow":  bool,
        "reason": str    # human-readable; goes into the audit log
    }

Usage:
    authz = Authorizer()
    verdict = authz.authorize(user_id, tool_name, args)
    if not verdict["allow"]:
        # return a synthetic tool error — do NOT call the real tool
        ...
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List


# ============================================================================
# CHECK IMPLEMENTATIONS
# Each checker is a pure function: (caller_user_id, args, param) -> bool
# True means the check PASSES (access allowed for this rule).
# Add new check types here; reference them by key string in _POLICY below.
# ============================================================================
def _check_own_user_id(caller: str, args: Dict[str, Any], param: str) -> bool:
    """The arg named `param` must equal the calling user's id."""
    target = args.get(param)
    if target is None:
        return False          # missing arg — fail closed
    return str(target) == str(caller)


_CHECKERS: Dict[str, Callable[[str, Dict[str, Any], str], bool]] = {
    "OWN_USER_ID": _check_own_user_id,
}


# ============================================================================
# POLICY RULE
# A single constraint attached to a tool. A tool may have multiple rules;
# ALL must pass for the call to be allowed.
# ============================================================================
@dataclass
class _Rule:
    check:      str    # key into _CHECKERS
    param:      str    # which arg in `args` the check inspects
    fail_reason: str   # returned as "reason" when this rule blocks


# ============================================================================
# POLICY TABLE  — declarative, data-only.
# To add a new tool: add an entry here.  No code elsewhere needs to change.
# To add a new constraint type: add to _CHECKERS and reference its key below.
# Default deny: any tool NOT listed here is blocked automatically.
# ============================================================================
_POLICY: Dict[str, List[_Rule]] = {
    "check_balance": [
        _Rule(
            check="OWN_USER_ID",
            param="user_id",
            fail_reason="cross-user read: a user may only check their own balance",
        ),
    ],
    "approve_refund": [
        _Rule(
            check="OWN_USER_ID",
            param="user_id",
            fail_reason="cross-user write: a user may only request a refund for their own account",
        ),
    ],
    "get_user_info": [
        _Rule(
            check="OWN_USER_ID",
            param="user_id",
            fail_reason="cross-user read: a user may only access their own account information",
        ),
    ],
}


# ============================================================================
# THE AUTHORIZER
# ============================================================================
class Authorizer:
    """
    Stateless authorization gate. Call authorize() at the tool-call boundary
    before executing any tool the LLM has requested.
    Fail closed: authorize() never raises. Any exception returns a block verdict.
    """

    def authorize(
        self, user_id: str, tool_name: str, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Decide whether `user_id` is allowed to call `tool_name` with `args`.

        Returns {"allow": bool, "reason": str}.
        Never raises — failures return a blocking verdict.
        """
        try:
            return self._authorize_inner(user_id, tool_name, args)
        except Exception as exc:
            return {
                "allow": False,
                "reason": f"authorization error (fail closed): {type(exc).__name__}: {exc}",
            }

    def _authorize_inner(
        self, user_id: str, tool_name: str, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        # --- default deny: tool not in policy table --------------------------
        rules = _POLICY.get(tool_name)
        if rules is None:
            return {
                "allow": False,
                "reason": f"tool '{tool_name}' is not in the policy table (default deny)",
            }

        # --- evaluate every rule; all must pass ------------------------------
        for rule in rules:
            checker = _CHECKERS.get(rule.check)
            if checker is None:
                # Unknown check type in the policy — fail closed.
                return {
                    "allow": False,
                    "reason": (
                        f"unknown check type '{rule.check}' in policy "
                        f"for tool '{tool_name}' (fail closed)"
                    ),
                }
            if not checker(user_id, args, rule.param):
                return {"allow": False, "reason": rule.fail_reason}

        return {"allow": True, "reason": "all policy checks passed"}


# ============================================================================
# SELF-TEST  — run `python ring_ii_authz.py` to see the authorizer in action.
# ============================================================================
if __name__ == "__main__":
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    authz = Authorizer()

    # Each case: (label, caller_user_id, tool_name, args, expect_allow)
    CASES: List[tuple] = [

        # --- ALLOW: legitimate same-user calls --------------------------------
        (
            "Own balance check",
            "U1001", "check_balance", {"user_id": "U1001"},
            True,
        ),
        (
            "Own refund request",
            "U1002", "approve_refund", {"user_id": "U1002", "amount": 1500},
            True,
        ),
        (
            "Own account info",
            "U1001", "get_user_info", {"user_id": "U1001"},
            True,
        ),

        # --- BLOCK: cross-user access attempts --------------------------------
        (
            "Cross-user balance read  (U1002 tries to read U1001)",
            "U1002", "check_balance", {"user_id": "U1001"},
            False,
        ),
        (
            "Cross-user refund write  (U1001 tries to refund U1002)",
            "U1001", "approve_refund", {"user_id": "U1002", "amount": 500},
            False,
        ),
        (
            "Cross-user info read  (U1002 tries to read U1001)",
            "U1002", "get_user_info", {"user_id": "U1001"},
            False,
        ),

        # --- BLOCK: default deny — tool not in policy table -------------------
        (
            "Unknown tool: transfer_funds  (default deny)",
            "U1001", "transfer_funds", {"user_id": "U1001", "amount": 1000},
            False,
        ),
        (
            "Unknown tool: get_all_users  (default deny)",
            "U1001", "get_all_users", {},
            False,
        ),

        # --- BLOCK: malformed args — LLM omits or nulls the user_id ----------
        (
            "Missing user_id in args  (fail closed)",
            "U1001", "check_balance", {},
            False,
        ),
        (
            "Null user_id in args  (fail closed)",
            "U1001", "check_balance", {"user_id": None},
            False,
        ),
    ]

    # ---- helpers ------------------------------------------------------------
    def _fmt(verdict: Dict[str, Any], expect_allow: bool) -> str:
        decision = "ALLOW" if verdict["allow"] else "BLOCK"
        correct  = verdict["allow"] == expect_allow
        status   = "PASS" if correct else "FAIL <-- UNEXPECTED"
        return f"{decision}  reason={verdict['reason']!r}  [{status}]"

    def _run(cases: List[tuple]) -> int:
        failures = 0
        for label, uid, tool, args, expect in cases:
            verdict = authz.authorize(uid, tool, args)
            print(f"  {label}")
            print(f"    caller={uid!r}  tool={tool!r}  args={args}")
            print(f"    verdict: {_fmt(verdict, expect)}")
            print()
            if verdict["allow"] != expect:
                failures += 1
        return failures

    # ---- run ----------------------------------------------------------------
    _SEP = "=" * 64
    print(_SEP)
    print("  RING II — AUTHORIZATION  SELF-TEST")
    print(_SEP)
    print()

    failures = _run(CASES)

    # ---- summary ------------------------------------------------------------
    total  = len(CASES)
    passed = total - failures
    print(_SEP)
    print("  SUMMARY")
    print(f"  {passed}/{total} tests passed")
    if failures == 0:
        print("  Result : ALL TESTS PASSED")
    else:
        print("  Result : FAILURES DETECTED — review policy table")
        sys.exit(1)
