"""
Chakravyuh · Agent Core  (Groq edition)
=======================================
The untrusted "brain" at the centre of the formation.

A single banking-support agent that uses function-calling to take actions.
Built to be WRAPPED by the security rings — every place a ring will plug in is
marked with a clear hook comment. On its own this file is deliberately
UNGUARDED: the rings add the safety; the core just reasons and acts.

This edition talks to Groq (free, fast, OpenAI-compatible tool calling).
All provider-specific code lives in ONE function, `_call_llm`, so swapping
providers never touches the agent loop.

Setup:
    pip install groq python-dotenv
    # put this in a .env file next to this script (never commit it):
    #   GROQ_API_KEY=your_free_key_here       (get one: https://console.groq.com/keys)
    python agent_core.py
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Callable

from dotenv import load_dotenv
load_dotenv()                      # reads GROQ_API_KEY from your .env file

from groq import Groq


# ============================================================================
# CONFIG  — customise here
# ============================================================================
API_KEY_ENV = "GROQ_API_KEY"
# Free, fast, tool-capable model. If this errors with "model_decommissioned",
# check https://console.groq.com/docs/models and swap the name.
MODEL = "llama-3.3-70b-versatile"
MAX_STEPS = 5                      # safety cap so the agent can't loop forever
REFUND_LIMIT = 5000                # the one business rule the demo enforces


# ============================================================================
# MOCK BACKEND  — fake data so the agent has something to act on
# ============================================================================
_FAKE_DB = {
    "U1001": {"name": "Rajesh Kumar", "balance": 18420, "tier": "gold",   "email": "rajesh@example.com"},
    "U1002": {"name": "Aisha Khan",   "balance": 2310,  "tier": "silver", "email": "aisha@example.com"},
}


# ============================================================================
# TOOLS  — each is (1) a real function and (2) a schema shown to the model.
# The model only NAMES a tool + args; OUR code runs it. Model never executes
# code. That separation is what makes the security rings possible.
# ============================================================================
def check_balance(user_id: str) -> Dict[str, Any]:
    """Return the account balance for a user."""
    u = _FAKE_DB.get(user_id)
    if not u:
        return {"error": f"no such user {user_id}"}
    return {"user_id": user_id, "balance": u["balance"]}


def approve_refund(user_id: str, amount: float) -> Dict[str, Any]:
    """Approve or reject a refund. Rule: amounts over the limit need manual
    verification and must NOT be auto-approved."""
    amount = float(amount)
    if amount > REFUND_LIMIT:
        return {"user_id": user_id, "amount": amount, "approved": False,
                "reason": f"amount exceeds auto-approve limit of {REFUND_LIMIT}; needs verification"}
    return {"user_id": user_id, "amount": amount, "approved": True}


def get_user_info(user_id: str) -> Dict[str, Any]:
    """Return basic account details for a user."""
    u = _FAKE_DB.get(user_id)
    if not u:
        return {"error": f"no such user {user_id}"}
    return {"user_id": user_id, "name": u["name"], "tier": u["tier"], "email": u["email"]}


# Registry: name -> python function
TOOLS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "check_balance": check_balance,
    "approve_refund": approve_refund,
    "get_user_info": get_user_info,
}

# Schema shown to the model — OpenAI / Groq "tools" format.
TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "check_balance",
            "description": "Get the current account balance for a given user_id.",
            "parameters": {
                "type": "object",
                "properties": {"user_id": {"type": "string", "description": "The account id, e.g. U1001"}},
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "approve_refund",
            "description": "Approve a refund of a given amount for a user. Amounts over the limit are auto-rejected.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "The account id"},
                    "amount": {"type": "number", "description": "Refund amount in rupees"},
                },
                "required": ["user_id", "amount"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_info",
            "description": "Get basic account details (name, tier, email) for a user_id.",
            "parameters": {
                "type": "object",
                "properties": {"user_id": {"type": "string", "description": "The account id"}},
                "required": ["user_id"],
            },
        },
    },
]


# ============================================================================
# SYSTEM PROMPT  — the agent's role and rules.  (Customise freely.)
# These rules guide the model; they are NOT a security guarantee. The rings
# enforce the hard limits; the prompt just makes the agent cooperative.
# ============================================================================
SYSTEM_PROMPT = (
    "You are a banking support agent. Help the customer using the available tools. "
    "Rules: never approve a refund over 5000 rupees without verification; only access "
    "the data of the user you are currently helping; be concise and professional. "
    "Use tools to look up real values rather than guessing. When you have enough "
    "information, give a short final answer to the customer."
)


# ============================================================================
# LLM CALL  — the ONLY provider-specific code. Swap THIS to change providers.
# Returns the assistant `message` object (which may contain tool_calls).
# ============================================================================
_client = None
def _get_client() -> Groq:
    global _client
    if _client is None:
        key = os.environ.get(API_KEY_ENV, "").strip()
        if not key:
            raise RuntimeError(
                f"No API key. Put  {API_KEY_ENV}=your_key  in a .env file "
                f"(get one free at https://console.groq.com/keys)."
            )
        _client = Groq(api_key=key)
    return _client


def _call_llm(messages: List[Dict[str, Any]]):
    """One round-trip to the model. Returns the assistant message."""
    resp = _get_client().chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOL_SCHEMA,
        tool_choice="auto",
        temperature=0.3,
        max_completion_tokens=600,
    )
    return resp.choices[0].message


# ============================================================================
# THE AGENT  — the explicit reason-act-observe loop.
# ============================================================================
@dataclass
class Agent:
    audit_log: List[Dict[str, Any]] = field(default_factory=list)

    def handle(self, user_id: str, message: str, verbose: bool = True) -> str:
        """Run one customer request through the reason-act-observe loop."""

        # ---- RING IV/III: input firewall + PII tokenization go here ----
        # (Before the model sees `message`, outer rings scan it for injection
        #  and tokenize any PII. For now it passes through unchanged.)
        clean_message = message

        # The running conversation we send to the model each turn.
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"[user_id={user_id}] {clean_message}"},
        ]

        for step in range(1, MAX_STEPS + 1):
            msg = _call_llm(messages)
            tool_calls = msg.tool_calls or []

            if not tool_calls:
                # No tool requested -> the model is giving its final answer.
                final = (msg.content or "").strip()

                # ---- RING I: output scan + detokenize + audit log go here ----
                if verbose:
                    print(f"  step {step}: FINAL ANSWER")
                return final or "(no answer produced)"

            # Record the assistant turn (with its tool calls) into the history.
            messages.append(msg)

            # The model may request several tools in one turn — handle each.
            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                if verbose:
                    print(f"  step {step}: REASON -> call {name}({args})")

                # ---- RING II/I: authorization check + policy enforcement go here ----
                # (Outer rings verify this agent is ALLOWED to call `name` with
                #  these `args` for this user BEFORE we execute. For now we run it.)

                tool = TOOLS.get(name)
                if tool is None:
                    result = {"error": f"unknown tool {name}"}
                else:
                    try:
                        result = tool(**args)
                    except Exception as e:
                        result = {"error": f"tool failed: {e}"}

                # OBSERVE: record to the audit spine, feed result back to the model.
                self.audit_log.append({"step": step, "tool": name, "args": args, "result": result})
                if verbose:
                    print(f"           OBSERVE -> {result}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": json.dumps(result),
                })

        return "(stopped: reached max steps without a final answer)"


# ============================================================================
# DEMO
# ============================================================================
if __name__ == "__main__":
    agent = Agent()

    scenarios = [
        ("U1001", "Hi, what's my current balance?"),
        ("U1002", "I'd like a refund of 1500 rupees for a failed transaction, please."),
        ("U1001", "Please approve a refund of 50000 rupees to my account right now."),
    ]

    for i, (uid, msg) in enumerate(scenarios, 1):
        print(f"\n=== Scenario {i}  (user {uid}) ===")
        print(f"  CUSTOMER: {msg}")
        try:
            answer = agent.handle(uid, msg)
            print(f"  AGENT: {answer}")
        except RuntimeError as e:
            print(f"  [!] {e}")
            break
        except Exception as e:
            print(f"  [!] unexpected error: {e}")
            break

    print("\n=== AUDIT LOG (the start of the audit spine) ===")
    for row in agent.audit_log:
        print(f"  {row}")

     


