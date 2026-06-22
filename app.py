"""
Chakravyuh · API Gateway
=========================
FastAPI web service wrapping the full Chakravyuh agent + security rings.
All four rings (IV → III → Core → II → I) remain active via agent_core.py.
No ring logic is duplicated here — this file is purely the HTTP transport layer.

Run locally:
    uvicorn app:app --reload --port 8000

Example curl:
    curl -X POST http://localhost:8000/chat \
         -H "Content-Type: application/json" \
         -d '{"user_id": "U1001", "message": "What is my account balance?"}'
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Importing Agent pulls in all ring singletons from agent_core.py.
# The `if __name__ == "__main__"` demo block does NOT run on import.
from agent_core import Agent


# ============================================================================
# APP + MIDDLEWARE
# ============================================================================
app = FastAPI(
    title="Chakravyuh Banking Agent API",
    description=(
        "Zero-trust AI agent gateway with four deterministic security rings. "
        "Ring IV (input firewall) → Ring III (PII tokenisation) → "
        "LLM core → Ring II (authorisation) → Ring I (detokenise + audit)."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# SCHEMAS
# ============================================================================
class ChatRequest(BaseModel):
    user_id: str = Field(..., description="Authenticated caller id, e.g. 'U1001'")
    message: str = Field(..., description="Raw customer message (untrusted text)")


class ChatResponse(BaseModel):
    user_id:   str
    response:  str
    audit_log: List[Dict[str, Any]]


# ============================================================================
# ROUTES
# ============================================================================
@app.get("/health")
def health():
    """Liveness probe — deployment platforms poll this to verify the service is up."""
    return {"status": "ok", "service": "chakravyuh"}


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    """
    Run one customer message through the full Chakravyuh pipeline.

    A fresh Agent is created per request so each call gets its own audit log
    and vault — state never bleeds across sessions.
    verbose=False suppresses console prints; the audit_log in the response
    carries all per-request detail instead.
    """
    agent = Agent()
    try:
        answer = agent.handle(body.user_id, body.message, verbose=False)
    except RuntimeError as exc:
        # RuntimeError is raised when GROQ_API_KEY is absent.
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        )

    return ChatResponse(
        user_id=body.user_id,
        response=answer,
        audit_log=agent.audit_log,
    )


# ============================================================================
# FALLBACK ERROR HANDLER
# Catches anything that slips past the route-level try/except and ensures the
# response is always JSON — never an HTML traceback page.
# ============================================================================
@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "detail": str(exc)},
    )
