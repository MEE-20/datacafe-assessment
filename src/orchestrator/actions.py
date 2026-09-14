"""Deterministic /actions endpoint — no LLM involved."""

import time
from typing import Optional

from pydantic import BaseModel

from src.playbook.engine import run_scope


class ActionRequest(BaseModel):
    scope: Optional[str] = None


class ActionResponse(BaseModel):
    recommendations: list
    count: int
    latency_ms: float


def process_action_request(req: ActionRequest) -> dict:
    t0 = time.time()
    recommendations = run_scope(scope=req.scope)
    latency = round((time.time() - t0) * 1000, 2)
    return {
        "recommendations": recommendations,
        "count": len(recommendations),
        "latency_ms": latency,
    }