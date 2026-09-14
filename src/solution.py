"""F1 capstone solution entry point — FastAPI server for /ask and /actions."""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

# .env must be loaded BEFORE any other top-level imports that read env variables
load_dotenv()

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.orchestrator.graph import run_ask_pipeline
from src.orchestrator.actions import ActionRequest, process_action_request

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

if not GEMINI_API_KEY or GEMINI_API_KEY == "your-gemini-api-key-here":
    print("WARNING: GEMINI_API_KEY not set. /ask will fail at LLM calls.")
    print("Set it in .env or as environment variable.")

app = FastAPI(title="ACPL Sales Focus and Action Assistant", version="1.0.0")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str | None
    status: str
    evidence: list
    cost_usd: float
    latency_ms: float
    token_usage: dict | None = None
    explanation: str | None = None


class ActionRequestModel(BaseModel):
    scope: str | None = None


class ActionResponseModel(BaseModel):
    recommendations: list
    count: int
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question or not req.question.strip():
        return AskResponse(
            answer=None, status="NO_ANSWER", evidence=[],
            cost_usd=0.0, latency_ms=0.0,
            explanation="Empty question provided."
        )
    result = run_ask_pipeline(req.question.strip())
    return AskResponse(**result)


@app.post("/actions", response_model=ActionResponseModel)
def actions(req: ActionRequestModel):
    ar = ActionRequest(scope=req.scope)
    result = process_action_request(ar)
    return ActionResponseModel(**result)


def main():
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))
    print(f"Starting ACPL Sales Assistant on {host}:{port}")
    print(f"  Port and host can be overridden via HOST and PORT env variables.")
    print(f"  /ask  - POST with {{\"question\": \"...\"}}")
    print(f"  /actions - POST with {{\"scope\": \"West\"}}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()