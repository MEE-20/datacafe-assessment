"""LangGraph orchestration for the /ask endpoint.

LLM (Gemini) selects tools via function-calling, calls them,
and iteratively decides if more tools are needed, then synthesises the answer.
Max tool calls per question: 4.
"""

import os
from pathlib import Path
from dotenv import load_dotenv 
import sys
import time
import json

from typing import Optional, get_type_hints

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from src.analytics.sales_target import target_achievement, sales_total, national_total_sales, sales_by_region
from src.analytics.stockouts import stockout_events
from src.analytics.promotions import promotion_uplift
from src.analytics.distributor_sku import distributor_profile, sku_profile
from src.analytics.document_retrieval import retrieve
load_dotenv(Path(__file__).resolve().parents[3] / ".env")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MAX_TOOL_CALLS = 5

print("gemini model", GEMINI_MODEL)

# ---------------------------------------------------------------------------
# Pydantic schema for tool parameters — used by the LLM
# ---------------------------------------------------------------------------

class NationalTotalSalesParams(BaseModel):
    """Get the single national FY26 primary-sales total. No parameters needed."""

class SalesByRegionParams(BaseModel):
    """Aggregate sales by region. Use top_n=1 for highest region."""
    region: Optional[str] = Field(None, description="Filter to a specific region (North, South, East, West)")
    month: Optional[str] = Field(None, description="Filter to a month (YYYY-MM format)")
    top_n: Optional[int] = Field(None, description="Return only top N regions by sales value")

class TargetAchievementParams(BaseModel):
    """Get target achievement % (actual/target) per brand x region x month."""
    brand: Optional[str] = Field(None, description="Brand name (e.g., GlucoJoy, Aqualite)")
    region: Optional[str] = Field(None, description="Region (North, South, East, West)")
    month: Optional[str] = Field(None, description="Month (YYYY-MM format)")

class SalesTotalParams(BaseModel):
    """Get granular SKU-level sales data (units, value) with optional filters."""
    region: Optional[str] = Field(None, description="Region (North, South, East, West)")
    brand: Optional[str] = Field(None, description="Brand name")
    sku: Optional[str] = Field(None, description="SKU code (e.g., BS-0101)")
    month: Optional[str] = Field(None, description="Month (YYYY-MM format)")

class StockoutEventsParams(BaseModel):
    """Get stockout events with optional filters. Covers chronic (>6 weeks) and multi-SKU (>=3) detection."""
    region: Optional[str] = Field(None, description="Region (North, South, East, West)")
    brand: Optional[str] = Field(None, description="Brand name")
    distributor: Optional[str] = Field(None, description="Distributor ID (e.g., D032)")
    sku: Optional[str] = Field(None, description="SKU code")
    month: Optional[str] = Field(None, description="Month (YYYY-MM format)")

class PromotionUpliftParams(BaseModel):
    """Get promotion uplift analysis (% uplift, classification weak/moderate/strong)."""
    region: Optional[str] = Field(None, description="Region (North, South, East, West)")
    brand: Optional[str] = Field(None, description="Brand name")
    promo_id: Optional[str] = Field(None, description="Promotion ID (e.g., PR-2025-056)")

class DistributorProfileParams(BaseModel):
    """Get detailed profile of a single distributor including stockout history."""
    distributor_id: str = Field(..., description="Distributor ID (e.g., D001, D032)")

class SkuProfileParams(BaseModel):
    """Get detailed profile of a single SKU including sales and stockout stats."""
    sku_code: str = Field(..., description="SKU code (e.g., BS-0101, BV-0104)")

class DocumentRetrievalParams(BaseModel):
    """Search business documents for context about causes, competitor activity, supply issues, escalation rules."""
    query: str = Field(..., description="Search query, e.g., 'CremeDelight North competitor February'")

# ---------------------------------------------------------------------------
# Tool definitions for Gemini
# ---------------------------------------------------------------------------

TOOL_REGISTRY = {
    "national_total_sales": {
        "params": NationalTotalSalesParams,
        "description": "Get the single national FY26 primary-sales total in INR.",
        "exec": lambda p, q: _run_national_total(),
    },
    "sales_by_region": {
        "params": SalesByRegionParams,
        "description": "Aggregate primary-sales total by region. Use top_n=1 when asked for highest/top region.",
        "exec": lambda p, q: _run_sales_by_region(p, q),
    },
    "target_achievement": {
        "params": TargetAchievementParams,
        "description": "Get target achievement percentage (actual / target * 100) per brand x region x month. Use when asked about hitting/missing targets, achievement, or performance vs plan.",
        "exec": lambda p, q: _run_target_achievement(p),
    },
    "sales_total": {
        "params": SalesTotalParams,
        "description": "Get granular SKU-level primary sales (units, value) with optional filters. Use when you need raw sales numbers, not aggregated.",
        "exec": lambda p, q: _run_sales_total(p),
    },
    "stockout_events": {
        "params": StockoutEventsParams,
        "description": "Get distributor stockout events. Covers chronic (>6 consecutive weeks on same SKU) and multi-SKU (>=3 SKUs in a month) detection. Use for stockout-related questions.",
        "exec": lambda p, q: _run_stockout_events(p),
    },
    "promotion_uplift": {
        "params": PromotionUpliftParams,
        "description": "Get promotion uplift analysis: promo-period average weekly sales vs non-promo baseline, uplift %, and classification (weak <10%, moderate 10-25%, strong >25%).",
        "exec": lambda p, q: _run_promotion_uplift(p),
    },
    "distributor_profile": {
        "params": DistributorProfileParams,
        "description": "Get a single distributor's full profile: name, territory, region, total stockout events, unique SKUs with stockouts, and stockout history.",
        "exec": lambda p, q: _run_distributor_profile(p),
    },
    "sku_profile": {
        "params": SkuProfileParams,
        "description": "Get a single SKU's full profile: name, brand, category, pack size, MRP, total FY26 sales (units and value), stockout events.",
        "exec": lambda p, q: _run_sku_profile(p),
    },
    "document_retrieval": {
        "params": DocumentRetrievalParams,
        "description": "Search business documents (escalation SOP, market visit notes, distributor supply notes, promo circulars) for context on root causes like competitor activity, supply issues, or escalation rules. Use when the question asks WHY something happened.",
        "exec": lambda p, q: _run_document_retrieval(p),
    },
}


def _pydantic_to_schema(model_cls) -> types.Schema:
    """Convert a Pydantic model to a Gemini FunctionDeclaration Schema."""
    type_map = {
        str: types.Type.STRING,
        int: types.Type.INTEGER,
        float: types.Type.NUMBER,
        bool: types.Type.BOOLEAN,
    }
    properties = {}
    required = []
    for name, field in model_cls.model_fields.items():
        origin = field.annotation
        t = type_map.get(origin, types.Type.STRING)
        properties[name] = types.Schema(type=t, description=field.description or "")
        if field.is_required():
            required.append(name)
    return types.Schema(type=types.Type.OBJECT, properties=properties, required=required)


def _build_tool_declarations() -> list[types.FunctionDeclaration]:
    decls = []
    for name, reg in TOOL_REGISTRY.items():
        param_cls = reg["params"]
        schema = _pydantic_to_schema(param_cls)
        decls.append(types.FunctionDeclaration(
            name=name,
            description=reg["description"],
            parameters=schema,
        ))
    return decls


_TOOLS_LIB = types.Tool(function_declarations=_build_tool_declarations())


# ---------------------------------------------------------------------------
# Tool execution functions
# ---------------------------------------------------------------------------

def _result(tool_name: str, success: bool, evidence: list, error: str | None = None) -> dict:
    return {"tool_name": tool_name, "success": success, "evidence": evidence, "error": error}

def call_tool(tool_name: str, params: dict, question: str = "") -> dict:
    _log("call_tool", f"ENTER: {tool_name} params={json.dumps(params)}")
    reg = TOOL_REGISTRY.get(tool_name)
    if not reg:
        _log("call_tool", f"UNKNOWN TOOL: {tool_name}")
        return _result(tool_name, False, [], f"Unknown tool: {tool_name}")
    try:
        validated = reg["params"](**params)
        _log("call_tool", f"Validated params: {validated}")
        result = reg["exec"](validated, question)
        ev_count = len(result.get("evidence", []))
        _log("call_tool", f"DONE: {tool_name} success={result['success']} evidence_count={ev_count}")
        return result
    except Exception as e:
        _log("call_tool", f"ERROR: {tool_name} raised: {e}")
        return _result(tool_name, False, [], str(e))

def _run_national_total() -> dict:
    r = national_total_sales()
    d = r.to_dict()
    return _result("national_total_sales", d["success"], d.get("evidence", []), d.get("error"))

def _run_sales_by_region(p: SalesByRegionParams, q: str) -> dict:
    r = sales_by_region(region=p.region, month=p.month, top_n=p.top_n)
    return _result("sales_by_region", r.success, r.to_dict().get("evidence", []), r.error)

def _run_target_achievement(p: TargetAchievementParams) -> dict:
    r = target_achievement(region=p.region, brand=p.brand, month=p.month)
    d = r.to_dict()
    return _result("target_achievement", d["success"], d.get("evidence", []), d.get("error"))

def _run_sales_total(p: SalesTotalParams) -> dict:
    r = sales_total(region=p.region, brand=p.brand, sku=p.sku, month=p.month)
    d = r.to_dict()
    return _result("sales_total", d["success"], d.get("evidence", []), d.get("error"))

def _run_stockout_events(p: StockoutEventsParams) -> dict:
    r = stockout_events(region=p.region, brand=p.brand, distributor=p.distributor, sku=p.sku, month=p.month)
    return _result("stockout_events", r.success, r.evidence, r.error)

def _run_promotion_uplift(p: PromotionUpliftParams) -> dict:
    r = promotion_uplift(region=p.region, brand=p.brand, promo_id=p.promo_id)
    return _result("promotion_uplift", r.success, r.evidence, r.error)

def _run_distributor_profile(p: DistributorProfileParams) -> dict:
    r = distributor_profile(distributor_id=p.distributor_id)
    return _result("distributor_profile", r.success, r.evidence, r.error)

def _run_sku_profile(p: SkuProfileParams) -> dict:
    r = sku_profile(sku_code=p.sku_code)
    return _result("sku_profile", r.success, r.evidence, r.error)

def _run_document_retrieval(p: DocumentRetrievalParams) -> dict:
    r = retrieve(p.query)
    return _result("document_retrieval", r["success"], r.get("documents", []), None if r["success"] else "No docs found")

# ---------------------------------------------------------------------------
# Orchestrator state
# ---------------------------------------------------------------------------

class OrchestratorState(BaseModel):
    question: str = ""
    tool_results: list[dict] = Field(default_factory=list)
    evidence: list = Field(default_factory=list)
    answer: Optional[str] = None
    status: str = "OK"
    explanation: Optional[str] = None
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    tool_call_count: int = 0
    start_time: float = 0.0


def _estimate_cost(in_tokens: int, out_tokens: int) -> float:
    """Cost estimate using Gemini 2.0 Flash pricing (per 1M tokens).
    Input: $0.15/1M, Output: $0.60/1M. Plus $0.001 flat for deterministic compute."""
    input_cost = in_tokens * 0.15 / 1_000_000
    output_cost = out_tokens * 0.60 / 1_000_000
    deterministic_cost = 0.001
    return round(input_cost + output_cost + deterministic_cost, 6)


# ---------------------------------------------------------------------------
# Debug logging
# ---------------------------------------------------------------------------

_DEBUG = os.environ.get("DEBUG_ASK", "1") in ("1", "true", "yes")


def _log(node: str, msg: str) -> None:
    if _DEBUG:
        print(f"[DEBUG][{node}] {msg}", file=sys.stderr, flush=True)


def _log_state(state: OrchestratorState, node: str) -> None:
    if not _DEBUG:
        return
    tr_count = len(state.tool_results)
    ev_count = len(state.evidence)
    tr_names = [t["tool_name"] for t in state.tool_results]
    tr_success = [t["success"] for t in state.tool_results]
    ev_tools = list({e.get("tool") for e in state.evidence})
    _log(node, f"state: tool_results={tr_count} tools={tr_names} successes={tr_success} evidence={ev_count} evidence_tools={ev_tools} status={state.status} ans_len={len(state.answer or '')}")


# ---------------------------------------------------------------------------
# LangGraph pipeline nodes
# ---------------------------------------------------------------------------

def _record_usage(state: OrchestratorState, resp) -> None:
    """Track token usage from a Gemini response."""
    try:
        state.llm_input_tokens += resp.usage_metadata.prompt_token_count
        state.llm_output_tokens += resp.usage_metadata.candidates_token_count
    except (AttributeError, KeyError):
        pass


def llm_select_and_call_tools(state: OrchestratorState) -> OrchestratorState:
    """The LLM selects which tools to call based on the question, calls them,
    and iterates until satisfied (or MAX_TOOL_CALLS reached)."""
    _log("llm_select_and_call_tools", f"ENTER question={state.question[:80]}")

    system_instruction = (
        "You are a sales analytics assistant. Answer these types of questions:\n"
        "1. What is happening in sales performance?\n"
        "2. Why is it happening?\n"
        "3. What should be done next?\n\n"
        "You have a set of analytical tools available. For each user question:\n"
        "- Decide which tools are needed and call them with the right parameters.\n"
        "- You can call multiple tools — they will run and return results.\n"
        "- After seeing the results, decide if you need more information.\n"
        "- Call at most 5 tools total.\n"
        "- When you have enough information, answer the user's question using ONLY the evidence returned.\n"
        "- Do not calculate numbers yourself — use the tools.\n"
        "- If the question asks WHY (root cause), call document_retrieval after getting structured data.\n"
        "- If no tool can answer the question, say 'I cannot answer this question from available data.'"
    )

    config = types.GenerateContentConfig(
        tools=[_TOOLS_LIB],
        system_instruction=system_instruction,
    )

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        chat = client.chats.create(model=GEMINI_MODEL, config=config)
    except Exception as e:
        _log("llm_select_and_call_tools", f"FAILED client/chat init: {e}")
        state.status = "NO_ANSWER"
        state.explanation = f"Failed to initialize LLM: {e}. Check GEMINI_API_KEY and GEMINI_MODEL."
        return state

    # First call: LLM decides which tools to invoke
    try:
        resp = chat.send_message(state.question)
        _record_usage(state, resp)
        _log("llm_select_and_call_tools", f"INITIAL LLM response received. Usage: in={state.llm_input_tokens} out={state.llm_output_tokens}")
        if resp.candidates and resp.candidates[0].content.parts:
            for part in resp.candidates[0].content.parts:
                if part.function_call:
                    _log("llm_select_and_call_tools", f"LLM called: {part.function_call.name}({dict(part.function_call.args)})")
                elif part.text:
                    _log("llm_select_and_call_tools", f"LLM text response (no tool call): '{part.text[:150]}'")
        else:
            _log("llm_select_and_call_tools", "LLM returned no content parts")
    except Exception as e:
        _log("llm_select_and_call_tools", f"INITIAL LLM call FAILED: {e}")
        state.status = "NO_ANSWER"
        state.explanation = f"LLM call failed: {e}"
        return state

    while state.tool_call_count < MAX_TOOL_CALLS:
        try:
            fc = resp.candidates[0].content.parts[0].function_call if resp.candidates[0].content.parts and resp.candidates[0].content.parts[0].function_call else None
        except (IndexError, AttributeError):
            fc = None
        if fc is None:
            _log("llm_select_and_call_tools", "No more function calls from LLM — exiting loop")
            break

        tool_name = fc.name
        params = {k: v for k, v in fc.args.items()}
        _log("llm_select_and_call_tools", f"TOOL CALL #{state.tool_call_count + 1}: {tool_name}({params})")
        result = call_tool(tool_name, params, state.question)
        ev_len = len(result["evidence"])
        _log("llm_select_and_call_tools", f"TOOL RESULT: {tool_name} success={result['success']} evidence_count={ev_len} error={result.get('error')}")
        if _DEBUG and ev_len <= 3:
            _log("llm_select_and_call_tools", f"TOOL DATA: {json.dumps(result['evidence'], default=str)[:300]}")
        state.tool_results.append(result)
        state.tool_call_count += 1

        try:
            fc_part = types.Part(
                function_response=types.FunctionResponse(
                    name=tool_name,
                    response={"result": json.dumps(result["evidence"], default=str)[:4000], "success": result["success"]},
                )
            )
            resp = chat.send_message(fc_part)
            _record_usage(state, resp)
            _log("llm_select_and_call_tools", f"LLM follow-up received. Total usage: in={state.llm_input_tokens} out={state.llm_output_tokens}")
            if resp.candidates and resp.candidates[0].content.parts:
                for part in resp.candidates[0].content.parts:
                    if part.function_call:
                        _log("llm_select_and_call_tools", f"LLM next call: {part.function_call.name}({dict(part.function_call.args)})")
                    elif part.text:
                        _log("llm_select_and_call_tools", f"LLM partial answer: '{part.text[:200]}'")
        except Exception as e:
            _log("llm_select_and_call_tools", f"FOLLOW-UP LLM call FAILED: {e}")
            state.explanation = f"Error after tool call '{tool_name}': {e}"
            break

    # After tool loop, get final answer
    try:
        final_text = resp.candidates[0].content.parts[0].text if resp.candidates[0].content.parts else None
    except (IndexError, AttributeError):
        final_text = None

    if final_text:
        state.answer = final_text
        _log("llm_select_and_call_tools", f"FINAL answer ({len(final_text)} chars): '{final_text[:200]}'")
    else:
        state.answer = "I cannot answer this question from available data."
        state.status = "NO_ANSWER"
        _log("llm_select_and_call_tools", "NO final text — setting NO_ANSWER")

    state.cost_usd = _estimate_cost(state.llm_input_tokens, state.llm_output_tokens)
    _log_state(state, "llm_select_and_call_tools")
    return state


def build_evidence(state: OrchestratorState) -> OrchestratorState:
    """Build the evidence[] array from tool results."""
    _log("build_evidence", f"ENTER: {len(state.tool_results)} tool_results")
    for i, tr in enumerate(state.tool_results):
        _log("build_evidence", f"  tr[{i}]: tool={tr['tool_name']} success={tr['success']} ev_count={len(tr.get('evidence', []))} error={tr.get('error')}")

    evidence = []
    for tr in state.tool_results:
        if tr["success"] and tr["evidence"]:
            for item in tr["evidence"]:
                evidence.append({"tool": tr["tool_name"], "data": item})
        elif not tr["success"] and tr.get("error"):
            evidence.append({"tool": tr["tool_name"], "data": None, "error": tr["error"]})

    state.evidence = evidence
    _log("build_evidence", f"Built {len(evidence)} evidence items from {len(state.tool_results)} tool results")

    if not any(tr["success"] for tr in state.tool_results):
        state.status = "NO_ANSWER"
        errors = [tr.get("error", "") for tr in state.tool_results if tr.get("error")]
        state.explanation = f"Could not obtain evidence: {'; '.join(errors)}" if errors else "No evidence could be built."
        _log("build_evidence", f"ALL TOOLS FAILED — setting NO_ANSWER: {state.explanation}")
    else:
        _log("build_evidence", f"At least one tool succeeded — status stays OK")

    _log_state(state, "build_evidence")
    return state


def assemble_response(state: OrchestratorState) -> dict:
    _log("assemble_response", f"ENTER: status={state.status} evidence_count={len(state.evidence)} answer_len={len(state.answer or '')}")
    base = {
        "cost_usd": state.cost_usd,
        "token_usage": {
            "input_tokens": state.llm_input_tokens,
            "output_tokens": state.llm_output_tokens,
        },
    }
    if state.status == "NO_ANSWER":
        _log("assemble_response", f"Returning NO_ANSWER with {len(state.evidence)} evidence items")
        return {
            "answer": None,
            "status": "NO_ANSWER",
            "evidence": state.evidence,
            "explanation": state.explanation or "Question cannot be answered from available data.",
            **base,
        }
    _log("assemble_response", f"Returning OK with {len(state.evidence)} evidence items")
    return {
        "answer": state.answer,
        "status": "OK",
        "evidence": state.evidence,
        **base,
    }


def run_ask_pipeline(question: str) -> dict:
    _log("pipeline", f"START question='{question[:100]}'")
    state = OrchestratorState(question=question, start_time=time.time())

    state = llm_select_and_call_tools(state)
    _log("pipeline", f"AFTER llm_select_and_call_tools: tool_results={len(state.tool_results)} evidence={len(state.evidence)} status={state.status}")
    state = build_evidence(state)
    _log("pipeline", f"AFTER build_evidence: tool_results={len(state.tool_results)} evidence={len(state.evidence)} status={state.status}")
    result = assemble_response(state)

    elapsed_ms = round((time.time() - state.start_time) * 1000, 2)
    result["latency_ms"] = elapsed_ms
    _log("pipeline", f"END latency={elapsed_ms}ms status={result['status']} evidence_count={len(result['evidence'])} token_usage={state.llm_input_tokens}i/{state.llm_output_tokens}o")
    return result