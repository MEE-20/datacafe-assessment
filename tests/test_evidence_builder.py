"""Tests for evidence generation in the orchestrator pipeline."""

import pytest
from src.orchestrator.graph import OrchestratorState, build_evidence, run_ask_pipeline


class TestEvidenceBuilder:

    def test_successful_evidence_is_built(self):
        state = OrchestratorState(question="test")
        state.tool_results = [
            {
                "tool_name": "target_achievement",
                "success": True,
                "evidence": [{
                    "month": "2025-07",
                    "brand": "GlucoJoy",
                    "region": "North",
                    "target_value_inr": 2864000.0,
                    "actual_value_inr": 2736914.0,
                    "achievement_pct": 95.56,
                }],
                "error": None,
            }
        ]
        state = build_evidence(state)
        assert state.status == "OK"
        assert len(state.evidence) == 1
        assert state.evidence[0]["tool"] == "target_achievement"
        assert state.evidence[0]["data"]["achievement_pct"] == 95.56

    def test_no_evidence_sets_no_answer(self):
        state = OrchestratorState(question="test")
        state.tool_results = [
            {"tool_name": "target_achievement", "success": False, "evidence": [], "error": "No data found"}
        ]
        state = build_evidence(state)
        assert state.status == "NO_ANSWER"
        assert state.explanation is not None

    def test_partial_success_still_ok(self):
        state = OrchestratorState(question="test")
        state.tool_results = [
            {"tool_name": "target_achievement", "success": True, "evidence": [{"achievement_pct": 95.0}], "error": None},
            {"tool_name": "stockout_events", "success": False, "evidence": [], "error": "No stockouts"},
        ]
        state = build_evidence(state)
        assert state.status == "OK"
        assert len(state.evidence) == 2

    def test_all_failures_returns_no_answer(self):
        state = OrchestratorState(question="What is the meaning of life?")
        state.tool_results = [
            {"tool_name": "document_retrieval", "success": False, "evidence": [], "error": "No docs found"}
        ]
        state = build_evidence(state)
        assert state.status == "NO_ANSWER"
        assert state.explanation is not None

    def test_unsupported_has_no_evidence(self):
        result = run_ask_pipeline("What is the stock price of ACPL?")
        assert result["status"] == "NO_ANSWER"

    def test_empty_question(self):
        result = run_ask_pipeline("")
        assert "status" in result

    def test_cost_and_latency_present(self):
        result = run_ask_pipeline("What was sales in North?")
        assert "cost_usd" in result
        assert "latency_ms" in result
        assert isinstance(result["latency_ms"], (int, float))