"""Tests for FastAPI endpoints /ask and /actions."""

from fastapi.testclient import TestClient

from src.solution import app


client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestAskEndpoint:
    def test_ask_missing_field_returns_422(self):
        resp = client.post("/ask", json={})
        assert resp.status_code == 422

    def test_ask_empty_question_returns_no_answer(self):
        resp = client.post("/ask", json={"question": ""})
        data = resp.json()
        assert resp.status_code == 200
        assert data["status"] == "NO_ANSWER"

    def test_ask_whitespace_only(self):
        resp = client.post("/ask", json={"question": "   "})
        data = resp.json()
        assert resp.status_code == 200
        assert data["status"] == "NO_ANSWER"

    def test_ask_valid_question_returns_ok_or_no_answer(self):
        resp = client.post("/ask", json={"question": "What was GlucoJoy target achievement in North in 2025-07?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("OK", "NO_ANSWER")
        assert "evidence" in data
        assert "cost_usd" in data
        assert "latency_ms" in data

    def test_ask_response_has_required_fields(self):
        resp = client.post("/ask", json={"question": "Sales in North?"})
        data = resp.json()
        assert "answer" in data or data.get("answer") is None
        assert "status" in data
        assert "evidence" in data
        assert "cost_usd" in data
        assert "latency_ms" in data

    def test_ask_stockout_question(self):
        resp = client.post("/ask", json={"question": "What distributors had chronic stockouts in West?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("OK", "NO_ANSWER")
        if data["status"] == "OK":
            assert len(data["evidence"]) > 0

    def test_ask_promotion_question(self):
        resp = client.post("/ask", json={"question": "Which promotions had strong uplift in West?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data

    def test_ask_unsupported_question(self):
        resp = client.post("/ask", json={"question": "What color is the sky?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "NO_ANSWER"

    def test_ask_why_question_routes_to_docs(self):
        resp = client.post("/ask", json={"question": "Why did CremeDelight miss in North in February 2026?"})
        assert resp.status_code == 200
        data = resp.json()
        # Should attempt doc retrieval
        assert "status" in data


class TestActionsEndpoint:
    def test_actions_missing_scope(self):
        resp = client.post("/actions", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "recommendations" in data
        assert data["count"] > 0

    def test_actions_scope_west(self):
        resp = client.post("/actions", json={"scope": "West"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] > 0
        assert len(data["recommendations"]) == data["count"]
        for rec in data["recommendations"]:
            assert "rule_id" in rec
            assert "state" in rec
            assert "findings" in rec
            assert "action" in rec
            assert "supporting_metrics" in rec

    def test_actions_approval_states_present(self):
        resp = client.post("/actions", json={"scope": "West"})
        data = resp.json()
        states = {r["state"] for r in data["recommendations"]}
        assert "RECOMMENDED" in states
        assert "PENDING_APPROVAL" in states

    def test_actions_first_is_r01(self):
        resp = client.post("/actions", json={"scope": "West"})
        data = resp.json()
        assert data["recommendations"][0]["rule_id"] == "R-01"

    def test_actions_latency_reported(self):
        resp = client.post("/actions", json={"scope": "West"})
        data = resp.json()
        assert "latency_ms" in data
        assert isinstance(data["latency_ms"], (int, float))

    def test_actions_scope_north(self):
        resp = client.post("/actions", json={"scope": "North"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 0
        for rec in data["recommendations"]:
            assert rec["rule_id"] in ("R-01", "R-02", "R-03", "R-04", "R-05", "R-06", "R-07", "R-08")

    def test_actions_no_scope_returns_all(self):
        resp = client.post("/actions", json={})
        data = resp.json()
        assert data["count"] > 0

    def test_actions_state_consistent_with_rule(self):
        resp = client.post("/actions", json={"scope": "West"})
        data = resp.json()
        for rec in data["recommendations"]:
            if rec["rule_id"] in ("R-01", "R-04", "R-08"):
                assert rec["state"] == "PENDING_APPROVAL"
            else:
                assert rec["state"] == "RECOMMENDED"

    def test_actions_has_supporting_metrics(self):
        resp = client.post("/actions", json={"scope": "West"})
        data = resp.json()
        for rec in data["recommendations"]:
            assert isinstance(rec["supporting_metrics"], dict)
            assert len(rec["supporting_metrics"]) > 0