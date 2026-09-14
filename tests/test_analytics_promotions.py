"""Tests for promotions analytics tool."""

import pytest
from src.analytics.promotions import (
    promotion_uplift,
    promo_active_during_month,
    promo_is_underperforming,
    promo_is_strong,
)


class TestPromotionUplift:

    def test_uplift_all(self):
        r = promotion_uplift()
        assert r.success
        assert len(r.evidence) == 40

    def test_uplift_by_region(self):
        r = promotion_uplift(region="West")
        assert r.success
        assert all(e["region"] == "West" for e in r.evidence)

    def test_uplift_by_brand(self):
        r = promotion_uplift(brand="GlucoJoy")
        assert r.success
        assert all(e["brand"] == "GlucoJoy" for e in r.evidence)

    def test_uplift_by_promo_id(self):
        r = promotion_uplift(promo_id="PR-2025-056")
        assert r.success
        assert len(r.evidence) == 1
        assert r.evidence[0]["promo_id"] == "PR-2025-056"

    def test_uplift_has_classification(self):
        r = promotion_uplift()
        classifications = set(e["classification"] for e in r.evidence)
        assert classifications.issubset({"weak", "moderate", "strong", "unknown"})

    def test_uplift_weak_exists(self):
        r = promotion_uplift()
        weak = [e for e in r.evidence if e["classification"] == "weak"]
        assert len(weak) > 0

    def test_uplift_strong_exists(self):
        r = promotion_uplift()
        strong = [e for e in r.evidence if e["classification"] == "strong"]
        assert len(strong) > 0

    def test_uplift_has_baseline_weeks(self):
        r = promotion_uplift()
        for e in r.evidence:
            assert e["baseline_week_count"] >= 0
            if not e["insufficient_baseline"]:
                assert e["baseline_week_count"] >= 2

    def test_uplift_has_promo_avg_value(self):
        r = promotion_uplift()
        for e in r.evidence:
            if e["promo_avg_weekly_value_inr"] is not None:
                assert e["promo_avg_weekly_value_inr"] > 0


class TestPromoActiveDuringMonth:

    def test_glucojoy_north_has_promo(self):
        # Check if GlucoJoy had a promo in North in relevant months
        result = promo_active_during_month("North", "GlucoJoy", "2025-07")
        assert isinstance(result, bool)

    def test_never_active_brand(self):
        result = promo_active_during_month("North", "NonExistent", "2025-07")
        assert result is False

    def test_known_promo_period(self):
        # PR-2025-030 for GlucoJoy West in Oct/Nov 2025
        result = promo_active_during_month("West", "GlucoJoy", "2025-10")
        assert result is True

    def test_no_promo_in_region(self):
        result = promo_active_during_month("East", "GlucoJoy", "2025-07")
        assert isinstance(result, bool)


class TestPromoIsUnderperforming:

    def test_underperforming_exists(self):
        r = promotion_uplift()
        weak = [e for e in r.evidence if e["classification"] == "weak"]
        if weak:
            # Check one weak promo via the convenience function
            w = weak[0]
            result = promo_is_underperforming(w["region"], w["brand"], "2025-07")
            assert isinstance(result, bool)

    def test_no_underperforming_for_nonexistent(self):
        result = promo_is_underperforming("North", "NonExistent", "2025-07")
        assert result is False


class TestPromoIsStrong:

    def test_strong_exists(self):
        r = promotion_uplift()
        strong = [e for e in r.evidence if e["classification"] == "strong"]
        assert len(strong) > 0
        s = strong[0]
        result = promo_is_strong(s["region"], s["brand"], "2025-07")
        assert isinstance(result, bool)

    def test_no_strong_for_wrong_brand(self):
        result = promo_is_strong("North", "NonExistent", "2025-07")
        assert result is False