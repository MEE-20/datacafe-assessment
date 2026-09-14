"""Tests for sales_target analytics tool."""

import pytest
from src.analytics.sales_target import sales_total, target_achievement


class TestSalesTotal:

    def test_sales_total_no_filter(self):
        r = sales_total()
        assert r.success
        assert r.data is not None
        assert len(r.data) > 0

    def test_sales_total_by_region(self):
        r = sales_total(region="West")
        assert r.success
        regions = r.data["region"].unique()
        assert list(regions) == ["West"]

    def test_sales_total_by_brand(self):
        r = sales_total(brand="GlucoJoy")
        assert r.success
        brands = r.data["brand"].unique()
        assert list(brands) == ["GlucoJoy"]

    def test_sales_total_by_sku(self):
        r = sales_total(sku="BS-0101")
        assert r.success
        skus = r.data["sku_code"].unique()
        assert list(skus) == ["BS-0101"]

    def test_sales_total_by_month(self):
        r = sales_total(month="2025-07")
        assert r.success
        months = r.data["month"].unique()
        assert list(months) == ["2025-07"]

    def test_sales_total_brand_not_found(self):
        r = sales_total(brand="NonExistentBrand")
        assert r.success
        assert len(r.data) == 0


class TestTargetAchievement:

    def test_achievement_known_brand_region_month(self):
        r = target_achievement(brand="GlucoJoy", region="North", month="2025-07")
        assert r.success
        d = r.to_dict()
        assert d["success"]
        assert len(d["evidence"]) > 0
        ev = d["evidence"][0]
        assert ev["brand"] == "GlucoJoy"
        assert ev["region"] == "North"
        assert ev["month"] == "2025-07"
        assert ev["target_value_inr"] > 0
        assert ev["actual_value_inr"] > 0
        assert 0 < ev["achievement_pct"] < 200

    def test_achievement_no_filters_returns_all(self):
        r = target_achievement()
        assert r.success
        d = r.to_dict()
        # 15 brands × 4 regions × 12 months = 720 rows in targets
        assert len(d["evidence"]) == 720

    def test_achievement_by_region_only(self):
        r = target_achievement(region="South")
        assert r.success
        d = r.to_dict()
        for ev in d["evidence"]:
            assert ev["region"] == "South"

    def test_achievement_by_brand_only(self):
        r = target_achievement(brand="Aqualite")
        assert r.success
        d = r.to_dict()
        for ev in d["evidence"]:
            assert ev["brand"] == "Aqualite"

    def test_achievement_has_category(self):
        r = target_achievement(brand="GlucoJoy", region="North", month="2025-07")
        d = r.to_dict()
        assert d["evidence"][0]["category"] == "Biscuits"

    def test_achievement_over_delivery(self):
        r = target_achievement(region="East", month="2025-07")
        d = r.to_dict()
        over = [ev for ev in d["evidence"] if ev["achievement_pct"] > 100]
        assert len(over) > 0, "Expected some brands to over-deliver"

    def test_achievement_below_70(self):
        r = target_achievement()
        d = r.to_dict()
        below70 = [ev for ev in d["evidence"] if ev["achievement_pct"] < 70]
        assert len(below70) > 0, "Expected some brand/region/months to be below 70%"

    def test_no_false_data(self):
        r = target_achievement()
        d = r.to_dict()
        for ev in d["evidence"]:
            assert ev["target_value_inr"] > 0
            assert ev["actual_value_inr"] >= 0
            assert "month" in ev
            assert ev["month"] is not None