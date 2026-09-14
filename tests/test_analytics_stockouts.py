"""Tests for stockouts analytics tool."""

import pytest
from src.analytics.stockouts import (
    stockout_events,
    chronic_stockouts,
    multi_sku_stockouts,
    brand_has_repeated_stockouts,
)


class TestStockoutEvents:

    def test_stockout_all(self):
        r = stockout_events()
        assert r.success
        assert len(r.evidence) == 520

    def test_stockout_by_region(self):
        r = stockout_events(region="West")
        assert r.success
        assert all(e["region"] == "West" for e in r.evidence)

    def test_stockout_by_region_normalized(self):
        r = stockout_events(region="North")
        assert r.success
        assert all(e["region"] == "North" for e in r.evidence)

    def test_stockout_by_brand(self):
        r = stockout_events(brand="Aqualite")
        assert r.success
        assert all(e["brand"] == "Aqualite" for e in r.evidence)

    def test_stockout_by_sku(self):
        r = stockout_events(sku="BV-0104")
        assert r.success
        assert all(e["sku_code"] == "BV-0104" for e in r.evidence)

    def test_stockout_by_distributor(self):
        r = stockout_events(distributor="D032")
        assert r.success
        assert all(e["distributor_id"] == "D032" for e in r.evidence)

    def test_stockout_by_month(self):
        r = stockout_events(month="2025-07")
        assert r.success
        # Check first few
        assert len(r.evidence) > 0

    def test_stockout_days_range(self):
        r = stockout_events()
        for e in r.evidence:
            assert 1 <= e["days_out_of_stock"] <= 7

    def test_stockout_has_brand_and_category(self):
        r = stockout_events(sku="BV-0104")
        for e in r.evidence:
            assert e["brand"] is not None
            assert e["category"] is not None


class TestChronicStockouts:

    def test_chronic_count(self):
        r = chronic_stockouts()
        assert r.success
        # From data recon: D032 and D033 on BV-0104
        assert len(r.evidence) == 2

    def test_chronic_details(self):
        r = chronic_stockouts()
        for e in r.evidence:
            assert e["weeks_with_stockout"] > 6
            assert e["threshold_weeks"] == 6

    def test_chronic_by_region(self):
        r = chronic_stockouts(region="West")
        assert r.success
        assert len(r.evidence) >= 1
        assert all(e["region"] == "West" for e in r.evidence)

    def test_chronic_by_region_no_results(self):
        r = chronic_stockouts(region="North")
        assert r.success
        assert len(r.evidence) == 0

    def test_chronic_by_brand(self):
        r = chronic_stockouts(brand="Aqualite")
        assert r.success
        assert len(r.evidence) == 2

    def test_chronic_wrong_brand(self):
        r = chronic_stockouts(brand="GlucoJoy")
        assert r.success
        assert len(r.evidence) == 0


class TestMultiSkuStockouts:

    def test_multi_sku_count(self):
        r = multi_sku_stockouts()
        assert r.success
        assert len(r.evidence) > 0
        for e in r.evidence:
            assert e["sku_count"] >= 3

    def test_multi_sku_by_region(self):
        r = multi_sku_stockouts(region="West")
        assert r.success
        assert all(e["region"] == "West" for e in r.evidence)

    def test_multi_sku_by_month(self):
        r = multi_sku_stockouts(month="2026-06")
        assert r.success
        if r.evidence:
            assert all(e["month"] == "2026-06" for e in r.evidence)


class TestBrandHasRepeatedStockouts:

    def test_glucojoy_north_repeated(self):
        result = brand_has_repeated_stockouts("GlucoJoy", "North", "2025-07")
        assert result is True

    def test_cremedelight_north_false(self):
        result = brand_has_repeated_stockouts("CremeDelight", "North", "2025-07")
        assert result is False

    def test_aqualite_west_repeated(self):
        result = brand_has_repeated_stockouts("Aqualite", "West", "2026-06")
        assert result is True

    def test_unknown_brand(self):
        result = brand_has_repeated_stockouts("NonExistent", "North", "2025-07")
        assert result is False