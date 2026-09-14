"""Tests for distributor_sku analytics tool."""

import pytest
from src.analytics.distributor_sku import distributor_profile, sku_profile


class TestDistributorProfile:

    def test_known_distributor(self):
        r = distributor_profile("D001")
        assert r.success
        e = r.evidence[0]
        assert e["distributor_id"] == "D001"
        assert e["distributor_name"] == "Delhi Sales Corp 1"
        assert e["territory_code"] == "T-N1"
        assert e["city"] == "Delhi NCR"
        assert e["region"] == "North"

    def test_distributor_with_stockouts(self):
        r = distributor_profile("D032")
        assert r.success
        e = r.evidence[0]
        assert e["total_stockout_events"] > 0
        assert e["unique_skus_with_stockouts"] > 0
        assert e["region"] == "West"

    def test_distributor_with_multiple_skus(self):
        r = distributor_profile("D001")
        assert r.success
        e = r.evidence[0]
        assert e["unique_skus_with_stockouts"] > 0

    def test_unknown_distributor(self):
        r = distributor_profile("D999")
        assert not r.success
        assert r.error is not None

    def test_stockout_history_format(self):
        r = distributor_profile("D032")
        e = r.evidence[0]
        history = e["stockout_history"]
        assert len(history) > 0
        for entry in history:
            assert "sku_code" in entry
            assert "brand" in entry
            assert "days_out_of_stock" in entry
            assert "week_start" in entry


class TestSkuProfile:

    def test_known_sku(self):
        r = sku_profile("BS-0101")
        assert r.success
        e = r.evidence[0]
        assert e["sku_code"] == "BS-0101"
        assert e["sku_name"] == "GlucoJoy 50g"
        assert e["brand"] == "GlucoJoy"
        assert e["category"] == "Biscuits"
        assert e["total_sales_units_fy26"] > 0
        assert e["total_sales_value_fy26_inr"] > 0

    def test_sku_with_stockouts(self):
        r = sku_profile("BV-0104")
        assert r.success
        e = r.evidence[0]
        assert e["total_stockout_events"] > 0
        assert e["unique_distributors_with_stockouts"] > 0

    def test_sku_without_stockouts(self):
        r = sku_profile("BS-0202")
        assert r.success
        e = r.evidence[0]
        assert e["total_stockout_events"] == 0

    def test_sku_has_mrp(self):
        r = sku_profile("BS-0101")
        e = r.evidence[0]
        assert e["mrp_inr"] == 34

    def test_unknown_sku(self):
        r = sku_profile("ZZ-9999")
        assert not r.success
        assert r.error is not None