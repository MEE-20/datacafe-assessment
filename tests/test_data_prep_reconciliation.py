"""Tests for data preparation and reconciliation."""

import json

import pytest
import pandas as pd


class TestDataPrepReconciliation:

    def test_reconciliation_report_exists(self, reconciliation_report_path):
        assert reconciliation_report_path.exists()

    def test_reconciliation_report_contents(self, reconciliation_report_path):
        with open(reconciliation_report_path) as f:
            report = json.load(f)
        assert "national_fy26_primary_sales_total_inr" in report
        assert isinstance(report["national_fy26_primary_sales_total_inr"], float)
        assert report["national_fy26_primary_sales_total_inr"] > 0

    def test_no_reconciliation_mismatches(self, reconciliation_report_path):
        with open(reconciliation_report_path) as f:
            report = json.load(f)
        mismatches = report.get("reconciliation_mismatches", [])
        assert len(mismatches) == 0, f"Unexpected mismatches: {mismatches}"

    def test_all_sources_present(self, reconciliation_report_path):
        with open(reconciliation_report_path) as f:
            report = json.load(f)
        expected_sources = [
            "dim_geo", "dim_sku", "dim_distributor",
            "fact_primary_sales", "fact_targets", "promotions",
            "stockouts", "promotion_baselines",
        ]
        for s in expected_sources:
            assert s in report["data_sources"], f"Missing source: {s}"

    def test_row_counts_retained(self, reconciliation_report_path):
        with open(reconciliation_report_path) as f:
            report = json.load(f)
        expected = {
            "dim_geo": 12,
            "dim_sku": 120,
            "dim_distributor": 40,
            "fact_primary_sales": 74880,
            "fact_targets": 720,
            "promotions": 40,
            "stockouts": 520,
            "promotion_baselines": 40,
        }
        for name, count in expected.items():
            ds = report["data_sources"].get(name)
            assert ds is not None, f"Missing source: {name}"
            assert ds["raw_rows"] == count, f"{name}: expected {count}, got {ds['raw_rows']}"
            assert ds["cleaned_rows"] == count, f"{name}: expected {count} cleaned, got {ds['cleaned_rows']}"

    def test_national_total_plausible(self, reconciliation_report_path):
        with open(reconciliation_report_path) as f:
            report = json.load(f)
        total = report["national_fy26_primary_sales_total_inr"]
        # FY26 total should be in the billions for FMCG with 120 SKUs × 12 territories
        assert 1_000_000_000 < total < 2_000_000_000, f"Total {total} outside expected range"

    def test_stockout_region_normalization(self, reconciliation_report_path):
        with open(reconciliation_report_path) as f:
            report = json.load(f)
        dist = report.get("stockout_region_distribution_after_normalization", {})
        assert set(dist.keys()) == {"North", "South", "East", "West"}
        total = sum(dist.values())
        assert total == 520, f"Stockout events total: {total}, expected 520"

    def test_skus_never_in_stockouts(self, reconciliation_report_path):
        with open(reconciliation_report_path) as f:
            report = json.load(f)
        skus = report.get("skus_never_in_stockouts", [])
        assert len(skus) == 2
        assert "BS-0107" in skus
        assert "BS-0202" in skus

    def test_all_parquet_files_exist(self, cleaned_data_dir):
        expected = [
            "dim_geo.parquet", "dim_sku.parquet", "dim_distributor.parquet",
            "fact_primary_sales.parquet", "fact_targets.parquet",
            "promotions.parquet", "stockouts.parquet", "promotion_baselines.parquet",
        ]
        for f in expected:
            assert (cleaned_data_dir / f).is_file(), f"Missing {f}"

    def test_parquet_columns_match_report(self, cleaned_data_dir, reconciliation_report_path):
        with open(reconciliation_report_path) as f:
            report = json.load(f)
        cleaned = report.get("cleaned_data_columns", {})
        for name, expected_cols in cleaned.items():
            df = pd.read_parquet(cleaned_data_dir / f"{name}.parquet")
            assert list(df.columns) == expected_cols, f"{name}: columns mismatch"

    def test_no_duplicate_primary_keys_in_cleaned(self, cleaned_data_dir):
        pks = {
            "dim_geo": ["territory_code"],
            "dim_sku": ["sku_code"],
            "dim_distributor": ["distributor_id"],
            "fact_primary_sales": ["week_start", "sku_code", "territory_code"],
            "fact_targets": ["month", "brand", "region"],
            "promotions": ["promo_id"],
        }
        for name, pk_cols in pks.items():
            df = pd.read_parquet(cleaned_data_dir / f"{name}.parquet")
            dups = df.duplicated(subset=pk_cols).sum()
            assert dups == 0, f"{name}: {dups} duplicate PKs"

    def test_foreign_keys_valid(self, cleaned_data_dir):
        fps = pd.read_parquet(cleaned_data_dir / "fact_primary_sales.parquet")
        sku = set(pd.read_parquet(cleaned_data_dir / "dim_sku.parquet")["sku_code"])
        geo = set(pd.read_parquet(cleaned_data_dir / "dim_geo.parquet")["territory_code"])
        assert set(fps["sku_code"].unique()) - sku == set()
        assert set(fps["territory_code"].unique()) - geo == set()

        so = pd.read_parquet(cleaned_data_dir / "stockouts.parquet")
        dd = set(pd.read_parquet(cleaned_data_dir / "dim_distributor.parquet")["distributor_id"])
        assert set(so["sku_code"].unique()) - sku == set()
        assert set(so["distributor_id"].unique()) - dd == set()

    def test_stockouts_region_normalized_to_4(self, cleaned_data_dir):
        so = pd.read_parquet(cleaned_data_dir / "stockouts.parquet")
        assert len(so["region"].unique()) == 4
        assert set(so["region"].unique()) == {"North", "South", "East", "West"}

    def test_promotion_baselines_have_uplift(self, cleaned_data_dir):
        pb = pd.read_parquet(cleaned_data_dir / "promotion_baselines.parquet")
        assert len(pb) == 40
        assert "uplift_pct" in pb.columns
        assert "insufficient_baseline" in pb.columns

    def test_final_week_partial_note(self, cleaned_data_dir):
        fps = pd.read_parquet(cleaned_data_dir / "fact_primary_sales.parquet")
        last = fps["week_start"].max()
        assert last == pd.Timestamp("2026-06-23")
        june = fps[fps["month"] == "2026-06"]
        # June has 4 weeks (5 weekly rows for each SKU/territory that cross the boundary - but mapped by week_start)
        # Week starts: 2026-06-02, 2026-06-09, 2026-06-16, 2026-06-23 = 4 weeks
        june_weeks = june["week_start"].nunique()
        assert june_weeks == 4
        assert len(june) == 12 * 120 * 4  # 1440 rows per week × 4 weeks