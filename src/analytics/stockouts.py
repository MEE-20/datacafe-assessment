"""Deterministic stockout analytics tool."""

from pathlib import Path
from typing import Optional

import pandas as pd

CLEAN_DIR = Path("data_cleaned")

CHRONIC_THRESHOLD_WEEKS = 6
MULTI_SKU_THRESHOLD = 3
REPEATED_STOCKOUT_WEEKS = 2


class StockoutAnalysisResult:
    def __init__(self, success: bool, evidence: Optional[list] = None, error: Optional[str] = None):
        self.success = success
        self.evidence = evidence or []
        self.error = error

    def to_dict(self):
        return {"success": self.success, "evidence": self.evidence, "error": self.error}


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    so = pd.read_parquet(CLEAN_DIR / "stockouts.parquet")
    dd = pd.read_parquet(CLEAN_DIR / "dim_distributor.parquet")
    dg = pd.read_parquet(CLEAN_DIR / "dim_geo.parquet")
    ds = pd.read_parquet(CLEAN_DIR / "dim_sku.parquet")
    return so, dd, dg, ds


def stockout_events(
    region: Optional[str] = None,
    brand: Optional[str] = None,
    distributor: Optional[str] = None,
    sku: Optional[str] = None,
    month: Optional[str] = None,
) -> StockoutAnalysisResult:
    try:
        so, dd, dg, ds = _load()

        so = so.merge(dd[["distributor_id", "territory_code"]], on="distributor_id", how="left")
        so = so.merge(dg[["territory_code", "region"]], on="territory_code", how="left", suffixes=("", "_geo"))
        so["region"] = so["region_geo"].combine_first(so["region"])

        so = so.merge(ds[["sku_code", "brand", "category"]], on="sku_code", how="left")
        so["month"] = so["week_start"].dt.strftime("%Y-%m")

        if region:
            so = so[so["region"] == region]
        if brand:
            so = so[so["brand"] == brand]
        if distributor:
            so = so[so["distributor_id"] == distributor]
        if sku:
            so = so[so["sku_code"] == sku]
        if month:
            so = so[so["month"] == month]

        evidence = []
        for _, row in so.iterrows():
            evidence.append({
                "distributor_id": row["distributor_id"],
                "sku_code": row["sku_code"],
                "brand": row["brand"],
                "category": row["category"],
                "region": row["region"],
                "week_start": str(row["week_start"].date()),
                "days_out_of_stock": int(row["days_out_of_stock"]),
            })

        return StockoutAnalysisResult(success=True, evidence=evidence)
    except Exception as e:
        return StockoutAnalysisResult(success=False, error=str(e))


def chronic_stockouts(
    region: Optional[str] = None,
    brand: Optional[str] = None,
) -> StockoutAnalysisResult:
    try:
        so, dd, dg, ds = _load()

        so = so.merge(dd[["distributor_id", "territory_code"]], on="distributor_id", how="left")
        so = so.merge(dg[["territory_code", "region"]], on="territory_code", how="left", suffixes=("", "_geo"))
        so["region"] = so["region_geo"].combine_first(so["region"])
        so = so.merge(ds[["sku_code", "brand", "category"]], on="sku_code", how="left")

        if region:
            so = so[so["region"] == region]
        if brand:
            so = so[so["brand"] == brand]

        grp = so.groupby(["distributor_id", "sku_code", "region", "brand"]).size().reset_index(name="weeks_reported")
        chronic = grp[grp["weeks_reported"] > CHRONIC_THRESHOLD_WEEKS]

        evidence = []
        for _, row in chronic.iterrows():
            evidence.append({
                "distributor_id": row["distributor_id"],
                "sku_code": row["sku_code"],
                "brand": row["brand"],
                "region": row["region"],
                "weeks_with_stockout": int(row["weeks_reported"]),
                "threshold_weeks": CHRONIC_THRESHOLD_WEEKS,
            })

        return StockoutAnalysisResult(success=True, evidence=evidence)
    except Exception as e:
        return StockoutAnalysisResult(success=False, error=str(e))


def multi_sku_stockouts(
    region: Optional[str] = None,
    month: Optional[str] = None,
) -> StockoutAnalysisResult:
    try:
        so, dd, dg, ds = _load()

        so = so.merge(dd[["distributor_id", "territory_code"]], on="distributor_id", how="left")
        so = so.merge(dg[["territory_code", "region"]], on="territory_code", how="left", suffixes=("", "_geo"))
        so["region"] = so["region_geo"].combine_first(so["region"])
        so["month"] = so["week_start"].dt.strftime("%Y-%m")

        if region:
            so = so[so["region"] == region]
        if month:
            so = so[so["month"] == month]

        skus_per_dist_month = so.groupby(["distributor_id", "month", "region"])["sku_code"].nunique().reset_index()
        skus_per_dist_month.columns = ["distributor_id", "month", "region", "sku_count"]
        multi = skus_per_dist_month[skus_per_dist_month["sku_count"] >= MULTI_SKU_THRESHOLD]

        evidence = []
        for _, row in multi.iterrows():
            evidence.append({
                "distributor_id": row["distributor_id"],
                "region": row["region"],
                "month": row["month"],
                "sku_count": int(row["sku_count"]),
                "threshold_skus": MULTI_SKU_THRESHOLD,
            })

        return StockoutAnalysisResult(success=True, evidence=evidence)
    except Exception as e:
        return StockoutAnalysisResult(success=False, error=str(e))


def brand_has_repeated_stockouts(
    brand: str,
    region: str,
    month: str,
) -> bool:
    """Check if brand had >=2 stockout weeks in the region (used by R-01)."""
    so, dd, dg, ds = _load()

    so = so.merge(dd[["distributor_id", "territory_code"]], on="distributor_id", how="left")
    so = so.merge(dg[["territory_code", "region"]], on="territory_code", how="left", suffixes=("", "_geo"))
    so["region"] = so["region_geo"].combine_first(so["region"])
    so = so.merge(ds[["sku_code", "brand"]], on="sku_code", how="left")

    target_month_start = pd.to_datetime(f"{month}-01")
    relevant = so[
        (so["brand"] == brand)
        & (so["region"] == region)
        & (so["week_start"] >= target_month_start - pd.Timedelta(days=90))
    ]
    return relevant["week_start"].nunique() >= REPEATED_STOCKOUT_WEEKS