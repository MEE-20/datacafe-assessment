"""Deterministic sales-vs-target analytics tool."""

from pathlib import Path
from typing import Optional

import pandas as pd

CLEAN_DIR = Path("data_cleaned")


class SalesAnalysisResult:
    def __init__(self, success: bool, data: Optional[pd.DataFrame] = None, error: Optional[str] = None):
        self.success = success
        self.data = data
        self.error = error

    def to_dict(self):
        if not self.success:
            return {"success": False, "error": self.error, "evidence": []}
        evidence = []
        for _, row in self.data.iterrows():
            entry = {}
            for col in self.data.columns:
                val = row[col]
                if isinstance(val, float):
                    entry[col] = round(float(val), 2)
                elif pd.isna(val):
                    entry[col] = None
                else:
                    entry[col] = val
            evidence.append(entry)
        return {"success": True, "evidence": evidence}


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fps = pd.read_parquet(CLEAN_DIR / "fact_primary_sales.parquet")
    ft = pd.read_parquet(CLEAN_DIR / "fact_targets.parquet")
    ds = pd.read_parquet(CLEAN_DIR / "dim_sku.parquet")
    return fps, ft, ds


def sales_total(
    region: Optional[str] = None,
    brand: Optional[str] = None,
    sku: Optional[str] = None,
    month: Optional[str] = None,
) -> SalesAnalysisResult:
    try:
        fps, ft, ds = _load()
        dg = pd.read_parquet(CLEAN_DIR / "dim_geo.parquet")

        df = fps.merge(ds[["sku_code", "brand"]], on="sku_code", how="left")
        df = df.merge(dg[["territory_code", "region"]], on="territory_code", how="left")

        if region:
            df = df[df["region"] == region]
        if brand:
            df = df[df["brand"] == brand]
        if sku:
            df = df[df["sku_code"] == sku]
        if month:
            df = df[df["month"] == month]

        agg = df.groupby(["month", "brand", "region", "sku_code"]).agg(
            total_units=("units", "sum"),
            total_value_inr=("value_inr", "sum"),
        ).reset_index()

        return SalesAnalysisResult(success=True, data=agg)
    except Exception as e:
        return SalesAnalysisResult(success=False, error=str(e))


def national_total_sales() -> SalesAnalysisResult:
    """Return the single national FY26 primary-sales total."""
    try:
        fps, _, _ = _load()
        total = float(fps["value_inr"].sum())
        df = pd.DataFrame([{"metric": "national_fy26_total_inr", "value": total}])
        return SalesAnalysisResult(success=True, data=df)
    except Exception as e:
        return SalesAnalysisResult(success=False, error=str(e))


def sales_by_region(
    region: Optional[str] = None,
    month: Optional[str] = None,
    top_n: Optional[int] = None,
) -> SalesAnalysisResult:
    """Aggregate sales by region (and optionally by month). Returns total units and value per region."""
    try:
        fps, _, ds = _load()
        dg = pd.read_parquet(CLEAN_DIR / "dim_geo.parquet")

        df = fps.merge(dg[["territory_code", "region"]], on="territory_code", how="left")

        if region:
            df = df[df["region"] == region]
        if month:
            df = df[df["month"] == month]

        agg = df.groupby("region").agg(
            total_units=("units", "sum"),
            total_value_inr=("value_inr", "sum"),
        ).reset_index().sort_values("total_value_inr", ascending=False)

        if top_n:
            agg = agg.head(top_n)

        return SalesAnalysisResult(success=True, data=agg)
    except Exception as e:
        return SalesAnalysisResult(success=False, error=str(e))


def target_achievement(
    region: Optional[str] = None,
    brand: Optional[str] = None,
    month: Optional[str] = None,
) -> SalesAnalysisResult:
    try:
        fps, ft, ds = _load()
        dg = pd.read_parquet(CLEAN_DIR / "dim_geo.parquet")

        fps = fps.merge(ds[["sku_code", "brand", "category"]], on="sku_code", how="left")
        fps = fps.merge(dg[["territory_code", "region"]], on="territory_code", how="left")

        if region:
            fps = fps[fps["region"] == region]
            ft = ft[ft["region"] == region]
        if brand:
            fps = fps[fps["brand"] == brand]
            ft = ft[ft["brand"] == brand]
        if month:
            fps = fps[fps["month"] == month]
            ft = ft[ft["month"] == month]

        actual = fps.groupby(["month", "brand", "region"])["value_inr"].sum().reset_index()
        actual.columns = ["month", "brand", "region", "actual_value_inr"]

        merged = actual.merge(ft, on=["month", "brand", "region"], how="outer")
        merged["achievement_pct"] = (merged["actual_value_inr"] / merged["target_value_inr"]) * 100

        merged = merged.merge(ds[["brand", "category"]].drop_duplicates(), on="brand", how="left")

        return SalesAnalysisResult(success=True, data=merged)
    except Exception as e:
        return SalesAnalysisResult(success=False, error=str(e))