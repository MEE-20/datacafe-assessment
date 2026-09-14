"""Deterministic promotion uplift analytics tool."""

from pathlib import Path
from typing import Optional

import pandas as pd

CLEAN_DIR = Path("data_cleaned")

WEAK_UPLIFT_THRESHOLD = 10.0
STRONG_UPLIFT_THRESHOLD = 25.0


class PromotionAnalysisResult:
    def __init__(self, success: bool, evidence: Optional[list] = None, error: Optional[str] = None):
        self.success = success
        self.evidence = evidence or []
        self.error = error

    def to_dict(self):
        return {"success": self.success, "evidence": self.evidence, "error": self.error}


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    pb = pd.read_parquet(CLEAN_DIR / "promotion_baselines.parquet")
    fps = pd.read_parquet(CLEAN_DIR / "fact_primary_sales.parquet")
    return pb, fps


def promotion_uplift(
    region: Optional[str] = None,
    brand: Optional[str] = None,
    promo_id: Optional[str] = None,
) -> PromotionAnalysisResult:
    try:
        pb, fps = _load()
        ds = pd.read_parquet(CLEAN_DIR / "dim_sku.parquet")

        pb = pb.merge(ds[["sku_code", "brand", "category"]], on="sku_code", how="left")

        if region:
            pb = pb[pb["region"] == region]
        if brand:
            pb = pb[pb["brand"] == brand]
        if promo_id:
            pb = pb[pb["promo_id"] == promo_id]

        evidence = []
        for _, row in pb.iterrows():
            uplift = row.get("uplift_pct")
            if uplift is not None:
                if uplift < WEAK_UPLIFT_THRESHOLD:
                    classification = "weak"
                elif uplift > STRONG_UPLIFT_THRESHOLD:
                    classification = "strong"
                else:
                    classification = "moderate"
            else:
                classification = "unknown"

            evidence.append({
                "promo_id": row["promo_id"],
                "sku_code": row["sku_code"],
                "brand": row["brand"],
                "category": row["category"],
                "region": row["region"],
                "start_date": str(row["start_date"].date()) if pd.notna(row.get("start_date")) else None,
                "end_date": str(row["end_date"].date()) if pd.notna(row.get("end_date")) else None,
                "promo_avg_weekly_value_inr": row.get("promo_avg_weekly_value"),
                "baseline_avg_weekly_value_inr": row.get("baseline_avg_weekly_value"),
                "baseline_week_count": row.get("baseline_week_count"),
                "uplift_pct": uplift,
                "classification": classification,
                "insufficient_baseline": bool(row.get("insufficient_baseline", False)),
            })

        return PromotionAnalysisResult(success=True, evidence=evidence)
    except Exception as e:
        return PromotionAnalysisResult(success=False, error=str(e))


def promo_active_during_month(region: str, brand: str, month: str) -> bool:
    """Check if any promotion was active for this brand in this region during the month."""
    pb, fps = _load()
    ds = pd.read_parquet(CLEAN_DIR / "dim_sku.parquet")

    pb = pb.merge(ds[["sku_code", "brand"]], on="sku_code", how="left")
    month_start = pd.Timestamp(f"{month}-01")
    if month == "2026-06":
        month_end = pd.Timestamp("2026-06-30")
    else:
        month_end = month_start + pd.offsets.MonthEnd(0)

    relevant = pb[
        (pb["brand"] == brand)
        & (pb["region"] == region)
        & (pb["start_date"] <= month_end)
        & (pb["end_date"] >= month_start)
    ]
    return len(relevant) > 0


def promo_is_underperforming(region: str, brand: str, month: str) -> bool:
    """Check if an active promotion has weak uplift (<10%)."""
    pb, fps = _load()
    ds = pd.read_parquet(CLEAN_DIR / "dim_sku.parquet")

    pb = pb.merge(ds[["sku_code", "brand"]], on="sku_code", how="left")
    month_start = pd.Timestamp(f"{month}-01")
    if month == "2026-06":
        month_end = pd.Timestamp("2026-06-30")
    else:
        month_end = month_start + pd.offsets.MonthEnd(0)

    relevant = pb[
        (pb["brand"] == brand)
        & (pb["region"] == region)
        & (pb["start_date"] <= month_end)
        & (pb["end_date"] >= month_start)
    ]

    for _, row in relevant.iterrows():
        uplift = row.get("uplift_pct")
        if uplift is not None and uplift < WEAK_UPLIFT_THRESHOLD:
            return True
    return False


def promo_is_strong(region: str, brand: str, month: str) -> bool:
    """Check if any active promotion has strong uplift (>25%)."""
    pb, fps = _load()
    ds = pd.read_parquet(CLEAN_DIR / "dim_sku.parquet")

    pb = pb.merge(ds[["sku_code", "brand"]], on="sku_code", how="left")
    month_start = pd.Timestamp(f"{month}-01")
    if month == "2026-06":
        month_end = pd.Timestamp("2026-06-30")
    else:
        month_end = month_start + pd.offsets.MonthEnd(0)

    relevant = pb[
        (pb["brand"] == brand)
        & (pb["region"] == region)
        & (pb["start_date"] <= month_end)
        & (pb["end_date"] >= month_start)
    ]

    for _, row in relevant.iterrows():
        uplift = row.get("uplift_pct")
        if uplift is not None and uplift > STRONG_UPLIFT_THRESHOLD:
            return True
    return False