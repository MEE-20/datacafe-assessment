"""Deterministic distributor/SKU-level analytics tool."""

from pathlib import Path
from typing import Optional

import pandas as pd

CLEAN_DIR = Path("data_cleaned")


class DistributorSkuResult:
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


def distributor_profile(distributor_id: str) -> DistributorSkuResult:
    try:
        so, dd, dg, ds = _load()

        dist_info = dd[dd["distributor_id"] == distributor_id]
        if len(dist_info) == 0:
            return DistributorSkuResult(success=False, error=f"Distributor {distributor_id} not found")

        dist_info = dist_info.iloc[0]
        territory = dist_info["territory_code"]
        geo = dg[dg["territory_code"] == territory]
        region = geo.iloc[0]["region"] if len(geo) > 0 else None

        stockout_history = so[so["distributor_id"] == distributor_id].copy()
        stockout_history = stockout_history.merge(ds[["sku_code", "brand", "category"]], on="sku_code", how="left")
        stockout_history["month"] = stockout_history["week_start"].dt.strftime("%Y-%m")

        total_stockout_events = len(stockout_history)
        unique_skus = stockout_history["sku_code"].nunique()

        evidence = [
            {
                "distributor_id": distributor_id,
                "distributor_name": dist_info["distributor_name"],
                "territory_code": territory,
                "city": dist_info["city"],
                "region": region,
                "total_stockout_events": total_stockout_events,
                "unique_skus_with_stockouts": unique_skus,
                "stockout_history": sorted(stockout_history.to_dict("records"), key=lambda x: x["week_start"]) if total_stockout_events > 0 else [],
            }
        ]

        return DistributorSkuResult(success=True, evidence=evidence)
    except Exception as e:
        return DistributorSkuResult(success=False, error=str(e))


def sku_profile(sku_code: str) -> DistributorSkuResult:
    try:
        so, dd, dg, ds = _load()
        fps = pd.read_parquet(CLEAN_DIR / "fact_primary_sales.parquet")

        sku_info = ds[ds["sku_code"] == sku_code]
        if len(sku_info) == 0:
            return DistributorSkuResult(success=False, error=f"SKU {sku_code} not found")
        sku_info = sku_info.iloc[0]

        sales = fps[fps["sku_code"] == sku_code]
        total_units = int(sales["units"].sum())
        total_value = float(sales["value_inr"].sum())

        stockouts = so[so["sku_code"] == sku_code]
        stockouts = stockouts.merge(dd[["distributor_id", "distributor_name"]], on="distributor_id", how="left")
        stockouts["month"] = stockouts["week_start"].dt.strftime("%Y-%m")

        evidence = [
            {
                "sku_code": sku_code,
                "sku_name": sku_info["sku_name"],
                "brand": sku_info["brand"],
                "category": sku_info["category"],
                "pack_size": sku_info["pack_size"],
                "mrp_inr": int(sku_info["mrp_inr"]),
                "total_sales_units_fy26": total_units,
                "total_sales_value_fy26_inr": total_value,
                "total_stockout_events": len(stockouts),
                "unique_distributors_with_stockouts": stockouts["distributor_id"].nunique(),
            }
        ]

        return DistributorSkuResult(success=True, evidence=evidence)
    except Exception as e:
        return DistributorSkuResult(success=False, error=str(e))