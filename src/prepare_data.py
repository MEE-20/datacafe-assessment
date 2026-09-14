"""Offline data preparation: raw CSVs to cleaned analytical datasets."""

import json
import shutil
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/fmcg-sales-copilot-ai-engineer-mid-4to6")
CLEAN_DIR = Path("data_cleaned")
REPORT_PATH = Path("reconciliation_report.json")

STOCKOUT_REGION_MAP = {
    "North": "North",
    "NORTH": "North",
    "north": "North",
    "North Region": "North",
    "South": "South",
    "SOUTH": "South",
    "south": "South",
    "South Region": "South",
    "East": "East",
    "EAST": "East",
    "east": "East",
    "East Region": "East",
    "West": "West",
    "WEST": "West",
    "west": "West",
    "West Region": "West",
}

COLUMN_RENAMES = {
    "promotions": {"sku": "sku_code"},
    "stockouts": {"item_code": "sku_code"},
    "fact_targets": {"brand_name": "brand", "region_name": "region"},
}

REQUIRED_COLS = {
    "dim_geo": ["territory_code", "territory_name", "region", "state"],
    "dim_sku": ["sku_code", "sku_name", "brand", "category", "pack_size", "mrp_inr"],
    "dim_distributor": ["distributor_id", "distributor_name", "territory_code", "city"],
    "fact_primary_sales": ["week_start", "sku_code", "territory_code", "units", "value_inr"],
    "fact_targets": ["month", "brand_name", "region_name", "target_value_inr"],
    "promotions": ["promo_id", "sku", "region", "start_date", "end_date", "discount_pct", "mechanic"],
    "stockouts": ["distributor_id", "item_code", "week_start", "region", "days_out_of_stock"],
}

PK_COLS = {
    "dim_geo": ["territory_code"],
    "dim_sku": ["sku_code"],
    "dim_distributor": ["distributor_id"],
    "fact_primary_sales": ["week_start", "sku_code", "territory_code"],
    "fact_targets": ["month", "brand", "region"],
    "promotions": ["promo_id"],
    "stockouts": None,
}

DATE_COLS = {
    "fact_primary_sales": {"week_start": "%Y-%m-%d"},
    "promotions": {"start_date": "%d/%m/%Y", "end_date": "%d/%m/%Y"},
    "stockouts": {"week_start": "%Y-%m-%d"},
}

NUMERIC_COLS = {
    "fact_primary_sales": {"units": int, "value_inr": float},
    "fact_targets": {"target_value_inr": int},
    "promotions": {"discount_pct": int},
    "stockouts": {"days_out_of_stock": int},
}


def load_raw(name: str) -> pd.DataFrame:
    path = RAW_DIR / f"{name}.csv"
    df = pd.read_csv(path)
    return df


def validate_schema(df: pd.DataFrame, name: str) -> list:
    issues = []
    expected = REQUIRED_COLS.get(name, [])
    for col in expected:
        if col not in df.columns:
            issues.append(f"Missing column '{col}' in {name}")
    return issues


def validate_types(df: pd.DataFrame, name: str) -> list:
    issues = []
    for col, fmt in DATE_COLS.get(name, {}).items():
        if col not in df.columns:
            continue
        parsed = pd.to_datetime(df[col], format=fmt, errors="coerce")
        failed = parsed.isna().sum()
        if failed > 0:
            issues.append(f"{failed} unparseable dates in {name}.{col}")
        df[col] = parsed
    for col, expected_type in NUMERIC_COLS.get(name, {}).items():
        if col not in df.columns:
            continue
        if df[col].dtype != expected_type:
            df[col] = df[col].astype(expected_type)
    return issues


def ensure_month_as_string(df: pd.DataFrame, col: str = "month") -> pd.DataFrame:
    if col in df.columns and df[col].dtype != "object":
        df[col] = df[col].astype(str)
    return df


def validate_pks(df: pd.DataFrame, name: str) -> list:
    issues = []
    pk = PK_COLS.get(name)
    if pk is None:
        return issues
    dups = df.duplicated(subset=pk).sum()
    if dups > 0:
        issues.append(f"{dups} duplicate rows (PK={pk}) in {name}")
    return issues


def normalize_column_names(df: pd.DataFrame, name: str) -> pd.DataFrame:
    renames = COLUMN_RENAMES.get(name, {})
    return df.rename(columns=renames)


def normalize_stockout_regions(df: pd.DataFrame) -> pd.DataFrame:
    unmapped = df[~df["region"].isin(STOCKOUT_REGION_MAP.keys())]["region"].unique()
    if len(unmapped) > 0:
        print(f"  WARNING: unmapped stockout regions: {unmapped}")
    df["region"] = df["region"].map(STOCKOUT_REGION_MAP).fillna(df["region"])
    return df


def validate_foreign_keys(dfs: dict) -> list:
    issues = []
    fps = dfs["fact_primary_sales"]
    dg = dfs["dim_geo"]
    ds = dfs["dim_sku"]

    bad_territories = set(fps["territory_code"].unique()) - set(dg["territory_code"].unique())
    if bad_territories:
        issues.append(f"Territories in sales not in dim_geo: {bad_territories}")
    bad_skus = set(fps["sku_code"].unique()) - set(ds["sku_code"].unique())
    if bad_skus:
        issues.append(f"SKUs in sales not in dim_sku: {bad_skus}")

    so = dfs["stockouts"]
    dd = dfs["dim_distributor"]
    bad_distributors = set(so["distributor_id"].unique()) - set(dd["distributor_id"].unique())
    if bad_distributors:
        issues.append(f"Distributors in stockouts not in dim_distributor: {bad_distributors}")
    bad_skus_so = set(so["sku_code"].unique()) - set(ds["sku_code"].unique())
    if bad_skus_so:
        issues.append(f"SKUs in stockouts not in dim_sku: {bad_skus_so}")

    pr = dfs["promotions"]
    bad_skus_pr = set(pr["sku_code"].unique()) - set(ds["sku_code"].unique())
    if bad_skus_pr:
        issues.append(f"SKUs in promotions not in dim_sku: {bad_skus_pr}")
    bad_regions_pr = set(pr["region"].unique()) - set(dg["region"].unique())
    if bad_regions_pr:
        issues.append(f"Regions in promotions not in dim_geo: {bad_regions_pr}")

    ft = dfs["fact_targets"]
    bad_brands = set(ft["brand"].unique()) - set(ds["brand"].unique())
    if bad_brands:
        issues.append(f"Brands in targets not in dim_sku: {bad_brands}")
    bad_regions_ft = set(ft["region"].unique()) - set(dg["region"].unique())
    if bad_regions_ft:
        issues.append(f"Regions in targets not in dim_geo: {bad_regions_ft}")

    return issues


def build_canonical_dimensions(dfs: dict) -> dict:
    dims = {"dim_geo": dfs["dim_geo"].copy(), "dim_sku": dfs["dim_sku"].copy(), "dim_distributor": dfs["dim_distributor"].copy()}
    return dims


def map_week_to_month(df: pd.DataFrame) -> pd.DataFrame:
    df["month"] = df["week_start"].dt.strftime("%Y-%m")
    return df


def compute_promotion_baselines(sales_df: pd.DataFrame, promos_df: pd.DataFrame, geo_df: pd.DataFrame) -> pd.DataFrame:
    promos_df = promos_df.copy()
    promos_df["start_dt"] = pd.to_datetime(promos_df["start_date"], format="%d/%m/%Y", errors="coerce")
    promos_df["end_dt"] = pd.to_datetime(promos_df["end_date"], format="%d/%m/%Y", errors="coerce")
    baseline_results = []
    for _, promo in promos_df.iterrows():
        sku = promo["sku_code"]
        region = promo["region"]
        start = promo["start_dt"]
        end = promo["end_dt"]

        promo_sales = sales_df[
            (sales_df["sku_code"] == sku)
            & (sales_df["territory_code"].isin(
                geo_df[geo_df["region"] == region]["territory_code"]
            ))
            & (sales_df["week_start"] >= start)
            & (sales_df["week_start"] <= end)
        ]
        promo_avg_weekly = promo_sales.groupby("week_start")["value_inr"].sum().mean()
        promo_weekly_units = promo_sales.groupby("week_start")["units"].sum().mean()

        non_promo_sales = sales_df[
            (sales_df["sku_code"] == sku)
            & (sales_df["territory_code"].isin(
                geo_df[geo_df["region"] == region]["territory_code"]
            ))
            & ((sales_df["week_start"] < start) | (sales_df["week_start"] > end))
        ]
        baseline_weeks = non_promo_sales["week_start"].nunique()
        baseline_avg_weekly = non_promo_sales.groupby("week_start")["value_inr"].sum().mean()
        baseline_weekly_units = non_promo_sales.groupby("week_start")["units"].sum().mean()

        if pd.notna(baseline_avg_weekly) and baseline_avg_weekly > 0:
            uplift_pct = ((promo_avg_weekly - baseline_avg_weekly) / baseline_avg_weekly) * 100
        else:
            uplift_pct = None

        baseline_results.append({
            "promo_id": promo["promo_id"],
            "sku_code": sku,
            "region": region,
            "start_date": start,
            "end_date": end,
            "promo_avg_weekly_value": round(promo_avg_weekly, 2) if pd.notna(promo_avg_weekly) else None,
            "promo_avg_weekly_units": round(promo_weekly_units, 2) if pd.notna(promo_weekly_units) else None,
            "baseline_avg_weekly_value": round(baseline_avg_weekly, 2) if pd.notna(baseline_avg_weekly) else None,
            "baseline_avg_weekly_units": round(baseline_weekly_units, 2) if pd.notna(baseline_weekly_units) else None,
            "baseline_week_count": int(baseline_weeks),
            "uplift_pct": round(uplift_pct, 2) if uplift_pct is not None else None,
            "insufficient_baseline": baseline_weeks < 2,
        })
    return pd.DataFrame(baseline_results)


def generate_reconciliation_report(dfs: dict, issues: list, national_total: float) -> dict:
    report = {
        "national_fy26_primary_sales_total_inr": national_total,
        "data_sources": {},
        "reconciliation_mismatches": issues,
        "cleaned_data_columns": {},
    }
    for name, df in dfs.items():
        report["data_sources"][name] = {
            "raw_rows": len(df),
            "cleaned_rows": len(df),
            "columns": list(df.columns),
        }
        report["cleaned_data_columns"][name] = list(df.columns)

    stockout_region_counts = dfs["stockouts"]["region"].value_counts().to_dict()
    report["stockout_region_distribution_after_normalization"] = stockout_region_counts

    skus_never_in_stockouts = set(dfs["dim_sku"]["sku_code"].unique()) - set(dfs["stockouts"]["sku_code"].unique())
    report["skus_never_in_stockouts"] = sorted(skus_never_in_stockouts)

    return report


def main():
    print("=" * 60)
    print("ACPL Data Preparation Pipeline")
    print("=" * 60)

    if CLEAN_DIR.exists():
        shutil.rmtree(CLEAN_DIR)
    CLEAN_DIR.mkdir(parents=True)

    all_issues = []
    dfs = {}

    print("\n1. Loading raw data...")
    for name in ["dim_geo", "dim_sku", "dim_distributor", "fact_primary_sales", "fact_targets", "promotions", "stockouts"]:
        df = load_raw(name)
        dfs[name] = df
        print(f"   {name}: {len(df)} rows, {len(df.columns)} cols")

    print("\n2. Validating schemas (raw columns)...")
    for name in dfs:
        issues = validate_schema(dfs[name], name)
        all_issues.extend(issues)
        if issues:
            for i in issues:
                print(f"   SCHEMA ISSUE: {i}")
    print("   Schema validation complete.")

    print("\n3. Normalizing column names...")
    for name in dfs:
        before = list(dfs[name].columns)
        dfs[name] = normalize_column_names(dfs[name], name)
        after = list(dfs[name].columns)
        if before != after:
            print(f"   {name}: {before} -> {after}")
    print("   Column normalization complete.")

    print("\n4. Validating types and parsing dates...")
    for name in dfs:
        issues = validate_types(dfs[name], name)
        all_issues.extend(issues)
        if issues:
            for i in issues:
                print(f"   TYPE ISSUE: {i}")
    print("   Type validation complete.")

    print("\n4b. Ensuring month columns are strings...")
    dfs["fact_targets"] = ensure_month_as_string(dfs["fact_targets"])
    print("   Month columns normalized.")

    print("\n5. Validating primary keys...")
    for name in dfs:
        issues = validate_pks(dfs[name], name)
        all_issues.extend(issues)
        if issues:
            for i in issues:
                print(f"   PK ISSUE: {i}")
    print("   PK validation complete.")

    print("\n6. Normalizing stockout regions...")
    before_count = dfs["stockouts"]["region"].nunique()
    dfs["stockouts"] = normalize_stockout_regions(dfs["stockouts"])
    after_count = dfs["stockouts"]["region"].nunique()
    print(f"   Stockout region variants: {before_count} -> {after_count}")
    print(f"   Distribution: {dfs['stockouts']['region'].value_counts().to_dict()}")

    print("\n7. Validating foreign keys...")
    fk_issues = validate_foreign_keys(dfs)
    all_issues.extend(fk_issues)
    for i in fk_issues:
        print(f"   FK ISSUE: {i}")
    if not fk_issues:
        print("   All foreign keys valid.")

    print("\n8. Building canonical dimensions...")
    dims = build_canonical_dimensions(dfs)
    for name, df in dims.items():
        print(f"   {name}: {len(df)} rows, PK={PK_COLS.get(name)}")

    print("\n9. Mapping week_start to calendar month...")
    dfs["fact_primary_sales"] = map_week_to_month(dfs["fact_primary_sales"])
    months = sorted(dfs["fact_primary_sales"]["month"].unique())
    print(f"   Months: {months}")

    last_week = dfs["fact_primary_sales"]["week_start"].max()
    last_week_month = dfs["fact_primary_sales"].loc[
        dfs["fact_primary_sales"]["week_start"] == last_week, "month"
    ].iloc[0]
    print(f"   Last sales week: {last_week.date()} -> month {last_week_month}")
    print(f"   NOTE: Final week ends 2026-06-23, June target covers full month — partial coverage.")

    print("\n10. Computing promotion baselines...")
    promo_baselines = compute_promotion_baselines(dfs["fact_primary_sales"], dfs["promotions"], dfs["dim_geo"])
    print(f"    Computed baselines for {len(promo_baselines)} promotions")
    insufficient = promo_baselines["insufficient_baseline"].sum()
    if insufficient > 0:
        promos_insufficient = promo_baselines[promo_baselines["insufficient_baseline"]]
        print(f"    WARNING: {insufficient} promotions have <2 comparable baseline weeks")
        print(f"    Promo IDs: {promos_insufficient['promo_id'].tolist()}")
    dfs["promotion_baselines"] = promo_baselines

    print("\n11. Computing national FY26 primary-sales total...")
    national_total = dfs["fact_primary_sales"]["value_inr"].sum()
    print(f"    National FY26 Primary Sales Total: INR {national_total:,.2f}")

    print("\n12. Writing cleaned datasets...")
    for name in dfs:
        path = CLEAN_DIR / f"{name}.parquet"
        dfs[name].to_parquet(path, index=False)
        print(f"    {path.name}: {len(dfs[name])} rows written")

    print("\n13. Generating reconciliation report...")
    report = generate_reconciliation_report(dfs, all_issues, national_total)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"    Report written to {REPORT_PATH}")

    print("\n" + "=" * 60)
    print("Data preparation complete.")
    print(f"Cleaned data in: {CLEAN_DIR}/")
    print(f"Reconciliation report: {REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()