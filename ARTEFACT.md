# Data Reconciliation Report — ACPL Sales Focus and Action Assistant

## Preparation Command

```
python src/prepare_data.py
```

## Row Counts Retained after Data Preparation

All 7 source datasets retained 100% of rows with zero exclusions.

| Dataset | Raw Rows | Cleaned Rows | Retention |
|---------|----------|-------------|-----------|
| dim_geo | 12 | 12 | 100% |
| dim_sku | 120 | 120 | 100% |
| dim_distributor | 40 | 40 | 100% |
| fact_primary_sales | 74,880 | 74,880 | 100% |
| fact_targets | 720 | 720 | 100% |
| promotions | 40 | 40 | 100% |
| stockouts | 520 | 520 | 100% |
| promotion_baselines | — | 40 | (derived) |

One derived table was created: `promotion_baselines` (40 rows — one per promotion) containing computed uplift metrics.

## National FY26 Primary-Sales Total

**INR 1,357,631,078.74**

Computed as `SUM(value_inr)` across all 74,880 rows in `fact_primary_sales` for the full 52-week period (2025-07-01 to 2026-06-23).

## Reconciliation Mismatches

**Zero mismatches detected.** All validations passed:

- **Schema validation** — all required columns present in raw files
- **Primary keys** — no duplicate rows in any table
- **Foreign keys** — all referential integrity checks clean:
  - All territories in `fact_primary_sales` exist in `dim_geo`
  - All SKUs in `fact_primary_sales`, `stockouts`, and `promotions` exist in `dim_sku`
  - All distributors in `stockouts` exist in `dim_distributor`
  - All brands in `fact_targets` exist in `dim_sku.brand`
  - All regions in `fact_targets` and `promotions` exist in `dim_geo.region`
- **Date parsing** — all dates parsed successfully with zero errors

## Data Quality Corrections Applied

### 1. Column Name Normalization (3 tables)

| Table | Original Column | Normalized To |
|-------|----------------|---------------|
| fact_targets | `brand_name` | `brand` |
| fact_targets | `region_name` | `region` |
| promotions | `sku` | `sku_code` |
| stockouts | `item_code` | `sku_code` |

### 2. Stockout Region Normalization (16 variants → 4 canonical)

The `stockouts.csv` `region` column contained 16 different text representations for the 4 canonical regions. A mapping dictionary was applied:

| Canonical Region | Count of Stockout Events |
|-----------------|--------------------------|
| North | 146 |
| South | 136 |
| West | 125 |
| East | 113 |

Variants collapsed: `North`, `NORTH`, `north`, `North Region` → `North`, etc.

### 3. Week-to-Month Mapping

Per data dictionary rule: "a week belongs to the calendar month containing its `week_start`". Applied to `fact_primary_sales` to enable sales-to-target joins at month grain.

**Known limitation:** The final sales week starts 2026-06-23 (Tuesday) and ends 2026-06-29, but only 2026-06-23 falls within the FY26 target period. June 2026 sales totals will slightly undercount relative to the full-month target.

## Promotion Baseline Methodology

For each of the 40 promotions:
- **Promo period:** weeks where `week_start` falls within `[start_date, end_date]`
- **Baseline period:** all other weeks for the same SKU in the same region
- **Uplift:** `(promo_avg_weekly_value - baseline_avg_weekly_value) / baseline_avg_weekly_value * 100`
- **Insufficient baseline flag:** set when `<2` non-promo weeks are available for that SKU+region

## Additional Observations

- **2 SKUs never appear in stockouts:** `BS-0107` (GlucoJoy Twin 2x100g) and `BS-0202` (CremeDelight 100g) — both Biscuits category. This is a genuine zero-event observation, not a data quality issue.
- **9-week chronic stockout pattern:** D032 and D033 (Mumbai/West) recorded stockouts on `BV-0104` (Beverages 1L) for 9 consecutive weeks (Apr–Jun 2026), corroborated by `distributor_note_west.docx`.