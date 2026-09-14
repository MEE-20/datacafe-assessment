"""Deterministic playbook rule engine implementing R-01 through R-08.

Rules are evaluated in precedence order:
  Primary: R-01, R-04, R-08
  Secondary: R-02, R-03, R-05
  Tertiary: R-06, R-07

Every rule evaluation is purely deterministic using prepared analytical data.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from src.analytics.sales_target import target_achievement
from src.analytics.stockouts import (
    chronic_stockouts,
    multi_sku_stockouts,
    brand_has_repeated_stockouts,
)
from src.analytics.promotions import (
    promo_active_during_month,
    promo_is_underperforming,
    promo_is_strong,
)
from src.analytics.document_retrieval import DOC_SUMMARIES

CLEAN_DIR = Path("data_cleaned")

RULE_PRECEDENCE = ["R-01", "R-04", "R-08", "R-02", "R-03", "R-05", "R-06", "R-07"]

APPROVAL_RULES = {"R-01", "R-04", "R-08"}


class PlaybookFinding:
    def __init__(self, rule_id: str, condition: str, findings: str, action: str, needs_approval: bool, supporting_metrics: dict):
        self.rule_id = rule_id
        self.condition = condition
        self.findings = findings
        self.action = action
        self.needs_approval = needs_approval
        self.state = "PENDING_APPROVAL" if needs_approval else "RECOMMENDED"
        self.supporting_metrics = supporting_metrics

    @property
    def distributor_id(self) -> str | None:
        return self.supporting_metrics.get("distributor_id")

    @property
    def region(self) -> str | None:
        return self.supporting_metrics.get("region")

    @property
    def brand(self) -> str | None:
        return self.supporting_metrics.get("brand")

    @property
    def sku_code(self) -> str | None:
        return self.supporting_metrics.get("sku_code")

    def to_dict(self):
        return {
            "rule_id": self.rule_id,
            "condition": self.condition,
            "findings": self.findings,
            "action": self.action,
            "state": self.state,
            "supporting_metrics": self.supporting_metrics,
        }


def evaluate_r01(brand: str, region: str, month: str) -> Optional[PlaybookFinding]:
    ta = target_achievement(region=region, brand=brand, month=month)
    if not ta.success or ta.data is None or ta.data.empty:
        return None
    row = ta.data.iloc[0]
    achievement = row["achievement_pct"]
    if achievement >= 70:
        return None
    has_repeated_so = brand_has_repeated_stockouts(brand, region, month)
    if not has_repeated_so:
        return None

    return PlaybookFinding(
        rule_id="R-01",
        condition="Brand misses target (<70%) with repeated stock-outs",
        findings=(
            f"{brand} in {region} achieved {achievement:.1f}% of target in {month} "
            f"(below 70% threshold) AND has repeated stock-outs on its SKUs in the region."
        ),
        action="Expedite replenishment and escalate to the regional supply lead",
        needs_approval=True,
        supporting_metrics={
            "brand": brand,
            "region": region,
            "month": month,
            "achievement_pct": round(achievement, 2),
            "target_value_inr": float(row["target_value_inr"]),
            "actual_value_inr": float(row["actual_value_inr"]),
        },
    )


def evaluate_r02(brand: str, region: str, month: str) -> Optional[PlaybookFinding]:
    ta = target_achievement(region=region, brand=brand, month=month)
    if not ta.success or ta.data is None or ta.data.empty:
        return None
    row = ta.data.iloc[0]
    achievement = row["achievement_pct"]
    if achievement >= 80:
        return None
    has_promo = promo_active_during_month(region, brand, month)
    if not has_promo:
        return None
    underperforming = promo_is_underperforming(region, brand, month)
    if not underperforming:
        return None

    return PlaybookFinding(
        rule_id="R-02",
        condition="Brand misses target (<80%) with underperforming promotion (<10% uplift)",
        findings=(
            f"{brand} in {region} achieved {achievement:.1f}% of target in {month} "
            f"(below 80% threshold) while a promotion was running with weak uplift (<10%)."
        ),
        action="Review promo effectiveness with the brand team",
        needs_approval=False,
        supporting_metrics={
            "brand": brand,
            "region": region,
            "month": month,
            "achievement_pct": round(achievement, 2),
        },
    )


def evaluate_r03(brand: str, region: str, month: str) -> Optional[PlaybookFinding]:
    ta = target_achievement(region=region, brand=brand, month=month)
    if not ta.success or ta.data is None or ta.data.empty:
        return None
    row = ta.data.iloc[0]
    achievement = row["achievement_pct"]
    if achievement >= 80:
        return None
    has_promo = promo_active_during_month(region, brand, month)
    has_so = brand_has_repeated_stockouts(brand, region, month)
    if has_promo or has_so:
        return None

    return PlaybookFinding(
        rule_id="R-03",
        condition="Brand misses target (<80%) with NO stock-out and NO promotion",
        findings=(
            f"{brand} in {region} achieved {achievement:.1f}% of target in {month} "
            f"with no stock-out and no promotion — cause is external or unknown."
        ),
        action="Commission a market-visit / competitor check for the brand in that region",
        needs_approval=False,
        supporting_metrics={
            "brand": brand,
            "region": region,
            "month": month,
            "achievement_pct": round(achievement, 2),
        },
    )


def evaluate_r04(region: Optional[str] = None) -> list[PlaybookFinding]:
    results = chronic_stockouts(region=region)
    if not results.success:
        return []
    findings = []
    for ev in results.evidence:
        findings.append(PlaybookFinding(
            rule_id="R-04",
            condition="Single distributor out of stock for >6 weeks on a SKU",
            findings=(
                f"Distributor {ev['distributor_id']} has been out of stock on SKU {ev['sku_code']} "
                f"({ev['brand']}) for {ev['weeks_with_stockout']} weeks (threshold: 6 weeks)."
            ),
            action="Raise a replenishment order for that distributor",
            needs_approval=True,
            supporting_metrics=ev,
        ))
    return findings


def evaluate_r05(brand: str, region: str, month: str) -> Optional[PlaybookFinding]:
    ta = target_achievement(region=region, brand=brand, month=month)
    if not ta.success or ta.data is None or ta.data.empty:
        return None
    row = ta.data.iloc[0]
    achievement = row["achievement_pct"]
    if achievement <= 110:
        return None

    return PlaybookFinding(
        rule_id="R-05",
        condition="Brand over-delivers against target (>110%)",
        findings=(
            f"{brand} in {region} achieved {achievement:.1f}% of target in {month} "
            f"(above 110% threshold) — something is working well."
        ),
        action="Capture what worked and redeploy field effort elsewhere",
        needs_approval=False,
        supporting_metrics={
            "brand": brand,
            "region": region,
            "month": month,
            "achievement_pct": round(achievement, 2),
        },
    )


def evaluate_r06(brand: str, region: str, month: str) -> Optional[PlaybookFinding]:
    ta = target_achievement(region=region, brand=brand, month=month)
    if not ta.success or ta.data is None or ta.data.empty:
        return None
    row = ta.data.iloc[0]
    achievement = row["achievement_pct"]
    if achievement >= 80:
        return None
    has_promo = promo_active_during_month(region, brand, month)
    has_so = brand_has_repeated_stockouts(brand, region, month)
    if has_promo or has_so:
        return None

    has_doc = False
    for fname, summary in DOC_SUMMARIES.items():
        if brand.lower() in summary.lower() and region.lower() in summary.lower():
            has_doc = True
            break

    if has_doc:
        return None

    return PlaybookFinding(
        rule_id="R-06",
        condition="Brand misses target with no stock-out, no promotion, and no supporting note",
        findings=(
            f"{brand} in {region} achieved {achievement:.1f}% of target in {month} "
            f"with no stock-out, no promotion, and no supporting business document. "
            f"No cause is determinable from available data."
        ),
        action="Flag for manual review; do not auto-attribute a cause",
        needs_approval=False,
        supporting_metrics={
            "brand": brand,
            "region": region,
            "month": month,
            "achievement_pct": round(achievement, 2),
        },
    )


def evaluate_r07(region: Optional[str] = None) -> list[PlaybookFinding]:
    from src.analytics.promotions import promotion_uplift
    results = promotion_uplift(region=region)
    if not results.success:
        return []
    findings = []
    for ev in results.evidence:
        uplift = ev.get("uplift_pct")
        if uplift is not None and uplift > 25:
            findings.append(PlaybookFinding(
                rule_id="R-07",
                condition="Promotion delivered strong uplift (>25%)",
                findings=(
                    f"Promotion {ev['promo_id']} for {ev['brand']} SKU {ev['sku_code']} "
                    f"in {ev['region']} achieved {uplift:.1f}% uplift (threshold: >25%). The mechanic worked."
                ),
                action="Consider extending or replicating the mechanic in a comparable region",
                needs_approval=False,
                supporting_metrics=ev,
            ))
    return findings


def evaluate_r08(region: Optional[str] = None) -> list[PlaybookFinding]:
    results = multi_sku_stockouts(region=region)
    if not results.success:
        return []
    findings = []
    for ev in results.evidence:
        findings.append(PlaybookFinding(
            rule_id="R-08",
            condition="Distributor shows stock-outs across 3 or more SKUs in a month",
            findings=(
                f"Distributor {ev['distributor_id']} in {ev['region']} had stockouts on "
                f"{ev['sku_count']} SKUs in {ev['month']} (threshold: 3 SKUs). Distributor-level supply issue."
            ),
            action="Schedule a distributor stock-review call",
            needs_approval=True,
            supporting_metrics=ev,
        ))
    return findings


def run_scope(scope: Optional[str] = None) -> list[dict]:
    """Evaluate all rules for an optional scope (region name like 'West')."""

    all_findings = []

    ds = pd.read_parquet(CLEAN_DIR / "dim_sku.parquet")
    ft = pd.read_parquet(CLEAN_DIR / "fact_targets.parquet")

    if scope:
        unique_brand_regions = ft[ft["region"] == scope][["brand", "region"]].drop_duplicates()
    else:
        unique_brand_regions = ft[["brand", "region"]].drop_duplicates()

    months = sorted(ft["month"].unique())
    latest_month = months[-1] if months else None

    brand_region_pairs = unique_brand_regions.to_dict("records")

    # Primary precedence: R-01, R-04, R-08
    for br in brand_region_pairs:
        brand = br["brand"]
        region = br["region"]
        month = latest_month

        finding = evaluate_r01(brand, region, month)
        if finding:
            all_findings.append(finding.to_dict())

    if scope:
        for finding in evaluate_r04(region=scope):
            all_findings.append(finding.to_dict())
        for finding in evaluate_r08(region=scope):
            all_findings.append(finding.to_dict())
    else:
        for finding in evaluate_r04():
            all_findings.append(finding.to_dict())
        for finding in evaluate_r08():
            all_findings.append(finding.to_dict())

    # Secondary: R-02, R-03, R-05
    for br in brand_region_pairs:
        brand = br["brand"]
        region = br["region"]
        month = latest_month

        for r02 in [evaluate_r02(brand, region, month)]:
            if r02:
                all_findings.append(r02.to_dict())
        for r03 in [evaluate_r03(brand, region, month)]:
            if r03:
                all_findings.append(r03.to_dict())
        for r05 in [evaluate_r05(brand, region, month)]:
            if r05:
                all_findings.append(r05.to_dict())

    # Tertiary: R-06, R-07
    for br in brand_region_pairs:
        brand = br["brand"]
        region = br["region"]
        month = latest_month

        for r06 in [evaluate_r06(brand, region, month)]:
            if r06:
                all_findings.append(r06.to_dict())

    for finding in evaluate_r07(region=scope if scope else None):
        all_findings.append(finding.to_dict())

    return all_findings