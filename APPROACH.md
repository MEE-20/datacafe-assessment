# Approach

## Problem Summary

Build a Sales Focus and Action Assistant for ACPL (FMCG company) that answers three questions from weekly/monthly sales data, business documents, and a deterministic action playbook:

1. **What is happening?** — sales performance vs targets, stockout patterns, promotion impact
2. **Why is it happening?** — root causes from structured data (stockouts, promotions) or business documents (competitor activity, supply notes)
3. **What should be done next?** — actionable recommendations traced to the sanctioned playbook

Key constraints: raw data is never modified, the LLM never performs arithmetic or invents actions, all numerical calculations are deterministic, every OK answer carries verifiable evidence, and the system returns NO_ANSWER when it cannot confidently answer.

---

## System Architecture

Two phases: **offline data preparation** (one command, run once) and **runtime inference** (two API endpoints).

```
┌─────────────────────────────────────────────────────────────────────┐
│  OFFLINE (one command)                                              │
│                                                                     │
│  raw/data/ ──▶ 1. Validate schemas, keys, duplicates                │
│                2. Normalize column names                             │
│                3. Normalize stockout regions (16 variants → 4)      │
│                4. Map week_start → calendar month                    │
│                5. Build canonical dimensions                         │
│                6. Compute promotion baselines                        │
│                7. Write data_cleaned/ + validation metadata          │
│                                                                     │
│  Output: data_cleaned/ + reconciliation_report.json                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  RUNTIME — LangGraph orchestration                                  │
│                                                                     │
│  ┌──────────┐    ┌──────────────────────────────────────────────┐   │
│  │ POST     │    │  ORCHESTRATOR (LLM-as-router)                 │   │
│  │ /ask     │───▶│  Classifies question → structured | doc |     │   │
│  └──────────┘    │  both | unsupported                           │   │
│                  └───────┬──────────────────┬────────────────────┘   │
│                          │                  │                        │
│              ┌───────────▼────┐    ┌───────▼────────────┐            │
│              │ TOOLS LAYER    │    │ DOC RETRIEVAL      │            │
│              │ (deterministic) │    │ (deterministic)   │            │
│              │                │    │                    │            │
│              │ • sales_total  │    │ • relevant_passage │            │
│              │ • target_achvm │    │ • source_id        │            │
│              │ • stockout_mtr │    │ • snippet          │            │
│              │ • promo_uplift │    └────────┬───────────┘            │
│              │ • dist_sku_mtr │             │                        │
│              └────────┬───────┘             │                        │
│                       │                     │                        │
│              ┌────────▼─────────────────────▼────────┐               │
│              │  EVIDENCE BUILDER                      │               │
│              │  Merges tool results + doc passages    │               │
│              │  into evidence[] array with exact      │               │
│              │  figures and source identifiers        │               │
│              └────────────────┬───────────────────────┘               │
│                               │                                      │
│              ┌────────────────▼───────────────────────┐               │
│              │  LLM SYNTHESIS (constrained)            │               │
│              │  Given evidence[], produce natural-     │               │
│              │  language answer. No arithmetic.        │               │
│              │  If no evidence → NO_ANSWER             │               │
│              └────────────────┬───────────────────────┘               │
│                               │                                      │
│              ┌────────────────▼───────────────────────┐               │
│              │  Structured API Response                │               │
│              │  { answer, status, evidence[],          │               │
│              │    cost_usd, latency_ms }               │               │
│              └─────────────────────────────────────────┘               │
│                                                                       │
│  ┌──────────┐    ┌──────────────────────────────────────────────┐     │
│  │ POST     │    │  NO LLM — pure deterministic                  │     │
│  │ /actions │───▶│                                              │     │
│  └──────────┘    │  • Run analytics for scope (region/brand)    │     │
│                  │  • Evaluate R-01 through R-08 deterministically│    │
│                  │  • Apply precedence: R-01,R-04,R-08 first     │     │
│                  │  • Attach state: RECOMMENDED / PENDING_APPROVAL│   │
│                  │  • Build findings[] with supporting metrics   │     │
│                  └──────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Preparation (Offline)

A single command: `python src/prepare_data.py`

### Sequence

1. **Read raw** — load all CSV files from `data/` without modification
2. **Validate schemas** — check required columns, data types, date parseability, primary-key uniqueness, foreign-key referential integrity
3. **Normalize column names** — `promotions.sku → sku_code`, `stockouts.item_code → sku_code`, `fact_targets.brand_name → brand`, `fact_targets.region_name → region`
4. **Normalize stockout regions** — map 16 variants (`North`, `NORTH`, `north`, `North Region`, `South`, ...) to canonical `North | South | East | West`
5. **Validate normalized regions** — cross-check against `dim_geo.region`; flag unmapped values
6. **Establish canonical dimensions** — produce deduplicated dimension views with consistent keys
7. **Map week_start to month** — per data dictionary: "a week belongs to the calendar month containing its week_start"
8. **Preserve final partial June week** — week 2026-06-23 is the last sales week but doesn't span to June 30; document this limitation for June month calculations
9. **Compute promotion baselines** — for each SKU+region+promotion, compute average weekly sales in non-promo weeks (same SKU, same region) as primary baseline; note cases where insufficient comparable data exists
10. **Output** — write `data_cleaned/` Parquet files + `reconciliation_report.json` with row counts retained, excluded rows, detected mismatches, national FY26 primary-sales total

### Row Counts Retained (expected)

| Source | Raw Rows | Cleaned Rows | Notes |
|--------|----------|-------------|-------|
| fact_primary_sales | 74,880 | 74,880 | No exclusions expected |
| fact_targets | 720 | 720 | No exclusions expected |
| dim_sku | 120 | 120 | No exclusions expected |
| dim_geo | 12 | 12 | No exclusions expected |
| dim_distributor | 40 | 40 | No exclusions expected |
| promotions | 40 | 40 | No exclusions expected |
| stockouts | 520 | 520 | Region normalization applied, no row loss |

### National FY26 Primary-Sales Total

Computed as `SUM(value_inr)` across all 74,880 rows in `fact_primary_sales`.

### Reconciliation Mismatches

Expected findings:
- 16 stockout region variants → 4 canonical values — all map successfully
- 2 SKUs (`BS-0202`, `BS-0107`) never appear in stockouts — documented as "no events" not "missing data"
- Promo end dates may fall on non-Monday dates (not a sales week boundary) — documented

---

## Runtime Implementation Detail

### Orchestrator (LangGraph)

The orchestrator is an LLM node that classifies the user's natural language question into one of four intents:

| Intent | Routes To |
|--------|-----------|
| `structured_data` | Tool layer (analytics) → evidence builder |
| `business_documents` | Doc retrieval → evidence builder |
| `structured_and_documents` | Both tools + doc retrieval → evidence builder |
| `unsupported` | Direct NO_ANSWER response |

The LLM selects from available tools by function-calling. Tools return structured dicts with exact figures — the LLM does not perform calculations.

### Tools (Deterministic, LangGraph Tool Nodes)

**1. Sales Analytics Tool**
- Input: optional filters (`region`, `brand`, `sku`, `month_start`, `month_end`)
- Capabilities:
  - Sales totals (national, region, brand, SKU)
  - Target achievement % per brand×region×month
  - Achievement categories: `<70%`, `<80%`, `80-110%`, `>110%`

**2. Stockout Metrics Tool**
- Input: optional filters (`region`, `brand`, `distributor`, `sku`, `month`)
- Capabilities:
  - List stockout events with duration
  - Detect chronic stockouts (>6 consecutive weeks on same SKU for a distributor)
  - Detect multi-SKU stockouts (≥3 SKUs for a distributor in a month)
  - Map stockouts to brand via SKU dimension

**3. Promotion Uplift Tool**
- Input: optional filters (`region`, `brand`, `promo_id`, `month`)
- Capabilities:
  - Compute uplift % = (promo-period avg weekly sales − non-promo baseline) / baseline × 100
  - Classify: `<10%` (weak), `10-25%` (moderate), `>25%` (strong)
  - Baseline: same SKU + same territory, average of non-promo weeks in the dataset
  - Where baseline has <2 comparable weeks, label `insufficient_baseline`

**4. Distributor/SKU Metrics Tool**
- Input: `distributor_id` or `sku_code`
- Capabilities:
  - Stockout history for a distributor
  - Sales performance for a SKU
  - Territory mapping

**5. Document Retrieval Tool**
- Fixed document store (ingested at startup from 4 relevant .docx files)
- Returns: `passage`, `source_file`, `source_snippet`

**6. Playbook Evaluation Tool**
- Evaluates all 8 rules for a given scope (region, brand, or global)
- Returns findings with supporting evidence

### Evidence Builder

- Aggregates outputs from all invoked tools into `evidence[]` array
- Each evidence item: `{"metric": "...", "value": ..., "source": "...", "period": "..."}`
- No LLM involvement — pure deterministic merge

### LLM Synthesis (Constrained)

- Receives: question + evidence[] array
- Produces: natural language answer
- Constrained by: "do not add numbers not present in evidence", "if evidence is empty return NO_ANSWER"
- Template: "Based on the data, [synthesized claim]. Evidence: [cite figures]."

### /actions Endpoint (No LLM)

1. Accept `scope` (e.g., `{"scope": "West"}`)
2. Run analytics for the scope (all tools above, filtered)
3. Evaluate R-01 through R-08 deterministically:
   - For each rule, check if condition is met using computed metrics
   - Apply precedence ordering: R-01 → R-04 → R-08 → R-02 → R-03 → R-05 → R-06 → R-07
   - Where multiple rules match the same entity, list primary then secondary
4. Assign state:
   - `needs_approval = Yes` → `PENDING_APPROVAL`
   - `needs_approval = No` → `RECOMMENDED`
5. Output: `[{"findings": "...", "rule_id": "R-xx", "action": "...", "state": "..."}]`

### Implementation Conventions (not in the playbook)

- **"Repeated stock-outs" (R-01):** ≥2 distinct stockout weeks for one or more SKUs belonging to the brand within the relevant region
- **Promotion uplift baseline:** same SKU + same territory, non-promo weeks average; if <2 comparable weeks, mark insufficient_baseline
- **Rule precedence:** R-01, R-04, R-08 → R-02, R-03, R-05 → R-06 → R-07

---

## Data Flow for Each Question Type

### Structured Data Only (e.g., "What was GlucoJoy's target achievement in North in July?")

```
/ask → Orchestrator classifies as structured_data
     → Sales Analytics Tool (brand=GlucoJoy, region=North, month=2025-07)
     → Evidence Builder: [metric=achievement%, value=89.3%, ...]
     → LLM synthesizes: "GlucoJoy achieved 89.3% of target in North in July 2025."
     → { answer, status: OK, evidence: [...], cost_usd, latency_ms }
```

### Documents Only (e.g., "Why did CremeDelight miss in North in February?")

```
/ask → Orchestrator classifies as business_documents
     → Document Retrieval Tool (keywords: CremeDelight, North, February)
     → Returns: visit_note_north_feb2026 passage
     → Evidence Builder: [source=visit_note_north_feb2026.docx, ...]
     → LLM synthesizes: "Competitor activity in mid-pack biscuits..."
     → { answer, status: OK, evidence: [...], ... }
```

### Both (e.g., "What caused the stockout on BV-0104 in West?")

```
/ask → Orchestrator classifies as structured_and_documents
     → Stockout Metrics Tool (sku=BV-0104, region=West)
     → Document Retrieval Tool (keywords: BV-0104, West)
     → Evidence Builder merges both
     → LLM synthesizes combined answer
```

### Unsupported

```
/ask → Orchestrator classifies as unsupported
     → { answer: null, status: NO_ANSWER, evidence: [],
         Explanation: "No data available to answer this question." }
```

---

## Trade-offs & Limitations

| Trade-off | Rationale |
|-----------|-----------|
| LLM never calculates | Prevents hallucinated numbers; all figures come from deterministic code |
| Region normalization is manual (16→4) | Stockout CSV has no standard; normalization requires a hardcoded mapping dict |
| Promotion baseline limited to non-promo weeks | Simple and reproducible; may undercount in regions with heavy promo coverage |
| "Repeated stock-outs" threshold (≥2 weeks) is an implementation convention | The playbook does not define it numerically; documented and traceable |
| Final June week truncated | Week 2026-06-23 doesn't cover June 23-30; June totals will slightly undercount |
| Document context is static (ingested at startup) | Documents are fixed artifacts; no live updates |
| No multi-turn conversation | Each /ask is stateless; no follow-up context |
| LangGraph overhead | Adds complexity vs direct Python, but provides explicit orchestration and tool routing |

---

## AI Tool Usage

The LLM (via Cadra) is used only in:
1. **Orchestrator node** — classify question intent and select tools
2. **Synthesis node** — produce natural-language answer from pre-computed evidence

The LLM is NOT used for:
- Numerical calculations or aggregations
- Playbook rule evaluation or action selection
- Generating arbitrary SQL or data queries
- Inventing facts, figures, or causes not present in the evidence

Prompts are structured to constrain the LLM to the evidence provided, with explicit instructions to return NO_ANSWER when evidence is insufficient.

---

## Deliverables Summary

- `src/prepare_data.py` — single-command offline data preparation
- `data_cleaned/` — prepared analytical datasets
- `reconciliation_report.json` — row counts, mismatches, national total
- `src/analytics/` — deterministic tool implementations (sales_target, stockouts, promotions, distributor_sku)
- `src/playbook/engine.py` — deterministic R-01 through R-08 evaluation
- `src/documents/loader.py` — .docx ingestion and retrieval
- `src/orchestrator/graph.py` — LangGraph state machine
- `main.py` — FastAPI server with /ask and /actions
- `APPROACH.md` — this document
- `README.md` — how to prepare data, run services, field notes