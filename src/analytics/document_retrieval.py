"""Document retrieval tool for grounding answers in business documents."""

from pathlib import Path
from typing import Optional

from docx import Document

DOCUMENTS_DIR = Path("data/fmcg-sales-copilot-ai-engineer-mid-4to6/documents")

RELEVANT_DOCS = [
    "escalation_sop.docx",
    "visit_note_north_feb2026.docx",
    "distributor_note_west.docx",
    "promo_circular_h2fy26.docx",
]


def _load_doc(filename: str) -> str:
    path = DOCUMENTS_DIR / filename
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def _build_index() -> list[dict]:
    index = []
    for fname in RELEVANT_DOCS:
        text = _load_doc(fname)
        lines = text.strip().split("\n")
        current_section = "general"
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            if len(line) < 60 and line.isupper() and i > 0:
                current_section = line.lower().replace(" ", "_")[:30]
            index.append({
                "source": fname,
                "section": current_section,
                "text": line,
                "line": i,
            })
    return index


_DOC_INDEX = None


def _get_index():
    global _DOC_INDEX
    if _DOC_INDEX is None:
        _DOC_INDEX = _build_index()
    return _DOC_INDEX


DOC_SUMMARIES = {
    "escalation_sop.docx": (
        "Sales Operations Escalation SOP. Every action must cite its playbook rule. "
        "Actions that notify another team or change a commitment (supply escalations, replenishment orders, "
        "distributor calls) require manager approval. Analysis/review actions do not. "
        "When cause is not clear, do not invent a reason."
    ),
    "visit_note_north_feb2026.docx": (
        "North Region market visit note (Feb 2026). A competitor ran a deep price-off on mid-pack biscuits "
        "in Jaipur and Lucknow, causing the CremeDelight range in the North to lose ~25% of expected February "
        "offtake. No supply issue; not visible in stock-out or promotion data. Recommends a targeted counter-promotion."
    ),
    "distributor_note_west.docx": (
        "West Region distributor supply note. Mumbai distributors D032 and D033 had repeated out-of-stocks "
        "on Beverages 1L line through April-June 2026, coinciding with a promotion. Depot replenishment was slow."
    ),
    "promo_circular_h2fy26.docx": (
        "H2 FY26 Trade Promotions Circular. Beverages: 'Buy 2 Get 1' on 1L pack in West through May-June. "
        "Snacks: price-off on 90g pack in North in November. All other schemes handled locally."
    ),
}


STOPWORDS = {"a", "an", "in", "on", "to", "do", "no", "is", "it", "be", "by", "at", "or", "of", "as", "so", "if", "up", "we", "he", "she", "they", "not", "but", "the", "and", "for", "its", "are", "was", "had", "has", "can", "all", "may", "out", "you", "that", "this", "with", "from", "about", "into", "over", "also", "does", "will", "more", "been", "each", "than", "some"}


def retrieve(query: str) -> dict:
    query_lower = query.lower()
    query_words = [w for w in query_lower.split() if len(w) >= 4 and w not in STOPWORDS]
    results = []

    for fname in RELEVANT_DOCS:
        summary = DOC_SUMMARIES.get(fname, "")
        summary_words = [w for w in summary.lower().split() if len(w) >= 4 and w not in STOPWORDS]
        if query_words and any(kw in query_lower for kw in summary_words):
            results.append({
                "source": fname,
                "relevance": "high",
                "summary": summary,
            })

    if not results:
        index = _get_index()
        for entry in index:
            if query_lower in entry["text"].lower():
                results.append({
                    "source": entry["source"],
                    "relevance": "medium",
                    "snippet": entry["text"][:500],
                })

    return {
        "success": len(results) > 0,
        "documents": results[:5],
        "note": "Documents are static business context files, not curated structured data.",
    }


def get_all_document_context() -> list[dict]:
    results = []
    for fname, summary in DOC_SUMMARIES.items():
        results.append({"source": fname, "summary": summary})
    return results