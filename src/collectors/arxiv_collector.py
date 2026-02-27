"""
arXiv RSS Collector for Jordan HEDGE Curiosity Engine.
Fetches recent papers from relevant categories, filters by topic,
and ingests new_evidence_available signals for matching threads.
"""

import xml.etree.ElementTree as ET
import urllib.request
import logging
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

from ..engine.hedge_engine import HedgeEngine, SignalItem

logger = logging.getLogger(__name__)

STATE_FILE = Path(__file__).parent.parent.parent / ".arxiv_state.json"

# arXiv categories to monitor
ARXIV_FEEDS = [
    "https://rss.arxiv.org/rss/cs.AI",
    "https://rss.arxiv.org/rss/cs.LG",
    "https://rss.arxiv.org/rss/cs.HC",
    "https://rss.arxiv.org/rss/q-bio.QM",
]

# Keywords that map to Jordan's domains
TOPIC_THREAD_MAP = {
    "MEL Cloud Deployment": [
        "clinical decision support", "medical ai", "clinical nlp",
        "ehr", "electronic health record", "medical transcription",
        "clinical note", "prior authorization",
    ],
    "MEL Ontology Quality": [
        "medical ontology", "clinical knowledge graph", "bayesian diagnosis",
        "probabilistic reasoning", "differential diagnosis", "llr",
        "log-likelihood", "clinical reasoning", "knowledge graph medical",
        "federated learning health", "federated medical",
    ],
    "Ridgeline L6 Promotion": [
        "knowledge management", "investment ai", "portfolio optimization",
        "financial machine learning", "regime detection", "factor model",
        "natural language finance", "earnings call",
    ],
    "Meniscus Tear Recovery": [
        "meniscus", "cartilage repair", "bpc-157", "tb-500", "peptide therapy",
        "musculoskeletal", "tendon", "ligament", "zone 2", "vo2max",
        "longevity", "aging", "healthspan",
    ],
}

# High-value cross-domain topics (seed curiosity threads)
CURIOSITY_KEYWORDS = [
    "autonomous agent", "agentic", "multi-agent", "reasoning under uncertainty",
    "bayesian", "ontology", "knowledge graph", "federated learning",
    "large language model", "retrieval augmented", "mcp", "model context protocol",
    "constraint satisfaction", "optimization", "causal inference",
    "explainability", "calibration", "uncertainty quantification",
]


class ArxivCollector:

    def __init__(self, engine: HedgeEngine = None):
        self.engine = engine or HedgeEngine()
        self.state = self._load_state()

    def collect(self) -> dict:
        summary = {}
        papers = self._fetch_all_papers()
        logger.info(f"[ArxivCollector] Fetched {len(papers)} papers")

        thread_signals: dict[str, list] = {}
        curiosity_hits = []

        for paper in papers:
            if self._is_seen(paper["id"]):
                continue

            text = f"{paper['title']} {paper['summary']}".lower()

            # Match to threads
            for thread_name, keywords in TOPIC_THREAD_MAP.items():
                if any(kw in text for kw in keywords):
                    thread_signals.setdefault(thread_name, []).append(SignalItem(
                        signal_name="new_evidence_available",
                        value="present",
                        confidence=0.7,
                        source="arxiv",
                        raw_data={"title": paper["title"], "id": paper["id"],
                                  "url": paper.get("link", ""), "date": paper.get("date", "")},
                    ))

            # Track curiosity hits separately
            if any(kw in text for kw in CURIOSITY_KEYWORDS):
                curiosity_hits.append({
                    "title": paper["title"],
                    "url": paper.get("link", ""),
                    "date": paper.get("date", ""),
                    "id": paper["id"],
                })

            self._mark_seen(paper["id"])

        # Ingest thread signals (deduplicated)
        for thread_name, signals in thread_signals.items():
            seen_sigs = set()
            unique = [s for s in signals if s.signal_name not in seen_sigs and not seen_sigs.add(s.signal_name)]
            if unique:
                count = self.engine.ingest_signals(thread_name, unique)
                self.engine.generate_actions(thread_name)
                summary[thread_name] = {"signals": [s.signal_name for s in unique], "ingested": count}

        # Store curiosity hits for digest
        self.state["latest_curiosity"] = curiosity_hits[:10]
        self._save_state()

        logger.info(f"[ArxivCollector] {len(curiosity_hits)} curiosity hits, signals for {len(summary)} threads")
        return {"thread_signals": summary, "curiosity_hits": curiosity_hits[:5]}

    def _fetch_all_papers(self) -> list[dict]:
        papers = []
        seen_ids = set()
        for feed_url in ARXIV_FEEDS:
            try:
                req = urllib.request.Request(feed_url, headers={"User-Agent": "JordanHEDGE/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    root = ET.fromstring(resp.read())
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                # Try RSS format first, then Atom
                items = root.findall(".//item") or root.findall("atom:entry", ns)
                for item in items:
                    paper = self._parse_paper(item, ns)
                    if paper and paper["id"] not in seen_ids:
                        papers.append(paper)
                        seen_ids.add(paper["id"])
            except Exception as e:
                logger.warning(f"[ArxivCollector] Failed to fetch {feed_url}: {e}")
        return papers

    def _parse_paper(self, item, ns) -> dict:
        def get(tag, ns_prefix=None):
            el = item.find(tag) or (item.find(f"{ns_prefix}:{tag}", ns) if ns_prefix else None)
            return (el.text or "").strip() if el is not None else ""

        title   = get("title")
        summary = get("description") or get("summary", "atom")
        link    = get("link") or get("id")
        date    = get("pubDate") or get("updated", "atom") or get("published", "atom")
        arxiv_id = link.split("/")[-1] if link else title[:40]

        if not title:
            return None
        return {"id": arxiv_id, "title": title, "summary": summary[:500], "link": link, "date": date}

    def _load_state(self) -> dict:
        if STATE_FILE.exists():
            try:
                return json.loads(STATE_FILE.read_text())
            except Exception:
                return {}
        return {}

    def _save_state(self):
        STATE_FILE.write_text(json.dumps(self.state, indent=2))

    def _is_seen(self, paper_id: str) -> bool:
        return paper_id in self.state.get("seen_ids", [])

    def _mark_seen(self, paper_id: str):
        seen = self.state.setdefault("seen_ids", [])
        if paper_id not in seen:
            seen.append(paper_id)
        self.state["seen_ids"] = seen[-2000:]
