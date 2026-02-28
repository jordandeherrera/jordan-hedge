"""
Hacker News Collector for Jordan HEDGE Curiosity Engine.
Uses Algolia HN API (no key required).
Fetches Show HN + Ask HN + topic-relevant posts.
"""

import urllib.request
import json
import logging
from pathlib import Path

from ..engine.hedge_engine import HedgeEngine, SignalItem

logger = logging.getLogger(__name__)
STATE_FILE = Path(__file__).parent.parent.parent / ".hn_state.json"

ALGOLIA_API = "https://hn.algolia.com/api/v1"

TOPIC_THREAD_MAP = {
    "MEL Cloud Deployment":   ["medical ai", "clinical ai", "prior auth", "ehr", "healthcare ai", "medical scribe"],
    "MEL Ontology Quality":   ["bayesian", "ontology", "knowledge graph", "probabilistic", "federated learning", "differential diagnosis"],
    "Ridgeline L6 Promotion": ["investment", "fintech", "portfolio", "knowledge management", "mcp server", "model context protocol"],
    "Meniscus Tear Recovery": ["longevity", "peptide", "biohacking", "zone 2", "vo2", "rapamycin", "cartilage", "injury"],
}

CURIOSITY_KEYWORDS = [
    "show hn", "autonomous agent", "reasoning", "bayesian", "ontology",
    "knowledge graph", "federated", "mcp", "llm", "agentic",
    "prior auth", "insurance", "healthcare", "estate", "probate",
    "medical error", "calibration", "uncertainty",
]


class HNCollector:

    def __init__(self, engine: HedgeEngine = None):
        self.engine = engine or HedgeEngine()
        self.state = self._load_state()

    def collect(self) -> dict:
        posts = self._fetch_recent_posts()
        logger.info(f"[HNCollector] Fetched {len(posts)} posts")

        thread_signals: dict[str, list[SignalItem]] = {}
        curiosity_hits = []

        for post in posts:
            if self._is_seen(post["id"]):
                continue
            text = f"{post.get('title','')} {post.get('text','')}".lower()

            for thread_name, keywords in TOPIC_THREAD_MAP.items():
                if any(kw in text for kw in keywords):
                    thread_signals.setdefault(thread_name, []).append(SignalItem(
                        signal_name="new_evidence_available",
                        value="present",
                        confidence=0.65,
                        source="hackernews",
                        raw_data={"title": post.get("title"), "url": post.get("url",""),
                                  "score": post.get("score",0), "id": post["id"]},
                    ))

            if any(kw in text for kw in CURIOSITY_KEYWORDS):
                curiosity_hits.append({
                    "title": post.get("title",""),
                    "url": post.get("url") or f"https://news.ycombinator.com/item?id={post['id']}",
                    "score": post.get("score", 0),
                    "id": post["id"],
                    "type": "show_hn" if "show hn" in text else "ask_hn" if "ask hn" in text else "hn",
                })

            self._mark_seen(post["id"])

        summary = {}
        for thread_name, signals in thread_signals.items():
            seen_sigs = set()
            unique = [s for s in signals if s.signal_name not in seen_sigs and not seen_sigs.add(s.signal_name)]
            if unique:
                count = (self.engine.ingest_signals(thread_name, unique) or {}).get("processed", 0)
                self.engine.generate_actions(thread_name)
                summary[thread_name] = {"signals": [s.signal_name for s in unique], "ingested": count}

        self.state["latest_curiosity_hn"] = sorted(curiosity_hits, key=lambda x: -x["score"])[:10]
        self._save_state()
        return {"thread_signals": summary, "curiosity_hits": curiosity_hits[:5]}

    def _fetch_recent_posts(self) -> list[dict]:
        posts = []
        seen_ids = set()
        queries = [
            f"{ALGOLIA_API}/search?tags=show_hn&hitsPerPage=30",
            f"{ALGOLIA_API}/search?tags=ask_hn&query=ai+health+legal+agent&hitsPerPage=20",
            f"{ALGOLIA_API}/search?query=bayesian+ontology+agent+healthcare+insurance&hitsPerPage=20&numericFilters=points>5",
        ]
        for url in queries:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "JordanHEDGE/1.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read())
                for hit in data.get("hits", []):
                    pid = str(hit.get("objectID",""))
                    if pid and pid not in seen_ids:
                        posts.append({"id": pid, "title": hit.get("title",""),
                                      "url": hit.get("url",""), "text": hit.get("story_text","") or "",
                                      "score": hit.get("points", 0)})
                        seen_ids.add(pid)
            except Exception as e:
                logger.debug(f"[HNCollector] {e}")
        return posts

    def _load_state(self) -> dict:
        if STATE_FILE.exists():
            try: return json.loads(STATE_FILE.read_text())
            except: return {}
        return {}

    def _save_state(self):
        STATE_FILE.write_text(json.dumps(self.state, indent=2))

    def _is_seen(self, pid: str) -> bool:
        return str(pid) in self.state.get("seen_ids", [])

    def _mark_seen(self, pid: str):
        seen = self.state.setdefault("seen_ids", [])
        if str(pid) not in seen:
            seen.append(str(pid))
        self.state["seen_ids"] = seen[-3000:]
