"""
Reddit Collector for Jordan HEDGE Curiosity Engine.
Uses public .json API (no auth required, 10 QPM limit).
Monitors relevant subreddits for product signals and research.
"""

import urllib.request
import json
import logging
import time
from pathlib import Path

from ..engine.hedge_engine import HedgeEngine, SignalItem

logger = logging.getLogger(__name__)

STATE_FILE = Path(__file__).parent.parent.parent / ".reddit_state.json"

# Subreddits + their domain relevance
SUBREDDIT_THREAD_MAP = {
    "MachineLearning":    ["MEL Ontology Quality", "Ridgeline L6 Promotion"],
    "artificial":         ["MEL Ontology Quality", "Ridgeline L6 Promotion"],
    "healthIT":           ["MEL Cloud Deployment", "MEL Ontology Quality"],
    "medicine":           ["MEL Cloud Deployment", "Meniscus Tear Recovery"],
    "Noctor":             ["MEL Cloud Deployment"],
    "legaladvice":        ["Estate Inventory Deadline", "Medical Malpractice — Renown"],
    "Pathofexile":        [],   # skip
    "longevity":          ["Meniscus Tear Recovery"],
    "Biohackers":         ["Meniscus Tear Recovery"],
    "investing":          ["Ridgeline L6 Promotion"],
    "startups":           [],   # curiosity only
    "SideProject":        [],   # curiosity only
}

# Keywords that indicate product gap / user hack (high curiosity signal)
HACK_SIGNALS = [
    "i built", "show reddit", "i made", "launched", "open source",
    "anyone else use", "hack i use", "workaround", "pain point",
    "wish there was", "frustrated with", "no good tool",
    "spreadsheet for", "notion template", "obsidian",
    "prior auth", "insurance denied", "appeal", "EOB", "billing error",
    "estate", "probate", "beneficiary",
]

CURIOSITY_KEYWORDS = [
    "bayesian", "uncertainty", "agent", "llm", "ontology",
    "knowledge graph", "federated", "prior auth", "insurance denial",
    "medical error", "clinical ai", "longevity", "peptide", "bpc",
]


class RedditCollector:

    def __init__(self, engine: HedgeEngine = None):
        self.engine = engine or HedgeEngine()
        self.state = self._load_state()

    def collect(self) -> dict:
        summary = {}
        thread_signals: dict[str, list[SignalItem]] = {}
        curiosity_hits = []
        total_fetched = 0

        for subreddit, thread_names in SUBREDDIT_THREAD_MAP.items():
            posts = self._fetch_subreddit(subreddit, limit=10)
            total_fetched += len(posts)
            time.sleep(0.1)  # respect 10 QPM

            for post in posts:
                if self._is_seen(post["id"]):
                    continue

                text = f"{post['title']} {post.get('selftext','')}".lower()

                # Thread signals
                for thread_name in thread_names:
                    thread_signals.setdefault(thread_name, []).append(SignalItem(
                        signal_name="new_evidence_available",
                        value="present",
                        confidence=0.6,
                        source="reddit",
                        raw_data={"title": post["title"], "url": post.get("url",""),
                                  "subreddit": subreddit, "score": post.get("score",0),
                                  "id": post["id"]},
                    ))

                # Hack/product gap signals (high curiosity value)
                if any(kw in text for kw in HACK_SIGNALS):
                    curiosity_hits.append({
                        "title": post["title"],
                        "url": f"https://reddit.com{post.get('permalink','')}",
                        "subreddit": subreddit,
                        "score": post.get("score", 0),
                        "id": post["id"],
                        "type": "hack" if any(h in text for h in ["i built","i made","show reddit"]) else "pain_point",
                    })

                # General curiosity
                elif any(kw in text for kw in CURIOSITY_KEYWORDS):
                    curiosity_hits.append({
                        "title": post["title"],
                        "url": f"https://reddit.com{post.get('permalink','')}",
                        "subreddit": subreddit,
                        "score": post.get("score", 0),
                        "id": post["id"],
                        "type": "research",
                    })

                self._mark_seen(post["id"])

        # Ingest deduplicated thread signals
        for thread_name, signals in thread_signals.items():
            seen_sigs = set()
            unique = [s for s in signals if s.signal_name not in seen_sigs and not seen_sigs.add(s.signal_name)]
            if unique:
                count = self.engine.ingest_signals(thread_name, unique)
                self.engine.generate_actions(thread_name)
                summary[thread_name] = {"signals": [s.signal_name for s in unique], "ingested": count}

        self.state["latest_curiosity_reddit"] = sorted(curiosity_hits, key=lambda x: -x["score"])[:10]
        self._save_state()

        logger.info(f"[RedditCollector] {total_fetched} posts, {len(curiosity_hits)} curiosity hits, signals for {len(summary)} threads")
        return {"thread_signals": summary, "curiosity_hits": curiosity_hits[:5]}

    def _fetch_subreddit(self, subreddit: str, limit: int = 10) -> list[dict]:
        url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "JordanHEDGE/1.0 (personal research tool)"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            return [p["data"] for p in data.get("data", {}).get("children", [])]
        except Exception as e:
            logger.debug(f"[RedditCollector] Failed r/{subreddit}: {e}")
            return []

    def _load_state(self) -> dict:
        if STATE_FILE.exists():
            try: return json.loads(STATE_FILE.read_text())
            except: return {}
        return {}

    def _save_state(self):
        STATE_FILE.write_text(json.dumps(self.state, indent=2))

    def _is_seen(self, post_id: str) -> bool:
        return post_id in self.state.get("seen_ids", [])

    def _mark_seen(self, post_id: str):
        seen = self.state.setdefault("seen_ids", [])
        if post_id not in seen:
            seen.append(post_id)
        self.state["seen_ids"] = seen[-5000:]
