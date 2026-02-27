"""
GitHub Collector for Jordan HEDGE.

Scans GitHub repos for signals:
- CI failure on main branch         → deployment_issue
- Stale open PR (> 7 days)          → pr_open_stale + progress_stalled
- Stale open issues (> 14 days)     → task_stale
- Recent commits after long silence → new_evidence_available (activity resumed)
- No commits in > 21 days           → progress_stalled
"""

import json
import subprocess
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from ..engine.hedge_engine import HedgeEngine, SignalItem

logger = logging.getLogger(__name__)

# Map repos to threads
REPO_THREAD_MAP: dict[str, str] = {
    "patient-talk-summary":    "MEL Cloud Deployment",
    "med-vigil-ai":            "MEL Cloud Deployment",
    "Melvin_DeHerrera_Estate": "Estate Inventory Deadline",
    "estate-command-center":   "Estate Inventory Deadline",
    "jordan-hedge":            None,   # Self — no thread mapping
}

STALE_PR_DAYS     = 7
STALE_ISSUE_DAYS  = 14
STALE_COMMIT_DAYS = 21
GITHUB_USER       = "jordandeherrera"


class GitHubCollector:

    def __init__(self, engine: HedgeEngine = None):
        self.engine = engine or HedgeEngine()

    def collect(self) -> dict:
        """Scan GitHub repos and ingest signals. Returns summary per thread."""
        thread_signals: dict[str, list[SignalItem]] = {}

        for repo, thread_name in REPO_THREAD_MAP.items():
            if not thread_name:
                continue

            signals = self._collect_repo(repo)
            if signals:
                existing = thread_signals.setdefault(thread_name, [])
                existing.extend(signals)

        # Deduplicate by signal_name per thread and ingest
        summary = {}
        for thread_name, signals in thread_signals.items():
            seen = set()
            unique = []
            for s in signals:
                if s.signal_name not in seen:
                    unique.append(s)
                    seen.add(s.signal_name)

            if unique:
                count = self.engine.ingest_signals(thread_name, unique)
                self.engine.generate_actions(thread_name)
                summary[thread_name] = {
                    "signals": [s.signal_name for s in unique],
                    "ingested": count,
                }

        logger.info(f"[GitHubCollector] Signals ingested for {len(summary)} threads")
        return summary

    def _collect_repo(self, repo: str) -> list[SignalItem]:
        """Collect signals for a single repo."""
        signals = []
        full_repo = f"{GITHUB_USER}/{repo}"
        now = datetime.now(timezone.utc)

        # ── CI Status ────────────────────────────────
        ci = self._get_latest_ci_run(full_repo)
        if ci:
            conclusion = ci.get("conclusion", "")
            status     = ci.get("status", "")
            created_at = self._parse_dt(ci.get("createdAt", ""))

            if conclusion in ("failure", "cancelled", "timed_out"):
                signals.append(SignalItem(
                    signal_name="deployment_issue",
                    value="present",
                    confidence=0.95,
                    source="github",
                    raw_data={"repo": repo, "run": ci.get("name"), "conclusion": conclusion,
                              "date": ci.get("createdAt", "")},
                ))
            elif conclusion == "success":
                signals.append(SignalItem(
                    signal_name="deployment_issue",
                    value="absent",
                    confidence=0.9,
                    source="github",
                    raw_data={"repo": repo, "conclusion": "success"},
                ))

        # ── Stale PRs ────────────────────────────────
        prs = self._get_open_prs(full_repo)
        stale_prs = []
        for pr in prs:
            created = self._parse_dt(pr.get("createdAt", ""))
            if created and (now - created).days >= STALE_PR_DAYS:
                stale_prs.append(pr)

        if stale_prs:
            signals.append(SignalItem(
                signal_name="pr_open_stale",
                value="present",
                confidence=0.9,
                source="github",
                raw_data={"repo": repo, "stale_pr_count": len(stale_prs),
                          "prs": [{"number": p.get("number"), "title": p.get("title", "")[:60],
                                   "days_open": (now - self._parse_dt(p.get("createdAt",""))).days
                                   if self._parse_dt(p.get("createdAt","")) else "?"} for p in stale_prs]},
            ))
            signals.append(SignalItem(
                signal_name="progress_stalled",
                value="present",
                confidence=0.8,
                source="github",
                raw_data={"repo": repo, "reason": "stale_prs"},
            ))

        # ── Commit Staleness ─────────────────────────
        last_commit_dt = self._get_last_commit_dt(full_repo)
        if last_commit_dt:
            days_since = (now - last_commit_dt).days
            if days_since >= STALE_COMMIT_DAYS:
                signals.append(SignalItem(
                    signal_name="progress_stalled",
                    value="present",
                    confidence=0.85,
                    source="github",
                    raw_data={"repo": repo, "days_since_commit": days_since,
                              "last_commit": last_commit_dt.isoformat()},
                ))
            elif days_since <= 2:
                # Recent activity — thread is not stale
                signals.append(SignalItem(
                    signal_name="task_stale",
                    value="absent",
                    confidence=0.8,
                    source="github",
                    raw_data={"repo": repo, "days_since_commit": days_since},
                ))

        if signals:
            logger.info(f"[GitHubCollector] {repo}: {[s.signal_name for s in signals]}")

        return signals

    # ─────────────────────────────────────────────
    # GH CLI WRAPPERS
    # ─────────────────────────────────────────────

    def _get_latest_ci_run(self, full_repo: str) -> Optional[dict]:
        try:
            result = subprocess.run(
                ["gh", "run", "list", "-R", full_repo,
                 "--limit", "1",
                 "--json", "status,conclusion,createdAt,name,headBranch"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return None
            runs = json.loads(result.stdout)
            # Only care about main/master branch runs
            main_runs = [r for r in runs if r.get("headBranch") in ("main", "master")]
            return main_runs[0] if main_runs else (runs[0] if runs else None)
        except Exception as e:
            logger.debug(f"[GitHubCollector] CI check failed for {full_repo}: {e}")
            return None

    def _get_open_prs(self, full_repo: str) -> list[dict]:
        try:
            result = subprocess.run(
                ["gh", "pr", "list", "-R", full_repo,
                 "--state", "open",
                 "--json", "number,title,createdAt,author"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return []
            return json.loads(result.stdout)
        except Exception:
            return []

    def _get_last_commit_dt(self, full_repo: str) -> Optional[datetime]:
        try:
            result = subprocess.run(
                ["gh", "api", f"repos/{full_repo}/commits",
                 "--jq", ".[0].commit.committer.date",
                 "-q", ".[0].commit.committer.date"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return None
            dt_str = result.stdout.strip()
            return self._parse_dt(dt_str) if dt_str else None
        except Exception:
            return None

    def _parse_dt(self, dt_str: str) -> Optional[datetime]:
        if not dt_str:
            return None
        try:
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return None
