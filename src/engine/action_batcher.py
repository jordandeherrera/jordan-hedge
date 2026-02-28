"""
Action Batcher — Effort-aware clustering of next_actions across HEDGE threads.

Core insight: many actions share a research substrate (the underlying knowledge
work required). Batching those actions saves effort, but batching has coordination
cost. This module decides when to batch and when to act individually.

Decision rule:
    batch_value      = (n_threads - 1) × overlap_coefficient × action_cost_estimate
    coordination_cost = FIXED_OVERHEAD + n_threads × domain_heterogeneity_penalty
    batch_if: batch_value > coordination_cost

Substrate detection (Option C): embedding similarity between action
descriptions. Actions that cluster in embedding space share substrate
implicitly — no ontology maintenance required, gets smarter as action
corpus grows.

Embedding backend: Voyage AI (voyage-3-lite, 512-dim). Backend is
abstracted behind EmbeddingBackend so it can be swapped without
touching the clustering logic.
"""

from __future__ import annotations

import json
import math
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# TUNING CONSTANTS
# ─────────────────────────────────────────────

# Cosine similarity threshold for neural (Voyage) embeddings
SUBSTRATE_SIMILARITY_THRESHOLD_NEURAL = 0.82

# Cosine similarity threshold for TF-IDF fallback
SUBSTRATE_SIMILARITY_THRESHOLD_TFIDF = 0.55

# Minimum cluster size to consider batching worthwhile
MIN_BATCH_SIZE = 2

# Fixed overhead of assembling and coordinating a batch (in action-cost units)
FIXED_COORDINATION_OVERHEAD = 0.4

# Per-thread penalty for heterogeneous domains within a batch
DOMAIN_HETEROGENEITY_PENALTY = 0.15

# Cross-domain batches have an extra synthesis cost (shared research, separate apply)
CROSS_DOMAIN_SYNTHESIS_OVERHEAD = 0.25

# Assumed normalized cost of a single action (1.0 = one focused work unit)
DEFAULT_ACTION_COST = 1.0

# Entropy threshold: below this, the thread "knows what to do" → act individually
LOW_ENTROPY_THRESHOLD = 0.35

# Impact threshold: above this urgency_score, never wait for batch
HIGH_URGENCY_THRESHOLD = 0.75


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class ActionNode:
    """A pending action enriched with thread context and embedding."""
    action_id: int
    thread_id: int
    thread_name: str
    domain: str
    action_type: str
    title: str
    description: str
    expected_utility: float
    urgency_score: float
    thread_entropy: float           # current entropy of the thread's top hypothesis
    research_topic: str = "product_strategy"
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class ActionBatch:
    """
    A cluster of actions that share a research substrate.
    Includes the batcher's recommendation: batch or act individually.
    """
    cluster_id: int
    actions: list[ActionNode]
    centroid_embedding: Optional[np.ndarray] = field(default=None, repr=False)
    substrate_label: str = ""           # human-readable substrate description
    intra_similarity: float = 0.0       # avg pairwise cosine within cluster
    estimated_effort_solo: float = 0.0  # total cost if each action done separately
    estimated_effort_batch: float = 0.0 # total cost if batched
    coordination_cost: float = 0.0
    net_savings: float = 0.0
    recommendation: str = "individual"  # "batch" | "individual" | "urgent_solo"
    rationale: str = ""

    @property
    def n(self) -> int:
        return len(self.actions)

    @property
    def threads(self) -> list[str]:
        return [a.thread_name for a in self.actions]

    @property
    def domains(self) -> set[str]:
        return {a.domain for a in self.actions}

    @property
    def max_urgency(self) -> float:
        return max((a.urgency_score for a in self.actions), default=0.0)

    @property
    def total_utility(self) -> float:
        return sum(a.expected_utility for a in self.actions)


# ─────────────────────────────────────────────
# EMBEDDING BACKEND PROTOCOL
# ─────────────────────────────────────────────

@runtime_checkable
class EmbeddingBackend(Protocol):
    def embed(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a list of texts, return unit-normalized vectors."""
        ...


class VoyageBackend:
    """
    Voyage AI embedding backend (voyage-3-lite, 512-dim).
    Batches API calls; caches embeddings in the DB to avoid redundant calls.
    """
    MODEL = "voyage-3-lite"

    def __init__(self, api_key: Optional[str] = None):
        import voyageai
        self._client = voyageai.Client(api_key=api_key or os.environ.get("VOYAGE_API_KEY", ""))

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        result = self._client.embed(texts, model=self.MODEL, input_type="document")
        vecs = [np.array(e, dtype=np.float32) for e in result.embeddings]
        # Unit-normalize for cosine similarity via dot product
        return [v / (np.linalg.norm(v) + 1e-10) for v in vecs]


class TFIDFBackend:
    """
    Fallback backend: TF-IDF vectors computed in-process with numpy only.
    No external dependencies. Lower quality than a neural backend but works
    offline and costs nothing.
    """

    def __init__(self):
        self._vocab: dict[str, int] = {}
        self._idf: Optional[np.ndarray] = None
        self._corpus_texts: list[str] = []

    def _tokenize(self, text: str) -> list[str]:
        import re
        return re.findall(r'[a-z]+', text.lower())

    def _fit(self, texts: list[str]):
        """Build vocabulary and IDF from corpus."""
        docs = [self._tokenize(t) for t in texts]
        # Build vocab
        vocab: dict[str, int] = {}
        for doc in docs:
            for tok in doc:
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        self._vocab = vocab
        n = len(docs)
        V = len(vocab)
        # Compute document frequency
        df = np.zeros(V, dtype=np.float32)
        for doc in docs:
            for tok in set(doc):
                if tok in vocab:
                    df[vocab[tok]] += 1
        self._idf = np.log((n + 1) / (df + 1)) + 1.0

    def _vectorize(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        V = len(self._vocab)
        tf = np.zeros(V, dtype=np.float32)
        for tok in tokens:
            if tok in self._vocab:
                tf[self._vocab[tok]] += 1
        if tokens:
            tf /= len(tokens)
        vec = tf * self._idf
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-10)

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        self._fit(texts)
        return [self._vectorize(t) for t in texts]


def build_backend() -> EmbeddingBackend:
    """
    Return the best available embedding backend.
    Tries Voyage AI first (probe with a single embed call), falls back to TF-IDF.
    """
    voyage_key = os.environ.get("VOYAGE_API_KEY", "")
    if voyage_key:
        try:
            b = VoyageBackend(voyage_key)
            # Probe with a single call to confirm the key actually works
            b.embed(["probe"])
            logger.info("ActionBatcher: using VoyageBackend")
            return b
        except Exception as e:
            logger.warning(f"VoyageBackend unavailable: {e}; falling back to TF-IDF")

    logger.info("ActionBatcher: using TFIDFBackend (set VOYAGE_API_KEY for better semantic clustering)")
    return TFIDFBackend()


# ─────────────────────────────────────────────
# COSINE UTILITIES
# ─────────────────────────────────────────────

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))   # already unit-normalized


def pairwise_similarity_matrix(vecs: list[np.ndarray]) -> np.ndarray:
    M = np.stack(vecs)              # shape (n, d)
    sim = M @ M.T                   # cosine similarity matrix (unit-normalized)
    np.fill_diagonal(sim, 1.0)
    return sim


# ─────────────────────────────────────────────
# CLUSTERING
# ─────────────────────────────────────────────

def greedy_cluster(
    nodes: list[ActionNode],
    threshold: float = SUBSTRATE_SIMILARITY_THRESHOLD_NEURAL,
) -> list[list[ActionNode]]:
    """
    Single-pass greedy clustering by embedding similarity.
    Each action is assigned to the first existing cluster whose centroid
    is within `threshold` cosine similarity, or starts a new cluster.

    O(n²) — fine for dozens of pending actions.
    """
    clusters: list[list[ActionNode]] = []
    centroids: list[np.ndarray] = []

    for node in nodes:
        if node.embedding is None:
            # No embedding — singleton cluster
            clusters.append([node])
            centroids.append(np.zeros(1))
            continue

        best_cluster = -1
        best_sim = threshold - 1e-9

        for i, centroid in enumerate(centroids):
            if centroid.shape[0] == 1:
                continue  # sentinel for no-embedding node
            sim = cosine(node.embedding, centroid)
            if sim > best_sim:
                best_sim = sim
                best_cluster = i

        if best_cluster >= 0:
            clusters[best_cluster].append(node)
            # Update centroid as running mean
            n = len(clusters[best_cluster])
            centroids[best_cluster] = (
                centroids[best_cluster] * (n - 1) / n
                + node.embedding / n
            )
            # Re-normalize
            norm = np.linalg.norm(centroids[best_cluster])
            centroids[best_cluster] /= norm + 1e-10
        else:
            clusters.append([node])
            centroids.append(node.embedding.copy())

    return clusters


# ─────────────────────────────────────────────
# BATCH ECONOMICS
# ─────────────────────────────────────────────

def compute_batch_economics(cluster: list[ActionNode], topic: str = "") -> dict:
    """
    Compute the effort economics for a candidate batch.

    Savings model:
        effort_solo  = n × action_cost
        effort_batch = action_cost + (n-1) × action_cost × (1 - overlap_coefficient)
        coordination = FIXED_OVERHEAD + n × domain_heterogeneity_penalty × heterogeneity

    overlap_coefficient: derived from intra-cluster similarity.
    domain_heterogeneity: fraction of domains that differ from the majority domain.
    """
    n = len(cluster)
    if n < 2:
        return {
            "effort_solo": DEFAULT_ACTION_COST,
            "effort_batch": DEFAULT_ACTION_COST,
            "coordination_cost": 0.0,
            "net_savings": 0.0,
            "intra_similarity": 1.0,
        }

    # Intra-cluster similarity
    vecs = [a.embedding for a in cluster if a.embedding is not None]
    if len(vecs) >= 2:
        sim_matrix = pairwise_similarity_matrix(vecs)
        # Average of upper triangle
        upper = [sim_matrix[i, j] for i in range(len(vecs)) for j in range(i+1, len(vecs))]
        intra_sim = float(np.mean(upper))
    else:
        intra_sim = 0.5

    # Overlap coefficient: how much of the research effort is shared
    # Normalized: sim=1.0 → 100% overlap; sim=threshold → ~0% overlap
    overlap = max(0.0, (intra_sim - SUBSTRATE_SIMILARITY_THRESHOLD_NEURAL) / (1.0 - SUBSTRATE_SIMILARITY_THRESHOLD_NEURAL))

    # Domain heterogeneity
    domains = [a.domain for a in cluster]
    majority_count = max(domains.count(d) for d in set(domains))
    heterogeneity = 1.0 - majority_count / n

    effort_solo = n * DEFAULT_ACTION_COST
    effort_batch = DEFAULT_ACTION_COST + (n - 1) * DEFAULT_ACTION_COST * (1.0 - overlap)

    # Cross-domain batches add synthesis overhead (shared research, but each
    # domain still needs its own apply/decision step).
    from .topic_inference import is_cross_domain_batchable
    n_domains = len({a.domain for a in cluster})
    cross_domain_overhead = (
        CROSS_DOMAIN_SYNTHESIS_OVERHEAD * (n_domains - 1)
        if is_cross_domain_batchable(topic) and n_domains > 1
        else 0.0
    )

    coordination = (
        FIXED_COORDINATION_OVERHEAD
        + n * DOMAIN_HETEROGENEITY_PENALTY * heterogeneity
        + cross_domain_overhead
    )
    net_savings = effort_solo - effort_batch - coordination

    return {
        "effort_solo": round(effort_solo, 2),
        "effort_batch": round(effort_batch + coordination, 2),
        "coordination_cost": round(coordination, 2),
        "cross_domain_overhead": round(cross_domain_overhead, 2),
        "n_domains": n_domains,
        "net_savings": round(net_savings, 2),
        "intra_similarity": round(intra_sim, 3),
        "overlap_coefficient": round(overlap, 3),
    }


# ─────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ─────────────────────────────────────────────

def recommend(cluster: list[ActionNode], economics: dict) -> tuple[str, str]:
    """
    Return (recommendation, rationale) for a cluster.

    Rules (in priority order):
    1. Any action with urgency > HIGH_URGENCY_THRESHOLD AND low entropy
       → urgent_solo (never wait for batch assembly)
    2. Single-action clusters → individual
    3. net_savings > 0 → batch
    4. Otherwise → individual
    """
    # Rule 1: urgent + low entropy = act now, don't wait
    urgent_solos = [
        a for a in cluster
        if a.urgency_score >= HIGH_URGENCY_THRESHOLD and a.thread_entropy <= LOW_ENTROPY_THRESHOLD
    ]
    if urgent_solos:
        names = ", ".join(a.thread_name for a in urgent_solos)
        return (
            "urgent_solo",
            f"High urgency + low entropy on [{names}] — act immediately, don't wait for batch."
        )

    if len(cluster) == 1:
        a = cluster[0]
        if a.thread_entropy <= LOW_ENTROPY_THRESHOLD:
            return (
                "individual",
                f"Single action, low entropy (belief converged) — execute directly on [{a.thread_name}]."
            )
        return (
            "individual",
            f"Single action — no batch opportunity found for [{a.thread_name}]."
        )

    if economics["net_savings"] > 0:
        savings_pct = int(economics["net_savings"] / economics["effort_solo"] * 100)
        threads = ", ".join(a.thread_name for a in cluster)
        return (
            "batch",
            (
                f"Shared substrate across {len(cluster)} threads [{threads}]. "
                f"Estimated {savings_pct}% effort savings "
                f"(solo: {economics['effort_solo']:.1f} units → "
                f"batch: {economics['effort_batch']:.1f} units incl. {economics['coordination_cost']:.1f} coordination). "
                f"Substrate overlap: {economics['overlap_coefficient']:.0%}."
            )
        )

    # Batch not worth it
    threads = ", ".join(a.thread_name for a in cluster)
    return (
        "individual",
        (
            f"Substrate similarity detected [{threads}] but coordination cost "
            f"({economics['coordination_cost']:.1f}) exceeds savings "
            f"({economics['effort_solo'] - economics['effort_batch']:.1f}). "
            f"Act individually."
        )
    )


# ─────────────────────────────────────────────
# MAIN BATCHER
# ─────────────────────────────────────────────

class ActionBatcher:
    """
    Main entry point. Load pending actions from DB, embed, cluster, score,
    and return ActionBatch recommendations.

    Usage:
        batcher = ActionBatcher(db_conn)
        batches = batcher.compute_batches()
        for b in batches:
            print(b.recommendation, b.rationale)
    """

    def __init__(self, db_conn, backend: Optional[EmbeddingBackend] = None):
        self._conn = db_conn
        self._backend = backend or build_backend()

    def load_actions(self, limit: int = 50) -> list[ActionNode]:
        """
        Load pending actions with their thread entropy context.
        Thread entropy is the mean entropy of the thread's active hypotheses.
        """
        rows = self._conn.execute("""
            SELECT
                na.id,
                na.thread_id,
                t.name        AS thread_name,
                t.domain,
                t.description AS thread_description,
                na.action_type,
                na.title,
                COALESCE(na.description, '') AS description,
                na.expected_utility,
                na.urgency_score,
                COALESCE(na.research_topic, 'product_strategy') AS research_topic,
                COALESCE(avg_h.avg_entropy, 0.5) AS thread_entropy
            FROM next_actions na
            JOIN threads t ON t.id = na.thread_id
            LEFT JOIN (
                SELECT
                    thread_id,
                    AVG(belief_entropy(log_odds_posterior)) AS avg_entropy
                FROM hypotheses
                WHERE status = 'active'
                GROUP BY thread_id
            ) avg_h ON avg_h.thread_id = na.thread_id
            WHERE na.status = 'pending'
            ORDER BY na.expected_utility DESC, na.urgency_score DESC
            LIMIT ?
        """, (limit,)).fetchall()

        return [ActionNode(
            action_id=r["id"],
            thread_id=r["thread_id"],
            thread_name=r["thread_name"],
            domain=r["domain"],
            action_type=r["action_type"],
            title=r["title"],
            description=f"{r['description']} context:{r['thread_description'] or ''}",
            expected_utility=r["expected_utility"],
            urgency_score=r["urgency_score"],
            thread_entropy=r["thread_entropy"],
            research_topic=r["research_topic"],
        ) for r in rows]

    @staticmethod
    def _action_text(node: ActionNode) -> str:
        """
        Build the embedding input text for an action.
        Uses domain + action_type + thread name + description to give the
        embedding model enough semantic signal beyond the templated title.
        """
        return (
            f"domain:{node.domain} "
            f"type:{node.action_type} "
            f"thread:{node.thread_name} "
            f"{node.description}"
        )

    def embed_actions(self, nodes: list[ActionNode]) -> None:
        """
        Embed each action in-place using richer semantic text.
        Checks DB cache first; stores new embeddings back to DB.
        """
        # Check which action_ids already have cached embeddings
        ids = [n.action_id for n in nodes]
        if not ids:
            return

        placeholders = ",".join("?" * len(ids))
        cached = {
            row["action_id"]: np.frombuffer(row["embedding"], dtype=np.float32)
            for row in self._conn.execute(
                f"SELECT action_id, embedding FROM action_embeddings WHERE action_id IN ({placeholders})",
                ids
            ).fetchall()
        }

        # Separate cached from uncached
        to_embed: list[tuple[int, str]] = []
        for n in nodes:
            if n.action_id in cached:
                n.embedding = cached[n.action_id]
            else:
                to_embed.append((n.action_id, self._action_text(n)))

        if not to_embed:
            return

        # Embed uncached
        texts = [t for _, t in to_embed]
        try:
            vecs = self._backend.embed(texts)
        except Exception as e:
            logger.warning(f"Embedding failed: {e}; proceeding without embeddings")
            return

        # Store in DB and assign to nodes
        node_by_id = {n.action_id: n for n in nodes}
        for (action_id, _), vec in zip(to_embed, vecs):
            node_by_id[action_id].embedding = vec
            self._conn.execute(
                "INSERT OR REPLACE INTO action_embeddings (action_id, embedding) VALUES (?, ?)",
                (action_id, vec.tobytes())
            )
        self._conn.commit()

    def compute_batches(self, limit: int = 50) -> list[ActionBatch]:
        """
        Two-phase clustering pipeline:
          Phase 1 — Group by research_topic (hard, explicit substrate signal).
                     Cross-domain batchable topics pull together actions from
                     different domains if they share the same research work.
          Phase 2 — Within each topic group, use embedding similarity to split
                     broad topics that are too heterogeneous to batch usefully.

        Returns ActionBatch list sorted by total_utility descending.
        """
        from .topic_inference import is_cross_domain_batchable, TOPIC_LABELS

        nodes = self.load_actions(limit)
        if not nodes:
            return []

        self.embed_actions(nodes)

        is_neural = isinstance(self._backend, VoyageBackend)
        emb_threshold = (
            SUBSTRATE_SIMILARITY_THRESHOLD_NEURAL if is_neural
            else SUBSTRATE_SIMILARITY_THRESHOLD_TFIDF
        )

        # ── Phase 1: group by research_topic ──────────────────────────────
        topic_groups: dict[str, list[ActionNode]] = {}
        for n in nodes:
            topic_groups.setdefault(n.research_topic, []).append(n)

        # ── Phase 2: within each topic group, split by embedding similarity
        #            if the topic is broad (cross-domain batchable) and the
        #            group spans multiple domains with low intra-similarity.
        raw_clusters: list[tuple[str, list[ActionNode]]] = []  # (topic, nodes)

        for topic, group in topic_groups.items():
            if len(group) <= 1:
                raw_clusters.append((topic, group))
                continue

            with_emb = [n for n in group if n.embedding is not None]
            without_emb = [n for n in group if n.embedding is None]

            if len(with_emb) < 2:
                raw_clusters.append((topic, group))
                continue

            # Only split cross-domain batchable topics — thread-specific topics
            # should stay together (they're all about the same external context).
            if is_cross_domain_batchable(topic):
                sub_clusters = greedy_cluster(with_emb, threshold=emb_threshold)
            else:
                sub_clusters = [with_emb]

            for sc in sub_clusters:
                raw_clusters.append((topic, sc + without_emb if sc is sub_clusters[-1] else sc))

        # ── Build ActionBatch objects ──────────────────────────────────────
        batches: list[ActionBatch] = []
        for idx, (topic, cluster) in enumerate(raw_clusters):
            economics = compute_batch_economics(cluster, topic)
            rec, rationale = recommend(cluster, economics)

            vecs = [a.embedding for a in cluster if a.embedding is not None]
            centroid = np.mean(np.stack(vecs), axis=0) if vecs else None
            if centroid is not None:
                centroid = centroid / (np.linalg.norm(centroid) + 1e-10)

            batches.append(ActionBatch(
                cluster_id=idx,
                actions=cluster,
                centroid_embedding=centroid,
                substrate_label=TOPIC_LABELS.get(topic, topic),
                intra_similarity=economics.get("intra_similarity", 0.0),
                estimated_effort_solo=economics["effort_solo"],
                estimated_effort_batch=economics["effort_batch"],
                coordination_cost=economics["coordination_cost"],
                net_savings=economics["net_savings"],
                recommendation=rec,
                rationale=rationale,
            ))

        batches.sort(key=lambda b: b.total_utility, reverse=True)
        return batches


def _label_substrate(cluster: list[ActionNode]) -> str:
    """
    Generate a human-readable substrate label from cluster action types and domains.
    """
    action_types = list({a.action_type for a in cluster})
    domains = list({a.domain for a in cluster})
    if len(cluster) == 1:
        return f"{cluster[0].action_type} / {cluster[0].domain}"
    return f"{' + '.join(sorted(action_types))} [{' + '.join(sorted(domains))}]"
