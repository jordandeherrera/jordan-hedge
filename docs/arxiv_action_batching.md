# Substrate-Aware Action Batching in Personal Reasoning Systems: A HEDGE Framework Extension

**Jordan DeHerrera**
Independent Research
jordan@deherrera.dev

---

## Abstract

Personal AI reasoning systems face a scheduling problem that production task managers ignore: actions across distinct strategic threads often share underlying research substrate, making sequential execution wasteful. We present a two-phase action batching architecture that (1) infers explicit *research topics* from the semantic context of each action using a structured inference layer, and (2) clusters actions by topic-first grouping before refining with embedding similarity. The system is built as an extension to the HEDGE framework (Hypothesis-Evidence-Driven-Graph-Evaluation), a Bayesian belief-state engine for personal and clinical reasoning. We show that explicit topic inference consistently identifies cross-domain batch opportunities that embedding-only clustering misses, while a principled economics model—accounting for per-topic coordination cost and cross-domain synthesis overhead—prevents over-batching. The implementation uses neural embeddings (Voyage voyage-3-lite) with a TF-IDF fallback and integrates with SQLite-backed belief state persistence. Evaluated on 38 pending actions spanning 13 active threads across 7 domains (legal, finance, health, two product domains, an open-source project, and an employer context), the system surfaces 3 high-confidence batch opportunities and correctly identifies 4 urgent-solo clusters that should not wait for batch assembly, with a net estimated effort saving of ~1.1 normalized work units.

---

## 1. Introduction

Personal AI assistants operating on behalf of a single user across multiple life domains accumulate pending actions that span heterogeneous contexts: legal proceedings, product strategy, clinical decisions, infrastructure operations. Conventional task management treats each action atomically. This is suboptimal when actions across distinct threads share the same underlying *knowledge work*—the competitive landscape research required to decide on a healthcare AI product positioning is the same research required to evaluate a patient navigation startup; reading it once and applying it twice saves real effort.

The insight is not new in project management (the concept of "batching" similar work), but it has not been formalized in AI reasoning systems in a way that is (a) grounded in a belief-state model, (b) sensitive to urgency and uncertainty, and (c) capable of detecting cross-domain substrate overlap that semantic metadata alone cannot see.

This paper presents an action batching system built as an extension to HEDGE, a personal Bayesian reasoning framework. The contribution is threefold:

1. A **topic inference layer** that maps (domain, hypothesis type, action type, thread name) → research topic, making implicit substrate explicit before clustering.
2. A **two-phase clustering algorithm** that groups by explicit research topic first, then uses neural embedding similarity to split heterogeneous topic groups.
3. A **batch economics model** that penalizes cross-domain batches with a synthesis overhead term, preventing false positive batch recommendations where research is shared but decision context is not.

---

## 2. Background: The HEDGE Framework

HEDGE (Hypothesis-Evidence-Driven-Graph-Evaluation) is a Bayesian reasoning framework originally developed for clinical decision support and extended to personal strategic reasoning. Its core primitives are:

- **Thread**: a persistent reasoning context (e.g., "Medical Malpractice — Renown", "MEL Prior Auth Voice Agent")
- **Hypothesis**: a competing explanation with a prior and a log-odds posterior, updated via log-likelihood ratios (LLRs) as evidence arrives
- **Fact**: an observation (concept + value + confidence + source)
- **Relation**: how a fact affects a hypothesis (LLR_pos, LLR_neg)
- **Next Action**: a recommended action derived from the current belief state of a thread

The use of log-odds is deliberate: LLRs are additive in log space, making belief updates clean, explainable, and composable across evidence sources.

A thread's **entropy** measures residual uncertainty:

$$H(\theta) = -p \log_2 p - (1-p) \log_2(1-p)$$

where *p* is the posterior probability of the leading hypothesis. Low entropy + high urgency = high-confidence, time-sensitive action. High entropy = gather more evidence before acting.

Action generation maps (hypothesis_type, entropy, urgency) → action recommendations with expected utility scores:

$$U(a) = p \cdot w_r \cdot (1 - 0.5 \cdot H)$$

where $w_r$ is the hypothesis risk weight and $H$ is thread entropy. This is the signal the batching layer operates on.

---

## 3. Problem Statement

Given a set of pending actions $A = \{a_1, \ldots, a_n\}$ with associated threads, domains, utilities, and urgency scores, find a partition $\mathcal{C} = \{C_1, \ldots, C_k\}$ of $A$ into clusters such that:

1. **Correct urgency routing**: Actions with high urgency and low thread entropy are never delayed for batch assembly.
2. **Substrate matching**: Actions within a cluster share research substrate—executing them together is cheaper than executing them sequentially.
3. **Economics discipline**: The effort savings from batching exceed the coordination + synthesis costs.

The challenge is that "research substrate" is not directly observable from action metadata. A `research` action on thread "Advocate — Patient Navigation AI" (domain: advocate) and a `research` action on thread "MEL Prior Auth Voice Agent" (domain: MEL) share the substrate "competitive landscape of prior authorization automation," but their thread names, domains, and action titles give no obvious shared signal to a clustering algorithm.

---

## 4. Architecture

### 4.1 Topic Inference Layer

The topic inference layer assigns each action a **research topic** from an 11-topic taxonomy before any clustering occurs:

| Topic | Description |
|---|---|
| `competitive_landscape` | Market research, competitor analysis, positioning |
| `regulatory_environment` | Insurance regulations, legal compliance |
| `clinical_literature` | Medical evidence, treatment protocols, ontology |
| `technical_architecture` | System design, infrastructure decisions |
| `product_strategy` | GTM, pricing, roadmap, entity structure |
| `stakeholder_outreach` | Follow-ups, escalations to external parties |
| `legal_proceedings` | Court filings, attorney coordination, evidence |
| `financial_modeling` | Pricing, unit economics, benefits analysis |
| `personal_health` | Medical decisions, recovery tracking |
| `infrastructure_ops` | Deployment, cloud, DevOps, reliability |
| `knowledge_capture` | Documentation, open-source prep, write-up |

Topics are inferred via a priority-ordered lookup:

1. **Thread metadata override**: explicit `research_topics` array in thread metadata
2. **Thread name override**: a lookup table mapping canonical thread names to topics (e.g., "MEL Prior Auth Voice Agent" → `competitive_landscape`)
3. **Specific rule**: a (domain, hypothesis_type, action_type) triple lookup (e.g., `("advocate", "OPPORTUNITY_EMERGING", "research")` → `competitive_landscape`)
4. **Domain default**: a per-domain fallback (e.g., `estate` → `legal_proceedings`)
5. **Global fallback**: `product_strategy`

The taxonomy also distinguishes **cross-domain batchable** topics (those where research is genuinely shared across domain boundaries—`competitive_landscape`, `technical_architecture`, `knowledge_capture`, `product_strategy`, `regulatory_environment`, `clinical_literature`) from **thread-specific** topics (those where the research context is inherently bound to a single thread—`legal_proceedings`, `personal_health`, `financial_modeling`, `infrastructure_ops`, `stakeholder_outreach`).

This distinction drives Phase 2 clustering behavior and the economics model.

### 4.2 Two-Phase Clustering

**Phase 1 — Topic Grouping (hard)**

All actions are grouped by `research_topic`. This is a deterministic, zero-ambiguity operation. Actions sharing a topic are candidate batch members regardless of domain.

**Phase 2 — Embedding Refinement (soft)**

For each Phase 1 group tagged as a cross-domain batchable topic:
- Compute neural embeddings for each action (embed text = domain + action_type + thread_name + thread_description + action_description)
- Apply greedy centroid-based clustering with cosine similarity threshold $\tau$
- $\tau = 0.82$ for Voyage neural embeddings; $\tau = 0.55$ for TF-IDF fallback

For thread-specific topics, Phase 2 is skipped—all actions in the group remain clustered together (they share the same external context regardless of embedding distance).

The Phase 2 split prevents false positive batches within broad topics. For example, `product_strategy` covers both Ridgeline L6 Promotion (career strategy) and MEL GTM (healthcare product launch)—they share a topic label but not a substrate. Embedding similarity correctly separates them.

### 4.3 Embedding Backend

The system implements an `EmbeddingBackend` protocol with two implementations:

**`VoyageBackend`** (primary): Uses Voyage AI's `voyage-3-lite` model. Embeddings are cached in a SQLite `action_embeddings` table keyed by `(action_id, model_name)` with CASCADE delete on action deletion. This avoids redundant API calls on repeated runs.

**`TFIDFBackend`** (fallback): Pure numpy TF-IDF with cosine similarity. Activates automatically when `VOYAGE_API_KEY` is absent. Uses a lower similarity threshold to compensate for lower embedding quality.

### 4.4 Batch Economics Model

For a cluster $C$ of $n$ actions:

$$\text{effort\_solo} = n \cdot c_0$$

$$\text{effort\_batch} = c_0 + (n-1) \cdot c_0 \cdot (1 - \phi)$$

$$\text{coordination} = \delta_f + n \cdot \delta_h \cdot \eta + \delta_s \cdot (d - 1) \cdot \mathbb{1}[\text{cross-domain batchable}]$$

$$\text{net\_savings} = \text{effort\_solo} - \text{effort\_batch} - \text{coordination}$$

where:
- $c_0 = 1.0$ — normalized cost of a single action (one focused work unit)
- $\phi$ — substrate overlap coefficient, derived from intra-cluster cosine similarity
- $\delta_f = 0.4$ — fixed coordination overhead (assembling a batch)
- $\delta_h = 0.15$ — per-action domain heterogeneity penalty
- $\eta$ — domain heterogeneity (0 if all same domain, 1 if all different)
- $\delta_s = 0.25$ — cross-domain synthesis overhead per extra domain
- $d$ — number of distinct domains in the cluster

The synthesis overhead term $\delta_s$ captures a real cost: even when research is shared across domains, each domain requires its own decision step and application of the research. Without this term, the model would over-recommend cross-domain batches where the research fraction is small relative to the domain-specific decision work.

### 4.5 Urgency Routing

Before evaluating batch economics, actions are screened for urgent-solo routing:

An action is routed `urgent_solo` if:
- `urgency_score > 0.75` AND `thread_entropy < 0.35`

The intuition: high urgency + low entropy means the system is confident about what needs to happen, and waiting for batch assembly introduces delay with no epistemic benefit. Time-sensitive legal deadlines, imminent deployment failures, and open medical decisions all fall into this category and should never be batched.

The `recommend()` function applies this check as a priority-zero rule before evaluating net savings.

---

## 5. Results

Evaluated against 38 pending actions across 13 active threads spanning 7 domains. Threads are labeled Thread-A through Thread-M; domain labels are generalized to preserve anonymity while retaining structural properties relevant to the evaluation.

**Thread inventory:**

| Thread | Domain | Research Topic | Description |
|---|---|---|---|
| Thread-A | product-alpha | `competitive_landscape` | AI product focused on automating insurance prior authorization |
| Thread-B | product-beta | `competitive_landscape` | Patient-side navigation platform for insurance appeals |
| Thread-C | product-alpha | `infrastructure_ops` | Cloud deployment and infrastructure for product-alpha |
| Thread-D | product-alpha | `clinical_literature` | Clinical ontology quality for product-alpha's reasoning layer |
| Thread-E | legal | `legal_proceedings` | Estate administration: asset inventory and court deadline |
| Thread-F | legal | `legal_proceedings` | Estate administration: fraud investigation |
| Thread-G | legal | `legal_proceedings` | Insurance policy investigation for estate |
| Thread-H | legal | `legal_proceedings` | Medical malpractice proceedings |
| Thread-I | finance | `financial_modeling` | Insurance benefits optimization |
| Thread-J | health | `personal_health` | Musculoskeletal injury recovery decision |
| Thread-K | oss | `knowledge_capture` | Open-source release of core reasoning framework |
| Thread-L | employer | `technical_architecture` | Enterprise knowledge graph product |
| Thread-M | employer | `product_strategy` | Career advancement strategy |

### 5.1 Routing Summary

| Recommendation | Clusters | Actions | Notes |
|---|---|---|---|
| `urgent_solo` | 4 | 24 | Legal (Threads E–H), infrastructure ops (Thread-C), financial (Thread-I), personal health (Thread-J) |
| `batch` | 3 | 8 | Clinical literature (Thread-D), knowledge capture (Thread-K), technical architecture (Thread-L) |
| `individual` | 2 | 6 | Competitive landscape (Threads A+B, cross-domain penalty tips negative), product strategy (Thread-M) |

### 5.2 Key Findings

**Topic inference resolves cross-domain substrate that embeddings miss.** Thread-A (prior authorization automation, domain: product-alpha) and Thread-B (patient navigation, domain: product-beta) were correctly grouped under `competitive_landscape` by the topic inference layer. Both threads require the same competitive research—who else is building AI for insurance prior authorization—despite belonging to different product domains with different vocabularies. Prior to topic inference, embedding-only clustering kept them in separate clusters because their thread descriptions and domain context dominated the embedding signal.

**Legal proceedings consolidation.** 15 actions across 4 distinct legal threads (Threads E, F, G, H) spanning two domain contexts (estate administration and malpractice proceedings) were consolidated into a single `urgent_solo` cluster. Without topic-first grouping, these formed 4 separate clusters with fragmented rationales. The consolidated view correctly signals: all legal threads require coordinated attention under time pressure.

**Economics model correctly rejects marginal cross-domain batches.** The `competitive_landscape` cluster (Threads A and B) routes `individual` because coordination cost (0.9, including cross-domain synthesis overhead) exceeds estimated net savings (-1.0). This is correct: the research substrate is shared, but the decision contexts diverge after the research step—Thread-A's decision is about product architecture, Thread-B's is about go-to-market entity structure. The synthesis overhead term correctly prices in this divergence. As more competitive research actions accumulate on each thread (~3 per thread is the empirically estimated flip point), batch savings will exceed coordination cost and the recommendation will flip to `batch`.

**Two-phase approach outperforms embedding-only on recall.** With embedding-only clustering (prior implementation), 38 actions produced 13 clusters; with two-phase, 9 clusters—a 31% reduction. More importantly, cluster quality improved: same-topic actions that were embedding-distant due to different thread vocabulary are now correctly grouped (Threads E–H under legal proceedings), while superficially similar actions with different decision contexts are correctly separated (Thread-K knowledge capture isolated from Thread-L technical architecture, despite both living in the broader product development space).

---

## 6. Discussion

### 6.1 The Taxonomy Stability Problem

The 11-topic taxonomy is manually curated. As new threads are added, new topics may be needed, or existing topic boundaries may prove too coarse. The system handles this via three escape hatches: (a) thread name overrides that bypass the taxonomy entirely, (b) thread metadata overrides for precise per-thread control, and (c) specific (domain × hypothesis_type × action_type) rules that can override domain defaults.

In practice, the taxonomy has been stable across 13 threads spanning 7 domains. The binding constraint on taxonomy quality is not the number of topics but the precision of the `CROSS_DOMAIN_BATCHABLE` vs. `THREAD_SPECIFIC_TOPICS` classification. Misclassifying a thread-specific topic as cross-domain batchable causes spurious batch recommendations; misclassifying a cross-domain batchable topic as thread-specific misses real batch opportunities.

### 6.2 The Synthesis Overhead Calibration Problem

The cross-domain synthesis overhead ($\delta_s = 0.25$ per extra domain) is a calibrated constant. The correct value depends on how different the decision contexts are—a `competitive_landscape` batch combining two products in the same vertical (e.g., Threads A and B, both in healthcare insurance automation) has lower synthesis overhead than one combining a product-domain thread with an open-source release thread, where the research is nominally shared but the apply step is entirely different. Future work could make $\delta_s$ a function of inter-domain embedding distance, allowing the economics model to be more precise about which cross-domain batches are genuinely cheap to synthesize.

### 6.3 Relation to Prior Work

The problem of identifying shared research substrate across tasks is related to the *task similarity* literature in multi-task learning [Caruana 1997, Ruder 2017] but differs in two important ways: (a) we are scheduling *actions*, not training models, so the cost structure is different, and (b) the "shared substrate" is about knowledge acquisition cost, not gradient sharing. The closest prior work is in automated planning (STRIPS-family batch planning [Fikes & Nilsson 1971]) but that work assumes explicit precondition structures; here, substrate overlap must be inferred.

The use of belief-state entropy as a gating signal for urgency routing is inspired by active learning literature [Settles 2009] and information-theoretic planning [Itti & Baldi 2009]. The HEDGE framework's log-odds update model is equivalent to a Naive Bayes classifier operating in log space, which has well-understood convergence and calibration properties.

---

## 7. Implementation Notes

The system is implemented in Python with SQLite as the backing store. Key design decisions:

- **SQLite custom functions**: `logit_to_prob()`, `prob_to_logit()`, `belief_entropy()` registered as Python-side SQLite scalar functions, allowing them to be used in SQL queries for in-database belief state computation.
- **Embedding cache**: Stored as BLOBs in SQLite with a model name tag. Cache invalidation is explicit (action deletion cascades). This avoids the Voyage API on repeat runs.
- **Backfill migration**: `backfill_research_topics()` on the engine handles existing actions that predate topic inference. Migration 002 adds the `research_topic` column with a safe default.
- **Auto env loading**: `.env` is loaded at module import time in `db.py`, making `VOYAGE_API_KEY` available across all entry points without explicit environment variable management.

---

## 8. Conclusion

We have presented a substrate-aware action batching system that extends the HEDGE personal reasoning framework with a two-phase clustering pipeline. The key insight is that *explicit topic inference* is a prerequisite for reliable cross-domain batching—embedding similarity alone cannot recover the shared knowledge work signal when thread vocabulary diverges. The economics model disciplines the system against over-batching by pricing in cross-domain synthesis overhead.

The immediate next step is to make $\delta_s$ (synthesis overhead) a function of inter-domain embedding distance rather than a fixed constant, and to accumulate empirical calibration data on actual batch execution times to validate the normalized cost model. The longer-term question is whether the topic taxonomy can be learned from the belief graph structure itself—if two threads share a common hypothesis type cluster and their evidence graphs overlap semantically, that is a candidate for the same research topic, derived automatically.

---

## References

- Caruana, R. (1997). Multitask Learning. *Machine Learning*, 28(1), 41–75.
- Fikes, R. E., & Nilsson, N. J. (1971). STRIPS: A New Approach to the Application of Theorem Proving to Problem Solving. *Artificial Intelligence*, 2(3–4), 189–208.
- Itti, L., & Baldi, P. (2009). Bayesian Surprise Attracts Human Attention. *Vision Research*, 49(10), 1295–1306.
- Ruder, S. (2017). An Overview of Multi-Task Learning in Deep Neural Networks. arXiv:1706.05098.
- Settles, B. (2009). Active Learning Literature Survey. Computer Sciences Technical Report 1648, University of Wisconsin–Madison.

---

*Source code: https://github.com/jordandeherrera/jordan-hedge*
*Submitted to arXiv cs.AI*
