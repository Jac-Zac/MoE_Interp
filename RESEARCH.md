## Interpretability of Mixture-of-Experts (MoE)

### Models

#### [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066)

Introduces fine-grained experts by splitting standard MoE experts into smaller sub-experts, increasing specialization without increasing per-token compute. Includes always-on shared experts to capture common knowledge and reduce redundancy.

---

#### [Mixtral of Experts](https://arxiv.org/abs/2401.04088)

High-performance sparse MoE built on the Mistral architecture, using top-2 routing over feed-forward experts while keeping attention dense.

---

#### [Open Mixture-of-Experts Language Models (OLMoE)](https://arxiv.org/abs/2409.02060)

Fully open-source MoE models with extensive analysis of routing behavior during training, showing strong expert specialization and stable routing patterns.

---

### MoE Mechanistic Interpretability

#### [What Gets Activated: Uncovering Domain and Driver Experts in MoE Language Models](https://arxiv.org/abs/2601.10159)

Establishes a functional hierarchy among experts using entropy-based specialization metrics and causal mediation analysis. The paper distinguishes between:

- **Domain experts**, which specialize in specific semantic or structural domains (e.g., math, code, legal text) and exhibit low routing entropy.
- **Driver experts**, which exert disproportionately large causal influence on next-token probabilities.

**Key findings:**

- Domain specialization and causal influence are distinct properties.
- Driver experts are more frequently activated by earlier tokens in a sequence.
- Adjusting the relative contribution of domain and driver experts yields significant performance improvements.

---

#### [Beyond Benchmarks: Understanding Mixture-of-Experts Models through Internal Mechanisms](https://arxiv.org/abs/2509.23933)

Uses the **Model Utilization Index (MUI)**, measuring the fraction of internal capacity (neurons or experts) used to solve a task.
Across multiple public MoE families, as models improve MUI often _decreases_, suggesting that stronger generalization corresponds to feature compression and more efficient expert utilization.
Moreover, task performance typically emerges from _collaborative expert activity_, not single-expert dominance.

---

#### [Sparsity and Superposition in Mixture of Experts](https://arxiv.org/abs/2510.23671)

Shows that experts often encode multiple unrelated features (superposition).
Proposes metrics for cross-expert feature overlap and demonstrates that stronger sparsity and routing constraints reduce superposition and improve interpretability.

> Expert Specialization

#### [Probing Semantic Routing in Large Mixture-of-Expert Models](https://arxiv.org/abs/2502.10928)

Demonstrates that MoE routers make semantic, not purely load-balancing, decisions.
Routing patterns correlate strongly with semantic categories of input tokens.

#### [Multilingual Routing in Mixture-of-Experts](https://arxiv.org/abs/2510.04694)

Finds a U-shaped specialization pattern across layers: early and late layers are language-specific, while middle layers show cross-lingual alignment.
Middle-layer alignment correlates with multilingual performance.

---

### Model Editing

#### [MoEEdit: Efficient and Routing-Stable Knowledge Editing for Mixture-of-Experts LLMs](https://openreview.net/forum?id=BV4oHxGBx7)

Enables targeted editing of individual experts while preserving routing decisions, avoiding routing shifts and downstream performance degradation.

---

### Additional Related Work

#### [Understanding Routing Mechanism in Mixture-of-Experts Language Models](https://openreview.net/forum?id=BqyPLOkxFY)

Analyzes router convergence, expert collapse, and the role of initialization and auxiliary losses in shaping expert diversity.

---

#### [Mixture of Experts Made Intrinsically Interpretable](https://arxiv.org/abs/2503.07639)

Proposes architectural constraints that enforce monosemantic experts by design, embedding interpretability directly into the MoE structure.

---

### Notes

1. Specialization operates at multiple levels: routing, representation (superposition), and causal influence.
2. Lower utilization and stronger specialization often correlate with better generalization and interpretability.
