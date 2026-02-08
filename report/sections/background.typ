= Background and Theory

== Transformer Architecture and Residual Stream

A transformer-based language model processes a sequence of tokens by passing them through $L$ successive layers, each composed of a multi-head self-attention sublayer and a feed-forward sublayer. 
Following the framework of Elhage et al. @elhage2021mathematical, the model's computation can be understood through the _residual stream_: a $d$-dimensional vector associated with each token position that accumulates additive contributions from every sublayer. 
Concretely, for a token at position $i$, the residual stream after layer $l$ is:

$ bold(x)_i^l = bold(x)_i^(l-1) + "Attn"^l (bold(x)_i^(l-1)) + "FFN"^l (bold(x)_i^(l-1)) $

At the final layer, the residual stream is projected onto the _unembedding matrix_ $bold(D) in RR^(v times d)$ (where $v$ is the vocabulary size) to produce logits over the token vocabulary. 
Each row of $bold(D)$ is a $d$-dimensional vector that represents a vocabulary token in the model's latent space.

== Mixture-of-Experts Layers

In MoE transformer architectures @shazeer2017outrageously @fedus2022switch, the dense feed-forward sublayer is replaced by a _Mixture-of-Experts_ layer consisting of $E$ parallel expert networks ${f_1, dots, f_E}$, each an independent feed-forward network (FFN), together with a learned _router_ (or _gating function_) $G$ that selects which experts process each token.

For a given input token representation $bold(x) in RR^d$, the router computes a score for each expert and selects the top-$k$ experts with the highest scores. 
The output of the MoE layer is the weighted sum of the selected expert outputs:

$ "MoE"(bold(x)) = sum_(e in cal(T)_k (bold(x))) g_e (bold(x)) dot f_e (bold(x)) $ <eq:moe>

where $cal(T)_k (bold(x))$ denotes the set of top-$k$ experts selected by the router for input $bold(x)$, $g_e (bold(x))$ is the gating weight assigned to expert $e$, and $f_e (bold(x))$ is the FFN output of expert $e$. 
Experts not in $cal(T)_k (bold(x))$ contribute exactly zero to the residual stream for that token.

Different MoE architectures vary in the number of total experts $E$, the number of active experts $k$, and the routing mechanism. 
For example, Mixtral @jiang2024mixtral uses $E = 8$ experts per layer with $k = 2$; OLMoE @muennighoff2024olmoe uses $E = 64$ experts with $k = 8$; and DeepSeekMoE @dai2024deepseekmoe introduces fine-grained sub-experts alongside shared always-on experts.

== Residual Stream Decomposition for Experts

In Head Pursuit @basile2025headpursuit, the contribution of each attention head is isolated by decomposing the residual stream at head-level granularity. 
Specifically, the output written by attention head $h$ at layer $l$ into the residual stream is modeled as a matrix $bold(H)_(h,l) in RR^(n times d)$, following Elhage et al. @elhage2021mathematical, where $n$ is the number of samples and $d$ is the model dimension.

We propose an analogous decomposition for MoE experts. For each expert $e$ at layer $l$, we define the _expert activation matrix_ $bold(E)_(e,l) in RR^(n_e times d)$, where $n_e$ is the number of tokens routed to expert $e$ across the dataset. 
Each row of $bold(E)_(e,l)$ is the _gated output_ of expert $e$ for a token it processed:

$ bold(E)_(e,l) [i] = g_e (bold(x)_i) dot f_e (bold(x)_i) quad "for" bold(x)_i in {bold(x) : e in cal(T)_k (bold(x))} $ <eq:expert-activation>

This is the quantity that actually gets added to the residual stream (see @eq:moe), making it the natural unit of analysis for understanding what an expert contributes to the model's computation. 
Only tokens for which expert $e$ was selected by the router appear as rows in $bold(E)_(e,l)$; all other tokens have zero contribution from this expert.

To obtain a per-document (or per-sample) representation suitable for SOMP, we aggregate the gated outputs by averaging over the routed tokens within each document. 
Given a document $j$ with tokens ${bold(x)_1, dots, bold(x)_(T_j)}$, let $cal(R)_(e,l)^j = {i : e in cal(T)_k (bold(x)_i)}$ be the set of token positions routed to expert $e$ at layer $l$. 
The aggregated expert representation for document $j$ is:

$ macron(bold(E))_(e,l)^j = frac(1, |cal(R)_(e,l)^j|) sum_(i in cal(R)_(e,l)^j) g_e (bold(x)_i) dot f_e (bold(x)_i) $ <eq:expert-agg>

Stacking these across $n$ documents yields the aggregated expert matrix $macron(bold(E))_(e,l) in RR^(n times d)$ used as input to SOMP.

== Sparse Coding via Simultaneous Orthogonal Matching Pursuit

To identify interpretable directions that characterize each expert's function, we apply Simultaneous Orthogonal Matching Pursuit (SOMP) @tropp2006algorithms, a classical sparse coding algorithm. SOMP is a multi-sample extension of Orthogonal Matching Pursuit (OMP) @pati1993orthogonal, itself a refinement of the original Matching Pursuit algorithm @mallat1993matching. 
Rather than analyzing each sample independently, SOMP jointly considers all samples and selects the dictionary directions that are most informative across the entire representation.

As a dictionary, we adopt the unembedding matrix $bold(D) in RR^(v times d)$ of the language model, since its rows are naturally aligned with semantically meaningful outputs each row is a $d$-dimensional vector representing a vocabulary token in the model's latent space.

Given an aggregated expert activation matrix $macron(bold(E)) in RR^(n times d)$ and dictionary $bold(D) in RR^(v times d)$, SOMP iteratively constructs a column-sparse coefficient matrix $bold(W)^* in RR^(n times v)$ such that:

$ macron(bold(E)) approx bold(W)^* bold(D) $ <eq:somp-approx>

At each iteration $t$, the algorithm selects the dictionary atom (row of $bold(D)$) that maximally correlates with the current residuals across all samples:

$ p^t = arg max_j norm(bold(D)[j] bold(R)^(t T))_1 $ <eq:somp-select>

where the residual matrix $bold(R)^t in RR^(n times d)$ is the difference between the original signal and its current reconstruction: $bold(R)^t = macron(bold(E)) - macron(bold(E))_r^t$. 
The selected index $p^t$ is added to the support set $SS^(t+1)$, and the coefficients are refit by solving a least-squares problem restricted to the current support:

$ bold(W)^t = arg min_(bold(W)) norm(macron(bold(E)) - bold(W) bold(D)[SS^(t+1)])_F $ <eq:somp-refit>

The reconstruction is updated as $macron(bold(E))_r^(t+1) = bold(W)^t bold(D)[SS^(t+1)]$, and the residuals are recomputed. 
This process continues until a predefined sparsity level (number of iterations) is reached. The resulting decomposition expresses each expert's aggregated output using a sparse set of semantically meaningful vocabulary tokens, yielding an interpretable approximation of its behavior.

== From Head Pursuit to Expert Pursuit

While the mathematical framework of SOMP applies identically to both attention head outputs and expert FFN outputs, several structural differences between the two settings are worth noting, as they shape both the methodology and the interpretation of results:

+ *Routing and coverage.* Every attention head processes all tokens in the input sequence. In contrast, each MoE expert only processes the tokens routed to it by the gating mechanism. This means the expert activation matrix is naturally sparse, with most documents contributing only a subset of their tokens to any given expert.

+ *Selection bias.* The tokens an expert receives are not a random sample they are precisely the tokens that the router determined are most suited to that expert's learned function. This routing-induced selection bias means that the aggregated expert representation is inherently more semantically focused than the corresponding attention head representation, and we may expect SOMP to find tighter semantic clusters.

+ *Gating weights.* Expert outputs are scaled by the router's gating weights before entering the residual stream. This natural weighting, absent in the attention head setting, reflects the router's confidence in the expert's relevance to each token. By analyzing gated outputs directly, we capture the expert's actual contribution rather than its raw computation.

These differences motivate treating Expert Pursuit not as a trivial extension of Head Pursuit, but as an adaptation that accounts for the unique structure of sparse expert routing in MoE architectures.
