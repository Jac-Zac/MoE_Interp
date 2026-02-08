= Introduction

Large-scale language models have achieved remarkable performance on a wide spectrum of tasks, from open-ended text generation @brown2020languagemodelsfewshotlearners to question answering and code synthesis. 
As these models scale
Currently a growing number of modern LLms have adopted a Mixture-of-Experts (MoE) architecture @shazeer2017outrageously @fedus2022switch, in which only a subset of the model's parameters is activated for each input token. 
This sparse activation strategy allows MoE models to store significantly more factual knowledge in their parameters while keeping inference cost relatively low, since only a small fraction of experts is selected at each layer. 
Notable examples include Mixtral @jiang2024mixtral, which routes each token to its top-2 feed-forward experts while keeping attention dense; OLMoE @muennighoff2024olmoe, a fully open-source MoE model; and DeepSeekMoE @dai2024deepseekmoe, which introduces fine-grained sub-experts and shared always-on experts to reduce redundancy.

Despite the success of MoE architectures, the internal mechanisms by which experts organize and represent knowledge remain incompletely understood. 
Recent work has begun to investigate expert specialization from several complementary perspectives. Wang et al. @wang2025whatgets establish a functional hierarchy among experts using entropy-based specialization metrics and causal mediation analysis, distinguishing between _domain experts_ that specialize in specific semantic areas and _driver experts_ that exert disproportionate causal influence on next-token probabilities. 
Jin et al. @jin2025probing demonstrate that MoE routers make genuinely semantic, rather than purely load-balancing, decisions, with routing patterns correlating strongly with input token categories. 
Lecomte et al. @lecomte2025sparsity show that experts often encode multiple unrelated features through superposition, and that stronger sparsity constraints reduce this overlap and improve interpretability. 
At a coarser level, Gao et al. @gao2025beyond propose the Model Utilization Index (MUI), finding that stronger generalization often corresponds to _lower_ internal utilization, suggesting efficient feature compression within expert representations.

However, these techniques typically focus on _which_ experts are selected by the router analyzing routing frequencies, entropy, and activation patterns rather than on _what_ individual experts actually compute. 
This leaves a gap between understanding expert selection and understanding expert function.

In this work, we address this gap by adapting the Head Pursuit framework @basile2025headpursuit to analyze the internal computations of MoE experts. 
Head Pursuit, recently introduced for attention head interpretability in both language and vision-language transformers, applies Simultaneous Orthogonal Matching Pursuit (SOMP) @tropp2006algorithms a classical sparse signal recovery algorithm to decompose attention head outputs using the model's unembedding matrix as a dictionary of interpretable directions. 
This yields a sparse set of vocabulary tokens that characterize what each head writes into the residual stream, and enables ranking heads by their relevance to target semantic concepts. 
We propose _Expert Pursuit_, which extends this approach from attention heads to MoE feed-forward experts. 
By capturing each expert's gated output (the FFN output weighted by the router gate) and applying SOMP, we obtain a principled, semantically grounded characterization of each expert's function, moving beyond routing metadata to directly analyze the information that experts contribute to the model's computation.

A key structural difference between the attention head setting and the MoE expert setting is that attention heads process all tokens in a sequence, whereas each expert only processes the tokens routed to it by the gating mechanism. 
We aggregating gated outputs only over routed tokens, and we note that this routing-induced selection bias may actually produce tighter semantic clusters than in the dense head setting, since the router has already pre-selected tokens that match the expert's learned specialization.

Our contributions are as follows:
- We extend the Head Pursuit framework from attention heads to MoE experts, proposing _Expert Pursuit_ as a method to characterize the semantic function of individual experts using sparse decomposition over the unembedding matrix.
- We formalize how SOMP applies to expert gated outputs, accounting for the sparse routing structure inherent in MoE architectures.
- We propose an expert ranking and intervention protocol that enables targeted suppression or enhancement of semantic concepts by rescaling the contributions of concept-specific experts.
