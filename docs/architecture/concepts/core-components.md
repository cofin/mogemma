# Core Components: A Conceptual Guide

This guide introduces the core building blocks of the mogemma inference engine. It is designed for system engineers and developers who are new to Large Language Models (LLMs) and want to understand how the underlying math and data structures map to the systems they build.

## Token Projection & Embeddings

Before a model can process text, the text is broken down into discrete chunks called **tokens**. 
In mogemma, the vocabulary consists of thousands of distinct tokens, each represented by a unique integer ID.

When a token ID enters the model, it must be converted into a format the math operations can understand: a continuous, high-dimensional vector. This process is called **token projection** or **embedding**. 

Conceptually, this is a simple table lookup. The `embed_tokens` matrix contains a vector for every possible token in the vocabulary. The model selects the vector corresponding to the input ID and passes it into the first transformer layer. This vector represents the "meaning" of the token in a mathematical space.

## RMSNorm (Root Mean Square Normalization)

As data (the token vector) passes through the many layers of the model, its numerical values can grow uncontrollably large or shrink to zero. This instability makes the model's predictions unreliable.

**RMSNorm** is a normalization technique applied throughout the model (e.g., before attention blocks, before the feed-forward networks, and at the very end). 

It works by:
1. Calculating the Root Mean Square of all values in the vector.
2. Dividing each value by this RMS, effectively standardizing the scale of the vector's values.
3. Multiplying the normalized vector by a learned scaling weight.

By keeping the vectors constrained to a stable numerical range, RMSNorm ensures that the model's math operations remain mathematically sound across all layers.

## The KV Cache (Key-Value Cache)

Generating text is an **autoregressive** process—the model predicts the next token based on all previous tokens, and then repeats the process using the newly generated token. 

Naively recalculating the attention scores for the entire sequence at every step is computationally expensive. The **KV Cache** is the fundamental optimization that solves this.

During the attention mechanism:
- The **Query (Q)** represents what the current token is "looking for".
- The **Key (K)** represents what a token "contains".
- The **Value (V)** is the actual data the token provides if matched.

Because past tokens do not change, their Key and Value tensors remain static. The KV Cache stores these K and V tensors in memory for every token processed so far. 
When a new token is evaluated, the model only needs to compute the new token's Q, K, and V, and then compare its Q against the *cached* K's of all previous tokens. This reduces the complexity of each generation step from $O(N^2)$ to $O(N)$, drastically speeding up inference at the cost of increased memory usage.