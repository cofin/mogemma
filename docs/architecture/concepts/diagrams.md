# Architecture Diagrams

The following Mermaid diagrams illustrate the flow of data through the mogemma inference engine.

## Standard Gemma 3 Forward Pass

This diagram shows the lifecycle of a single token as it passes through the standard Gemma 3 model.

```mermaid
flowchart TD
    Token[Token ID] --> Embed[embed_tokens]
    Embed --> Hidden[Hidden State Vector]
    
    subgraph Layer 1
        Hidden --> Norm1[RMSNorm]
        Norm1 --> Attn[Self-Attention]
        Attn -- Query, Key, Value --> KV[KV Cache]
        Attn --> Add1[+]
        Add1 --> Norm2[RMSNorm]
        Norm2 --> MLP[Feed Forward Network]
        MLP --> Add2[+]
    end
    
    Hidden -- Residual --> Add1
    Add1 -- Residual --> Add2
    
    Add2 --> LayerN[... N Layers ...]
    LayerN --> FinalNorm[Final RMSNorm]
    FinalNorm --> LMHead[lm_head Projection]
    LMHead --> Logits[Vocabulary Logits]
```

## Nano Architecture (AltUp Data Flow)

This diagram highlights the complex, multi-stream data flow inside a single Nano layer.

```mermaid
flowchart TD
    InputStreams[Input Streams: 4 Modalties]
    
    subgraph Nano Layer
        InputStreams --> Predict[AltUp Router Prediction]
        Predict --> Route[Select Active Stream]
        
        Route -- Active Stream --> Laurel[Laurel Down/Up Projection]
        Laurel --> Norm1[RMSNorm]
        Norm1 --> Attn[Nano Self-Attention]
        Attn --> MLP[Feed Forward Network]
        
        Route -- Inactive Streams --> Wait[Bypass Math Ops]
        
        MLP --> Correct[AltUp Correction]
        Wait --> Correct
        
        Correct --> PerLayer[Per-Layer Mapping / Token Re-injection]
    end
    
    PerLayer --> OutputStreams[Output Streams: 4 Modalties]
```

## KV Cache Sharing Boundary (Nano)

This diagram visualizes how the Nano architecture shares KV cache to save memory.

```mermaid
flowchart LR
    subgraph Early Layers
        L0[Layer 0] --> KV0[(KV Slot 0)]
        L1[Layer 1] --> KV1[(KV Slot 1)]
    end
    
    Boundary((kv_share_start))
    
    subgraph Shared Layers
        L2[Layer 2] --> ReadKV1[Reads KV Slot 1]
        L3[Layer 3] --> ReadKV1
        L4[Layer 4] --> ReadKV1
    end
    
    L1 --> Boundary
    Boundary --> L2
```