# metal-inference

![Swift](https://img.shields.io/badge/Swift-5.9+-F05138?style=flat&logo=swift&logoColor=white)
![Metal](https://img.shields.io/badge/Apple%20Metal-GPU-A2AAAD?style=flat&logo=apple&logoColor=white)
![macOS](https://img.shields.io/badge/macOS-13%2B-000000?style=flat&logo=apple&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat&logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Apple%20Silicon-black?style=flat&logo=apple)
![License](https://img.shields.io/badge/License-BSL--1.1-blue?style=flat)

INT4 quantized transformer inference engine for Apple Silicon. Metal compute shaders run QKV projection, grouped-query attention, SwiGLU MLP, RoPE, and LoRA delta directly on the GPU. An entropy-based routing gate decides local vs. remote execution per request. The M5 Deterministic Memory Gateway provides a lock-controlled OS-layer buffer for deterministic memory access.

---

## Architecture

```mermaid
flowchart TD
    subgraph INPUT["Input Layer"]
        T[Token IDs] --> EMB[Embedding Lookup\nMetal kernel]
    end

    subgraph GATE["Entropy Gate · Python"]
        LG[Logits] --> ENT[logit_entropy]
        ENT -->|h ≤ 1.8| LOCAL[Route: LOCAL]
        ENT -->|h > 1.8| REMOTE[Route: REMOTE]
        SIZE[Prompt bytes > 16 KB] --> REMOTE
    end

    subgraph TRANSFORMER["Transformer Stack · Metal GPU"]
        EMB --> NORM1[RMSNorm]
        NORM1 --> QKV[INT4 QKV Projection\ngroup_size=32]
        QKV --> ROPE[RoPE Positional Encoding]
        ROPE --> GQA[Grouped-Query Attention\nGQA strided · KV Cache]
        GQA --> RES1[Residual Add]
        RES1 --> NORM2[RMSNorm]
        NORM2 --> MLP[SwiGLU MLP\nINT4 gate · up · down]
        MLP --> RES2[Residual Add]
        RES2 -->|next layer| NORM1
    end

    subgraph OUTPUT["Output"]
        RES2 --> FNORM[Final RMSNorm]
        FNORM --> PROJ[INT4 Output Projection]
        PROJ --> GREEDY[Greedy Token\nargmax Float16]
    end

    subgraph LORA["LoRA Adapters · optional"]
        LORA_D[LoRA Delta Kernel] -.->|add delta| QKV
        LORA_D -.->|add delta| PROJ
    end

    subgraph M5["M5 Memory Gateway · Swift"]
        CMD[Command Parser] --> LOCK[Lock Controller]
        LOCK --> BUF[4 KB Deterministic\nMemory Buffer]
        BUF --> WAT[m5_gateway.wat\nWASM bridge]
    end

    LOCAL --> TRANSFORMER
    GATE --> INPUT
```

---

## Components

### `Sources/Kernels/` — Metal Compute Shaders

| Shader | Purpose |
|---|---|
| `int4_transformer.metal` | Full INT4 transformer pass — QKV, attention, MLP |
| `fused_qkv_tiled.metal` | Tiled fused QKV projection |
| `gqa_strided.metal` | Grouped-query attention with strided KV heads |
| `kv_cache.metal` | KV cache read/write |
| `attention.metal` | Attention score computation |
| `rope.metal` | Rotary positional encoding |
| `embedding.metal` | Token embedding lookup |
| `lora_delta.metal` | LoRA weight delta injection |
| `normalization.metal` | RMSNorm |
| `matmul.metal` | General matrix multiply |
| `linear.metal` | Linear projection |
| `mlp.metal` | MLP forward pass |
| `residual.metal` | Residual connection |
| `decode.metal` | Decode-step utilities |
| `output.metal` | Output projection |
| `swiglu` _(in int4)_ | SwiGLU activation |

### `Sources/Host/` — Swift + ObjC++ Runtime

| File | Purpose |
|---|---|
| `MetalTransformer.swift` | Pipeline manager — builds all 13 compute states, dispatches kernels |
| `int4_host_runtime.mm` | INT4 weight packing, dequantization host side |
| `block_layout.h` | INT4 block layout — group_size=32, scale per group |
| `adapter_slot.h` | LoRA adapter slot management |
| `safety.h` | Safety bounds checking |

**Model defaults:** `hidden=3072` · `intermediate=8192` · `heads=24` · `kv_heads=8` · `head_dim=128` · `vocab=32000` · `context=4096`

### `Sources/Gate/` — Entropy Routing Gate (Python)

```mermaid
flowchart LR
    P[Prompt] --> SIZE{bytes > 16 KB?}
    SIZE -->|yes| R[REMOTE]
    SIZE -->|no| H[compute entropy\nH = -Σ p·log p]
    H --> THRESH{H ≤ 1.8 nats?}
    THRESH -->|yes| L[LOCAL · Metal]
    THRESH -->|no| R
```

- `entropy.py` — numerically stable softmax entropy with temperature
- `routing.py` — `decide(logits, policy, envelope) → Route`
- `session.py` — session state
- `directive_guard.py` — directive safety guard
- `crypto.py` — Ed25519 request signing

### `Sources/M5Gateway/` — Deterministic Memory Gateway

```mermaid
flowchart TD
    CMD[Swift Command] --> PARSE[Command Parser]
    PARSE --> LOCK{Memory Locked?}
    LOCK -->|yes| ERR[MEMORY_LOCKED error]
    LOCK -->|no| OP{Operation}
    OP -->|READ| READ[Read word from\n4096-byte buffer]
    OP -->|WRITE| WRITE[Write word to\nbuffer at addr]
    OP -->|LOCK| SETLOCK[Lock buffer]
    OP -->|DUMP| DUMP[Hex dump\nall 512 words]
    READ --> WAT[m5_gateway.wat\nWASM bridge]
    WRITE --> WAT
```

4 KB deterministic buffer · 8-byte word size · 512 addressable words · WASM bridge via `m5_gateway.wat`

---

## Build

Requires macOS 13+ with Apple Silicon and Xcode 15+.

```bash
git clone https://github.com/SNAPKITTYWEST/metal-inference.git
cd metal-inference
swift build
swift test
```

Python gate (optional):

```bash
pip install -e "Sources/Gate"
```

---

## INT4 Quantization Format

Weights are packed as nibbles — two INT4 values per byte. Each group of 32 elements shares a `half2` scale factor `(scale, zero_point)`. The host runtime in `int4_host_runtime.mm` handles packing; Metal dequantizes on-the-fly during projection.

```
[ b0_lo:4 | b0_hi:4 ] [ b1_lo:4 | b1_hi:4 ] ... (16 bytes per group of 32)
  └─ scale: half2 per group ─────────────────────┘
```
