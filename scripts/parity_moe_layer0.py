"""Extract Gemma 4 26B-A4B-it layer-0 MoE weights + compute numpy-reference FFN output.

Builds the parity fixture consumed by `src/mo/tests/test_moe_forward_real_weights.mojo`.
The numpy reference implements the exact HF `Gemma4TextDecoderLayer` FFN forward
(documented in `.agents/specs/moe-mojo-runtime/spec.md` Phase 0.1c).

Output: `src/mo/tests/data/moe_layer0_26b.npz` with
  - packed layer-0 weights (down-cast to F32 via OrbaxLoader.open_tensor)
  - a fixed-seed input residual tensor
  - the numpy-reference FFN-only output (residual-add + layer_scalar applied)

The Mojo test loads this fixture, runs forward_moe_layer with attention zeroed
(so x1 == input residual), and compares the Mojo output to the numpy reference.

Usage:
    uv run python scripts/parity_moe_layer0.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from mogemma.orbax_loader import OrbaxLoader

# Gemma 4 26B-A4B-it architecture constants (from .agents/knowledge/gemma4-models.md)
HIDDEN_SIZE = 2816
NUM_EXPERTS = 128
TOP_K = 8
I_MOE = 704
I_DENSE = 11264
NUM_HEADS = 22
NUM_KV_HEADS = 4
HEAD_DIM = 128

CHECKPOINT_ROOT = Path.home() / ".cache" / "mogemma" / "google--gemma-4-26B-A4B-it"
FIXTURE_OUT = Path("src/mo/tests/data/moe_layer0_26b.npz")


def _open(name: str) -> np.ndarray:
    """Load one Orbax tensor, cast BF16→F32, return a contiguous copy."""
    arr = OrbaxLoader.open_tensor(CHECKPOINT_ROOT, name)
    return np.ascontiguousarray(arr.astype(np.float32))


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Gemma-style RMSNorm: `y = x * rsqrt(mean(x²) + eps) * (1 + weight)`."""
    rms = np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)
    return np.asarray((x / rms) * (1.0 + weight))


def rms_noscale(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    rms = np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)
    return np.asarray(x / rms)


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    """Tanh-approximated GELU (Gemma convention)."""
    return np.asarray(0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3))))


def reference_moe_ffn(x1: np.ndarray, w: dict[str, np.ndarray]) -> np.ndarray:
    """Pure-numpy MoE FFN block matching HF Gemma4TextDecoderLayer.

    Args:
        x1: Pre-FFN residual, shape `(H,)` (post-attention).
        w: Layer-0 weights dict.

    Returns:
        Post-FFN output of shape `(H,)`, including residual add and layer_scalar.
    """
    # Dense branch — pre-norm operates on the residual.
    dense_in = rms_norm(x1, w["pre_feedforward_layernorm"])
    gate = w["dense_gate_proj"] @ dense_in  # (I_DENSE,)
    up = w["dense_up_proj"] @ dense_in  # (I_DENSE,)
    dense_hidden = gelu_tanh(gate) * up
    h1_raw = w["dense_down_proj"] @ dense_hidden  # (H,)
    h1 = rms_norm(h1_raw, w["post_feedforward_layernorm_1"])

    # MoE branch — pre-norm operates on the RAW flattened residual.
    # Router input is also the raw residual (router does internal rmsnorm_noscale).
    r = rms_noscale(x1) * w["router_scale"] / np.sqrt(HIDDEN_SIZE)
    logits = w["router_proj"] @ r  # (E,)
    probs = _softmax(logits)
    top_k_idx = np.argpartition(-probs, TOP_K)[:TOP_K]
    top_k_w = probs[top_k_idx]
    top_k_w = top_k_w / top_k_w.sum()
    top_k_w = top_k_w * w["per_expert_scale"][top_k_idx]

    moe_in = rms_norm(x1, w["pre_feedforward_layernorm_2"])
    h2_raw = np.zeros(HIDDEN_SIZE, dtype=np.float32)
    for tw_p, ti_p in zip(top_k_w, top_k_idx, strict=True):
        gate_up = w["expert_gate_up_proj"][ti_p] @ moe_in  # (2*I_MOE,)
        gate_e, up_e = gate_up[:I_MOE], gate_up[I_MOE:]
        expert_out = w["expert_down_proj"][ti_p] @ (gelu_tanh(gate_e) * up_e)  # (H,)
        h2_raw = h2_raw + tw_p * expert_out
    h2 = rms_norm(h2_raw, w["post_feedforward_layernorm_2"])

    # Combine + H-A post-norm + residual add + layer_scalar (Orbax skip_scale).
    combined = rms_norm(h1 + h2, w["post_feedforward_layernorm"])
    out = (x1 + combined) * w["layer_scalar"][0]
    return np.asarray(out, dtype=np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    m = x.max()
    e = np.exp(x - m)
    return np.asarray(e / e.sum())


def extract_layer0() -> dict[str, np.ndarray]:
    """Load the 20 Orbax tensors that define MoE layer 0."""
    pfx = "layer_0"
    return {
        # Attention (unused in FFN-only parity; still captured for completeness)
        "input_layernorm": _open(f"{pfx}.pre_attention_norm.scale"),
        "post_attention_layernorm": _open(f"{pfx}.post_attention_norm.scale"),
        # Dense branch
        "pre_feedforward_layernorm": _open(f"{pfx}.pre_ffw_norm.scale"),
        "dense_gate_proj": _open(f"{pfx}.mlp2.gating_einsum.w")[0],  # [2, I_d, H] -> [I_d, H]
        "dense_up_proj": _open(f"{pfx}.mlp2.gating_einsum.w")[1],
        "dense_down_proj": _open(f"{pfx}.mlp2.linear.w").T,  # [I_d, H] -> [H, I_d]
        "post_feedforward_layernorm_1": _open(f"{pfx}.post_ffw1_norm.scale"),
        # MoE branch
        "pre_feedforward_layernorm_2": _open(f"{pfx}.pre_ffw2_norm.scale"),
        "router_proj": _open(f"{pfx}.mlp.router_logits.w").T,  # [H, E] -> [E, H]
        "router_scale": _open(f"{pfx}.mlp.router_scale"),
        "per_expert_scale": _open(f"{pfx}.mlp.per_expert_scale"),
        "expert_gate_up_proj": _open(f"{pfx}.mlp.gating_einsum.w").reshape(NUM_EXPERTS, 2 * I_MOE, HIDDEN_SIZE),
        "expert_down_proj": _open(f"{pfx}.mlp.linear.w").transpose(0, 2, 1),  # [E, I, H] -> [E, H, I]
        "post_feedforward_layernorm_2": _open(f"{pfx}.post_ffw2_norm.scale"),
        # Combined-branch norm (H-A placement)
        "post_feedforward_layernorm": _open(f"{pfx}.post_ffw_norm.scale"),
        # layer_scalar (Orbax skip_scale)
        "layer_scalar": _open(f"{pfx}.skip_scale"),
    }


def main() -> int:
    print(f"Loading layer-0 weights from {CHECKPOINT_ROOT}", file=sys.stderr)
    w = extract_layer0()
    for k, v in w.items():
        print(f"  {k:35s} {v.shape} {v.dtype}", file=sys.stderr)

    rng = np.random.default_rng(seed=20260416)
    x1 = rng.normal(size=HIDDEN_SIZE).astype(np.float32)

    print("Running numpy reference forward...", file=sys.stderr)
    expected = reference_moe_ffn(x1, w)
    print(f"  output: shape={expected.shape} mean={expected.mean():+.6f} std={expected.std():.6f}", file=sys.stderr)
    print(f"          min={expected.min():+.6f} max={expected.max():+.6f}", file=sys.stderr)
    if not np.all(np.isfinite(expected)):
        print("ERROR: non-finite values in reference output", file=sys.stderr)
        return 1

    FIXTURE_OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURE_OUT, x1=x1, expected=expected, **w)  # type: ignore[arg-type]
    size_mb = FIXTURE_OUT.stat().st_size / (1024 * 1024)
    print(f"Wrote {FIXTURE_OUT} ({size_mb:.1f} MiB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
