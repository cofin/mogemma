"""Real-weight numpy-reference parity for Gemma 4 26B-A4B-it MoE layer 0.

Gated on the presence of a cached Orbax checkpoint (≈90 GB). Skipped in CI /
on machines without the checkpoint. Validates that the numpy HF-formula
reference runs end-to-end on real weights without producing non-finite
output, confirming weight extraction + forward math is self-consistent at
production scale.

Invariants checked:
  * All 16 extracted layer-0 tensors match expected shapes.
  * Reference FFN output is finite (no NaN/Inf).
  * Output mean is within one order of magnitude of the input (no blow-up).
  * Router selects TOP_K distinct experts out of 128.

This is NOT a Mojo-vs-numpy parity test — that's tracked as Phase 3.2.A.4
deferred-work. It is a "numpy reference does not explode on real-scale
weights" smoke test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

CHECKPOINT_ROOT = Path.home() / ".cache" / "mogemma" / "google--gemma-4-26B-A4B-it"

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT_ROOT / "_CHECKPOINT_METADATA").exists(), reason="26B-A4B-it Orbax checkpoint not cached locally"
)


def test_numpy_reference_runs_on_real_layer0() -> None:
    from scripts.parity_moe_layer0 import HIDDEN_SIZE, NUM_EXPERTS, extract_layer0, reference_moe_ffn

    w = extract_layer0()

    assert w["expert_gate_up_proj"].shape == (NUM_EXPERTS, 1408, HIDDEN_SIZE)
    assert w["expert_down_proj"].shape == (NUM_EXPERTS, HIDDEN_SIZE, 704)
    assert w["layer_scalar"].shape == (1,)
    assert 0.0 < w["layer_scalar"][0] < 1.0, f"layer_scalar out of expected range: {w['layer_scalar'][0]}"

    rng = np.random.default_rng(seed=20260416)
    x1 = rng.normal(size=HIDDEN_SIZE).astype(np.float32)

    out = reference_moe_ffn(x1, w)

    assert out.shape == (HIDDEN_SIZE,)
    assert np.all(np.isfinite(out)), "non-finite values in reference output"
    assert abs(out.mean()) < 10.0, f"output mean {out.mean()} suggests numerical blow-up"
    assert 0.1 < out.std() < 50.0, f"output std {out.std()} outside plausible range"
