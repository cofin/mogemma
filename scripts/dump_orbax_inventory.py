"""Dump Orbax tensor inventory for a Gemma 4 variant.

Ephemeral — delete after Phase 1 of `orbax-safetensors-conversion` closes.
Writes `<name>\\t<shape>\\t<dtype>` lines to stdout so we can freeze tensor
shapes for PLE/MoE transforms before writing conversion code.

Usage:
    uv run python scripts/dump_orbax_inventory.py google/gemma-4-e2b-it \\
        > .agents/specs/orbax-safetensors-conversion/e2b-inventory.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

from mogemma.hub import HubManager
from mogemma.orbax_loader import OrbaxLoader


def main(model_id: str) -> int:
    hub = HubManager()
    local_dir = hub.download_sync(model_id)
    print(f"# model_id={model_id}", file=sys.stderr)
    print(f"# local_dir={local_dir}", file=sys.stderr)

    files_in_dir = sorted(p.name for p in Path(local_dir).iterdir())
    print(f"# top_level={files_in_dir}", file=sys.stderr)

    stub = OrbaxLoader.__new__(OrbaxLoader)
    stub.model_path = Path(local_dir)
    names = stub._enumerate_tensor_names()
    print(f"# tensor_count={len(names)}", file=sys.stderr)

    for name in names:
        arr = stub._open_tensor(name)
        print(f"{name}\t{tuple(arr.shape)}\t{arr.dtype}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "google/gemma-4-e2b-it"))
