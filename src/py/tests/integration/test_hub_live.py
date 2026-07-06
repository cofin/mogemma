"""Live-network smoke: every :data:`KNOWN_GCS_MODELS` entry has objects in GCS.

Not wired to CI. Run manually or on a cron-triggered workflow to catch
upstream changes (bucket renames, access changes, model deprecation).

Usage:

.. code-block:: shell

    MOGEMMA_LIVE_GCS=1 uv run pytest src/py/tests/test_hub_live.py -v
"""

from __future__ import annotations

import os

import obstore as obs
import pytest
from obstore.store import GCSStore

from mogemma.hub import KNOWN_GCS_MODELS, HubManager

_RUN_LIVE = os.environ.get("MOGEMMA_LIVE_GCS") == "1"


pytestmark = pytest.mark.skipif(not _RUN_LIVE, reason="Set MOGEMMA_LIVE_GCS=1 to opt into live gs://gemma-data probes.")


@pytest.fixture(scope="module")
def store() -> GCSStore:
    return GCSStore(bucket="gemma-data", skip_signature=True)


@pytest.mark.gcs
@pytest.mark.parametrize("model_id", sorted(KNOWN_GCS_MODELS))
def test_known_model_prefix_has_objects(store: GCSStore, model_id: str) -> None:
    clean = HubManager._clean_model_id(model_id)
    prefix = HubManager._gcs_checkpoint_prefix(clean)

    object_count = 0
    for batch in obs.list(store, prefix=prefix):
        object_count += len(batch)
        if object_count > 0:
            break

    assert object_count > 0, (
        f"Catalog entry {model_id!r} resolves to gs://gemma-data/{prefix} but has 0 objects. "
        "Remove from KNOWN_GCS_MODELS or publish the checkpoint upstream."
    )
