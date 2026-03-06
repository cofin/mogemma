import pytest

from mogemma.backends import parse_device_spec, resolve_device_selection
from mogemma.config import EmbeddingConfig, GenerationConfig
from mogemma.model import EmbeddingModel, SyncGemmaModel


def test_parse_device_spec_accepts_cpu_gpu_and_indexed_gpu() -> None:
    assert parse_device_spec("cpu") == ("cpu", None)
    assert parse_device_spec("gpu") == ("gpu", None)
    assert parse_device_spec("gpu:2") == ("gpu", 2)


def test_parse_device_spec_rejects_malformed_values() -> None:
    with pytest.raises(ValueError, match="Supported backends: cpu, gpu, gpu:<index>"):
        parse_device_spec("banana")
    with pytest.raises(ValueError, match="Supported backends: cpu, gpu, gpu:<index>"):
        parse_device_spec("gpu:abc")
    with pytest.raises(ValueError, match="Supported backends: cpu, gpu, gpu:<index>"):
        parse_device_spec("gpu:")


def test_resolve_device_selection_returns_normalized_descriptor() -> None:
    selection = resolve_device_selection("gpu:3", gpu_available=True)

    assert selection.requested == "gpu:3"
    assert selection.backend == "gpu"
    assert selection.device_kind == "gpu"
    assert selection.device_index == 3
    assert selection.strict is True
    assert selection.availability_source == "override"
    assert selection.effective_backend_id == "gpu"
    assert selection.effective_device == "gpu:3"


def test_resolve_device_selection_rejects_unavailable_gpu_without_fallback() -> None:
    with pytest.raises(RuntimeError, match="Requested device 'gpu:1' is unavailable on this host."):
        resolve_device_selection("gpu:1", gpu_available=False)


def test_generation_model_init_rejects_invalid_device_spec() -> None:
    with pytest.raises(ValueError, match="Supported backends: cpu, gpu, gpu:<index>"):
        SyncGemmaModel(GenerationConfig(model_path="dummy", device="gpu:abc"))


def test_embedding_model_init_rejects_invalid_device_spec() -> None:
    with pytest.raises(ValueError, match="Supported backends: cpu, gpu, gpu:<index>"):
        EmbeddingModel(EmbeddingConfig(model_path="dummy", device="banana"))
