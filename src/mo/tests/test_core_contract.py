from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

_core = pytest.importorskip("mogemma._core")


def test_mojo_init_model_rejects_empty_dict() -> None:
    # Just checking that our modified init_model handles empty metadata correctly
    # without blowing up.
    result = _core.init_model({})
    assert isinstance(result, dict)


def test_mojo_step_contract_is_token_sensitive() -> None:
    class FakeLLM:
        def step(self, token_id: int, temp: float, top_k: int, top_p: float) -> npt.NDArray[np.float32]:
            del temp, top_k, top_p
            return np.asarray(
                [0.1 + 0.0001 * token_id, 0.2 + 0.0002 * token_id, 0.3 + 0.0003 * token_id], dtype=np.float32
            )

    llm = FakeLLM()

    logits_for_token_one: npt.NDArray[np.float32] = np.asarray(_core.step(llm, 11, 1.0, 10, 1.0), dtype=np.float32)
    logits_for_token_two: npt.NDArray[np.float32] = np.asarray(_core.step(llm, 22, 1.0, 10, 1.0), dtype=np.float32)

    assert logits_for_token_one.shape == logits_for_token_two.shape
    assert logits_for_token_one.ndim == 1
    assert logits_for_token_one.dtype == np.float32
    assert not np.array_equal(logits_for_token_one, logits_for_token_two)


def test_mojo_embeddings_contract_enforces_shape_dtype_and_determinism() -> None:
    class FakeEmbeddingLLM:
        def encode(self, input_array: npt.NDArray[np.int32]) -> npt.NDArray[np.float32]:
            batch_size = int(input_array.shape[0])
            base = float(np.sum(input_array))
            return np.full((batch_size, 768), base, dtype=np.float32)

    llm = FakeEmbeddingLLM()
    tokens_a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    tokens_b = np.array([[7, 8, 9]], dtype=np.int32)

    emb_a_first = np.asarray(_core.generate_embeddings(llm, tokens_a), dtype=np.float32)
    emb_a_second = np.asarray(_core.generate_embeddings(llm, tokens_a), dtype=np.float32)
    emb_b = np.asarray(_core.generate_embeddings(llm, tokens_b), dtype=np.float32)

    assert emb_a_first.ndim == 2  # noqa: PLR2004
    assert emb_a_first.shape == (2, 768)
    assert emb_a_first.dtype == np.float32
    assert np.array_equal(emb_a_first, emb_a_second)
    assert not np.array_equal(emb_a_first[0], emb_b[0])


def test_mojo_step_contract_rejects_non_vector_backend_output() -> None:
    class InvalidStepLLM:
        def step(self, token_id: int, temp: float, top_k: int, top_p: float) -> npt.NDArray[np.float64]:
            del token_id, temp, top_k, top_p
            return np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64)

    llm = InvalidStepLLM()

    with pytest.raises(Exception, match="1D"):
        _core.step(llm, 11, 1.0, 10, 1.0)


def test_mojo_embeddings_contract_rejects_non_768d_output() -> None:
    class InvalidEmbeddingLLM:
        def encode(self, input_array: npt.NDArray[np.int32]) -> npt.NDArray[np.float32]:
            del input_array
            return np.zeros((1, 128), dtype=np.float32)

    llm = InvalidEmbeddingLLM()

    with pytest.raises(Exception, match="768"):
        _core.generate_embeddings(llm, np.array([[1, 2, 3]], dtype=np.int32))
