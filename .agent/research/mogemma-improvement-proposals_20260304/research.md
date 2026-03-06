# Research: `mogemma` improvement proposals for deterministic extraction

Date: 2026-03-04
Type: New feature + performance hardening
Constraint: Prefer non-MAX implementation path due licensing concerns.

## Executive Summary

The four requested improvements are all feasible in `mogemma`, but the **highest-performance path without MAX** is to move decode-time control into the Mojo core (not Python wrappers):

1. Add per-call generation overrides (`temperature`, `max_new_tokens`, `top_k`, `top_p`) on `generate()` / `generate_stream()`.
2. Add a first-class chat API that formats Gemma turns correctly and handles system instructions safely.
3. Add stop criteria at both token-id and string levels, with immediate stream termination.
4. Add constrained decoding by masking invalid tokens before sampling. For peak performance, implement token-mask application and sampling in Mojo.

Most important architectural finding: today, `_core.step()` computes logits in Mojo, but sampling/termination logic is still Python-side (`_sample_next_token` + token decode loop). That is the core reason deterministic extraction features are currently harder/slower than they need to be.

## Codebase Analysis

### Relevant modules

- `src/py/mogemma/model.py`
- `src/py/mogemma/config.py`
- `src/mo/mogemma/core.mojo`
- `src/py/mogemma/hub.py`
- `src/py/tests/test_gemma_model.py`
- `src/py/tests/test_async.py`

### Current behavior and gaps

1. Sampling parameters are config-level only, not per-call.
- `GenerationConfig` exposes `temperature`, `top_k`, `top_p`, `max_tokens`.
- `SyncGemmaModel.generate(prompt)` and `generate_stream(prompt)` do not accept per-call overrides.

2. Instruction formatting exists, but only as plain-string wrapping.
- `_format_instruction_prompt()` wraps prompt with `<start_of_turn>user ... <end_of_turn> <start_of_turn>model`.
- No `chat(messages)` / message-role API.

3. Stop behavior is limited.
- Generation stops on EOS token id and empty decoded text.
- No user-provided stop sequences, stop token IDs, or multi-condition stopping.

4. Sampling is Python-hot-path, not Mojo-hot-path.
- `_core.step(...)` returns logits.
- `_sample_next_token(...)` runs in Python/NumPy for each token.
- `step_mojo(...)` receives `temp_obj`, `top_k_obj`, `top_p_obj` but does not apply sampling logic internally.

5. Tokenizer/chat-template limitations in current download path.
- Hub download normalizes to `tokenizer.model` (SentencePiece file).
- Current tokenizer wrapper does not expose template metadata loading.
- This means chat templating must be explicit in `mogemma`, not delegated to existing local tokenizer metadata.

## Library Documentation and External Evidence

### Gemma formatting/system constraints

- Gemma IT uses turn tokens `<start_of_turn>` / `<end_of_turn>` with `user` and `model` roles.
- Google docs state Gemma IT supports only `user` and `model` roles; `system` should be folded into initial user content.

### Standard generation controls and stop API patterns

- Transformers `GenerationConfig` documents `max_new_tokens`, `temperature`, `top_k`, `top_p`, and `stop_strings`.
- This matches the requested API direction and provides familiar names.

### Chat template best practices

- Transformers `apply_chat_template()` shows message-list APIs and emphasizes correct control-token formatting and generation prompts.
- Incorrect template/control tokens reduce instruction following quality.

### Constrained decoding prior art and performance signals

- `llguidance` (MIT) reports ~50 microseconds CPU mask computation per token for 128k vocab and no significant startup cost.
- `xgrammar` (Apache-2.0) reports near-zero overhead for JSON structured generation and supports JSON schema / regex / CFG.
- `outlines` (Apache-2.0) is widely used but is more Python-oriented orchestration than low-level kernel path.

### MAX-specific note (ruled out as primary path here)

- MAX structured output uses llguidance but currently targets MAX-model GPU deployments; CPU and PyTorch support are still in progress.
- Since this flow prefers avoiding MAX coupling/licensing concerns, MAX is not the recommended implementation path for this repo.

### Mojo/current version context

- Mojo changelog currently shows stable `v26.1` (2026-01-29) and nightly `v0.26.2` in progress.
- Current interop APIs in this repo (`Int(py=...)`, `String(py=...)`, `PythonModuleBuilder`) are aligned with recent Mojo guidance.

## Prior Art

### Internal prior art

- Commit `b7be075` added instruction prompt wrapping and config constructor ergonomics.
- Existing tests already validate EOS stop and deterministic decode behavior (`temperature=0.0` paths), which provides a solid base for extension.

### External prior art to emulate

1. Message-list chat API + template application (Transformers-style UX).
2. `stop_strings`/multi-stop criteria (Transformers-style naming).
3. Constrained decoding via token-mask engines (`llguidance`, `xgrammar`) for correctness guarantees.

## Risk Assessment

1. API compatibility risk (medium)
- Adding many kwargs can create ambiguity with existing `GenerationConfig` fields.
- Mitigation: define precedence (`per-call overrides` > `config defaults`) and document it.

2. Performance regression risk (high if done in Python)
- Python-side regex filtering or post-hoc validation can degrade throughput and still allow malformed intermediate output.
- Mitigation: perform constraints as token-masking before sampling, ideally in Mojo.

3. Gemma role semantics risk (medium)
- Treating `system` as a native role conflicts with Gemma IT guidance.
- Mitigation: normalize `system` into first `user` turn for Gemma IT models.

4. Constraint-language scope risk (high)
- Full regex/CFG engines are complex to implement from scratch.
- Mitigation: phase rollout with a Mojo-native fast subset first (enum/literal + prefix/trie + simple regex subset), then optional advanced engine integration.

5. Licensing/compliance risk (project constraint)
- Pulling in MAX runtime can add policy/process implications for some deployments.
- Mitigation: keep default path pure `mogemma` + permissive OSS dependencies; expose optional plugin adapters behind extras.

## Recommended Approach

### Recommendation: non-MAX, Mojo-first decode control

Implement decode controls where tokens are chosen, not after decode. In this repo that means moving from:
- `Mojo logits -> Python sampling/stop`

to:
- `Mojo logits -> Mojo token-mask/stop/sampling -> Python yields decoded token`

This is the highest-performance architecture while staying aligned with your no-MAX preference.

### Phase plan

1. **Phase A: API surface (low risk, immediate value)**
- Extend sync+async methods with per-call kwargs:
  - `temperature`, `top_k`, `top_p`, `max_new_tokens`
  - `stop_sequences`, `stop_token_ids`
- Add `chat(messages, **kwargs)` and `chat_stream(messages, **kwargs)`.
- Add explicit `GenerationOptions` dataclass for normalized per-call options.

2. **Phase B: Correct stop handling (medium risk)**
- Pre-tokenize stop token sequences once per request.
- Track rolling token window for token-based stops.
- Track rolling string buffer for textual stop sequences and cut output exactly at stop boundary.
- Keep EOS alias handling as hard stop.

3. **Phase C: Mojo-native sampling and mask application (highest performance)**
- Add a new `_core` entrypoint that performs sampling in Mojo (or returns sampled token + done flags), removing Python NumPy sampling from hot loop.
- Add optional per-step token allow-mask input (bitset/int8 mask) for constrained decoding.
- Keep Python side as orchestration only.

4. **Phase D: constrained decoding levels**
- Level 1 (default, fast): literal/enum constraints and anchored prefix automata.
- Level 2: bounded regex subset compiled to DFA/trie-compatible transitions.
- Level 3 (optional plugin): advanced grammar engine bridge for full JSON schema/complex regex where needed.

### Proposed API sketch

```python
# per-call overrides
text = model.generate(
    prompt,
    temperature=0.0,
    max_new_tokens=20,
    top_k=1,
    stop_sequences=["<end_of_turn>", "\n"],
)

# chat API with role normalization for Gemma IT
reply = await model.chat(
    [
        {"role": "system", "content": "Output EXACTLY one class label."},
        {"role": "user", "content": "SKU: ..."},
    ],
    temperature=0.0,
    max_new_tokens=12,
)

# constrained decoding (phase D)
label = await model.generate(
    prompt,
    temperature=0.0,
    choices=["CP-ABC-123", "IGNORE"],
)
```

## Source Links

- Local code:
  - `src/py/mogemma/model.py`
  - `src/py/mogemma/config.py`
  - `src/mo/mogemma/core.mojo`
  - `src/py/mogemma/hub.py`
  - `src/py/tests/test_gemma_model.py`
  - `src/py/tests/test_async.py`

- Gemma prompt/system formatting:
  - https://ai.google.dev/gemma/docs/core/prompt-structure

- Hugging Face chat templates:
  - https://huggingface.co/docs/transformers/chat_templating

- Hugging Face generation parameters (`stop_strings`, sampling):
  - https://huggingface.co/docs/transformers/main_classes/text_generation

- llguidance (MIT, constrained decoding performance details):
  - https://github.com/guidance-ai/llguidance

- xgrammar (Apache-2.0, low-overhead structured generation):
  - https://github.com/mlc-ai/xgrammar

- outlines (Apache-2.0, structured generation framework):
  - https://github.com/dottxt-ai/outlines

- MAX structured output capability and limitations (for comparison):
  - https://docs.modular.com/max/serve/structured-output/

- Mojo changelog (version/date context):
  - https://docs.modular.com/mojo/changelog/

- Modular community license terms (licensing context):
  - https://www.modular.com/legal/community
