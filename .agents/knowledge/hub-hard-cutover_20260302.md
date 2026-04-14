# Learnings

1. The active runtime `obstore` surface is narrow: only [hub.py](/home/cody/code/c/mogemma/src/py/mogemma/hub.py), [pyproject.toml](/home/cody/code/c/mogemma/pyproject.toml), and `uv.lock` currently reference it. That makes the hard cutover primarily a hub-contract rewrite plus manifest cleanup.
2. [model.py](/home/cody/code/c/mogemma/src/py/mogemma/model.py) is a downstream contract consumer, not a migration site. The cutover should preserve strict hub resolution and committed-cache tokenizer expectations there instead of moving fallback logic into model initialization.
3. Failure mapping needs three separate operator signals: upstream transport problems should be retryable `ConnectionError`s, local disk problems should stay `OSError`s, and committed-content violations should fail closed as non-retryable integrity errors.
4. The clean handoff boundary is request-level, not artifact-list-level: Python should pass normalized model/cache intent, while native code owns remote artifact and tokenizer selection and returns only a committed cache directory plus compact summary flags.
5. The release gate should focus on caller-visible invariants, not implementation mechanics: warm-cache must stay local and fast, cold-path failures must leave no valid cache tree, and Python orchestration must never grow back into per-artifact scheduling.
