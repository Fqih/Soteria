"""Live integration tests for built-in providers.

These tests hit real HTTP endpoints. They are NOT part of the default ``pytest``
suite — they are gated behind the ``--run-live`` flag and an explicit
per-provider skip when the required environment variable is unset, so the
core CI run stays offline and free of paid API traffic.

Run locally with::

    # All four providers, skipping any whose key is missing
    pytest tests/integration --run-live -v

    # Just ollama (local)
    pytest tests/integration/test_ollama_live.py --run-live -v

    # Just minimax (with both API styles)
    HERNNESS_PROVIDER=minimax HERNNESS_MINIMAX_API_KEY=... \\
        pytest tests/integration/test_minimax_live.py --run-live -v

Required environment variables (per provider):

- ollama:   ``HERNNESS_OLLAMA_BASE_URL`` (default ``http://localhost:11434``),
            and a model the local daemon already has pulled.
- minimax:  ``HERNNESS_MINIMAX_API_KEY`` + ``HERNNESS_MINIMAX_MODEL``
            (defaults to ``MiniMax-M3``).
- anthropic: ``HERNNESS_ANTHROPIC_API_KEY`` + ``HERNNESS_ANTHROPIC_MODEL``.
- openai:   ``HERNNESS_OPENAI_API_KEY`` + ``HERNNESS_OPENAI_MODEL``.

Every test prints the resolved endpoint, the request URL, the response
status, and (on failure) the unredacted error body so a 400 / 401 / 429
can be diagnosed without re-running the script with extra logging. API
keys themselves are never printed.
"""
