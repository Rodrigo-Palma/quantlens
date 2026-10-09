"""Offline evaluation suite: faithfulness of LLM output, guardrail and retrieval.

Every eval runs without network or GPU. LLM outputs are scored from recorded
cassettes (``data/cassettes``), which ``python -m quantlens.evals.record``
regenerates against a local Ollama server.
"""
