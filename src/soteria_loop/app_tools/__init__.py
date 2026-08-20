"""Application tools: workspace sandboxing, shell execution, and approval.

Application tools build on the existing ``FunctionTool`` / ``ToolRegistry``
contract documented in ``src/soteria_loop/tools.py``. They do not modify
``AgentRuntime`` or the state machine; they plug in via the existing
``approval_callback`` parameter on ``AgentRuntime``.

The sandbox module is optional and lives behind the ``[sandbox]`` extra in
``pyproject.toml`` because it requires the ``docker`` package.
"""

from __future__ import annotations

__all__ = []
