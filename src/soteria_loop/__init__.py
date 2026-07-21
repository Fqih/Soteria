"""Public API for Soteria 0.1."""

from soteria_loop.events import AgentEvent, EventType
from soteria_loop.integrations import LetheMemoryAdapter, MemoryProvider
from soteria_loop.models import (
    Checkpoint,
    ModelRequest,
    ModelResponse,
    RunRecord,
    RunResult,
    TokenUsage,
    ToolCall,
    ToolMetadata,
    ToolResult,
)
from soteria_loop.policies import LoopPolicy
from soteria_loop.progress import ProgressDetector
from soteria_loop.providers.fake import FakeProvider, ScriptItem
from soteria_loop.runtime import AgentRuntime
from soteria_loop.state import RunState, StopReason
from soteria_loop.tools import FunctionTool, Tool, ToolRegistry
from soteria_loop.tracing import RunTrace, TraceEntry, TraceInspector

__version__ = "0.1.0"

__all__ = [
    "AgentEvent",
    "AgentRuntime",
    "Checkpoint",
    "EventType",
    "FakeProvider",
    "FunctionTool",
    "LetheMemoryAdapter",
    "LoopPolicy",
    "MemoryProvider",
    "ModelRequest",
    "ModelResponse",
    "ProgressDetector",
    "RunRecord",
    "RunResult",
    "RunState",
    "RunTrace",
    "ScriptItem",
    "StopReason",
    "TokenUsage",
    "Tool",
    "ToolCall",
    "ToolMetadata",
    "ToolRegistry",
    "ToolResult",
    "TraceEntry",
    "TraceInspector",
]
