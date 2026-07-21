"""Public API for Soteria 0.1."""

from soteria.events import AgentEvent, EventType
from soteria.integrations import LetheMemoryAdapter, MemoryProvider
from soteria.models import (
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
from soteria.policies import LoopPolicy
from soteria.progress import ProgressDetector
from soteria.runtime import AgentRuntime
from soteria.state import RunState, StopReason
from soteria.tools import FunctionTool, Tool, ToolRegistry
from soteria.tracing import RunTrace, TraceEntry, TraceInspector

__version__ = "0.1.0"

__all__ = [
    "AgentEvent",
    "AgentRuntime",
    "Checkpoint",
    "EventType",
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
