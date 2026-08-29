"""Exception hierarchy for Hernness."""

from __future__ import annotations


class HernnessError(Exception):
    """Base class for expected Hernness failures."""


class InvalidStateTransitionError(HernnessError):
    """Raised when a state-machine transition is not allowed."""


class RunNotFoundError(HernnessError):
    """Raised when a requested run does not exist."""


class RunAlreadyTerminalError(HernnessError):
    """Raised when execution is requested for a terminal run."""


class CheckpointNotFoundError(HernnessError):
    """Raised when a run has no checkpoint from which it can resume."""


class UnsafeResumeError(HernnessError):
    """Raised when a started tool has no durable result and cannot be retried safely."""


class DuplicateToolError(HernnessError):
    """Raised when two tools use the same registry name."""


class ToolNotFoundError(HernnessError):
    """Raised when a model requests a tool that is not registered."""


class ToolValidationError(HernnessError):
    """Raised when tool arguments or output fail validation."""


class ToolExecutionError(HernnessError):
    """Raised when a tool callable fails during execution."""


class ToolAlreadyCompletedError(HernnessError):
    """Raised when an already-completed tool-call identifier is invoked."""


class StorageError(HernnessError):
    """Raised when an event-store operation fails."""


class EventInvariantError(StorageError):
    """Raised when an append would violate an event-log invariant."""


class ProviderError(HernnessError):
    """Raised for deterministic or adapter-specific provider failures."""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class FakeProviderExhaustedError(ProviderError):
    """Raised when a fake provider has no scripted response remaining."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)
