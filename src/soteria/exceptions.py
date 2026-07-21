"""Exception hierarchy for Soteria."""

from __future__ import annotations


class SoteriaError(Exception):
    """Base class for expected Soteria failures."""


class InvalidStateTransitionError(SoteriaError):
    """Raised when a state-machine transition is not allowed."""


class RunNotFoundError(SoteriaError):
    """Raised when a requested run does not exist."""


class RunAlreadyTerminalError(SoteriaError):
    """Raised when execution is requested for a terminal run."""


class CheckpointNotFoundError(SoteriaError):
    """Raised when a run has no checkpoint from which it can resume."""


class UnsafeResumeError(SoteriaError):
    """Raised when a started tool has no durable result and cannot be retried safely."""


class DuplicateToolError(SoteriaError):
    """Raised when two tools use the same registry name."""


class ToolNotFoundError(SoteriaError):
    """Raised when a model requests a tool that is not registered."""


class ToolValidationError(SoteriaError):
    """Raised when tool arguments or output fail validation."""


class ToolExecutionError(SoteriaError):
    """Raised when a tool callable fails during execution."""


class ToolAlreadyCompletedError(SoteriaError):
    """Raised when an already-completed tool-call identifier is invoked."""


class StorageError(SoteriaError):
    """Raised when an event-store operation fails."""


class EventInvariantError(StorageError):
    """Raised when an append would violate an event-log invariant."""


class ProviderError(SoteriaError):
    """Raised for deterministic or adapter-specific provider failures."""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class FakeProviderExhaustedError(ProviderError):
    """Raised when a fake provider has no scripted response remaining."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)
