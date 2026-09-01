"""Deprecation helper for the public API surface.

Use :func:`deprecated` to mark a callable as scheduled for removal.
The wrapper emits a :class:`DeprecationWarning` on every call,
forwards the original behavior, and keeps the symbol discoverable in
``__all__`` plus the deprecation index for grep-driven audits.

Example::

    @deprecated(since="0.2.0", removal="0.4.0", replacement="new_thing")
    def old_thing() -> None:
        ...

See :doc:`docs/semver.md` for the deprecation policy.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

from avo.exceptions import AvoError

__all__ = ["DeprecatedSymbol", "deprecated", "deprecation_index"]

_F = TypeVar("_F", bound=Callable[..., Any])


class DeprecatedSymbol(AvoError):
    """Raised when callers ignore the soft deprecation warning.

    The exception is opt-in via :func:`deprecated`'s ``strict=True``
    flag. By default deprecations only emit a warning so existing
    callers continue to work until the removal milestone.
    """


_DEPRECATIONS: list[dict[str, str]] = []


def deprecation_index() -> list[dict[str, str]]:
    """Return a snapshot of every deprecation registered so far."""

    return [dict(entry) for entry in _DEPRECATIONS]


def deprecated(
    *,
    since: str,
    removal: str,
    replacement: str | None = None,
    strict: bool = False,
) -> Callable[[_F], _F]:
    """Decorate a callable as scheduled for removal.

    Parameters
    ----------
    since:
        Version in which the symbol was first deprecated.
    removal:
        Version in which the symbol will be removed. ``since`` and
        ``removal`` must differ; the wrapper checks both are
        non-empty strings.
    replacement:
        Name of the symbol callers should migrate to. ``None`` means
        there is no direct replacement (the symbol has no successor).
    strict:
        When ``True``, the wrapper raises
        :class:`DeprecatedSymbol` instead of emitting a warning.
        Useful in tests that assert the deprecation contract.
    """

    if not since or not removal:
        raise ValueError("`since` and `removal` are required.")
    if since == removal:
        raise ValueError("`since` and `removal` must differ.")

    def decorate(func: _F) -> _F:
        message = (
            f"{func.__qualname__} is deprecated since {since} and will be removed in {removal}."
        )
        if replacement:
            message += f" Use {replacement} instead."

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if strict:
                raise DeprecatedSymbol(message)
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        _DEPRECATIONS.append(
            {
                "name": func.__qualname__,
                "since": since,
                "removal": removal,
                "replacement": replacement or "",
            }
        )
        return cast(_F, wrapper)

    return decorate
