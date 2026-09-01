"""Notification system — webhook + desktop notifier on terminal events.

Two transports:

* :class:`WebhookNotifier` — POSTs a JSON payload to a configured URL.
  Uses :mod:`httpx` when available; otherwise raises ``NotifierError``
  so callers can degrade gracefully.
* :class:`DesktopNotifier` — invokes ``notify-send`` (Linux) /
  ``osascript`` (macOS) / ``msg`` (Windows) with a short message.

Both notifiers share :class:`NotificationDispatcher`, the registry the
runtime calls on terminal events. The dispatcher never raises — a
broken transport logs and continues so a transient webhook outage does
not block the run.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from avo.exceptions import ToolExecutionError

NotifierError = ToolExecutionError

try:
    import httpx as _httpx
except ModuleNotFoundError:  # pragma: no cover
    _httpx = None  # type: ignore[assignment]


@dataclass(frozen=True)
class Notification:
    """A message to deliver."""

    title: str
    body: str
    level: str = "info"  # "info" | "warning" | "error"

    def to_dict(self) -> dict[str, str]:
        return {"title": self.title, "body": self.body, "level": self.level}


@runtime_checkable
class Notifier(Protocol):
    """Common surface every transport implements."""

    def send(self, notification: Notification) -> None: ...


class WebhookNotifier:
    """POST notifications to a remote URL."""

    def __init__(self, url: str, *, timeout: float = 5.0) -> None:
        self._url = url
        self._timeout = timeout

    def send(self, notification: Notification) -> None:
        if _httpx is None:
            raise NotifierError(
                "WebhookNotifier requires httpx; install avo with the [providers] extra."
            )
        try:
            with _httpx.Client(timeout=self._timeout) as client:
                client.post(self._url, json=notification.to_dict())
        except Exception as exc:  # bad transport never crashes the run
            raise NotifierError(f"webhook delivery failed: {exc}") from exc


class DesktopNotifier:
    """Native OS notifications — best-effort, no-op when no helper exists."""

    def __init__(self) -> None:
        self._command = self._detect_command()

    @staticmethod
    def _detect_command() -> list[str] | None:
        if shutil.which("notify-send"):
            return ["notify-send"]
        if shutil.which("osascript"):
            return ["osascript", "-e", 'display notification "%s" with title "%s"']
        if shutil.which("msg"):
            return ["msg", "*"]
        return None

    def send(self, notification: Notification) -> None:
        if self._command is None:
            return
        try:
            if self._command[0] == "osascript":
                # macOS path: substitute placeholders.
                script = self._command[2].replace("%s", "{}", 1)
                payload = script.format(notification.body, notification.title)
                subprocess.run(
                    ["osascript", "-e", payload],
                    check=False,
                    capture_output=True,
                )
            else:
                subprocess.run(
                    [*self._command, notification.title, notification.body],
                    check=False,
                    capture_output=True,
                )
        except OSError:
            pass  # best-effort; swallow transport errors


class NotificationDispatcher:
    """Fan-out registry — call :meth:`send` once, all transports get it."""

    def __init__(self, notifiers: Iterable[Notifier] | None = None) -> None:
        self._notifiers: list[Notifier] = list(notifiers or [])

    def register(self, notifier: Notifier) -> None:
        self._notifiers.append(notifier)

    def send(self, notification: Notification) -> None:
        for notifier in self._notifiers:
            try:
                notifier.send(notification)
            except NotifierError as exc:
                _ = exc  # future: structured logging
            except Exception:
                # never let a buggy transport crash the runtime
                continue


def from_env() -> NotificationDispatcher:
    """Build a dispatcher from env vars.

    ``AVO_NOTIFY_WEBHOOK`` — URL to POST to.
    ``AVO_NOTIFY_DESKTOP`` — set to ``1`` to enable desktop notifications.
    """

    dispatcher = NotificationDispatcher()
    webhook = os.environ.get("AVO_NOTIFY_WEBHOOK", "").strip()
    if webhook:
        dispatcher.register(WebhookNotifier(webhook))
    if os.environ.get("AVO_NOTIFY_DESKTOP", "").strip() == "1":
        dispatcher.register(DesktopNotifier())
    return dispatcher


__all__ = [
    "DesktopNotifier",
    "Notification",
    "NotificationDispatcher",
    "NotifierError",
    "WebhookNotifier",
    "from_env",
]
