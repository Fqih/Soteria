"""Tests for the notification system."""

from __future__ import annotations

import pytest

from soteria_loop.notifications import (
    DesktopNotifier,
    Notification,
    NotificationDispatcher,
    NotifierError,
    WebhookNotifier,
    from_env,
)


class _FakeResponse:
    status_code: int = 200


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, str]]] = []

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def post(self, url: str, *, json: dict[str, str]) -> _FakeResponse:
        self.calls.append((url, json))
        return _FakeResponse()


def test_notification_to_dict() -> None:
    note = Notification(title="Run done", body="ok", level="info")
    assert note.to_dict() == {"title": "Run done", "body": "ok", "level": "info"}


def test_webhook_notifier_posts_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    import soteria_loop.notifications as mod

    fake = _FakeClient()
    fake_httpx = type("_H", (), {"Client": lambda *a, **kw: fake})
    monkeypatch.setattr(mod, "_httpx", fake_httpx)
    WebhookNotifier("https://example/hook").send(Notification("t", "b"))
    assert fake.calls == [("https://example/hook", {"title": "t", "body": "b", "level": "info"})]


def test_webhook_notifier_requires_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    import soteria_loop.notifications as mod

    monkeypatch.setattr(mod, "_httpx", None)
    with pytest.raises(NotifierError, match="requires httpx"):
        WebhookNotifier("https://x/").send(Notification("t", "b"))


def test_dispatcher_fans_out_to_all_notifiers() -> None:
    calls_a: list[Notification] = []
    calls_b: list[Notification] = []

    class _A:
        def send(self, note: Notification) -> None:
            calls_a.append(note)

    class _B:
        def send(self, note: Notification) -> None:
            calls_b.append(note)

    dispatcher = NotificationDispatcher([_A(), _B()])
    dispatcher.send(Notification("t", "b"))
    assert len(calls_a) == 1
    assert len(calls_b) == 1


def test_dispatcher_continues_on_failure() -> None:
    success_calls: list[Notification] = []

    class _Boom:
        def send(self, note: Notification) -> None:
            raise NotifierError("broken")

    class _Ok:
        def send(self, note: Notification) -> None:
            success_calls.append(note)

    dispatcher = NotificationDispatcher([_Boom(), _Ok()])
    dispatcher.send(Notification("t", "b"))
    assert len(success_calls) == 1


def test_dispatcher_register_appends() -> None:
    dispatcher = NotificationDispatcher()
    calls: list[Notification] = []

    class _A:
        def send(self, note: Notification) -> None:
            calls.append(note)

    dispatcher.register(_A())
    dispatcher.send(Notification("t", "b"))
    assert len(calls) == 1


def test_desktop_notifier_no_command_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import soteria_loop.notifications as mod

    monkeypatch.setattr(mod.shutil, "which", lambda _name: None)
    notifier = DesktopNotifier()
    assert notifier._command is None  # type: ignore[attr-defined]
    notifier.send(Notification("t", "b"))  # no raise


def test_from_env_returns_empty_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SOTERIA_NOTIFY_WEBHOOK", raising=False)
    monkeypatch.delenv("SOTERIA_NOTIFY_DESKTOP", raising=False)
    dispatcher = from_env()
    assert dispatcher._notifiers == []  # type: ignore[attr-defined]


def test_from_env_registers_webhook_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOTERIA_NOTIFY_WEBHOOK", "https://x/hook")
    monkeypatch.delenv("SOTERIA_NOTIFY_DESKTOP", raising=False)
    dispatcher = from_env()
    assert any(isinstance(n, WebhookNotifier) for n in dispatcher._notifiers)  # type: ignore[attr-defined]


def test_from_env_registers_desktop_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SOTERIA_NOTIFY_WEBHOOK", raising=False)
    monkeypatch.setenv("SOTERIA_NOTIFY_DESKTOP", "1")
    dispatcher = from_env()
    assert any(isinstance(n, DesktopNotifier) for n in dispatcher._notifiers)  # type: ignore[attr-defined]
