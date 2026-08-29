"""Skills system — Markdown prompts loaded from a directory.

A *skill* is a Markdown file under a configurable root whose contents
are injected verbatim into the conversation when the operator triggers
``/skill <name>``. The runtime exposes two pieces:

* :class:`SkillRegistry` — discovers skills from disk, exposes the
  available names, and returns the body for a given name.
* :func:`load_skills` — convenience that returns a ``{name: body}``
  mapping (mostly useful in tests and CLI surfaces).

Skills are pure data: no network, no side effects, no model calls. The
runtime already has the approval seam in :mod:`hernness.permissions`
and the hooks seam in :mod:`hernness.hooks` — those cover policy;
this module only delivers the bytes.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Final

from hernness.exceptions import ToolExecutionError

SkillError = ToolExecutionError

# Skill names must be slug-safe so they can be invoked as ``/skill <name>``.
_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_SKILL_SUFFIX: Final[str] = ".md"


class SkillRegistry:
    """Discover and load skills from a directory."""

    __slots__ = ("_root",)

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def _path_for(self, name: str) -> Path:
        if not _NAME_PATTERN.match(name):
            raise SkillError(f"invalid skill name {name!r}; must match {_NAME_PATTERN.pattern}")
        return self._root / f"{name}{_SKILL_SUFFIX}"

    def exists(self, name: str) -> bool:
        return self._path_for(name).is_file()

    def load(self, name: str) -> str:
        """Return the body of ``name`` or raise :class:`SkillError`."""

        path = self._path_for(name)
        if not path.is_file():
            raise SkillError(f"unknown skill: {name}")
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SkillError(f"failed to read skill {name!r}: {exc}") from exc

    def names(self) -> list[str]:
        """Return sorted skill names discovered under :attr:`root`."""

        if not self._root.is_dir():
            return []
        return sorted(
            path.stem
            for path in self._root.iterdir()
            if path.is_file() and path.suffix == _SKILL_SUFFIX
        )

    def __iter__(self) -> Iterator[tuple[str, str]]:
        for name in self.names():
            yield name, self.load(name)


def load_skills(root: Path | str) -> dict[str, str]:
    """Return ``{name: body}`` for every skill under ``root``.

    ``root`` is created if it does not already exist so first-run use
    does not crash on a fresh checkout.
    """
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    return dict(SkillRegistry(root_path))


__all__ = ["SkillError", "SkillRegistry", "load_skills"]
