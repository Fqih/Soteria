"""Workspace indexer — file map + mtime cache for project awareness.

The indexer walks the workspace once and produces a small in-memory map
of relative paths + mtimes. It is intentionally cheap to rebuild so a
``workspace_map`` tool call always returns current data, while a
cached ``recent_files`` query uses the same data without re-walking.

Paths are stored relative to ``Workspace.root`` and never absolute so a
caller cannot leak the host filesystem layout.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from hernness.exceptions import ToolExecutionError

IndexerError = ToolExecutionError

_DEFAULT_MAX_ENTRIES = 2_000
_DEFAULT_MAX_DEPTH = 16


@dataclass(frozen=True)
class IndexEntry:
    """One file in the workspace index."""

    path: str
    size: int
    mtime_ns: int

    def to_dict(self) -> dict[str, int | str]:
        return {"path": self.path, "size": self.size, "mtime_ns": self.mtime_ns}


@dataclass(frozen=True)
class WorkspaceIndex:
    """Snapshot of a workspace tree."""

    root: Path
    entries: tuple[IndexEntry, ...]
    truncated: bool

    def map_text(self, *, max_entries: int) -> str:
        """Render a compact ``path`` list — one per line, capped."""

        lines = [str(entry.path) for entry in self.entries[:max_entries]]
        if self.truncated or len(self.entries) > max_entries:
            lines.append(f"... +{max(self.entries.__len__() - max_entries, 0)} more")
        return "\n".join(lines)

    def recent_files(self, *, limit: int) -> tuple[IndexEntry, ...]:
        """Return the ``limit`` most-recently-modified entries."""

        return tuple(sorted(self.entries, key=lambda e: e.mtime_ns, reverse=True)[:limit])


class WorkspaceIndexer:
    """Walk a workspace once, return a :class:`WorkspaceIndex`."""

    __slots__ = ("_root",)

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def build(
        self,
        *,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        max_depth: int = _DEFAULT_MAX_DEPTH,
        follow_symlinks: bool = False,
    ) -> WorkspaceIndex:
        root = self._root
        if not root.is_dir():
            raise IndexerError(f"workspace root {root} is not a directory")
        truncated = False
        out: list[IndexEntry] = []
        for entry in _walk(root, max_depth=max_depth, follow_symlinks=follow_symlinks):
            if len(out) >= max_entries:
                truncated = True
                break
            stat = entry.stat()
            try:
                rel = entry.resolve().relative_to(root)
            except ValueError:
                continue  # pragma: no cover - guarded by walk
            out.append(
                IndexEntry(path=rel.as_posix(), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
            )
        return WorkspaceIndex(root=root, entries=tuple(out), truncated=truncated)


def _walk(root: Path, *, max_depth: int, follow_symlinks: bool) -> Iterator[Path]:
    """Yield file paths under ``root`` with a hard depth cap."""

    base_depth = len(root.parts)
    stack: list[tuple[Path, bool]] = [(root, True)]
    while stack:
        current, is_first = stack.pop()
        try:
            for entry in os.scandir(current):
                entry_path = Path(entry.path)
                depth = len(entry_path.resolve().parts) - base_depth
                if depth > max_depth:
                    continue
                if entry.is_dir(follow_symlinks=follow_symlinks):
                    stack.append((entry_path, False))
                elif entry.is_file(follow_symlinks=follow_symlinks):
                    yield entry_path
        except OSError:
            continue
        finally:
            _ = is_first  # placeholder for future hooks


__all__ = [
    "IndexEntry",
    "IndexerError",
    "WorkspaceIndex",
    "WorkspaceIndexer",
]
