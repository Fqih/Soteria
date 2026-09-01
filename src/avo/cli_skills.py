"""`avo skill` — install, list, show, remove skill markdown files.

Skills are plain markdown with a YAML frontmatter block. They live
under a directory named after the skill, with a single ``SKILL.md``
file inside (the same convention as Claude Code's
``~/.claude/skills/<name>/SKILL.md`` and Codex's skill packs).

Example skill file::

    ---
    name: python-style
    description: Project-specific Python style rules for the avo codebase.
    ---

    # python-style

    Use ruff defaults; line-length 100; mypy strict; etc.

This CLI installs skills into ``~/.avo/skills/<name>/SKILL.md`` so the
chat REPL picks them up via :class:`avo.skills.SkillRegistry`.
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from avo.exceptions import AvoError

USER_SKILLS_ROOT = Path.home() / ".avo" / "skills"


class SkillCliError(AvoError):
    """User-facing failure in `avo skill`."""


@dataclass(frozen=True)
class SkillInfo:
    """One installed skill."""

    name: str
    description: str
    path: Path

    @property
    def exists(self) -> bool:
        return self.path.exists()


def _validate_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise SkillCliError(f"invalid skill name {name!r}; use letters, digits, '.', '-', '_'.")
    return name


def _parse_frontmatter(text: str) -> tuple[str, str]:
    """Return ``(name, description)`` extracted from a YAML frontmatter block.

    The block must start with ``---`` on its own line and end with the
    next ``---``. Keys we recognise are ``name`` (defaults to the
    file's directory name) and ``description`` (defaults to empty).
    """

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return "", ""
    try:
        end = lines.index("---", 1)
    except ValueError:
        return "", ""
    name = ""
    description = ""
    for raw in lines[1:end]:
        line = raw.strip()
        if line.startswith("name:"):
            name = line.split(":", 1)[1].strip().strip('"').strip("'")
        elif line.startswith("description:"):
            description = line.split(":", 1)[1].strip().strip('"').strip("'")
    return name, description


def _resolve_source(source: str) -> tuple[Path, str]:
    """Resolve ``source`` to ``(path, inferred_name)``.

    Accepts a local path (file or directory) or a path inside a git
    checkout. The CLI itself does not clone — users should `git clone`
    first then ``avo skill install <local-path>``.
    """

    src = Path(source).expanduser().resolve()
    if not src.exists():
        raise SkillCliError(f"skill source does not exist: {src}")
    if src.is_dir():
        candidate = src / "SKILL.md"
        if candidate.exists():
            return candidate, src.name
        skill_files = sorted(src.glob("*.md"))
        if not skill_files:
            raise SkillCliError(f"directory {src} has no SKILL.md or *.md file to install.")
        return skill_files[0], skill_files[0].stem
    return src, src.stem


def install(source: str, *, name: str | None = None) -> SkillInfo:
    """Install a skill from a local path into :data:`USER_SKILLS_ROOT`."""

    src, inferred = _resolve_source(source)
    skill_name = _validate_name(name or inferred)
    target_dir = USER_SKILLS_ROOT / skill_name
    if target_dir.exists():
        raise SkillCliError(
            f"skill {skill_name!r} already installed at {target_dir}; "
            f"`avo skill remove {skill_name}` first."
        )
    USER_SKILLS_ROOT.mkdir(parents=True, exist_ok=True)
    target = target_dir / "SKILL.md"
    shutil.copyfile(src, target)
    body = target.read_text(encoding="utf-8")
    _, description = _parse_frontmatter(body)
    return SkillInfo(name=skill_name, description=description, path=target)


def remove(name: str) -> None:
    """Remove a skill by name."""

    name = _validate_name(name)
    target = USER_SKILLS_ROOT / name
    if not target.exists():
        raise SkillCliError(f"skill {name!r} is not installed.")
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()


def list_installed() -> tuple[SkillInfo, ...]:
    """Return all installed skills (sorted by name)."""

    if not USER_SKILLS_ROOT.exists():
        return ()
    out: list[SkillInfo] = []
    for entry in sorted(USER_SKILLS_ROOT.iterdir()):
        if not entry.is_dir():
            continue
        skill_file = entry / "SKILL.md"
        description = ""
        if skill_file.exists():
            _, description = _parse_frontmatter(skill_file.read_text(encoding="utf-8"))
        out.append(SkillInfo(name=entry.name, description=description, path=skill_file))
    return tuple(out)


def show(name: str) -> SkillInfo:
    """Return one installed skill."""

    name = _validate_name(name)
    target = USER_SKILLS_ROOT / name / "SKILL.md"
    if not target.exists():
        raise SkillCliError(f"skill {name!r} is not installed.")
    description = _parse_frontmatter(target.read_text(encoding="utf-8"))[1]
    return SkillInfo(name=name, description=description, path=target)


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="avo skill",
        description="Install, list, and remove avo skills (~/.avo/skills).",
    )
    sub = parser.add_subparsers(dest="skill_command", required=True)

    install_p = sub.add_parser("install", help="Install a skill from a local path.")
    install_p.add_argument("source", help="Path to a SKILL.md file or directory.")
    install_p.add_argument("--name", default=None, help="Override the skill name.")

    sub.add_parser("list", help="List installed skills.")

    show_p = sub.add_parser("show", help="Print one installed skill.")
    show_p.add_argument("name")

    remove_p = sub.add_parser("remove", help="Remove an installed skill.")
    remove_p.add_argument("name")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.skill_command == "install":
        info = install(args.source, name=args.name)
        print(f"Installed skill {info.name!r} → {info.path}")
        if info.description:
            print(f"  {info.description}")
        return 0

    if args.skill_command == "list":
        skills = list_installed()
        if not skills:
            print(f"No skills installed at {USER_SKILLS_ROOT}.")
            return 0
        print(f"{'NAME':24}  DESCRIPTION")
        for skill in skills:
            print(f"{skill.name:24}  {skill.description}")
        return 0

    if args.skill_command == "show":
        info = show(args.name)
        print(info.path.read_text(encoding="utf-8"))
        return 0

    if args.skill_command == "remove":
        remove(args.name)
        print(f"Removed skill {args.name!r}.")
        return 0

    raise SkillCliError(f"unknown skill subcommand: {args.skill_command}")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "USER_SKILLS_ROOT",
    "SkillCliError",
    "SkillInfo",
    "install",
    "list_installed",
    "main",
    "remove",
    "show",
]
