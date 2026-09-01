"""Confirm-before-destructive behavior for ``avo plugin/skill/mcp remove``."""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def tmp_avo_root(monkeypatch: pytest.MonkeyPatch) -> Path:
    root = Path(tempfile.mkdtemp(prefix="avo-confirm-test-"))
    monkeypatch.setattr("avo.cli_plugins.PLUGIN_ROOT", root)
    monkeypatch.setattr("avo.cli_plugins.PLUGIN_INDEX", root / "index.json")
    monkeypatch.setattr("avo.cli_skills.USER_SKILLS_ROOT", root / "skills")
    monkeypatch.setattr("avo.cli_mcp.MCP_CONFIG_PATH", root / "mcp.json")
    return root


def test_plugin_remove_aborts_without_yes(tmp_avo_root: Path) -> None:
    from avo.cli_plugins import main as plugin_main

    (tmp_avo_root / "demo").mkdir()
    (tmp_avo_root / "index.json").write_text(
        json.dumps(
            {
                "demo": {
                    "source": "https://example.com/demo.git",
                    "path": str(tmp_avo_root / "demo"),
                    "editable": True,
                    "kind": "git",
                }
            }
        ),
        encoding="utf-8",
    )
    stdin = io.StringIO("n\n")
    stdout = io.StringIO()
    with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
        code = plugin_main(["remove", "demo"])
    assert code == 1
    assert "Aborted." in stdout.getvalue()
    assert (tmp_avo_root / "demo").exists()


def test_plugin_remove_proceeds_with_yes(tmp_avo_root: Path) -> None:
    from avo.cli_plugins import main as plugin_main

    (tmp_avo_root / "demo").mkdir()
    (tmp_avo_root / "index.json").write_text(
        json.dumps(
            {
                "demo": {
                    "source": "https://example.com/demo.git",
                    "path": str(tmp_avo_root / "demo"),
                    "editable": True,
                    "kind": "git",
                }
            }
        ),
        encoding="utf-8",
    )
    stdout = io.StringIO()
    with patch("sys.stdout", stdout):
        code = plugin_main(["remove", "demo", "--yes"])
    assert code == 0
    assert "Removed plugin 'demo'" in stdout.getvalue()
    assert not (tmp_avo_root / "demo").exists()


def test_skill_remove_aborts_without_yes(tmp_avo_root: Path) -> None:
    from avo.cli_skills import main as skill_main

    skill_dir = tmp_avo_root / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: demo\n---\nbody\n", encoding="utf-8")

    stdin = io.StringIO("\n")
    stdout = io.StringIO()
    with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
        code = skill_main(["remove", "demo"])
    assert code == 1
    assert skill_dir.exists()


def test_skill_remove_proceeds_with_yes(tmp_avo_root: Path) -> None:
    from avo.cli_skills import main as skill_main

    skill_dir = tmp_avo_root / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: demo\n---\nbody\n", encoding="utf-8")

    stdout = io.StringIO()
    with patch("sys.stdout", stdout):
        code = skill_main(["remove", "demo", "--yes"])
    assert code == 0
    assert not skill_dir.exists()


def test_mcp_remove_aborts_without_yes(tmp_avo_root: Path) -> None:
    from avo.cli_mcp import main as mcp_main

    (tmp_avo_root / "mcp.json").write_text(
        json.dumps({"demo": {"command": ["echo", "hi"], "env": {}}}),
        encoding="utf-8",
    )
    stdin = io.StringIO("no\n")
    stdout = io.StringIO()
    with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
        code = mcp_main(["remove", "demo"])
    assert code == 1
    payload = json.loads((tmp_avo_root / "mcp.json").read_text(encoding="utf-8"))
    assert "demo" in payload


def test_mcp_remove_proceeds_with_yes(tmp_avo_root: Path) -> None:
    from avo.cli_mcp import main as mcp_main

    (tmp_avo_root / "mcp.json").write_text(
        json.dumps({"demo": {"command": ["echo", "hi"], "env": {}}}),
        encoding="utf-8",
    )
    stdout = io.StringIO()
    with patch("sys.stdout", stdout):
        code = mcp_main(["remove", "demo", "--yes"])
    assert code == 0
    payload = json.loads((tmp_avo_root / "mcp.json").read_text(encoding="utf-8"))
    assert "demo" not in payload


def test_mcp_add_masks_secret_env_values(
    capsys: pytest.CaptureFixture[str],
    tmp_avo_root: Path,
) -> None:
    """Secret-looking env keys must be masked in `mcp add` output."""

    from avo.cli_mcp import _looks_secret
    from avo.cli_mcp import main as mcp_main

    assert _looks_secret("API_TOKEN") is True
    assert _looks_secret("GITHUB_TOKEN") is True
    assert _looks_secret("AWS_SECRET_ACCESS_KEY") is True
    assert _looks_secret("FOO") is False
    assert _looks_secret("PATH") is False

    stdin = io.StringIO("")
    stdout = io.StringIO()
    with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
        code = mcp_main(
            [
                "add",
                "demo",
                "--env",
                "FOO=plain",
                "--env",
                "API_TOKEN=sk-xyz",
                "echo",
                "hi",
            ]
        )
    assert code == 0
    out = stdout.getvalue()
    assert "FOO" in out
    assert "plain" in out
    assert "API_TOKEN" in out
    assert "sk-xyz" not in out
    assert "***" in out
