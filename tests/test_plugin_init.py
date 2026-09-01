"""Tests for `avo plugin init` scaffold generation."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from avo.cli_plugins import PluginCliError, init_scaffold
from avo.cli_plugins import main as plugin_main


def test_init_scaffold_creates_directory_and_files(tmp_path: Path) -> None:
    path = init_scaffold("hello-plugin", directory=tmp_path, description="say hi")
    assert path == tmp_path / "hello-plugin"
    assert (path / "pyproject.toml").exists()
    assert (path / "README.md").exists()
    assert (path / ".gitignore").exists()
    assert (path / "hello_plugin" / "__init__.py").exists()


def test_init_scaffold_pyproject_declares_entry_point(tmp_path: Path) -> None:
    path = init_scaffold("cool", directory=tmp_path)
    data = tomllib.loads((path / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["name"] == "cool"
    eps = data["project"]["entry-points"]["avo.tools"]
    assert eps["cool"] == "cool:register"


def test_init_scaffold_module_register_returns_function_tool(tmp_path: Path) -> None:
    path = init_scaffold("widget", directory=tmp_path)
    spec_path = path / "widget" / "__init__.py"
    namespace: dict[str, object] = {}
    exec(compile(spec_path.read_text(encoding="utf-8"), str(spec_path), "exec"), namespace)
    register = namespace["register"]
    tools = register()  # type: ignore[operator]
    assert len(tools) == 1
    tool = tools[0]
    assert tool.metadata.name.startswith("plugin_widget_")
    assert tool.metadata.description


def test_init_scaffold_rejects_existing_directory_without_force(tmp_path: Path) -> None:
    init_scaffold("dup", directory=tmp_path)
    with pytest.raises(PluginCliError, match="already exists"):
        init_scaffold("dup", directory=tmp_path)


def test_init_scaffold_force_overwrites(tmp_path: Path) -> None:
    init_scaffold("dup", directory=tmp_path)
    path = init_scaffold("dup", directory=tmp_path, force=True, description="new desc")
    data = tomllib.loads((path / "pyproject.toml").read_text(encoding="utf-8"))
    assert "new desc" in data["project"]["description"]


def test_init_scaffold_rejects_invalid_name(tmp_path: Path) -> None:
    with pytest.raises(PluginCliError, match="invalid plugin name"):
        init_scaffold("../escape", directory=tmp_path)


def test_init_cli_init_invokes_scaffold(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = plugin_main(["init", "scaffolded", "--directory", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Scaffolded plugin" in out
    assert (tmp_path / "scaffolded" / "pyproject.toml").exists()


def test_init_cli_help_lists_force_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        plugin_main(["init", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--force" in out
    assert "--directory" in out
