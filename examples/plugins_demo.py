"""Demonstrate the avo plugin lifecycle.

Walks through the same flow a user would hit from the CLI:

1. Discover plugins that are already published via Python entry points
   (``avo.tools``, ``avo.providers``, ``avo.notifiers``).
2. Install a local path-based plugin (here, a tiny synthetic package we
   build on the fly into a temp directory).
3. List installed plugins and confirm the new one is present.
4. Remove the plugin and verify it is gone.

The install/remove calls hit :func:`avo.cli_plugins.install` /
:func:`avo.cli_plugins.remove` directly so the demo runs without a
real PyPI mirror or git remote. We override ``PLUGIN_ROOT`` so the
demo never touches ``~/.avo/plugins``.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import textwrap
from pathlib import Path

# Force the demo to use a temp PLUGIN_ROOT instead of ~/.avo/plugins so
# it never leaves residue on the host. Insert the override before any
# avo module reads the constant.
import avo.cli_plugins as cli_plugins

DEMO_ROOT = Path(tempfile.mkdtemp(prefix="avo-plugins-demo-"))
cli_plugins.PLUGIN_ROOT = DEMO_ROOT
cli_plugins.PLUGIN_INDEX = DEMO_ROOT / "index.json"

from avo.plugins import (  # noqa: E402  (deliberately after the override)
    ALL_GROUPS,
    PROVIDER_GROUP,
    TOOL_GROUP,
    discover_all,
)

# `pip install -e` writes to the active interpreter; skip it for local
# synthetic packages to keep the demo hermetic. Patch `_pip_install`
# to be a no-op when the source path lives inside DEMO_ROOT.
_original_pip_install = cli_plugins._pip_install


def _noop_pip_install(target: Path, *, editable: bool) -> None:
    if str(target).startswith(str(DEMO_ROOT)):
        return
    _original_pip_install(target, editable=editable)


cli_plugins._pip_install = _noop_pip_install


def _build_synthetic_plugin(tmp: Path, name: str) -> Path:
    pkg = tmp / name
    (pkg / name).mkdir(parents=True)
    (pkg / name / "__init__.py").write_text("SYNTH = {'hello': 'world'}\n", encoding="utf-8")
    (pkg / "pyproject.toml").write_text(
        textwrap.dedent(
            f"""
            [project]
            name = "{name}"
            version = "0.0.1"
            description = "Synthetic plugin used by examples/plugins_demo.py"

            [project.entry-points."avo.tools"]
            synth = "{name}:SYNTH"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return pkg


def main() -> int:
    print(f"Demo PLUGIN_ROOT: {DEMO_ROOT}")
    print()

    print("--- Step 1: discover already-installed plugins ---")
    discovered = discover_all()
    by_group: dict[str, list[str]] = {group: [] for group in ALL_GROUPS}
    for entry in discovered:
        by_group[entry.group].append(entry.name)
    for group, names in by_group.items():
        print(f"  {group}: {names or '(none)'}")
    print(f"  total entries: {len(discovered)}")
    print()

    # Step 2: build a synthetic plugin in another tempdir and "install" it.
    print("--- Step 2: install a synthetic plugin from a local path ---")
    with tempfile.TemporaryDirectory() as source_tmp:
        source_path = _build_synthetic_plugin(Path(source_tmp), "avo_demo_plugin")
        installed = cli_plugins.install(str(source_path), name="avo-demo-plugin")
        print(f"  installed: {installed.name} -> {installed.path}")
        print(f"  description: {installed.description or '(none)'}")
    print()

    print("--- Step 3: list installed plugins ---")
    for plugin in cli_plugins.list_installed():
        print(f"  {plugin.name}: source={plugin.source} editable={plugin.editable}")
    print()

    print("--- Step 4: discover entries by group ---")
    tools_after = sorted(p.name for p in cli_plugins.list_installed())
    print(f"  installed plugin names: {tools_after}")
    synth_entries = [entry.name for entry in discover_all() if entry.package == "avo_demo_plugin"]
    print(f"  entry-point discovery for synthetic package: {synth_entries}")
    print()

    print("--- Step 5: remove the plugin ---")
    cli_plugins.remove("avo-demo-plugin")
    remaining = cli_plugins.list_installed()
    print(f"  installed after remove: {[p.name for p in remaining]}")
    print()

    # The synthetic plugin also declares an `avo.tools` entry-point. We
    # can show the `discover` API without going through pip by inspecting
    # the package metadata directly.
    from avo.plugins import discover as _discover

    print("--- Step 6: TOOL_GROUP discovery ---")
    print(f"  registered tool names: {sorted(e.name for e in _discover(TOOL_GROUP))}")
    print(f"  PROVIDER_GROUP names:  {sorted(e.name for e in _discover(PROVIDER_GROUP))}")

    return 0


if __name__ == "__main__":
    try:
        rc = main()
    finally:
        shutil.rmtree(DEMO_ROOT, ignore_errors=True)
    sys.exit(rc)
