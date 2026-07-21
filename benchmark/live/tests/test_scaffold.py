import tomllib
from pathlib import Path


def test_live_benchmark_dependencies_are_optional() -> None:
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    optional = data["project"]["optional-dependencies"]["live-benchmark"]
    required = data["project"]["dependencies"]
    assert "httpx>=0.27" in optional
    assert "matplotlib>=3.8" in optional
    assert all(not item.startswith(("httpx", "matplotlib")) for item in required)
