"""Tests for ``avo.deprecation``."""

from __future__ import annotations

import pytest

from avo.deprecation import DeprecatedSymbol, deprecated, deprecation_index


def test_deprecated_emits_warning_and_forwards_result() -> None:
    @deprecated(since="0.2.0", removal="0.4.0", replacement="new_thing")
    def old() -> str:
        return "ok"

    with pytest.warns(DeprecationWarning, match="old is deprecated since 0.2.0"):
        assert old() == "ok"


def test_deprecated_records_entry_in_index() -> None:
    @deprecated(since="0.2.0", removal="0.4.0")
    def another() -> None:
        return None

    index = deprecation_index()
    assert any(
        entry["name"] == "test_deprecated_records_entry_in_index.<locals>.another"
        for entry in index
    )


def test_deprecated_strict_raises() -> None:
    @deprecated(since="0.2.0", removal="0.4.0", strict=True)
    def banned() -> None:
        return None

    with pytest.raises(DeprecatedSymbol, match="banned"):
        banned()


def test_deprecated_rejects_identical_versions() -> None:
    with pytest.raises(ValueError, match="must differ"):
        deprecated(since="0.1.0", removal="0.1.0")


def test_deprecated_rejects_empty_versions() -> None:
    with pytest.raises(ValueError, match="required"):
        deprecated(since="", removal="0.4.0")


def test_deprecated_preserves_metadata() -> None:
    @deprecated(since="0.2.0", removal="0.4.0")
    def sample() -> str:
        """Sample docstring."""
        return "x"

    assert sample.__name__ == "sample"
    assert "Sample docstring." in (sample.__doc__ or "")


def test_deprecation_index_returns_copy() -> None:
    snap = deprecation_index()
    snap.append({"name": "fake"})
    assert not any(entry["name"] == "fake" for entry in deprecation_index())


def test_deprecated_without_replacement() -> None:
    @deprecated(since="0.2.0", removal="0.4.0")
    def lone() -> None:
        return None

    with pytest.warns(DeprecationWarning) as records:
        lone()
    assert "instead" not in str(records[0].message)
