"""Tests for typed content blocks and per-provider translators."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Content-block models
# ---------------------------------------------------------------------------


def test_image_block_from_path_loads_and_encodes(tmp_path: Path) -> None:
    from avo.content_blocks import ImageBlock

    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
    img = tmp_path / "tiny.png"
    img.write_bytes(png_bytes)
    block = ImageBlock.from_path(img)
    assert block.media_type == "image/png"
    assert base64.b64decode(block.data) == png_bytes


def test_image_block_from_path_rejects_missing_file(tmp_path: Path) -> None:
    from avo.content_blocks import ImageBlock

    with pytest.raises(FileNotFoundError):
        ImageBlock.from_path(tmp_path / "nope.png")


def test_image_block_from_path_rejects_unknown_media_type(tmp_path: Path) -> None:
    from avo.content_blocks import ImageBlock

    img = tmp_path / "thing.png"
    img.write_bytes(b"not an image but extension is png")
    with pytest.raises(ValueError, match="unsupported image media_type"):
        ImageBlock.from_path(img, media_type="application/pdf")


def test_image_block_as_anthropic_source_shape() -> None:
    from avo.content_blocks import ImageBlock

    block = ImageBlock(media_type="image/png", data="AAAA")
    assert block.as_anthropic_source() == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
    }


def test_message_to_dict_serializes_blocks() -> None:
    from avo.content_blocks import ImageBlock, Message, TextBlock

    msg = Message(
        role="user",
        content=[
            TextBlock(text="what is this?"),
            ImageBlock(media_type="image/png", data="AAAA"),
        ],
    )
    dumped = msg.to_dict()
    assert dumped == {
        "role": "user",
        "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image", "media_type": "image/png", "data": "AAAA"},
        ],
    }


def test_message_rejects_unknown_role() -> None:
    from pydantic import ValidationError

    from avo.content_blocks import Message, TextBlock

    with pytest.raises(ValidationError):
        Message(role="narrator", content=[TextBlock(text="x")])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Provider translators
# ---------------------------------------------------------------------------


def test_anthropic_passes_image_block_through_verbatim() -> None:
    from avo.providers.anthropic import _anthropic_message

    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "see attached"},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
            },
        ],
    }
    out = _anthropic_message(message)
    assert out["role"] == "user"
    assert out["content"] == message["content"]


def test_openai_translates_image_block_to_data_url() -> None:
    from avo.providers.http_common import _openai_message

    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "see attached"},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": "BBBB"},
            },
        ],
    }
    out = _openai_message(message)
    assert out == {
        "role": "user",
        "content": [
            {"type": "text", "text": "see attached"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,BBBB"},
            },
        ],
    }


def test_openai_passes_through_plain_string_content() -> None:
    from avo.providers.http_common import _openai_message

    out = _openai_message({"role": "user", "content": "hello"})
    assert out == {"role": "user", "content": "hello"}


def test_ollama_extracts_images_into_sibling_array() -> None:
    from avo.providers.ollama import _ollama_message

    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "describe"},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
            },
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "BBBB"},
            },
        ],
    }
    out = _ollama_message(message)
    assert out["role"] == "user"
    assert out["content"] == "describe"
    assert out["images"] == ["AAAA", "BBBB"]


def test_ollama_text_only_block_list_has_no_images_sibling() -> None:
    from avo.providers.ollama import _ollama_message

    out = _ollama_message({"role": "user", "content": [{"type": "text", "text": "no images here"}]})
    assert out == {"role": "user", "content": "no images here"}
    assert "images" not in out


def test_ollama_passes_through_plain_string_content() -> None:
    from avo.providers.ollama import _ollama_message

    out = _ollama_message({"role": "user", "content": "hi"})
    assert out == {"role": "user", "content": "hi"}
