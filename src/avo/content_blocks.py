"""Provider-neutral typed content blocks for multimodal messages.

Anthropic, OpenAI, and Ollama all accept image inputs but use different
on-the-wire shapes. This module defines one canonical typed model —
``TextBlock`` / ``ImageBlock`` — plus per-provider dumpers that map each
block to the dict shape its target API expects.

The intent is to give agent code a single, validated way to attach an
image to a turn (``Message(role="user", content=[text, image])``)
without writing provider-specific JSON for each call.
"""

from __future__ import annotations

import base64
import binascii
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

_ALLOWED_IMAGE_MEDIA_TYPES: tuple[str, ...] = (
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
)


class _StrictModel(BaseModel):
    """Strict base shared by content-block models."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class TextBlock(_StrictModel):
    """Plain-text content block."""

    type: Literal["text"] = "text"
    text: str = Field(min_length=1)


class ImageBlock(_StrictModel):
    """Base64-encoded image block.

    The provider dumpers translate ``data`` into the URL form (OpenAI)
    or sibling ``images`` array (Ollama) — this model stays
    representation-agnostic.
    """

    type: Literal["image"] = "image"
    media_type: str
    data: str  # base64 (no data: prefix)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        media_type: str | None = None,
    ) -> ImageBlock:
        """Load an image from disk and return a typed block.

        ``media_type`` defaults to the file extension when omitted; the
        caller can override to pin a MIME type explicitly.
        """

        file_path = Path(path).expanduser().resolve()
        if not file_path.is_file():
            raise FileNotFoundError(f"image not found: {file_path}")
        inferred = media_type or _media_type_for(file_path)
        if inferred not in _ALLOWED_IMAGE_MEDIA_TYPES:
            raise ValueError(
                f"unsupported image media_type {inferred!r}; "
                f"allowed: {', '.join(_ALLOWED_IMAGE_MEDIA_TYPES)}"
            )
        raw = file_path.read_bytes()
        try:
            encoded = base64.b64encode(raw).decode("ascii")
        except binascii.Error as exc:  # pragma: no cover - b64encode accepts all bytes
            raise ValueError(f"failed to base64-encode image at {file_path}: {exc}") from exc
        return cls(media_type=inferred, data=encoded)

    def as_anthropic_source(self) -> dict[str, Any]:
        """Render the block as an Anthropic ``image`` content block."""

        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": self.media_type,
                "data": self.data,
            },
        }


Block = Annotated[TextBlock | ImageBlock, Field(discriminator="type")]


class Message(_StrictModel):
    """A single chat message composed of typed content blocks."""

    role: Literal["system", "user", "assistant", "tool"]
    content: list[Block]

    def to_dict(self) -> dict[str, Any]:
        """Render the message as the canonical dict form.

        Mirrors what ``runtime.run`` already accepts, so messages built
        with this module drop straight into the existing provider paths
        with no runtime changes.
        """

        return {"role": self.role, "content": [block.model_dump() for block in self.content]}


def _media_type_for(path: Path) -> str:
    """Infer an image MIME type from the file extension."""

    suffix = path.suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(suffix, "image/png")


def dump_block_anthropic(block: Block) -> dict[str, Any]:
    """Render a single block as Anthropic Messages API content."""

    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    return block.as_anthropic_source()


def dump_block_openai(block: Block) -> dict[str, Any]:
    """Render a single block as an OpenAI Chat Completions content part."""

    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{block.media_type};base64,{block.data}"},
    }


def dump_block_ollama(block: Block) -> tuple[str, str | None]:
    """Render a single block as an Ollama ``(text, base64_or_none)`` pair."""

    if isinstance(block, TextBlock):
        return block.text, None
    return "", block.data


__all__ = [
    "Block",
    "ImageBlock",
    "Message",
    "TextBlock",
    "dump_block_anthropic",
    "dump_block_ollama",
    "dump_block_openai",
]
