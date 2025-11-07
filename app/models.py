from __future__ import annotations

import os
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, HttpUrl, model_validator


class WatermarkRegion(BaseModel):
    width: int = Field(..., gt=0, description="Width of the watermark region in pixels")
    height: int = Field(..., gt=0, description="Height of the watermark region in pixels")
    offset_x: int = Field(0, ge=0, description="Horizontal offset from the right edge in pixels")
    offset_y: int = Field(0, ge=0, description="Vertical offset from the bottom edge in pixels")


def _default_device() -> str:
    return os.getenv("LAMA_DEVICE", "cpu")


class WatermarkRemovalRequest(BaseModel):
    images: Union[HttpUrl, List[HttpUrl]]
    watermark: WatermarkRegion = Field(
        default_factory=lambda: WatermarkRegion(width=120, height=120, offset_x=0, offset_y=0)
    )
    device: Literal["cpu", "cuda"] = Field(
        default_factory=_default_device,
        description="Device to run the inpainting model",
    )
    response_format: Literal["base64", "file"] = Field(
        "base64", description="Return base64 JSON payload or downloadable file(s)"
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_images(cls, values):
        images = values.get("images")
        if isinstance(images, list):
            if not images:
                raise ValueError("images list must not be empty")
            values["images"] = images
        elif images is None:
            raise ValueError("images field is required")
        else:
            values["images"] = [images]
        return values


class WatermarkedImageResult(BaseModel):
    source_url: HttpUrl
    cleaned_image_base64: str


class WatermarkRemovalResponse(BaseModel):
    results: List[WatermarkedImageResult]
