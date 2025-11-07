from __future__ import annotations

import asyncio
import base64
import io
import threading
from typing import Callable, Optional

import httpx
import numpy as np
from PIL import Image, ImageDraw

from .models import WatermarkRegion


class WatermarkRemovalError(Exception):
    pass


def _default_model_factory(device: str):
    try:
        from simple_lama_inpainting import SimpleLama
    except ImportError as exc:
        raise WatermarkRemovalError(
            "simple-lama-inpainting is not installed. Install it via pip before running the service."
        ) from exc

    return SimpleLama(device=device)


def create_mask(image_size: tuple[int, int], region: WatermarkRegion) -> Image.Image:
    width, height = image_size
    rect_width = min(region.width, width)
    rect_height = min(region.height, height)
    right = width - region.offset_x
    bottom = height - region.offset_y
    left = max(0, right - rect_width)
    top = max(0, bottom - rect_height)

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([(left, top), (right, bottom)], fill=255)
    return mask


class LamaWatermarkRemover:
    def __init__(
        self,
        device: str = "cpu",
        model_factory: Optional[Callable[[str], object]] = None,
    ) -> None:
        self._device = device
        self._model_factory = model_factory or _default_model_factory
        self._model: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
        self._lock = threading.Lock()

    def _ensure_model(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._model_factory(self._device)
        return self._model

    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        model = self._ensure_model()
        np_image = np.array(image.convert("RGB"))
        np_mask = np.array(mask.convert("L"))
        result = model(np_image, np_mask)
        if isinstance(result, Image.Image):
            return result
        return Image.fromarray(result)


async def fetch_image(url: str, timeout: float = 10.0) -> Image.Image:
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


async def remove_watermark_from_url(
    url: str,
    region: WatermarkRegion,
    remover: LamaWatermarkRemover,
) -> Image.Image:
    try:
        image = await fetch_image(url)
    except Exception as exc:  # noqa: BLE001 - capture network/image errors
        raise WatermarkRemovalError(f"Failed to download image from {url}: {exc}") from exc

    mask = create_mask(image.size, region)

    loop = asyncio.get_event_loop()
    try:
        cleaned_image = await loop.run_in_executor(None, remover.inpaint, image, mask)
    except Exception as exc:  # noqa: BLE001 - propagate as domain error
        raise WatermarkRemovalError(f"Failed to inpaint image from {url}: {exc}") from exc

    return cleaned_image
