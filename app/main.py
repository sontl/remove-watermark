from __future__ import annotations

import asyncio
import io
import zipfile
from functools import lru_cache
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from .models import (
    WatermarkRemovalRequest,
    WatermarkRemovalResponse,
    WatermarkedImageResult,
    WatermarkRegion,
)
from .services import (
    LamaWatermarkRemover,
    WatermarkRemovalError,
    image_to_base64,
    image_to_bytes,
    remove_watermark_from_url,
)


app = FastAPI(title="Watermark Removal API", version="1.0.0")


@lru_cache(maxsize=2)
def _remover_for_device(device: str) -> LamaWatermarkRemover:
    return LamaWatermarkRemover(device=device)


@app.get("/healthz", summary="Health check")
async def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/v1/remove-watermark",
    response_model=WatermarkRemovalResponse,
    summary="Remove fixed-position watermark from remote images",
)
async def remove_watermark(request: WatermarkRemovalRequest) -> WatermarkRemovalResponse:
    remover = _remover_for_device(request.device)
    urls = [str(url) for url in request.images]
    region: WatermarkRegion = request.watermark

    tasks = [remove_watermark_from_url(url, region, remover) for url in urls]

    try:
        cleaned_images = await asyncio.gather(*tasks)
    except WatermarkRemovalError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if request.response_format == "base64":
        results = [
            WatermarkedImageResult(source_url=url, cleaned_image_base64=image_to_base64(image))
            for url, image in zip(urls, cleaned_images)
        ]
        return WatermarkRemovalResponse(results=results)

    if len(cleaned_images) == 1:
        image_bytes = image_to_bytes(cleaned_images[0])
        filename = "cleaned.png"
        return StreamingResponse(
            io.BytesIO(image_bytes),
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for index, image in enumerate(cleaned_images, start=1):
            archive.writestr(f"cleaned_{index}.png", image_to_bytes(image))
    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=cleaned_images.zip"},
    )
