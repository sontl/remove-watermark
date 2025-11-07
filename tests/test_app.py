from __future__ import annotations

import io
import zipfile
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.services import image_to_base64, image_to_bytes


client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_cache():
    # Ensure cached removers do not leak between tests
    from app.main import _remover_for_device

    _remover_for_device.cache_clear()
    yield
    _remover_for_device.cache_clear()


def test_remove_watermark_single_image(monkeypatch: pytest.MonkeyPatch) -> None:
    sample_image = Image.new("RGB", (2, 2), color=(255, 0, 0))

    async def fake_remove(url: str, region: Any, remover: Any) -> Image.Image:
        return sample_image

    monkeypatch.setattr("app.main.remove_watermark_from_url", fake_remove)

    response = client.post(
        "/v1/remove-watermark",
        json={"images": "https://example.com/image.jpg"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["results"][0]["source_url"] == "https://example.com/image.jpg"
    assert payload["results"][0]["cleaned_image_base64"] == image_to_base64(sample_image)


def test_remove_watermark_handles_service_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import WatermarkRemovalError

    async def fake_remove(url: str, region: Any, remover: Any) -> Image.Image:
        raise WatermarkRemovalError("boom")

    monkeypatch.setattr("app.main.remove_watermark_from_url", fake_remove)

    response = client.post(
        "/v1/remove-watermark",
        json={"images": ["https://example.com/a.jpg", "https://example.com/b.jpg"]},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "boom"


def test_remove_watermark_rejects_empty_list() -> None:
    response = client.post("/v1/remove-watermark", json={"images": []})
    assert response.status_code == 422


def test_remove_watermark_returns_file_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    sample_image = Image.new("RGB", (3, 3), color=(0, 255, 0))

    async def fake_remove(url: str, region: Any, remover: Any) -> Image.Image:
        return sample_image

    monkeypatch.setattr("app.main.remove_watermark_from_url", fake_remove)

    response = client.post(
        "/v1/remove-watermark",
        json={"images": "https://example.com/image.jpg", "response_format": "file"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert "attachment; filename=cleaned.png" in response.headers["content-disposition"]
    assert response.content == image_to_bytes(sample_image)


def test_remove_watermark_returns_zip_for_multiple_files(monkeypatch: pytest.MonkeyPatch) -> None:
    images = [
        Image.new("RGB", (4, 4), color=(0, 0, 255)),
        Image.new("RGB", (5, 5), color=(255, 255, 0)),
    ]
    iterator = iter(images)

    async def fake_remove(url: str, region: Any, remover: Any) -> Image.Image:
        return next(iterator)

    monkeypatch.setattr("app.main.remove_watermark_from_url", fake_remove)

    response = client.post(
        "/v1/remove-watermark",
        json={
            "images": ["https://example.com/a.jpg", "https://example.com/b.jpg"],
            "response_format": "file",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert "attachment; filename=cleaned_images.zip" in response.headers["content-disposition"]

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert sorted(archive.namelist()) == ["cleaned_1.png", "cleaned_2.png"]
        for idx, expected in enumerate(images, start=1):
            with archive.open(f"cleaned_{idx}.png") as member:
                data = member.read()
                assert data == image_to_bytes(expected)
