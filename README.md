# Watermark Removal API

FastAPI service that uses LaMa inpainting to remove a fixed bottom-right watermark from remote images.

## Prerequisites

- Python 3.11+
- (Optional) virtual environment tool such as `venv`

## Local Setup

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Run the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Visit `http://127.0.0.1:8000/docs` for the interactive OpenAPI UI.

## Docker (GPU)

Ensure the NVIDIA Container Toolkit is installed on the host.

```bash
docker compose up --build
```

The service listens on port `8000` (`http://127.0.0.1:8000`).

## Example Request

```bash
# Base64 response
curl -X POST http://127.0.0.1:8000/v1/remove-watermark \
  -H "Content-Type: application/json" \
  -d '{
        "images": ["https://example.com/photo.jpg"],
        "watermark": {"width": 120, "height": 120, "offset_x": 0, "offset_y": 0},
        "response_format": "base64"
      }'

# Download cleaned file directly (single image -> PNG, multiple -> ZIP)
curl -X POST http://127.0.0.1:8000/v1/remove-watermark \
  -H "Content-Type: application/json" \
  -d '{
        "images": ["https://example.com/photo.jpg"],
        "response_format": "file"
      }' \
  -o cleaned.png
```

When `response_format` is `base64`, cleaned images appear in `results[].cleaned_image_base64`; when `file`, the response is a downloadable PNG or ZIP archive.

## Tests

```bash
python3 -m pytest
```
