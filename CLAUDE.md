# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PaddleOCR Fusion v3** is an API REST layer built on top of Paco's PaddleOCR 3.x project. It maintains 100% of Paco's processing logic while adding professional REST endpoints for easier integration.

## Architecture

```
┌─────────────────────────────────────────────┐
│         API REST Layer (Added)              │
│  Lines ~1895-2895                           │
│  ┌──────┬──────┬────────┬─────────┬────┐   │
│  │  /   │/stats│/process│/analyze │... │   │
│  └──────┴──────┴────────┴─────────┴────┘   │
│                  ↓                          │
├─────────────────────────────────────────────┤
│   Core Processing (Paco's Base)             │
│  Lines 1-~1894                              │
│  • PaddleOCR 3.x via PaddleX               │
│  • OpenCV preprocessing                     │
│  • Perspective/orientation/deskew correction│
│  • Multi-page PDF processing                │
│  • n8n integration (/ocr endpoint)          │
└─────────────────────────────────────────────┘
```

## Build and Run Commands

```bash
# Build and start
docker-compose build && docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Full rebuild (no cache)
docker-compose down && docker-compose build --no-cache && docker-compose up -d

# Enter container shell
docker exec -it paddlepaddle-cpu bash
```

## Testing Endpoints

```bash
# Health check
curl http://localhost:8503/health | jq

# Statistics
curl http://localhost:8503/stats | jq

# Process file - Normal format (plain text)
curl -X POST http://localhost:8503/process -F "file=@document.pdf" -F "format=normal"

# Process file - Layout format (spatial reconstruction using bounding boxes)
curl -X POST http://localhost:8503/process -F "file=@invoice.pdf" -F "format=layout"

# Detailed analysis
curl -X POST http://localhost:8503/analyze -F "file=@document.pdf" | jq -r '.ultra_analysis'

# Original n8n endpoint (expects file in /home/n8n/in/)
curl -X POST http://localhost:8503/ocr -F "filename=/home/n8n/in/document.pdf"
```

## Code Structure (app.py - ~2895 lines)

### Paco's Base (DO NOT MODIFY unless fixing critical bugs)
- Lines 1-~1894: Core OCR processing
- `OPENCV_CONFIG`, `ROTATION_CONFIG`, `OCR_CONFIG` - Configuration from ENV
- `doc_preprocessor` - Orientation classification model
- `ocr_instance` - PaddleOCR instance (lazy loaded)
- `init_docpreprocessor()`, `init_ocr()` - Model initialization
- `proc_pdf_ocr()`, `create_spdf()` - PDF/image OCR processing
- `/health`, `/ocr` endpoints

### API REST Layer (Safe to modify)
- Lines ~1895-2895: WebComunica API layer
- `server_stats` - Request statistics tracking
- `format_text_with_layout()` - Spatial text reconstruction using coordinates
- `dashboard()` - Web UI with file upload
- `/stats`, `/process`, `/analyze` endpoints
- `/api/history`, `/api/history/clear` - OCR history management

## Output Formats

### Normal (default)
Plain text extraction, lines joined with newlines.

### Layout
Spatial text reconstruction using bounding box coordinates from OCR. Attempts to preserve visual structure (columns, tables) similar to LLMWhisperer approach.

**Known Issue:** Intermittent `std::exception` errors can cause Layout mode to fail. See `INVESTIGATION_NOTES.md` for debugging details. The error is related to the global `ocr_instance` potentially having concurrency issues.

## Key Configuration (docker-compose.yml)

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_PORT` | 8503 | Server port |
| `OPENCV_HSV_LOWER_V` | 140 | Document detection threshold |
| `OPENCV_INNER_SCALE_FACTOR` | 1.06 | Document crop scaling |
| `ROTATION_MIN_CONFIDENCE` | 0.7 | Orientation confidence threshold |
| `OCR_TEXT_DET_LIMIT_SIDE_LEN` | 960 | Max image dimension for detection |

## Important Volumes

| Path | Purpose |
|------|---------|
| `/home/n8n` | n8n integration (input/output files) |
| `/home/n8n/.paddlex` | PaddleX model cache |
| `/home/n8n/.paddleocr` | PaddleOCR model cache |
| `/app/dictionaries` | OCR correction dictionaries (custom) |

## OCR Dictionary System

The dictionary system automatically corrects common OCR errors in Spanish documents.

### Dictionary Files
- **Base Dictionary**: 60+ corrections built into `app.py` for cities, fiscal terms, common concepts
- **Custom Dictionary**: User-added corrections, persisted in `/app/dictionaries/custom_corrections.json`

### Dictionary API Endpoints

```bash
# Get all dictionaries
curl http://localhost:8503/api/dictionary | jq

# Add a correction
curl -X POST http://localhost:8503/api/dictionary/add \
  -H "Content-Type: application/json" \
  -d '{"wrong": "Cadlz", "correct": "Cádiz", "dictionary": "custom"}'

# Remove a correction
curl -X POST http://localhost:8503/api/dictionary/remove \
  -H "Content-Type: application/json" \
  -d '{"wrong": "Cadlz", "dictionary": "custom"}'

# Test corrections on text
curl -X POST http://localhost:8503/api/dictionary/test \
  -H "Content-Type: application/json" \
  -d '{"text": "Factura de Cad1z con TOTA1: 55:23 EUR"}'

# Analyze document for potential OCR errors
curl -X POST http://localhost:8503/api/dictionary/analyze \
  -F "file=@invoice.pdf"

# Reload dictionaries from files
curl -X POST http://localhost:8503/api/dictionary/reload
```

### Price Format Correction
The dictionary system also corrects price formats automatically:
- `55:23` → `55,23` (colon to comma in decimal numbers)

## Adding New Endpoints

Add after line ~2848 (before `start_model_loading()`). Pattern:

```python
@app.route('/your-endpoint', methods=['POST'])
def your_endpoint():
    global server_stats
    start_time = time.time()
    server_stats['total_requests'] += 1
    temp_file_path = None

    try:
        # 1. Validate and save file to /home/n8n/in/
        # 2. Call ocr() internally via test_request_context
        # 3. Cleanup temp files
        # 4. Return formatted response with processing_time
    except Exception as e:
        server_stats['failed_requests'] += 1
        # Cleanup and return error
```

## CPU Requirements

Requires CPU with **AVX/AVX2** support. Will fail with `Illegal instruction (core dumped)` on:
- Basic KVM virtualized VPS
- Old CPUs without AVX instructions

## Development Notes

- Lazy loading: Models load in background thread after server starts
- First OCR request may take ~2 minutes (model download + initialization)
- Subsequent requests: 1-2 seconds
- All new endpoints should call `/ocr` internally to avoid duplicating processing logic
- Original `/ocr` endpoint must remain unchanged for n8n compatibility
