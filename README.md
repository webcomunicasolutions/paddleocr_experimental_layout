# PaddleOCR Fusion v3.1 - Experimental Layout Edition

[![Version](https://img.shields.io/badge/version-3.1.0-blue.svg)](https://github.com/webcomunicasolutions/paddleocr-experimental-layout)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-3.x-orange.svg)](https://github.com/PaddlePaddle/PaddleOCR)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **IMPORTANTE: REQUISITOS DE CPU**
>
> Este proyecto requiere una CPU con soporte **AVX/AVX2**. NO funcionará en:
> - VPS con "Common KVM processor" (virtualizados básicos)
> - CPUs antiguas sin instrucciones AVX
>
> **Error típico si la CPU no es compatible:** `Illegal instruction (core dumped)`

## Descripción

**PaddleOCR Fusion v3.1** es una API REST profesional construida sobre PaddleOCR 3.x con:

- **Sistema de Diccionarios OCR** - Correcciones automáticas para español (407+ correcciones)
- **Mejora con IA (Gemini Vision)** - Corrección inteligente comparando imagen vs texto OCR
- **Modo Layout Experimental** - Reconstrucción espacial de documentos
- **Dashboard Web Interactivo** - 6 tabs para gestión completa
- **Auto-recuperación de errores** - Reintentos automáticos en std::exception

## Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                    API REST Layer                           │
│  ┌────────┬────────┬─────────┬──────────┬────────────────┐│
│  │   /    │ /stats │/process │/analyze  │ /api/dictionary ││
│  │        │        │         │          │ /api/config     ││
│  └────────┴────────┴─────────┴──────────┴────────────────┘│
│                          ↓                                  │
│  ┌────────────────────────────────────────────────────────┐│
│  │   Sistema de Diccionarios OCR                          ││
│  │   • BASE: 407 correcciones para español               ││
│  │   • CUSTOM: Personalizadas (persistentes)             ││
│  │   • Regex: Patrones (55:23 → 55,23)                   ││
│  └────────────────────────────────────────────────────────┘│
│                          ↓                                  │
│  ┌────────────────────────────────────────────────────────┐│
│  │   Core Processing (Base de Paco)                       ││
│  │   • PaddleOCR 3.x (PP-OCRv3)                          ││
│  │   • Preprocesamiento OpenCV                           ││
│  │   • Corrección perspectiva/orientación/inclinación    ││
│  │   • Procesamiento multi-página                        ││
│  └────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Instalación Rápida

```bash
# Clonar repositorio
git clone https://github.com/webcomunicasolutions/paddleocr-experimental-layout.git
cd paddleocr-experimental-layout

# Construir e iniciar
docker-compose build
docker-compose up -d

# Verificar estado
curl http://localhost:8503/health

# Abrir Dashboard
firefox http://localhost:8503/
```

## Endpoints API

### Core Endpoints

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/` | GET | Dashboard web interactivo |
| `/health` | GET | Estado del servidor |
| `/stats` | GET | Estadísticas detalladas |
| `/process` | POST | OCR con formatos (normal/layout) |
| `/analyze` | POST | Análisis detallado |
| `/ocr` | POST | Endpoint original n8n |

### Dictionary API

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/api/dictionary` | GET | Obtener todos los diccionarios |
| `/api/dictionary/add` | POST | Añadir corrección |
| `/api/dictionary/remove` | POST | Eliminar corrección |
| `/api/dictionary/reload` | POST | Recargar desde archivos |
| `/api/dictionary/test` | POST | Probar correcciones en texto |
| `/api/dictionary/analyze` | POST | Analizar documento para errores |
| `/api/dictionary/improve` | POST | Corregir con Gemini Vision |
| `/api/dictionary/import` | POST | Importar diccionarios externos |
| `/api/dictionary/available` | GET | Listar diccionarios predefinidos |

### Config API

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/api/config/apikey` | GET | Verificar si hay API key configurada |
| `/api/config/apikey` | POST | Guardar API key de Gemini |
| `/api/config/apikey/test` | POST | Probar si la API key funciona |

## Dashboard Web

El dashboard incluye 6 tabs:

1. **Estado** - Estadísticas del servidor en tiempo real
2. **Test OCR** - Subir y procesar documentos (Normal/Layout)
3. **Diccionario** - Ver y gestionar correcciones OCR
4. **Mejorar Diccionario** - Flujo con IA para añadir correcciones
5. **Configuración** - Gestión de API Key de Gemini
6. **Ayuda** - Documentación y ejemplos

## Sistema de Diccionarios

### Diccionario BASE (407 correcciones)

Incluye correcciones para:

- **Ciudades españolas**: Cádiz, Málaga, Córdoba, Sevilla, Almería, Jaén...
- **Términos fiscales**: IVA, NIF, CIF, IRPF, Total, Base Imponible...
- **Confusiones OCR**: 1↔l↔I, 0↔O, rn↔m, 5↔S, 8↔B...
- **Productos/servicios**: Gasóleo, Gasolina, Electricidad, Suministro...
- **Formas de pago**: Efectivo, Tarjeta, Transferencia...

### Diccionario CUSTOM (persistente)

- Se guarda en volumen Docker `/app/dictionaries`
- Sobrevive a reinicios del contenedor
- Editable desde el Dashboard

### Patrones Regex

- `55:23` → `55,23` (precios con dos puntos)
- Otros patrones configurables

## Mejora con IA (Gemini Vision)

### Flujo de trabajo

1. **Subir documento** (PDF/imagen)
2. **Ejecutar OCR** con PaddleOCR
3. **Corregir con Gemini** - envía imagen + texto OCR
4. **Comparar diferencias** - extrae correcciones automáticamente
5. **Añadir al diccionario** - selecciona y guarda

### Configuración

1. Obtener API Key gratis en: https://aistudio.google.com/app/apikey
2. Ir a Dashboard → Tab "Configuración"
3. Introducir API Key y guardar

## Volúmenes Docker

| Volumen | Ruta Contenedor | Propósito |
|---------|-----------------|-----------|
| `/home/n8n` | `/home/n8n` | Integración n8n, archivos I/O |
| `paddlex-models` | `/home/n8n/.paddlex` | Cache modelos PaddleX |
| `paddleocr-models` | `/home/n8n/.paddleocr` | Cache modelos PaddleOCR |
| `ocr-dictionaries` | `/app/dictionaries` | Diccionarios personalizados |
| `ocr-config` | `/app/config` | API keys y configuración |

## Variables de Entorno

```env
# Flask
FLASK_ENV=production
FLASK_PORT=8503
TZ=Europe/Madrid

# OpenCV
OPENCV_HSV_LOWER_H=0
OPENCV_HSV_LOWER_S=0
OPENCV_HSV_LOWER_V=140
OPENCV_HSV_UPPER_H=180
OPENCV_HSV_UPPER_S=60
OPENCV_HSV_UPPER_V=255
OPENCV_MIN_AREA_PERCENT=0.05
OPENCV_EPSILON_FACTOR=0.01
OPENCV_ERODE_ITERATIONS=1
OPENCV_DILATE_ITERATIONS=2
OPENCV_MIN_WIDTH=300
OPENCV_MIN_HEIGHT=400
OPENCV_EROSION_PERCENT=0.085
OPENCV_INNER_SCALE_FACTOR=1.06

# Rotación
ROTATION_MIN_CONFIDENCE=0.7
ROTATION_MIN_SKEW_ANGLE=0.2

# OCR
OCR_VERSION=PP-OCRv3
OCR_LANG=es
OCR_USE_DOC_ORIENTATION=false
OCR_USE_DOC_UNWARPING=false
OCR_USE_TEXTLINE_ORIENTATION=false
OCR_TEXT_DET_THRESH=0.1
OCR_TEXT_DET_BOX_THRESH=0.4
OCR_TEXT_DET_UNCLIP_RATIO=1.5
OCR_TEXT_DET_LIMIT_SIDE_LEN=960
OCR_TEXT_DET_LIMIT_TYPE=min
OCR_TEXT_RECOGNITION_BATCH_SIZE=6

# Optimización CPU
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
FLAGS_allocator_strategy=auto_growth
FLAGS_fraction_of_gpu_memory_to_use=0
CUDA_VISIBLE_DEVICES=""
```

## Ejemplos de Uso

### OCR Normal

```bash
curl -X POST http://localhost:8503/process \
  -F "file=@factura.pdf" \
  -F "format=normal"
```

### OCR con Layout

```bash
curl -X POST http://localhost:8503/process \
  -F "file=@factura.pdf" \
  -F "format=layout"
```

### Importar diccionario predefinido

```bash
curl -X POST http://localhost:8503/api/dictionary/import \
  -H "Content-Type: application/json" \
  -d '{"dictionary": "fiscal_terms"}'
```

### Diccionarios predefinidos disponibles

- `spanish_cities` - Ciudades españolas con tildes
- `fiscal_terms` - IVA, NIF, CIF y términos fiscales
- `common_ocr_errors` - Confusiones típicas OCR
- `products_services` - Productos y servicios comunes

## Solución de Problemas

### Error: `Illegal instruction (core dumped)`

**Causa:** CPU sin soporte AVX/AVX2.

**Verificar:**
```bash
cat /proc/cpuinfo | grep avx
```

**Solución:** Usar servidor con CPU dedicada (Hetzner CPX/CCX, DigitalOcean Dedicated, etc.)

### Primera petición lenta (~2 minutos)

**Normal.** Los modelos se cargan en la primera petición. Las siguientes serán rápidas (~1-2s).

### Error std::exception en OCR

El sistema tiene auto-recuperación con 5 reintentos automáticos.

## Changelog

### v3.1.0 (2025-12-07)

- **Nuevo:** Sistema de diccionarios OCR (BASE + CUSTOM)
- **Nuevo:** 407 correcciones para español
- **Nuevo:** Integración Gemini Vision para mejora de diccionarios
- **Nuevo:** API REST completa para gestión de diccionarios
- **Nuevo:** Tab "Configuración" para API keys
- **Nuevo:** Tab "Mejorar Diccionario" con flujo IA
- **Nuevo:** Importación de diccionarios externos
- **Nuevo:** Volumen `ocr-config` para persistir configuración
- **Mejorado:** Auto-recuperación de errores std::exception

### v3.0.0 (2025-12-06)

- **Nuevo:** Modo Layout experimental con coordenadas
- **Nuevo:** Reconstrucción espacial de texto
- **Mejorado:** Serialización JSON de numpy arrays

## Licencia

MIT License - Ver [LICENSE](LICENSE)

## Créditos

- **Base:** Proyecto de Paco (PaddleOCR 3.x + preprocesamiento)
- **API REST:** WebComunica
- **PaddleOCR:** PaddlePaddle Team
- **Desarrollo:** Claude Code

---

**Made with Claude Code by WebComunica**
