# CONTEXTO DE SESIÓN - PaddleOCR Perfection

**LEER ESTE ARCHIVO AL INICIO DE CADA INTERACCIÓN**

## Estado Actual del Proyecto

### Versión: 4.0 (Inteligencia Documental Completa)
### Próximo objetivo: Mejoras continuas según feedback

**ÚLTIMA ACTUALIZACIÓN:** 2025-12-07 - v4.0 completada con todas las fases del roadmap.

---

## PROGRESO COMPLETADO (v4.0)

### Fase 1: PP-Structure Instalado ✅
- `paddlex[ocr]==3.3.10` instalado
- `scikit-learn` instalado (para clustering DBSCAN)
- Pipelines funcionando: `table_recognition`, `layout_parsing`

### Fase 2-3: Endpoints Creados ✅

**`/structure`** - Análisis estructural de documentos
- Detecta regiones (tablas, texto, imágenes, charts)
- Extrae tablas como HTML usando SLANet
- Proporciona coordenadas de cada región

**`/extract`** - Extracción inteligente de campos
- Detecta tipo de documento (invoice, receipt, etc.)
- Extrae campos: vendor, NIF, invoice_number, date, total, tax_base, tax_rate, tax_amount
- NUEVO: customer_name, customer_nif, line_items
- Calcula confidence score

### Fase 4: DBSCAN Clustering ✅ (NUEVO)
- `detect_columns_dbscan()` implementado
- Detección inteligente de columnas en documentos
- `format_text_with_layout()` mejorado con clustering

### Fase 5: KIE Mejorado ✅
- Patrones regex para IVA mejorados (5 estrategias)
- Extracción correcta de: tax_base, tax_rate, tax_amount
- NUEVO: LINE_ITEMS - conceptos/productos de factura
- NUEVO: CUSTOMER_NAME, CUSTOMER_NIF
- Corrección OCR para NIFs (R→B, I→1, O→0)

### Fase 6: Detección Híbrida ✅
- PDFs vectoriales → `pdftotext -layout` (preserva estructura)
- PDFs escaneados/imágenes → OCR con PaddleOCR
- Selección automática del mejor método
- Campo `extraction_method` en respuesta

---

## RESULTADOS DE PRUEBAS v4.0

### ticket.pdf (ESCANEADO) - extraction_method: ocr ✅
```json
{"vendor": "E.S.CUATRO OLIVOS S.L.", "vendor_nif": "B11368479",
 "invoice_number": "2025003047",
 "total": 66.83, "tax_base": 55.23, "tax_rate": 21.0, "tax_amount": 11.6,
 "confidence": 1.0}
```

### Factura noviembre.pdf (OLIVENET) - extraction_method: pdftotext_layout ✅
```json
{"vendor": "OLIVENET NETWORK S.L.U.", "vendor_nif": "B93340198",
 "invoice_number": "ON2025-584267",
 "customer_name": "Juan Jose Sanchez Bernal", "customer_nif": "78971220F",
 "total": 94.74, "tax_base": 78.3, "tax_rate": 21.0, "tax_amount": 16.44,
 "line_items": 7, "confidence": 0.9}
```

### Factura_VFR25087570.pdf (VODAFONE/DMI) - extraction_method: pdftotext_layout ✅
```json
{"vendor": "DMI Computer S.A.", "vendor_nif": "A79522702",
 "invoice_number": "VFR25087570",
 "customer_nif": "78971220F",
 "total": 172.12, "tax_base": 142.25, "tax_rate": 21.0, "tax_amount": 29.87,
 "line_items": 5, "confidence": 1.0}
```

---

## Ubicaciones Clave en app.py (~5900 líneas)

| Función | Línea (aprox) | Descripción |
|---------|---------------|-------------|
| `detect_columns_dbscan()` | ~2417 | NUEVO - Clustering de columnas |
| `format_text_with_layout()` | ~2450 | Reconstrucción espacial mejorada |
| `/process` endpoint | ~2750 | Endpoint principal |
| `/ocr` endpoint | 1913 | Original de Paco (NO TOCAR) |
| `OCR_CORRECTIONS_BASE` | 2084 | Diccionario OCR (407 correcciones) |
| `init_pp_structure_pipelines()` | ~4983 | Inicializa PP-Structure |
| `/structure` endpoint | ~5012 | Análisis estructural |
| `extract_invoice_fields()` | ~5545 | KIE - extracción de campos |
| `/extract` endpoint | ~5400 | Extracción inteligente |

---

## Fases del Roadmap - TODAS COMPLETADAS

| Fase | Descripción | Estado |
|------|-------------|--------|
| 1 | Instalar dependencias PP-Structure | ✅ COMPLETADO |
| 2 | Layout Analysis Pipeline | ✅ COMPLETADO |
| 3 | Table Recognition (SLANet) | ✅ COMPLETADO |
| 4 | Mejorar format_text_with_layout (DBSCAN) | ✅ COMPLETADO |
| 5 | Key Information Extraction | ✅ COMPLETADO |
| 6 | Endpoint /extract estructurado | ✅ COMPLETADO |

---

## Comandos de Test Rápido

```bash
# Health check
curl http://localhost:8503/health

# Test /extract (extracción de campos con KIE)
curl -X POST http://localhost:8503/extract -F "file=@ticket.pdf"

# Test /structure (análisis estructural con HTML de tablas)
curl -X POST http://localhost:8503/structure -F "file=@factura.pdf"

# Test /process layout (OCR perfecto para IA)
curl -X POST http://localhost:8503/process -F "file=@factura.pdf" -F "format=layout"

# Ver logs
docker logs --tail 50 paddlepaddle-cpu

# Rebuild completo
docker-compose down && docker-compose build --no-cache && docker-compose up -d
```

---

## Dependencias en Dockerfile

```dockerfile
pip install "paddlex[ocr]==3.3.10" scikit-learn
```

---

## NOTAS IMPORTANTES

1. **NO MODIFICAR** el endpoint `/ocr` (compatibilidad n8n)
2. **El valor real** está en el modo `layout` para pasar a una IA
3. **KIE es un bonus** - los regex nunca cubrirán todos los formatos
4. **Para OCR perfecto**: usar `/process` con `format=layout`
5. **Para extracción estructurada**: usar `/extract` (best-effort)

---

## Arquitectura Final

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENTRADA: PDF/Imagen                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Detección Híbrida                                              │
│  - PDF vectorial → pdftotext -layout                            │
│  - PDF escaneado → PaddleOCR + DBSCAN clustering                │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│  /process (layout)      │     │  /extract (KIE)         │
│  → Texto estructurado   │     │  → JSON con campos      │
│  → Para pasar a IA      │     │  → Best-effort regex    │
└─────────────────────────┘     └─────────────────────────┘
```
