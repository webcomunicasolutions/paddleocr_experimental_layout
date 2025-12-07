# CONTEXTO DE SESIÓN - PaddleOCR Perfection

**LEER ESTE ARCHIVO AL INICIO DE CADA INTERACCIÓN**

## Estado Actual del Proyecto

### Versión: 3.3 (Detección Híbrida + KIE Mejorado)
### Objetivo: Llegar a v4.0 (Inteligencia Documental)

**ÚLTIMA ACTUALIZACIÓN:** Todos los campos fiscales se extraen correctamente de 3/4 facturas de prueba.

---

## PROGRESO COMPLETADO

### Fase 1: PP-Structure Instalado ✅
- `paddlex[ocr]==3.3.10` instalado
- `scikit-learn` instalado (para clustering futuro)
- Pipelines funcionando: `table_recognition`, `layout_parsing`

### Fase 2-3: Endpoints Creados ✅

**`/structure`** - Análisis estructural de documentos
- Detecta regiones (tablas, texto, imágenes, charts)
- Extrae tablas como HTML usando SLANet
- Proporciona coordenadas de cada región

**`/extract`** - Extracción inteligente de campos
- Detecta tipo de documento (invoice, receipt, etc.)
- Extrae campos: vendor, NIF, invoice_number, date, total, tax_base, tax_rate, tax_amount
- Calcula confidence score

### Fase 4-5: KIE Mejorado ✅
- Patrones regex para IVA mejorados (5 estrategias)
- Extracción correcta de: tax_base, tax_rate, tax_amount
- Verificación cruzada de valores fiscales
- Corrección OCR para NIFs (R→B, I→1, O→0)

### Fase 6: Detección Híbrida ✅
- PDFs vectoriales → `pdftotext -layout` (preserva estructura)
- PDFs escaneados/imágenes → OCR con PaddleOCR
- Selección automática del mejor método
- Campo `extraction_method` en respuesta

---

## RESULTADOS ACTUALES DE PRUEBAS

### ticket.pdf (ESCANEADO) - extraction_method: ocr ✅
```json
{"vendor": "E.S.CUATRO OLIVOS S.L.", "vendor_nif": "B11368479",
 "total": 66.83, "tax_base": 55.23, "tax_rate": 21.0, "tax_amount": 11.6,
 "confidence": 1.0}
```

### Factura noviembre.pdf (OLIVENET) - extraction_method: pdftotext_layout ✅
```json
{"vendor": "OLIVENET NETWORK S.L.U.", "vendor_nif": "B93340198",
 "total": 94.74, "tax_base": 78.3, "tax_rate": 21.0, "tax_amount": 16.44,
 "confidence": 0.9}
```

### Factura_VFR25087570.pdf (VODAFONE) - extraction_method: pdftotext_layout ✅
```json
{"vendor": "DMI Computer S.A.", "vendor_nif": "A79522702",
 "total": 172.12, "tax_base": 142.25, "tax_rate": 21.0, "tax_amount": 29.87,
 "confidence": 1.0}
```

---

## Archivos Críticos a Consultar

1. **ROADMAP_PERFECTION.md** - Plan completo con 6 fases
2. **INVESTIGATION_NOTES.md** - Notas técnicas del fix de Layout
3. **app.py** - Código principal (~5500 líneas ahora)

## Ubicaciones Clave en app.py

| Función | Línea (aprox) | Descripción |
|---------|---------------|-------------|
| `format_text_with_layout()` | 2416 | Reconstrucción espacial |
| `/process` endpoint | ~2750 | Endpoint principal |
| `/ocr` endpoint | 1913 | Original de Paco (NO TOCAR) |
| `OCR_CORRECTIONS_BASE` | 2084 | Diccionario OCR (407 correcciones) |
| `init_pp_structure_pipelines()` | 4983 | Inicializa PP-Structure |
| `/structure` endpoint | 5012 | Análisis estructural |
| `extract_invoice_fields()` | 5324 | KIE - extracción de campos |
| `/extract` endpoint | 5600+ | Extracción inteligente |

---

## Fases del Roadmap

| Fase | Descripción | Estado |
|------|-------------|--------|
| 1 | Instalar dependencias PP-Structure | ✅ COMPLETADO |
| 2 | Layout Analysis Pipeline | ✅ COMPLETADO |
| 3 | Table Recognition (SLANet) | ✅ COMPLETADO |
| 4 | Mejorar format_text_with_layout | ⏳ PARCIAL |
| 5 | Key Information Extraction | ✅ COMPLETADO |
| 6 | Endpoint /extract estructurado | ✅ COMPLETADO |

---

## PENDIENTE

### 1. Manejar error std::exception
PP-Structure a veces falla con `std::exception` en llamadas consecutivas.
Solución planificada: reintentos + reinicialización de pipeline.

### 2. Mejorar format_text_with_layout
Usar clustering de coordenadas X para detectar columnas reales.
Ver ROADMAP Fase 4 para detalles.

### 3. Probar con todas las facturas

---

## Comandos de Test Rápido

```bash
# Health check
curl http://localhost:8503/health

# Test /extract (extracción de campos)
curl -X POST http://localhost:8503/extract -F "file=@ticket.pdf"

# Test /structure (análisis estructural con HTML de tablas)
curl -X POST http://localhost:8503/structure -F "file=@factura.pdf"

# Test /process layout
curl -X POST http://localhost:8503/process -F "file=@factura.pdf" -F "format=layout"

# Ver logs
docker logs --tail 50 paddlepaddle-cpu
```

---

## Dependencias Instaladas

### En Dockerfile (permanente)
```dockerfile
pip install "paddlex[ocr]==3.3.10" scikit-learn
```

### Modelos cacheados en contenedor
- PP-LCNet_x1_0_doc_ori (orientación)
- UVDoc (document enhancement)
- PP-DocLayout-L (layout detection)
- SLANet_plus (table recognition)
- PP-OCRv4_server_det (text detection)
- PP-OCRv4_server_rec (text recognition)
- RT-DETR-H_layout_17cls (17-class layout)

---

## Facturas de Prueba

```
/mnt/c/PROYECTOS CLAUDE/paddleocr/facturas_prueba/
├── Factura noviembre.pdf     # PDF vectorial (Layout OK)
├── ticket.pdf                # Escaneado (KIE funciona ✅)
├── CamScanner*.pdf           # Escaneado (probar)
└── Factura_VFR*.pdf          # Otro formato (probar)
```

---

## IMPORTANTE

1. **NO MODIFICAR** el endpoint `/ocr` (compatibilidad n8n)
2. **Siempre verificar** que el contenedor funciona después de cambios
3. **Si pierdes contexto**: leer ROADMAP_PERFECTION.md primero
4. **Los paquetes instalados** se mantienen con `docker restart` pero se pierden con `docker-compose down && up`
