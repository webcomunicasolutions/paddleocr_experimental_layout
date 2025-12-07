# ROADMAP: PaddleOCR Fusion v4 - El Camino a la Perfección

**Fecha inicio:** 2025-12-07
**Objetivo:** Transformar el OCR básico en un sistema de Inteligencia Documental completo

---

## 1. CONTEXTO DEL PROYECTO

### 1.1 Qué tenemos (v3.1)
- **Contenedor:** `paddlepaddle-cpu` en puerto 8503
- **PaddleOCR:** 3.3.2 con PaddleX 3.3.10
- **Layout Mode:** Funciona pero es BÁSICO (solo pdftotext -layout o reconstrucción simple)
- **Diccionario OCR:** 407 correcciones para español

### 1.2 Qué falta para la perfección
1. **PP-Structure completo** - Layout Analysis + Table Recognition
2. **SLANet** - Reconstrucción de tablas a HTML/JSON
3. **KIE (Key Information Extraction)** - Extraer campos clave de facturas
4. **Detección inteligente de columnas** - Clustering de coordenadas X

### 1.3 Archivos clave
```
/mnt/c/PROYECTOS CLAUDE/paddleocr/paddleocr_experimental_layout/
├── app.py                    # Código principal (~2900 líneas)
├── Dockerfile                # Imagen Docker
├── docker-compose.yml        # Configuración contenedor
├── INVESTIGATION_NOTES.md    # Notas técnicas del Layout fix
└── ROADMAP_PERFECTION.md     # ESTE ARCHIVO - LEER SIEMPRE
```

---

## 2. PROBLEMAS IDENTIFICADOS

### 2.1 Tickets/Facturas escaneados
**Ejemplo:** `ticket.pdf` (CamScanner)
```
Resultado ACTUAL (malo):
E.S.CUATRO OLIVOS S.L.
Ctra..MeamKM.3        <-- Texto mal parseado
Fecha   &4 21-08-25   <-- Errores OCR
Base Imponible   %IVA   Cuota    Total Factura
55,23  21,00   11.60    66,83    <-- Tabla destruida (sin estructura)
```

**Resultado DESEADO:**
```json
{
  "vendor": "E.S. CUATRO OLIVOS S.L.",
  "address": "Ctra. Medina KM.3, Puerto Real, 11510 Cádiz",
  "invoice_number": "2025003047",
  "date": "21-08-2025",
  "items": [
    {"concept": "GASÓLEO A", "quantity": 1, "price": 64.83}
  ],
  "tax_base": 55.23,
  "tax_rate": 21.00,
  "tax_amount": 11.60,
  "total": 66.83
}
```

### 2.2 Facturas vectoriales (PDF digital)
**Ejemplo:** `Factura noviembre.pdf`
- Layout actual funciona BIEN con pdftotext -layout
- PERO no extrae campos estructurados
- PERO no detecta tablas como entidades

---

## 3. FASES DE IMPLEMENTACIÓN

### FASE 1: Instalar dependencias PP-Structure
**Estado:** PENDIENTE
**Qué hacer:**
```dockerfile
# Añadir al Dockerfile
RUN pip install "paddlex[ocr]==3.3.10"
```
**Verificar con:**
```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline='table_recognition')  # Debe funcionar
pipeline = create_pipeline(pipeline='layout_parsing')     # Debe funcionar
```

### FASE 2: Layout Analysis Pipeline
**Estado:** PENDIENTE
**Qué hacer:**
- Crear función `analyze_document_layout(image_path)` que detecte regiones:
  - `header` - Encabezado
  - `table` - Tablas
  - `text` - Párrafos
  - `footer` - Pie de página
**Código base:**
```python
from paddlex import create_pipeline
layout_pipeline = create_pipeline(pipeline='layout_parsing')
result = layout_pipeline.predict(image_path)
# result contiene regiones con bboxes y tipos
```

### FASE 3: Table Recognition con SLANet
**Estado:** PENDIENTE
**Qué hacer:**
- Para cada región detectada como "table", aplicar SLANet
- Extraer estructura HTML de la tabla
- Convertir a JSON estructurado
**Código base:**
```python
table_pipeline = create_pipeline(pipeline='table_recognition')
result = table_pipeline.predict(table_image)
# result['html'] = "<table><tr><td>...</td></tr></table>"
# result['cells'] = [{"bbox": [...], "text": "..."}]
```

### FASE 4: Mejorar format_text_with_layout
**Estado:** PENDIENTE
**Qué hacer:**
- Implementar clustering de coordenadas X para detectar columnas reales
- Usar DBSCAN o K-means para agrupar posiciones X
- Reconstruir con tabulación inteligente
**Ubicación:** `app.py` línea 2416

### FASE 5: Key Information Extraction (KIE)
**Estado:** PENDIENTE
**Qué hacer:**
- Definir entidades para facturas españolas:
  - VENDOR_NAME, VENDOR_NIF, VENDOR_ADDRESS
  - INVOICE_NUMBER, INVOICE_DATE
  - CUSTOMER_NAME, CUSTOMER_NIF
  - LINE_ITEM, QUANTITY, UNIT_PRICE, AMOUNT
  - TAX_BASE, TAX_RATE, TAX_AMOUNT, TOTAL
- Usar patrones regex + posición espacial para clasificar
**Nota:** LayoutXLM requiere fine-tuning. Empezar con heurísticas.

### FASE 6: Nuevo endpoint /extract
**Estado:** PENDIENTE
**Qué hacer:**
- Crear endpoint que devuelva JSON estructurado
- Combinar Layout Analysis + Table Recognition + KIE
```
POST /extract
{
  "document_type": "invoice",
  "vendor": {...},
  "customer": {...},
  "items": [...],
  "totals": {...},
  "raw_text": "...",
  "tables": [{"html": "...", "data": [...]}],
  "layout_regions": [...]
}
```

---

## 4. COMANDOS ÚTILES

### Docker
```bash
# Ver logs
docker logs -f paddlepaddle-cpu

# Entrar al contenedor
docker exec -it paddlepaddle-cpu bash

# Reconstruir
docker-compose down && docker-compose build --no-cache && docker-compose up -d
```

### Testing
```bash
# Test Layout mode
curl -X POST http://localhost:8503/process \
  -F "file=@factura.pdf" \
  -F "format=layout"

# Test con ticket escaneado
curl -X POST http://localhost:8503/process \
  -F "file=@ticket.pdf" \
  -F "format=layout"
```

### Verificar módulos Python
```bash
docker exec paddlepaddle-cpu python3 -c "
from paddlex import create_pipeline
print('table_recognition:', create_pipeline(pipeline='table_recognition'))
print('layout_parsing:', create_pipeline(pipeline='layout_parsing'))
"
```

---

## 5. ARQUITECTURA OBJETIVO

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENTRADA: PDF/Imagen                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  FASE 1: Preprocesamiento                                       │
│  - Corrección de orientación (ya implementado)                  │
│  - Deskew (ya implementado)                                     │
│  - Mejora de contraste                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  FASE 2: Layout Analysis (PP-Structure)                         │
│  - Detectar regiones: header, table, text, footer               │
│  - Generar bboxes por región                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│  Regiones TABLA         │     │  Regiones TEXTO         │
│  → SLANet               │     │  → OCR + Layout         │
│  → HTML/JSON            │     │  → Reconstrucción       │
└─────────────────────────┘     └─────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  FASE 3: Key Information Extraction                             │
│  - Clasificar entidades (NIF, Total, Fecha, etc.)               │
│  - Establecer relaciones (CLAVE → VALOR)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  SALIDA: JSON Estructurado                                      │
│  {                                                              │
│    "vendor": {...},                                             │
│    "invoice": {...},                                            │
│    "items": [...],                                              │
│    "totals": {...},                                             │
│    "tables": [...],                                             │
│    "raw_text": "..."                                            │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. CRITERIOS DE ÉXITO

### Para considerar FASE completada:
- [ ] El código compila sin errores
- [ ] Los tests pasan con las facturas de prueba
- [ ] El resultado es MEJOR que el anterior (medible)

### Métricas de calidad:
1. **Tickets escaneados:** Extracción de Total con >95% precisión
2. **Facturas vectoriales:** Layout preservado al 100%
3. **Tablas:** Estructura HTML válida, datos en JSON
4. **Campos KIE:** NIF, Fecha, Total extraídos correctamente

---

## 7. NOTAS IMPORTANTES

### NO MODIFICAR sin necesidad:
- Endpoint `/ocr` original (compatibilidad n8n)
- Sistema de diccionarios OCR (funciona bien)
- Preprocesamiento OpenCV (funciona bien)

### SIEMPRE verificar:
- Que el contenedor arranca correctamente
- Que los modelos se cargan (ver logs)
- Que no hay regresiones en funcionalidad existente

### Si la conversación se hace larga:
1. LEER ESTE ARCHIVO primero
2. Ver estado actual en los TODOs
3. Continuar desde donde se quedó

---

## 8. HISTORIAL DE CAMBIOS

| Fecha | Versión | Cambio |
|-------|---------|--------|
| 2025-12-07 | v3.1 | Layout mode arreglado (pdftotext del original) |
| 2025-12-07 | v3.1 | Regex \s{3,} desactivado para preservar layout |
| 2025-12-07 | - | Iniciado ROADMAP hacia v4.0 |

---

**RECORDATORIO FINAL:**
El objetivo es pasar de "OCR que extrae texto" a "Sistema de Inteligencia Documental que entiende facturas". Cada fase debe ser incremental y verificable.
