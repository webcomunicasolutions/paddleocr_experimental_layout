# CONTEXTO DE SESIÓN - PaddleOCR Perfection

**LEER ESTE ARCHIVO AL INICIO DE CADA INTERACCIÓN**

## Estado Actual del Proyecto

### Versión: 4.2 (LINE_ITEMS Avanzado)
### Próximo objetivo: v4.3 (Post-procesamiento y Normalización)

**ÚLTIMA ACTUALIZACIÓN:** 2025-12-07 - v4.2 completada con LINE_ITEMS multi-formato.

---

## VERSIONES COMPLETADAS

### v4.0 - Inteligencia Documental ✅
- OCR con PaddleOCR + diccionario 407 correcciones
- Modo layout con DBSCAN clustering para columnas
- Detección híbrida (pdftotext para vectoriales, OCR para escaneados)
- PP-Structure (tablas HTML con SLANet, layout analysis)
- KIE: vendor, NIF, total, tax_base, tax_rate, tax_amount
- customer_name, customer_nif, invoice_number, line_items básico

### v4.1 - Optimización de Rendimiento ✅
- **Circuit Breaker** para PP-Structure
  - 3 fallos consecutivos → bloquea llamadas
  - 60 segundos de reset timeout
  - Estados: closed, open, half-open
- **Timeout inteligente**
  - Base: 60s + 30s por cada 500KB
  - Máximo: 120s
  - Ejecución en thread separado
- **Fallback automático** a OCR básico si PP-Structure falla

### v4.2 - LINE_ITEMS Avanzado ✅
- **Detección de zona de productos** (headers: Detalle, Descripción, etc.)
- **4 patrones de extracción**:
  - Patrón 1: Vodafone/DMI (Código + Descripción + Cantidad + Precio + Importe)
  - Patrón 2: Cantidad primero (Qty + Descripción + Precio + Importe)
  - Patrón 3: Olivenet (Concepto + Importe, sin cantidad)
  - Patrón 4: Simple (Texto + Importe)
- **Detección de descuentos** (importes negativos o palabra DESCUENTO)
- **Validación de importes** (cantidad × precio_unitario ≈ importe)
- **Resultados de pruebas**:
  - Olivenet: 9 items (incluyendo descuentos) ✅
  - Vodafone: 2 items con validación ✅
  - Tickets escaneados: Pendiente mejora

---

## PRÓXIMO: v4.3 - Post-procesamiento y Normalización

### Objetivo
Limpieza y normalización del texto extraído:
- Eliminar artefactos de OCR
- Normalizar formatos de fecha (→ ISO YYYY-MM-DD)
- Normalizar importes (→ 1234.56)
- Validar NIFs (formato + dígito de control)

### Plan de implementación
1. Detectar y corregir palabras cortadas entre líneas
2. Unificar espacios y saltos de línea
3. Convertir fechas a formato ISO
4. Normalizar importes a formato numérico estándar
5. Validar NIFs españoles

---

## ENDPOINTS DISPONIBLES

| Endpoint | Descripción | Uso principal |
|----------|-------------|---------------|
| `/process` | OCR con formato | `format=layout` para IA |
| `/extract` | Extracción KIE | JSON estructurado con LINE_ITEMS |
| `/structure` | PP-Structure | Tablas HTML + layout |
| `/ocr` | Original Paco | Compatibilidad n8n |

---

## UBICACIONES CLAVE EN app.py (~6100 líneas)

| Función | Línea (aprox) | Descripción |
|---------|---------------|-------------|
| `pp_structure_circuit_breaker` | ~5103 | Circuit breaker config |
| `check_circuit_breaker()` | ~5113 | Verificar estado CB |
| `record_circuit_success/failure()` | ~5138 | Registrar éxito/fallo |
| `run_pp_structure_with_retry()` | ~5202 | Ejecutar con retry+timeout |
| `detect_columns_dbscan()` | ~2417 | Clustering de columnas |
| `format_text_with_layout()` | ~2450 | Reconstrucción espacial |
| `extract_invoice_fields()` | ~5600 | KIE - extracción campos |
| `extract_line_items()` | ~6005 | LINE_ITEMS v4.2 |

---

## LINE_ITEMS v4.2 - PATRONES

```python
# Patrón 1: Vodafone/DMI - Código + Descripción + Qty + Precio + Importe
r'^([A-Z]{2}\d+|\d+)\s+(.+?)\s+(\d+)\s+(\d+[.,]\d{2})\s+(\d+[.,]\d{2})\s*€?'

# Patrón 2: Cantidad primero
r'^(\d+)\s+(.+?)\s+(\d+[.,]\d{2})\s+(\d+[.,]\d{2})\s*€?'

# Patrón 3: Olivenet - Concepto + Importe
r'^(.+?)\s{2,}(-?\d+[.,]\d{2,4})\s*€?'

# Patrón 4: Simple - Texto + Importe
r'(.+?)\s+(-?\d+[.,]\d{2})\s*€?\s*$'

# Descuentos: amount < 0 OR 'DESCUENTO' in description
# Validación: |quantity × unit_price - amount| < 0.02
```

---

## COMANDOS DE TEST

```bash
# Health check
curl http://localhost:8503/health

# Test /extract (incluye LINE_ITEMS)
curl -X POST http://localhost:8503/extract -F "file=@factura.pdf"

# Test /process layout
curl -X POST http://localhost:8503/process -F "file=@factura.pdf" -F "format=layout"

# Ver logs (circuit breaker, timeouts)
docker logs --tail 100 paddlepaddle-cpu | grep -E "CIRCUIT|PP-STRUCTURE|timeout"

# Rebuild completo
docker-compose down && docker-compose build --no-cache && docker-compose up -d
```

---

## ARCHIVOS DE DOCUMENTACIÓN

| Archivo | Contenido |
|---------|-----------|
| `ROADMAP_v4.1.md` | Plan de mejoras v4.1, v4.2, v4.3 |
| `CHANGELOG_DETAILED.md` | Historial de cambios detallado |
| `CLAUDE.md` | Instrucciones para Claude Code |
| `.claude/SESSION_CONTEXT.md` | ESTE ARCHIVO - contexto rápido |

---

## CIRCUIT BREAKER - CONFIGURACIÓN

```python
pp_structure_circuit_breaker = {
    'failures': 0,           # Contador de fallos
    'last_failure': 0,       # Timestamp último fallo
    'state': 'closed',       # closed/open/half-open
    'threshold': 3,          # Fallos para abrir
    'reset_timeout': 60,     # Segundos para reintentar
}
```

### Estados:
- **closed**: Normal, PP-Structure disponible
- **open**: Bloqueado, no se llama a PP-Structure
- **half-open**: Probando si PP-Structure funciona de nuevo

---

## TIMEOUT INTELIGENTE

```python
# Fórmula:
timeout = min(60 + (file_size // 500000) * 30, 120)

# Ejemplos:
# 100KB  → 60s
# 600KB  → 90s
# 1.2MB  → 120s (máximo)
```

---

## FACTURAS DE PRUEBA

```
/mnt/c/PROYECTOS CLAUDE/paddleocr/facturas_prueba/
├── ticket.pdf                # Escaneado, gasolinera (LINE_ITEMS pendiente)
├── Factura noviembre.pdf     # Vectorial, Olivenet ✅ 9 items
├── Factura_VFR25087570.pdf   # Vectorial, Vodafone ✅ 2 items
└── CamScanner*.pdf           # Escaneado grande
```

---

## TAREAS PENDIENTES FUTURAS

1. **std::exception en PaddleOCR** - Investigar causa raíz
2. **Recuperación de errores C++** - Mejorar en OCR principal
3. **LINE_ITEMS para tickets** - El OCR no preserva estructura tabular

---

## SI SE CORTA LA COMUNICACIÓN

1. Leer este archivo primero
2. Ver `ROADMAP_v4.1.md` para próximos pasos
3. Estado actual: **v4.2 completada, listo para v4.3**
4. Siguiente tarea: **Post-procesamiento y normalización**
   - Limpieza de artefactos OCR
   - Normalización de fechas e importes
   - Validación de NIFs
