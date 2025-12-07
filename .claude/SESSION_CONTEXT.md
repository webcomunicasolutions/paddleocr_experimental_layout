# CONTEXTO DE SESIÓN - PaddleOCR Perfection

**LEER ESTE ARCHIVO AL INICIO DE CADA INTERACCIÓN**

## Estado Actual del Proyecto

### Versión: 4.1 (Optimización de Rendimiento)
### Próximo objetivo: v4.2 (LINE_ITEMS Avanzado)

**ÚLTIMA ACTUALIZACIÓN:** 2025-12-07 - v4.1 completada con circuit breaker y timeout inteligente.

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

---

## PRÓXIMO: v4.2 - LINE_ITEMS Avanzado

### Objetivo
Extraer conceptos/productos completos para reproducir facturas:
- Descripción del producto/servicio
- Cantidad
- Precio unitario
- Importe (cantidad × precio)
- IVA aplicado por línea (si aplica)

### Plan de implementación
1. Detectar cabeceras de tabla (Concepto, Cantidad, Precio, Importe)
2. Mapear columnas automáticamente usando coordenadas X
3. Extraer cada línea con todos sus campos
4. Validar coherencia (cantidad × precio = importe)

---

## ENDPOINTS DISPONIBLES

| Endpoint | Descripción | Uso principal |
|----------|-------------|---------------|
| `/process` | OCR con formato | `format=layout` para IA |
| `/extract` | Extracción KIE | JSON estructurado |
| `/structure` | PP-Structure | Tablas HTML + layout |
| `/ocr` | Original Paco | Compatibilidad n8n |

---

## UBICACIONES CLAVE EN app.py (~6000 líneas)

| Función | Línea (aprox) | Descripción |
|---------|---------------|-------------|
| `pp_structure_circuit_breaker` | ~5103 | Circuit breaker config |
| `check_circuit_breaker()` | ~5113 | Verificar estado CB |
| `record_circuit_success/failure()` | ~5138 | Registrar éxito/fallo |
| `run_pp_structure_with_retry()` | ~5202 | Ejecutar con retry+timeout |
| `detect_columns_dbscan()` | ~2417 | Clustering de columnas |
| `format_text_with_layout()` | ~2450 | Reconstrucción espacial |
| `extract_invoice_fields()` | ~5600 | KIE - extracción campos |

---

## COMANDOS DE TEST

```bash
# Health check
curl http://localhost:8503/health

# Test /extract
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
├── ticket.pdf                # Escaneado, gasolinera ✅
├── Factura noviembre.pdf     # Vectorial, Olivenet ✅
├── Factura_VFR25087570.pdf   # Vectorial, Vodafone ✅
└── CamScanner*.pdf           # Escaneado grande
```

---

## SI SE CORTA LA COMUNICACIÓN

1. Leer este archivo primero
2. Ver `ROADMAP_v4.1.md` para próximos pasos
3. Estado actual: **v4.1 completada, listo para v4.2**
4. Siguiente tarea: **LINE_ITEMS avanzado**
   - Extraer cantidad, precio unitario, importe por línea
   - Detectar estructura tabular de items
