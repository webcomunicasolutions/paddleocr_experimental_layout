# CONTEXTO DE SESIÓN - PaddleOCR Perfection

**LEER ESTE ARCHIVO AL INICIO DE CADA INTERACCIÓN**

## Estado Actual del Proyecto

### Versión: 4.3 (Normalización y Validación)
### Próximo objetivo: Sistema de validación con agentes

**ÚLTIMA ACTUALIZACIÓN:** 2025-12-07 - v4.3 completada con normalización de fechas, validación de NIFs y normalización de importes.

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

### v4.3 - Normalización y Validación ✅
- **normalize_date()**: Convierte fechas a DD-MM-YYYY
  - DD-MM-YY / DD/MM/YY → DD-MM-YYYY (expande años cortos)
  - DD-MM-YYYY / DD/MM/YYYY → DD-MM-YYYY
  - DD MMM YYYY (ej: "05 NOV 2025") → DD-MM-YYYY
- **validate_spanish_nif()**: Valida NIFs/CIFs españoles
  - NIF persona física: 8 dígitos + letra (validación de letra)
  - NIE extranjeros: X/Y/Z + 7 dígitos + letra
  - CIF empresas: letra + 7 dígitos + control
- **normalize_amount()**: Normaliza importes a float
  - Formato español (1.234,56) → 1234.56
  - Elimina símbolos de moneda
- **Integración automática** en endpoint /extract
  - Nuevos campos: date_normalized, vendor_nif_valid, vendor_nif_type

---

## PRÓXIMO: Sistema de Validación con Agentes

### Objetivo
Cuando el usuario suba ~100 facturas de prueba:
1. Procesar cada factura con PaddleOCR (/extract)
2. Usar Claude Vision para validar resultados
3. Comparar y generar reporte de precisión
4. Identificar patrones de errores

### Plan de implementación
1. Crear endpoint /batch-validate o script de validación
2. Procesar facturas en lote
3. Usar API de Claude Vision para verificar
4. Generar reporte con métricas de precisión

---

## ENDPOINTS DISPONIBLES

| Endpoint | Descripción | Uso principal |
|----------|-------------|---------------|
| `/process` | OCR con formato | `format=layout` para IA |
| `/extract` | Extracción KIE + normalización v4.3 | JSON estructurado |
| `/structure` | PP-Structure | Tablas HTML + layout |
| `/ocr` | Original Paco | Compatibilidad n8n |

---

## UBICACIONES CLAVE EN app.py (~6500 líneas)

| Función | Línea (aprox) | Descripción |
|---------|---------------|-------------|
| `pp_structure_circuit_breaker` | ~5103 | Circuit breaker config |
| `check_circuit_breaker()` | ~5113 | Verificar estado CB |
| `run_pp_structure_with_retry()` | ~5202 | Ejecutar con retry+timeout |
| `extract_invoice_fields()` | ~5648 | KIE - extracción campos |
| `normalize_date()` | ~6185 | Normalizar fechas DD-MM-YYYY |
| `validate_spanish_nif()` | ~6268 | Validar NIFs/CIFs |
| `normalize_amount()` | ~6366 | Normalizar importes |
| `normalize_fields()` | ~6421 | Aplicar toda normalización |

---

## NORMALIZACIÓN v4.3

### Fechas (normalize_date)
```python
# Entrada → Salida
"21-08-25"    → "21-08-2025"
"05/11/2025"  → "05-11-2025"
"19/11/25"    → "19-11-2025"
"05 NOV 2025" → "05-11-2025"
```

### NIFs (validate_spanish_nif)
```python
# Tipos detectados
"78971220F"  → NIF (persona física) - control_check: True/False
"B93340198"  → CIF (empresa) - control_check: True/False
"X1234567L"  → NIE (extranjero) - control_check: True/False
```

### Importes (normalize_amount)
```python
# Entrada → Salida
"94,74 €"    → 94.74
"1.234,56"   → 1234.56  # formato español
"-12.3967 €" → -12.40   # redondeo
```

---

## COMANDOS DE TEST

```bash
# Health check
curl http://localhost:8503/health

# Test /extract con normalización v4.3
curl -X POST http://localhost:8503/extract -F "file=@factura.pdf"

# Ver campos normalizados específicos
curl -X POST http://localhost:8503/extract -F "file=@factura.pdf" | python3 -c "
import sys,json
d=json.load(sys.stdin)
f=d['fields']
print(f'Fecha: {f.get(\"date\")} → {f.get(\"date_normalized\")}')
print(f'NIF Vendor: {f.get(\"vendor_nif\")} - Válido: {f.get(\"vendor_nif_valid\")}')
"

# Ver logs
docker logs --tail 100 paddlepaddle-cpu

# Rebuild completo
docker-compose down && docker-compose build --no-cache && docker-compose up -d
```

---

## FACTURAS DE PRUEBA

```
/mnt/c/PROYECTOS CLAUDE/paddleocr/facturas_prueba/
├── ticket.pdf                # Escaneado - 21-08-25 → 21-08-2025 ✅
├── Factura noviembre.pdf     # Olivenet - 05/11/2025 → 05-11-2025 ✅
├── Factura_VFR25087570.pdf   # Vodafone - 19/11/25 → 19-11-2025 ✅
└── CamScanner*.pdf           # Escaneado grande
```

---

## TAREAS PENDIENTES FUTURAS

1. **std::exception en PaddleOCR** - Investigar causa raíz
2. **Recuperación de errores C++** - Mejorar en OCR principal
3. **LINE_ITEMS para tickets** - El OCR no preserva estructura tabular
4. **Sistema de validación con agentes** - Para pruebas masivas con ~100 facturas

---

## SI SE CORTA LA COMUNICACIÓN

1. Leer este archivo primero
2. Estado actual: **v4.3 completada**
3. Siguiente tarea: **Sistema de validación con agentes**
   - Usuario quiere subir ~100 facturas para pruebas
   - Usar PaddleOCR + Claude Vision para validar
   - Generar reporte de precisión
