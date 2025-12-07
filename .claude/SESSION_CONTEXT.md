# CONTEXTO DE SESIÓN - PaddleOCR Perfection

**LEER ESTE ARCHIVO AL INICIO DE CADA INTERACCIÓN**

## Estado Actual del Proyecto

### Versión: 4.3 (Normalización y Validación) - ESTABLE
### Próximo objetivo: Sistema de validación con agentes (~100 facturas)

**ÚLTIMA ACTUALIZACIÓN:** 2025-12-07 - v4.3 completada y subida a GitHub.

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
- **normalize_date()**: Convierte fechas a DD/MM/YYYY
  - DD-MM-YY / DD/MM/YY → DD/MM/YYYY
  - DD-MM-YYYY / DD/MM/YYYY → DD/MM/YYYY
  - DD MMM YYYY (ej: "05 NOV 2025") → DD/MM/YYYY
- **validate_spanish_nif()**: Valida NIFs/CIFs españoles
  - NIF persona física: 8 dígitos + letra (validación de letra)
  - NIE extranjeros: X/Y/Z + 7 dígitos + letra
  - CIF empresas: letra + 7 dígitos + control
- **normalize_amount()**: Normaliza importes con moneda
  - Detecta moneda: EUR (€), USD ($), GBP (£), CHF, MXN
  - Formato: "66.83 EUR"
- **detect_currency()**: Detecta moneda predominante en documento
- **Campos añadidos**:
  - `date_normalized`: "21/08/2025"
  - `currency`: "EUR"
  - `total_formatted`: "66.83 EUR"
  - `vendor_nif_valid`: true/false
  - `vendor_nif_type`: "CIF"/"NIF"/"NIE"

---

## ENDPOINTS DISPONIBLES

| Endpoint | Descripción | Uso principal |
|----------|-------------|---------------|
| `/process` | OCR con formato | `format=layout` para IA |
| `/extract` | Extracción KIE + normalización v4.3 | JSON estructurado |
| `/structure` | PP-Structure | Tablas HTML + layout |
| `/ocr` | Original Paco | Compatibilidad n8n |

---

## UBICACIONES CLAVE EN app.py (~6600 líneas)

| Función | Línea (aprox) | Descripción |
|---------|---------------|-------------|
| `pp_structure_circuit_breaker` | ~5103 | Circuit breaker config |
| `check_circuit_breaker()` | ~5113 | Verificar estado CB |
| `run_pp_structure_with_retry()` | ~5202 | Ejecutar con retry+timeout |
| `extract_invoice_fields()` | ~5648 | KIE - extracción campos |
| `normalize_date()` | ~6188 | Normalizar fechas DD/MM/YYYY |
| `validate_spanish_nif()` | ~6271 | Validar NIFs/CIFs |
| `normalize_amount()` | ~6369 | Normalizar importes con moneda |
| `detect_currency()` | ~6442 | Detectar moneda del documento |
| `normalize_fields()` | ~6473 | Aplicar toda normalización |

---

## NORMALIZACIÓN v4.3 - EJEMPLOS

### Fechas (DD/MM/YYYY)
```
"21-08-25"    → "21/08/2025"
"05/11/2025"  → "05/11/2025"
"19/11/25"    → "19/11/2025"
"05 NOV 2025" → "05/11/2025"
```

### NIFs (con validación)
```
"78971220F"  → NIF (persona física) - control_check: true
"B93340198"  → CIF (empresa) - control_check: true
"X1234567L"  → NIE (extranjero) - control_check: true/false
```

### Importes (con moneda)
```
66.83     → "66.83 EUR"
"94,74 €" → "94.74 EUR"
"$150.00" → "150.00 USD"
"£75.50"  → "75.50 GBP"
"-12.40"  → "-12.40 EUR"
```

---

## RESPUESTA JSON DE /extract

```json
{
  "success": true,
  "document_type": "invoice",
  "extraction_method": "pdftotext_layout",
  "fields": {
    "vendor": "OLIVENET NETWORK S.L.U.",
    "vendor_nif": "B93340198",
    "vendor_nif_valid": true,
    "vendor_nif_type": "CIF",
    "customer_nif": "78971220F",
    "customer_nif_valid": true,
    "customer_nif_type": "NIF",
    "invoice_number": "ON2025-584267",
    "date": "05/11/2025",
    "date_normalized": "05/11/2025",
    "total": 94.74,
    "total_formatted": "94.74 EUR",
    "tax_base": 78.3,
    "tax_base_formatted": "78.30 EUR",
    "tax_rate": 21.0,
    "tax_amount": 16.44,
    "tax_amount_formatted": "16.44 EUR",
    "currency": "EUR",
    "line_items": [
      {
        "description": "Static IP (01/10/2025 - 31/10/2025)",
        "amount": 15.0,
        "amount_normalized": 15.0,
        "amount_formatted": "15.00 EUR"
      }
    ],
    "_normalized": {
      "version": "4.3",
      "currency": "EUR",
      "date": {"original": "05/11/2025", "normalized": "05/11/2025", "valid": true},
      "vendor_nif": {"original": "B93340198", "type": "CIF", "control_check": true},
      "amounts": {...}
    }
  },
  "confidence": {"score": 1.0, "level": "high"}
}
```

---

## COMANDOS DE TEST

```bash
# Health check
curl http://localhost:8503/health

# Test /extract con normalización v4.3
curl -X POST http://localhost:8503/extract -F "file=@factura.pdf"

# Ver campos normalizados
curl -X POST http://localhost:8503/extract -F "file=@factura.pdf" | python3 -c "
import sys,json
d=json.load(sys.stdin)
f=d['fields']
print(f'Fecha: {f.get(\"date\")} → {f.get(\"date_normalized\")}')
print(f'Moneda: {f.get(\"currency\")}')
print(f'Total: {f.get(\"total_formatted\")}')
print(f'NIF válido: {f.get(\"vendor_nif_valid\")}')
"

# Ver logs
docker logs --tail 100 paddlepaddle-cpu

# Rebuild completo
docker-compose down && docker-compose build --no-cache && docker-compose up -d

# Copiar app.py actualizado
docker cp app.py paddlepaddle-cpu:/app/app.py && docker restart paddlepaddle-cpu
```

---

## COMMITS EN GITHUB

```
7d89c58 fix: Formato de fecha DD/MM/YYYY y detección de moneda
945530f docs: Actualizar SESSION_CONTEXT.md con estado v4.3 completada
165e86b feat: v4.3 - Normalización y Validación de Campos
259d2f7 docs: Actualizar SESSION_CONTEXT.md con estado v4.2 completada
1b2ca50 feat: v4.2 - LINE_ITEMS Advanced con soporte multi-formato
bd946e1 docs: Actualizar SESSION_CONTEXT.md con estado v4.1
fb1a313 feat: v4.1 - Optimización de rendimiento PP-Structure
ab5cabb fix: Actualizar todas las referencias a v4 en el dashboard
```

---

## FACTURAS DE PRUEBA

```
/mnt/c/PROYECTOS CLAUDE/paddleocr/facturas_prueba/
├── ticket.pdf                # Escaneado - 21/08/2025 ✅ EUR
├── Factura noviembre.pdf     # Olivenet - 05/11/2025 ✅ EUR
├── Factura_VFR25087570.pdf   # Vodafone - 19/11/2025 ✅ EUR
└── CamScanner*.pdf           # Escaneado grande
```

---

## TAREAS PENDIENTES FUTURAS

1. **std::exception en PaddleOCR** - Investigar causa raíz
2. **Recuperación de errores C++** - Mejorar en OCR principal
3. **LINE_ITEMS para tickets** - El OCR no preserva estructura tabular
4. **Sistema de validación** - Para pruebas masivas con ~100 facturas
   - Procesar con PaddleOCR
   - Validar con Claude Vision
   - Generar reporte de precisión

---

## SI SE CORTA LA COMUNICACIÓN

1. Leer este archivo primero
2. Estado actual: **v4.3 ESTABLE - Todo subido a GitHub**
3. Branch: `main` - Sincronizado con origin
4. Siguiente tarea: **Sistema de validación con ~100 facturas**
   - Usuario subirá facturas de prueba
   - Usar PaddleOCR + Claude Vision para validar
   - Comparar y generar métricas de precisión
