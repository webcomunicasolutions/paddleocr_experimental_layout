# CONTEXTO DE SESIÓN - PaddleOCR Perfection

**LEER ESTE ARCHIVO AL INICIO DE CADA INTERACCIÓN**

## Estado Actual del Proyecto

### Versión: 4.6.1 (Detección de Tablas MEJORADA) - VALIDADO ✅
### Próximo objetivo: Commit a GitHub

**ÚLTIMA ACTUALIZACIÓN:** 2025-12-08 - v4.6.1 validado con 27 facturas. Listo para commit.

---

## CAMBIOS EN PRUEBAS (NO COMMITEADOS)

### v4.6.1 - Mejoras en Detección de Tablas (EN PRUEBAS)

**Cambios realizados:**
1. **Criterio de precios relajado**: De 2+ precios por fila a 1+ precio
   - Línea 2570: `if len(prices) >= 1:` (antes era 2)
   - Razón: Facturas con producto único no se detectaban

2. **Criterio de filas mínimas relajado**: De 2+ filas de datos a 1+ fila
   - Línea 2573: `len(result['data_rows']) >= 1` (antes era 2)
   - Razón: Facturas Energeeks con 1-3 productos no se detectaban

3. **Nuevos patrones de fin de tabla** (END_TABLE_PATTERNS):
   - `baseimponible` - Para detectar fin antes de totales
   - `%\s*iva` - Columna %IVA en línea de totales
   - `cuota` - Cuota IVA
   - `total\s*en\s*eur` - Total en EUR

**Problema resuelto:**
- ticket.pdf formateaba los totales (55,23 21,00 11,60) como fila de tabla
- Ahora detecta correctamente que es una línea de totales y NO la incluye

**Resultados de pruebas completas (27 facturas):**

| Categoría | Cantidad | Estado |
|-----------|----------|--------|
| Escaneadas con tabla detectada | 17 | ✅ OK |
| Gasolineras Markoil (sin formato tabla) | 4 | ✅ OK (correcto) |
| PDFs vectoriales (usa pdftotext) | 2 | ✅ OK (correcto) |
| Tickets (sin formato tabla) | 1 | ✅ OK (correcto) |
| Formato especial | 3 | ✅ OK |

**Detalle de facturas con tabla:**
- 400_1: 3, 400_2: 6, 400_3: 5, 400_4: 13, 400_5: 8
- 400_6: 11, 400_7: 3, 400_8: 3, 400_9: 5, 400_10: 3
- 400_15: 2, 400_16: 2, 400_17: 4, 400_18: 5
- 400_20: 9, 400_21: 8, 400_23: 6, 400_24: 2

**Validación COMPLETADA:** 0 falsos positivos, 0 regresiones.

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
- **validate_spanish_nif()**: Valida NIFs/CIFs españoles
- **normalize_amount()**: Normaliza importes con moneda
- **detect_currency()**: Detecta moneda predominante en documento

### v4.4 - Multi-idioma e Internacionalización ✅
- **customer_name Multi-label**: Extrae nombre de cliente con múltiples etiquetas
  - Español: "Cliente:", "Facturado a", "Datos de facturación", "Destinatario"
  - Inglés: "Bill to", "Sold to", "Customer", "Recipient"
  - Alemán: "Rechnungsadresse", "Kundenadresse"
  - Portugués/Francés: "Faturado a", "Facturé à"
- **Fechas internacionales**: Soporte para múltiples formatos
  - DD.MM.YYYY (alemán): "15.03.2025" → "15/03/2025"
  - MMM DD, YYYY (inglés): "Sep 1, 2025" → "01/09/2025"
  - DD MMM YYYY (internacional): "31 JUL 2025" → "31/07/2025"
  - YYYY-MM-DD (ISO): "2025-03-15" → "15/03/2025"
  - Meses en inglés: Jan, Feb, Mar... Dec
- **invoice_number Multi-idioma**: Reconoce etiquetas internacionales
  - Inglés: "Invoice Number:", "Invoice #:", "INV-"
  - Alemán: "Rechnung Nr.", "RE Nr."
  - Español: "Factura:", "Nº Factura"
  - Genérico: "Document Number", "Order Number"
- **LINE_ITEMS limpieza**: Elimina fechas de garantía de descripciones
  - Filtra líneas que solo son fechas (DD-MM-YYYY)
  - Limpia "Garantía: DD-MM-YYYY" al final de descripciones

### v4.5 - Layout Mejorado para IA ✅
- **Objetivo**: Generar texto optimizado para consumo por IA (Claude, GPT)
- **Preservación espacial absoluta**: Transforma coordenadas X del documento a posiciones de caracteres
- **Estrategia híbrida inteligente**:
  - PDFs vectoriales → pdftotext -layout (rápido y preciso)
  - PDFs escaneados → coordenadas OCR con posicionamiento absoluto
- **DBSCAN para columnas**: Detecta automáticamente columnas múltiples
- **Manejo de colisiones**: Si hay textos superpuestos, busca espacios disponibles
- **Resultados verificados**:
  - Olivenet: Estructura 2 columnas preservada ✅
  - DIGI: Layout complejo de 2 páginas preservado ✅
  - Escaneadas: Tablas con columnas alineadas ✅

### v4.6 - Detección y Formateo de Tablas ✅ (COMMITEADA)
- **Objetivo**: Mejorar la legibilidad de tablas de productos para extracción de line_items
- **Detección automática de tablas**:
  - Busca filas de headers (CODIGO, DESCRIPCION, CANTIDAD, PRECIO, IMPORTE, etc.)
  - Identifica filas de datos con precios
  - Detecta fin de tabla (FORMA DE PAGO, TOTAL FACTURA, IVA, GARANTÍA, etc.)
- **Formateo con separadores**:
  - Headers y datos con separadores `|`
  - Línea separadora `+---+---+` después del header
  - Anchos de columna proporcionales basados en posiciones X
- **Funciones implementadas**:
  ```python
  detect_table_structure(blocks, rows)  # Detecta headers, data_rows, columns
  calculate_table_column_widths()       # Calcula anchos proporcionales
  format_table_row()                    # Formatea fila con |separadores|
  format_table_separator()              # Genera +---+---+ línea
  ```

---

## ARQUITECTURA DEL SISTEMA DE TABLAS (v4.6)

### Flujo de Detección
```
1. format_text_with_layout() recibe bloques OCR
2. Para cada página:
   a. Agrupa bloques por filas (Y similar)
   b. Llama detect_table_structure(blocks, rows)
   c. Si is_table=True:
      - Formatea header con |col1|col2|...|
      - Añade separador +---+---+...+
      - Formatea data_rows con |val1|val2|...|
   d. Detecta END_TABLE_PATTERNS para terminar tabla
```

### Patrones de Detección

**Headers de tabla (HEADER_PATTERNS):**
```python
[
    r'(?i)\b(codigo|c[oó]d\.?|art[ií]culo|ref\.?)\b',
    r'(?i)\b(descripci[oó]n|concepto|detalle|producto)\b',
    r'(?i)\b(cantidad|cant\.?|qty|uds?\.?|unidades)\b',
    r'(?i)\b(precio|p\.?u\.?|pvp|importe\s*unit)\b',
    r'(?i)\b(dto\.?|desc\.?|descuento|%\s*dto)\b',
    r'(?i)\b(importe|total|neto|subtotal)\b',
    r'(?i)\b(iva|%\s*iva|tipo)\b',
]
```

**Fin de tabla (END_TABLE_PATTERNS):**
```python
[
    r'(?i)(forma\s*de\s*pago|vencimiento|total\s*factura)',
    r'(?i)(base\s*imponible|baseimponible|subtotal|importe\s*total)',
    r'(?i)(iva\s*\d+|%\s*iva|cuota|recargo|descuento\s*total)',
    r'(?i)(garantia|garant[ií]a|registro|domicilio\s*social)',
    r'(?i)(banco|c\.?c\.?c\.?|iban)',
    r'(?i)(total\s*en\s*eur|total\s*eur|total\s*€)',
]
```

**Detección de precios:**
```python
PRICE_PATTERN = re.compile(r'\d+[,\.]\d{2}')  # Detecta 45,50 o 45.50
```

---

## RESULTADOS VALIDACIÓN (Actualizados)

### Facturas Escaneadas (Mylar)
| Archivo | Productos | Tabla | Formato |
|---------|-----------|-------|---------|
| escaneadas 400_1 (Energeeks) | 3 | ✅ | Separadores OK |
| escaneadas 400_3 | 5 | ✅ | Separadores OK |
| escaneadas 400_5 | 8 | ✅ | Separadores OK |
| escaneadas 400_6 | 3 | ✅ | Separadores OK |

### Facturas Vectoriales
| Archivo | Productos | Tabla | Notas |
|---------|-----------|-------|-------|
| Olivenet | - | N/A | Usa pdftotext (correcto) |
| AnyDesk | - | N/A | Usa pdftotext (correcto) |
| Google Cloud | - | N/A | Usa pdftotext (correcto) |

### Tickets Escaneados
| Archivo | Tabla | Notas |
|---------|-------|-------|
| ticket.pdf | No tabla | Correcto - no tiene formato tabla |

---

## ENDPOINTS DISPONIBLES

| Endpoint | Descripción | Uso principal |
|----------|-------------|---------------|
| `/process` | OCR con formato | `format=layout` para IA |
| `/extract` | Extracción KIE + normalización v4.4 | JSON estructurado |
| `/structure` | PP-Structure | Tablas HTML + layout |
| `/ocr` | Original Paco | Compatibilidad n8n |

---

## COMANDOS DE TEST

```bash
# Health check
curl http://localhost:8503/health

# Test layout con detección de tablas
curl -s -X POST http://localhost:8503/process \
  -F "file=@factura.pdf" \
  -F "format=layout" | jq -r '.text'

# Test /extract
curl -X POST http://localhost:8503/extract -F "file=@factura.pdf" | python3 -c "
import sys,json
d=json.load(sys.stdin)
f=d['fields']
print(f'Vendor: {f.get(\"vendor\")}')
print(f'Customer: {f.get(\"customer_name\")}')
print(f'Invoice: {f.get(\"invoice_number\")}')
print(f'Fecha: {f.get(\"date\")} → {f.get(\"date_normalized\")}')
print(f'Total: {f.get(\"total_formatted\")}')
"

# Ver logs
docker logs --tail 100 paddlepaddle-cpu

# Copiar app.py actualizado
docker cp app.py paddlepaddle-cpu:/app/app.py && docker restart paddlepaddle-cpu
```

---

## FACTURAS DE PRUEBA

```
/mnt/c/PROYECTOS CLAUDE/paddleocr/facturas_prueba/
├── Factura noviembre.pdf          # Olivenet - vectorial
├── Factura_VFR25087570.pdf        # DMI/Vodafone
├── Factura-72011.pdf              # LucusHost
├── Factura-2025-7406.pdf          # Ginernet
├── 5315630091.pdf                 # Google Cloud
├── ticket.pdf                     # Escaneado gasolinera
├── escaneadas 400_1.pdf           # Energeeks - 3 productos
├── escaneadas 400_3.pdf           # Mylar - 5 productos
├── escaneadas 400_5.pdf           # Mylar - 8 productos
├── escaneadas 400_6.pdf           # Mylar - 3 productos
└── ... (175 facturas totales)
```

---

## TAREAS PENDIENTES

1. ~~**URGENTE: Probar 20+ facturas**~~ ✅ COMPLETADO (27 facturas probadas)
2. ~~**Validar que no hay regresiones**~~ ✅ COMPLETADO (0 regresiones)
3. **Commit v4.6.1** - LISTO PARA EJECUTAR

---

## SI SE CORTA LA COMUNICACIÓN

1. Leer este archivo primero
2. Estado actual: **v4.6.1 VALIDADO - Listo para commit**
3. Branch: `main`
4. Siguiente tarea: **Commit v4.6.1 a GitHub**
5. Cambios validados (pendientes de commit):
   - Línea 2570: `if len(prices) >= 1:` (relajado de 2)
   - Línea 2573: `len(result['data_rows']) >= 1` (relajado de 2)
   - END_TABLE_PATTERNS con más patrones para detectar fin de tabla
