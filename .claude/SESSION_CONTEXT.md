# CONTEXTO DE SESIÓN - PaddleOCR Perfection

**LEER ESTE ARCHIVO AL INICIO DE CADA INTERACCIÓN**

## Estado Actual del Proyecto

### Versión: 4.4 (Multi-idioma e Internacionalización) - ESTABLE
### Próximo objetivo: Continuar validación masiva (~175 facturas)

**ÚLTIMA ACTUALIZACIÓN:** 2025-12-07 - v4.4 completada. Validación piloto con 20 facturas realizada.

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

### v4.4 - Multi-idioma e Internacionalización ✅ (NUEVA)
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

---

## RESULTADOS VALIDACIÓN PILOTO (20 facturas)

| Formato | Score | customer_name | Notas |
|---------|-------|---------------|-------|
| Olivenet | 90% | ✅ | Mejor formato español |
| DMI/Vodafone | 85% | ✅ | LINE_ITEMS mejorado |
| LucusHost | 75% | ✅ | invoice_number mejorado |
| Ginernet | 70% | ✅ | Formato "/" en factura |
| Google Cloud | 60% | ⚠️ | Fecha inglesa funciona |
| Internacionales | ~30% | ⚠️ | Layouts complejos |
| Escaneadas | ~35% | ❌ | Problemas de calidad |

**Mejoras v4.4 verificadas:**
- ✅ customer_name ahora extrae "Juan Jose Sanchez Bernal" en DMI, LucusHost, Ginernet
- ✅ Fecha "31 JUL 2025" → "31/07/2025" (Google Cloud)
- ✅ invoice_number "72011" extraído de "Factura: 72011" (LucusHost)
- ✅ LINE_ITEMS sin fechas de garantía contaminando descripciones

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

# Test /extract con mejoras v4.4
curl -X POST http://localhost:8503/extract -F "file=@factura.pdf" | python3 -c "
import sys,json
d=json.load(sys.stdin)
f=d['fields']
print(f'Vendor: {f.get(\"vendor\")}')
print(f'Customer: {f.get(\"customer_name\")}')  # NUEVO v4.4
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
├── Factura noviembre.pdf     # Olivenet - 90% ✅
├── Factura_VFR25087570.pdf   # DMI/Vodafone - 85% ✅
├── Factura-72011.pdf         # LucusHost - 75% ✅
├── Factura-2025-7406.pdf     # Ginernet - 70% ✅
├── 5315630091.pdf            # Google Cloud - 60% ✅
├── ticket.pdf                # Escaneado gasolinera - 38%
├── pCloud Invoice Sept 2025.pdf  # Internacional - 29%
└── ... (175 facturas totales)
```

---

## TAREAS PENDIENTES

1. **Continuar validación masiva** - 155 facturas restantes
2. **std::exception en PP-Structure** - Investigar causa raíz
3. **Mejorar facturas internacionales** - Layouts muy diferentes
4. **LINE_ITEMS para tickets escaneados** - OCR no preserva estructura

---

## SI SE CORTA LA COMUNICACIÓN

1. Leer este archivo primero
2. Estado actual: **v4.4 ESTABLE - Multi-idioma funcionando**
3. Branch: `main`
4. Siguiente tarea: **Continuar validación con facturas restantes**
   - 20/175 facturas procesadas
   - Mejoras implementadas y verificadas
   - VALIDATION_REPORT.md tiene detalles completos
