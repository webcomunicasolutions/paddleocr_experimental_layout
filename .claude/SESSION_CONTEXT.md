# CONTEXTO DE SESIÓN - PaddleOCR Perfection

**LEER ESTE ARCHIVO AL INICIO DE CADA INTERACCIÓN**

## Estado Actual del Proyecto

### Versión: 4.7.1 (Multi-idioma + Fix Detección Totales) - COMMITEADO ✅
### Próximo objetivo: Mejorar extracción de line_items / Probar más facturas internacionales

**ÚLTIMA ACTUALIZACIÓN:** 2025-12-08 03:00 - v4.7.1 commiteado y pusheado.

---

## HISTORIAL DE VERSIONES RECIENTES

| Versión | Commit | Estado | Descripción |
|---------|--------|--------|-------------|
| **v4.7.1** | `70b1751` | ✅ ACTUAL | Multi-idioma + fix detección de totales |
| v4.7 | `00c16e2` | ⚠️ Regresiones | Headers multi-idioma (causó falsos positivos) |
| v4.6.1 | `ff12847` | ✅ Estable | Última versión pre-multi-idioma |
| v4.6 | `1c30dd2` | ✅ | Detección y formateo de tablas |

**Para rollback a v4.6.1:**
```bash
git checkout ff12847 -- app.py
docker-compose build && docker-compose up -d
```

---

## CAMBIOS EN v4.7.1 (COMMITEADO)

### 1. TOTALS_SECTION_PATTERNS (nuevo)
Detecta headers que son realmente secciones de totales (no tabla de productos):
```python
TOTALS_SECTION_PATTERNS = [
    r'(?i)\bbase\s*imponible\b',
    r'(?i)\b%\s*iva\b',
    r'(?i)\bcuota\s*(iva|imponible)?\b',
    r'(?i)\btotal\s*factura\b',
    r'(?i)\bimporte\s*total\b',
    r'(?i)\brecargo\b',
]
```

### 2. Validación mejorada de headers
Si un header detectado contiene patrones de totales → se descarta.

### 3. Detección inteligente de filas de datos
```python
# Si hay texto descriptivo = definitivamente fila de producto
if len(text_without_numbers) >= 4:
    result['data_rows'].append(row_idx)
# Si tiene exactamente 3 precios (qty, precio, total) y está cerca del header = fila de datos
elif len(prices) == 3 and result['header_row_idx'] >= 0 and row_idx - result['header_row_idx'] <= 20:
    result['data_rows'].append(row_idx)
# Si tiene 4+ precios sin texto = probable línea de totales, descartar
else:
    logger.info(f"[TABLE-DETECT] Fila descartada (solo precios)...")
```

### 4. TABLE_HEADER_PATTERNS Multi-idioma (de v4.7)
10 patrones con soporte ES, EN, DE, FR, PT:
- Código/Referencia, Descripción, Cantidad, Precio
- Descuento, Importe/Total, IVA/Tax
- Servicios, Matrícula/Vehículos, Conductor/Usuario

### 5. END_TABLE_PATTERNS Multi-idioma (de v4.7)
8 patrones con soporte internacional:
- Forma de pago / Payment method
- Base imponible / Tax base
- IVA/Tax totals
- Garantía/Legal, Banco/Bank
- Total en moneda, Transacciones

---

## RESULTADOS DE PRUEBAS v4.7.1

| Factura | v4.6.1 | v4.7 | v4.7.1 | Estado |
|---------|--------|------|--------|--------|
| **ticket.pdf** | 0 | 2 ❌ | **0** | ✅ Corregido |
| **400_20** | 9 | 5 | **5** | ✅ (4 productos reales) |
| **400_19** | 0 | 2 | **2** | ✅ Mejorado (detecta renting) |
| 400_1 | 3 | - | **3** | ✅ |
| 400_4 | 13 | - | **13** | ✅ |
| 400_5 | 8 | - | **8** | ✅ |
| 400_6 | 11 | - | **11** | ✅ |

**Conclusiones:**
- ticket.pdf ya NO formatea la sección de totales como tabla
- 400_20 tiene 4 productos reales (antes contaba descripciones multi-línea)
- 400_19 ahora detecta tablas de renting (mejora de v4.7)
- Sin regresiones en otras facturas probadas

---

## ANÁLISIS CON CLAUDE VISION (Sesión actual)

### ticket.pdf - Problema identificado y resuelto
**Causa raíz:** El OCR mezcla columnas de forma caótica. La línea `Base Imponible %IVA Cuota Total Factura` se detectaba como header porque contiene palabras clave de tablas (Importe, Cantidad, etc.).

**Solución:** TOTALS_SECTION_PATTERNS descarta headers que contienen patrones de totales.

### 400_20 - Aclaración
**Antes (9 líneas):** Las descripciones multi-línea se contaban por separado.
**Ahora (5 líneas):** 1 header + 4 productos reales. Es CORRECTO.

---

## TAREAS PENDIENTES / IDEAS PARA CONTINUAR

### Alta Prioridad
1. **Probar más facturas internacionales** - Verificar que los patrones multi-idioma funcionan
2. **Mejorar line_items extraction** - Usar las tablas formateadas para mejor extracción

### Media Prioridad
3. **Tickets de gasolinera** - El OCR mezcla columnas en documentos escaneados pequeños
4. **Facturas de renting** - Tablas con columnas especiales (Matrícula, Conductor, Periodo)

### Baja Prioridad
5. **PP-Structure tables** - Evaluar si SLANet mejora extracción de tablas
6. **Batch processing** - Procesar múltiples facturas en paralelo

---

## VERSIONES COMPLETADAS (Resumen)

| Versión | Descripción |
|---------|-------------|
| v4.0 | OCR + diccionario 407 correcciones + KIE básico |
| v4.1 | Circuit Breaker + timeout inteligente |
| v4.2 | LINE_ITEMS avanzado (4 patrones) |
| v4.3 | Normalización fechas + validación NIF |
| v4.4 | Multi-idioma básico (customer_name, invoice_number) |
| v4.5 | Layout mejorado (DBSCAN, preservación espacial) |
| v4.6 | Detección de tablas con formateo |
| v4.6.1 | Criterios relajados (1+ precio, 1+ fila) |
| v4.7 | Headers/end patterns multi-idioma |
| **v4.7.1** | **Fix detección totales vs productos** |

---

## COMANDOS ÚTILES

```bash
# Rebuild y restart
docker-compose build && docker-compose down && docker-compose up -d

# Test layout
curl -s -X POST http://localhost:8503/process \
  -F "file=@factura.pdf" -F "format=layout" | python3 -c "
import sys,json
d=json.load(sys.stdin)
t=d.get('text','')
lines=[l for l in t.split('\n') if '|' in l]
print(f'Lineas tabla: {len(lines)}')
print(t[:2000])"

# Ver logs de detección de tablas
docker logs paddlepaddle-cpu 2>&1 | grep -i "TABLE-DETECT"

# Commit hash actual
git log --oneline -5
```

---

## SI SE CORTA LA COMUNICACIÓN

1. **Leer este archivo primero**
2. **Estado actual:** v4.7.1 COMMITEADO y funcionando
3. **Branch:** `main`
4. **Último commit:** `70b1751` - fix: v4.7.1 Detección mejorada de tablas vs secciones de totales
5. **Container corriendo:** paddlepaddle-cpu en puerto 8503
6. **Siguiente tarea sugerida:** Probar más facturas internacionales o mejorar line_items
