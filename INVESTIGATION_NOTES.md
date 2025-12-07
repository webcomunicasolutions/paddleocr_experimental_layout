# Notas de Investigacion - std::exception en OCR

## Estado Actual (2025-12-07)

### Problema Principal
El OCR falla intermitentemente con `std::exception`, causando que el modo Layout no funcione correctamente.

### Sintomas Observados
- **Cuando funciona (Normal a las 10:35:17):**
  - `Bloques OCR con coordenadas: 10`
  - Tiempo: ~2.5 segundos
  - Coordenadas capturadas correctamente

- **Cuando falla (Layout a las 10:35:28):**
  - `std::exception` en todas las paginas
  - 5 reintentos por pagina (1 segundo entre cada uno)
  - `Bloques OCR con coordenadas: 0`
  - Tiempo: ~15 segundos (por los reintentos)
  - El texto se extrae via `pdftotext` del PDF (sin coordenadas)

### Logs del Error
```
[OCR] Error en pagina 1 (intento 1): std::exception
[OCR] Esperando 1 segundo antes del siguiente intento...
[OCR] Error en pagina 1 (intento 2): std::exception
... (5 intentos)
[OCR] Error definitivo tras 5 intentos
```

### Hipotesis a Investigar

1. **Problema de Concurrencia**
   - La instancia `ocr_instance` es global y no tiene mutex/lock
   - Si dos requests llegan simultaneamente, puede corromper el estado
   - Ubicacion: `app.py` linea 190: `ocr_instance = None`

2. **Estado Corrupto del Modelo**
   - Tras ciertos procesamientos, el modelo puede quedar en mal estado
   - Posible solucion: reinicializar `ocr_instance` tras errores consecutivos

3. **Memoria del Contenedor Docker**
   - Aunque el servidor es potente, Docker puede tener limites
   - Verificar: `docker stats` durante el procesamiento

4. **Problema con Imagenes Especificas**
   - Ciertas caracteristicas de la imagen pueden causar el crash
   - Probar con diferentes facturas/imagenes

### Codigo Relevante

**Ubicacion del error** - `app.py` lineas 1249-1272:
```python
# Reintentos para OCR
page_ocr_result = None
max_attempts = 5
for attempt in range(1, max_attempts + 1):
    try:
        page_ocr_result = ocr_instance.predict(out_png)  # <-- Aqui falla
        ...
    except Exception as e:
        logger.error(f"[OCR] Error en pagina {page_num} (intento {attempt}): {e}")
```

**Inicializacion OCR** - `app.py` lineas 230-292:
- `ocr_instance` se inicializa una vez al arrancar
- No hay mecanismo de reinicializacion tras errores

### Proximos Pasos

1. **Verificar estado del contenedor:**
   ```bash
   docker ps | grep paddleocr
   docker logs --tail 100 <container>
   ```

2. **Monitorear recursos durante procesamiento:**
   ```bash
   docker stats <container>
   ```

3. **Probar con mutex/lock:**
   - Agregar `threading.Lock()` para proteger `ocr_instance.predict()`

4. **Agregar logging mas detallado:**
   - Capturar el traceback completo del `std::exception`
   - Ver si hay patron (siempre misma pagina, mismo tipo de imagen, etc.)

5. **Probar reinicializacion automatica:**
   - Si hay 3+ errores consecutivos, reinicializar `ocr_instance`

### Archivos Modificados en Esta Sesion

- `app.py` - Implementacion de modo Layout con coordenadas
  - `format_text_with_layout()` - Funcion de reconstruccion espacial
  - `parse_paddleocr_result()` - Conversion de numpy a listas Python
  - `create_spdf()` - Retorna coordenadas OCR
  - `proc_pdf_ocr()` - Acumula coordenadas de todas las paginas
  - `/ocr` endpoint - Incluye `ocr_blocks` y `coordinates` en respuesta
  - `/process` endpoint - Aplica layout cuando se solicita

### Comandos Utiles

```bash
# Reconstruir Docker
docker-compose down && docker-compose build --no-cache && docker-compose up -d

# Ver logs en tiempo real
docker logs -f <container>

# Probar endpoint Normal
curl -X POST http://localhost:8503/process -F "file=@factura.pdf" -F "format=normal"

# Probar endpoint Layout
curl -X POST http://localhost:8503/process -F "file=@factura.pdf" -F "format=layout"

# Entrar al contenedor
docker exec -it <container> bash
```

### Notas Adicionales

- El servidor es potente, no es problema de recursos fisicos
- El error es intermitente - a veces funciona, a veces no
- El modo Normal y Layout usan el mismo pipeline OCR
- La diferencia es que Layout necesita las coordenadas para reconstruir el texto

---

# Test Directo OCR - Verificación de Coordenadas

**Fecha:** 2025-12-07 11:00
**Archivo de prueba:** Factura noviembre.pdf
**Contenedor:** paddlepaddle-cpu (PaddleOCR 3.x via PaddleX)

## Resumen Ejecutivo

✅ **CONFIRMADO: El OCR SÍ genera coordenadas correctamente**

- **Total bloques detectados:** 67 bloques en página 1
- **Coordenadas disponibles:** Cada bloque tiene `rec_polys` con 4 puntos (bounding box)
- **Estructura:** `[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]` en formato numpy array
- **Scores de confianza:** Disponibles en `rec_scores` (0.0 a 1.0)

## Estructura del Resultado OCR

```python
result = ocr.predict('/tmp/factura-test.pdf')
# result[0] = página 1

page = result[0]
page.keys() = {
    'input_path',           # Ruta del archivo
    'page_index',           # Índice de página
    'doc_preprocessor_res', # Resultado preprocesado
    'dt_polys',            # Polígonos detectados
    'model_settings',       # Configuración del modelo
    'text_det_params',      # Parámetros detección
    'text_type',           # Tipo de texto
    'text_rec_score_thresh', # Umbral score
    'return_word_box',      # Si retornar word boxes
    'rec_texts',           # ⭐ TEXTOS RECONOCIDOS
    'rec_scores',          # ⭐ SCORES DE CONFIANZA
    'rec_polys',           # ⭐ COORDENADAS (POLYGONS)
    'vis_fonts',           # Fuentes para visualización
    'textline_orientation_angles', # Ángulos de orientación
    'rec_boxes'            # Bounding boxes
}
```

## Datos de Ejemplo

### Bloque 0
```json
{
  "text": "mn",
  "confidence": 0.38,
  "bbox": {"x": 251, "y": 11, "width": 144, "height": 70},
  "polygon": [[251,11], [395,11], [395,81], [251,81]]
}
```

### Bloque 1
```json
{
  "text": "OLIVENET NETWORK S.L.U.",
  "confidence": 0.99,
  "bbox": {"x": 180, "y": 132, "width": 225, "height": 29},
  "polygon": [[180,132], [405,137], [405,161], [180,156]]
}
```

### Bloque 2
```json
{
  "text": "Dirección: CALLE PEPITA BARRIENTOS, n°7,",
  "confidence": 0.99,
  "bbox": {"x": 180, "y": 176, "width": 355, "height": 28},
  "polygon": [[180,176], [535,180], [535,204], [180,199]]
}
```

## Ordenamiento Espacial

Los bloques están ordenados de **izquierda a derecha, arriba a abajo**:

```
Primeros 15 bloques ordenados:
 0. [ 251, 11] mn
 1. [ 180,132] OLIVENET NETWORK S.L.U.
 2. [ 180,176] Dirección: CALLE PEPITA BARRIENTOS, n°7,
 3. [ 693,171] Ref. Cliente: 221598
 4. [ 693,199] Nombre: Juan Jose Sanchez Bernal
 5. [ 180,203] OFICINA 313 29004 MALAGA
 6. [ 182,231] CIF: B93340198
 7. [ 693,225] Domicilio: PI. De La Ermita, Bldg. 006,
 8. [ 691,252] Tienda
 9. [ 182,260] N° de factura: ON2025-584267
10. [ 692,277] Población: Monda - 29110
11. [ 184,289] Fecha de emisión: 05/11/2025
12. [ 692,304] Provincia: Málaga
13. [ 692,330] Forma de pago : Domiciliado
14. [ 690,350] N° de cuenta: ****** 6067
```

## Agrupación por Filas

Con umbral vertical de ±30px, se detectan **22 filas**:

```
Fila  0: mn
Fila  1: OLIVENET NETWORK S.L.U.
Fila  2: Dirección: CALLE PEPITA BARRIENTOS, n°7, | Ref. Cliente: 221598
Fila  3: OFICINA 313 29004 MALAGA | Nombre: Juan Jose Sanchez Bernal
Fila  4: CIF: B93340198 | Domicilio: PI. De La Ermita, Bldg. 006,
Fila  5: N° de factura: ON2025-584267 | Tienda
Fila  6: Fecha de emisión: 05/11/2025 | Población: Monda - 29110 | Provincia: Málaga
Fila  7: N° de cuenta: ****** 6067 | Forma de pago : Domiciliado
Fila  8: N.I.F.: 78971220F
Fila  9: IVA (21%) | Total Factura | Total (Base Imponible) | 78.30 € | 94.74 € | 16.44 €
```

**Promedio:** 3.0 bloques por fila

## Análisis de Tablas

En el área de tabla (y > 400px) se detectan **51 bloques** agrupados en múltiples filas:

```
Fila  0: IVA (21%) (x=113) | Total Factura (x=113) | Total (Base Imponible) (x=115) | 78.30 € (x=1031) | 94.74 € (x=1031) | 16.44 € (x=1034)
Fila  1: Detalle por productos (x=68) | Concepto (x=82) | Importe (x=1051)
Fila  2: Static IP (01/10/2025 - 31/10/2025) (x=87) | 15.0000 € (x=1034)
Fila  3: Total de días facturados: 31 (x=86) | Central Virtual (01/10/2025  31/10/2025) (x=86) | 12.3967 € (x=1037)
...
```

**Observación importante:** Las coordenadas X permiten distinguir claramente:
- Columna izquierda (conceptos): x ≈ 80-100
- Columna derecha (importes): x ≈ 1030-1070

## Conclusiones

### ✅ Datos Disponibles
1. **Textos:** `rec_texts` - array de strings
2. **Coordenadas:** `rec_polys` - array de polygons (4 puntos cada uno)
3. **Confianza:** `rec_scores` - array de floats (0.0 a 1.0)

### ✅ Formato de Coordenadas
- **Tipo:** numpy.ndarray de shape (4, 2)
- **Formato:** `[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]`
- **Orden:** Top-left, top-right, bottom-right, bottom-left

### ✅ Posibilidades de Layout
1. **Agrupación por filas:** Ordenar por Y, agrupar con umbral ±30px
2. **Detección de columnas:** Analizar distribución X de bloques
3. **Reconstrucción de tablas:** Usar coordenadas para alinear conceptos e importes
4. **Formato estructurado:** JSON con bbox + polygon + text + confidence

## Siguiente Paso

**Implementar endpoint `/process?format=layout`** que:

1. Ordene bloques por posición (Y primero, luego X)
2. Agrupe bloques en filas (umbral Y ±30px)
3. Detecte columnas dentro de cada fila
4. Retorne texto estructurado con alineación preservada

**Ejemplo de salida esperada:**
```
OLIVENET NETWORK S.L.U.

Dirección: CALLE PEPITA BARRIENTOS, n°7,          Ref. Cliente: 221598
OFICINA 313 29004 MALAGA                          Nombre: Juan Jose Sanchez Bernal
CIF: B93340198                                    Domicilio: PI. De La Ermita, Bldg. 006,

Detalle por productos                             Concepto                        Importe
Static IP (01/10/2025 - 31/10/2025)                                              15.0000 €
Central Virtual (01/10/2025 - 31/10/2025)                                        12.3967 €
...
                                                  Total (Base Imponible)         78.30 €
                                                  IVA (21%)                      16.44 €
                                                  Total Factura                  94.74 €
```

## Comando de Test Completo

```bash
# Copiar factura al contenedor
docker cp "/mnt/c/PROYECTOS CLAUDE/paddleocr/facturas_prueba/Factura noviembre.pdf" paddlepaddle-cpu:/tmp/factura-test.pdf

# Test básico
docker exec paddlepaddle-cpu python3 -c "
from paddleocr import PaddleOCR
ocr = PaddleOCR(lang='es')
result = ocr.predict('/tmp/factura-test.pdf')
page = result[0]
print(f'Total bloques: {len(page[\"rec_texts\"])}')
print(f'Coordenadas disponibles: {len(page[\"rec_polys\"])}')
"
```

---

**Ultima actualizacion:** 2025-12-07 11:00
**Estado:** Coordenadas verificadas y funcionando correctamente
**Proxima sesion:** Implementar modo Layout con ordenamiento espacial
