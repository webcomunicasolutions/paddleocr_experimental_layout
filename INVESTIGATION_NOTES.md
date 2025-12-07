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
**Ultima actualizacion:** 2025-12-07
**Proxima sesion:** Continuar investigacion desde WSL con Docker
