# CHANGELOG DETALLADO - PaddleOCR Perfection

**Este archivo documenta TODOS los cambios realizados para no perder progreso**

---

## Sesión: 2025-12-07

### 1. DEPENDENCIAS INSTALADAS

#### En el contenedor (ejecutar si se recrea):
```bash
docker exec paddlepaddle-cpu pip install "paddlex[ocr]==3.3.10" scikit-learn
```

#### Ya incluido en Dockerfile (líneas 23-40):
```dockerfile
RUN python3.10 -m pip install --upgrade pip && \
    pip install --break-system-packages --no-cache-dir \
    numpy \
    decord \
    opencv-python==4.6.0.66 \
    paddleocr \
    "paddlex[ocr]==3.3.10" \
    pdf2image==1.16.3 \
    reportlab==4.0.4 \
    pdfplumber \
    Pillow>=10.0.0 \
    PyPDF2 \
    PyMuPDF \
    flask \
    waitress \
    google-generativeai \
    scikit-learn
```

---

### 2. ENDPOINTS NUEVOS CREADOS EN app.py

#### 2.1 Variables globales PP-Structure (línea ~4978)
```python
# Variables globales para pipelines PP-Structure (lazy loading)
pp_table_pipeline = None
pp_layout_pipeline = None
pp_structure_initialized = False
```

#### 2.2 Función init_pp_structure_pipelines (línea ~4983)
```python
def init_pp_structure_pipelines():
    """Inicializa los pipelines de PP-Structure bajo demanda"""
    global pp_table_pipeline, pp_layout_pipeline, pp_structure_initialized

    if pp_structure_initialized:
        return True

    try:
        from paddlex import create_pipeline

        logger.info("[PP-STRUCTURE] Inicializando pipelines...")

        # Pipeline para reconocimiento de tablas (SLANet)
        logger.info("[PP-STRUCTURE] Cargando table_recognition...")
        pp_table_pipeline = create_pipeline(pipeline='table_recognition')

        # Pipeline para análisis de layout
        logger.info("[PP-STRUCTURE] Cargando layout_parsing...")
        pp_layout_pipeline = create_pipeline(pipeline='layout_parsing')

        pp_structure_initialized = True
        logger.info("[PP-STRUCTURE] Pipelines inicializados correctamente")
        return True

    except Exception as e:
        logger.error(f"[PP-STRUCTURE] Error inicializando pipelines: {e}")
        return False
```

#### 2.3 Endpoint /structure (línea ~5012)
- Detecta regiones del documento (tablas, texto, imágenes)
- Extrae tablas como HTML estructurado usando SLANet
- Devuelve coordenadas de cada región

#### 2.4 Endpoint /extract (línea ~5600+)
- Extracción inteligente de campos de facturas
- Usa `extract_invoice_fields()` para KIE
- Devuelve JSON estructurado con confidence score

---

### 3. FUNCIÓN extract_invoice_fields MEJORADA (línea ~5324)

Esta función extrae campos de facturas españolas. Incluye:

#### 3.1 Patrones NIF con corrección OCR:
```python
nif_patterns = [
    r'(?:N\.?I\.?F\.?|C\.?I\.?F\.?)[:\s]*([A-Z]\d{8})',  # B12345678
    r'(?:N\.?I\.?F\.?|C\.?I\.?F\.?)[:\s]*(\d{8}[A-Z])',  # 12345678A
    r'(?:N\.?I\.?F\.?|C\.?I\.?F\.?)[:\s]*([A-Z]?\d{7,8}[A-Z]?)',
]
# Corrección OCR: nif.replace('R', 'B').replace('I', '1').replace('O', '0')
```

#### 3.2 Extracción de IVA con 5 estrategias (línea ~5418):
```python
# Estrategia 1: Patrón "IVA (21%)" seguido de importe
# Estrategia 2: Etiquetas en líneas separadas (formato tickets)
#   - Busca "%IVA" seguido de número en líneas siguientes
#   - Busca "CUOTA" para encontrar tax_amount
# Estrategia 3: Verificación cruzada (total - tax_base = tax_amount)
# Estrategia 4: Cálculo si tenemos base y tasa
# Estrategia 5: Validación de coherencia
```

---

### 4. RESULTADO VERIFICADO CON ticket.pdf

```bash
curl -X POST http://localhost:8503/extract -F "file=@ticket.pdf"
```

```json
{
  "vendor": "E.S.CUATRO OLIVOS S.L.",
  "vendor_nif": "B11368479",
  "invoice_number": "2025003047",
  "date": "21-08-25",
  "total": 66.83,
  "tax_base": 55.23,
  "tax_rate": 21.0,
  "tax_amount": 11.6,
  "confidence": {"score": 1.0, "level": "high"}
}
```

---

### 5. MODELOS PP-STRUCTURE CACHEADOS

Los modelos se descargan automáticamente y se cachean en `/root/.paddlex/official_models/`:
- PP-LCNet_x1_0_doc_ori
- UVDoc
- PP-DocLayout-L
- SLANet_plus
- PP-OCRv4_server_det
- PP-OCRv4_server_rec_doc
- RT-DETR-H_layout_17cls

---

### 6. COMANDOS ÚTILES

```bash
# Verificar paquetes instalados
docker exec paddlepaddle-cpu python3 -c "from paddlex import create_pipeline; print('OK')"

# Test extracción
curl -X POST http://localhost:8503/extract -F "file=@ticket.pdf"

# Test estructura
curl -X POST http://localhost:8503/structure -F "file=@factura.pdf"

# Copiar app.py actualizado al contenedor
docker cp app.py paddlepaddle-cpu:/app/app.py && docker restart paddlepaddle-cpu

# Ver logs
docker logs --tail 100 paddlepaddle-cpu
```

---

### 7. MANEJO DE ERROR std::exception (IMPLEMENTADO)

#### Problema:
- PP-Structure falla intermitentemente con `std::exception` en llamadas consecutivas
- Primera llamada funciona, segunda puede fallar

#### Solución implementada (línea ~5024):
```python
def run_pp_structure_with_retry(pipeline, file_path, max_retries=2):
    """
    Ejecuta un pipeline de PP-Structure con manejo de errores std::exception.
    Si falla con std::exception, reintenta reinicializando el pipeline.
    """
    global pp_table_pipeline, pp_layout_pipeline

    for attempt in range(max_retries + 1):
        try:
            if pipeline == 'table':
                pp = pp_table_pipeline
            else:
                pp = pp_layout_pipeline

            if pp is None:
                if not init_pp_structure_pipelines(force_reinit=True):
                    return None
                pp = pp_table_pipeline if pipeline == 'table' else pp_layout_pipeline

            results = list(pp.predict(file_path))
            return results

        except Exception as e:
            error_str = str(e)
            is_cpp_error = 'std::exception' in error_str or 'Segmentation' in error_str

            if is_cpp_error and attempt < max_retries:
                logger.warning(f"[PP-STRUCTURE] Error C++ detectado, reinicializando...")
                if init_pp_structure_pipelines(force_reinit=True):
                    continue
                else:
                    return None
            else:
                return None

    return None
```

#### También se modificó init_pp_structure_pipelines() para aceptar force_reinit=True

---

### 7.5 DETECCIÓN HÍBRIDA PDF VECTORIAL vs ESCANEADO (IMPLEMENTADO)

El endpoint `/extract` ahora detecta automáticamente el mejor método:

1. **PDFs vectoriales** → `pdftotext -layout` (preserva estructura con espacios)
2. **PDFs escaneados/imágenes** → OCR con PaddleOCR

```python
# Para PDFs, intentar primero con pdftotext -layout
if ext == '.pdf':
    try:
        result = subprocess.run(
            ['pdftotext', '-layout', temp_file_path, '-'],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            pdftotext_output = result.stdout.strip()
            if len(pdftotext_output.replace('\n', '').replace(' ', '')) > 50:
                raw_text = pdftotext_output
                extraction_method = "pdftotext_layout"
    except Exception as e:
        pass

# Si pdftotext no funcionó, usar OCR
if not raw_text:
    # ... llamar OCR interno
    extraction_method = "ocr"
```

**Ventaja**: Los PDFs vectoriales ahora extraen correctamente porque `pdftotext -layout` preserva:
- Alineación de columnas
- Espaciado entre etiqueta y valor
- Estructura tabular

---

### 8. RESULTADOS DE PRUEBAS CON FACTURAS

#### ticket.pdf (ESCANEADO - GASOLINERA) ✅ PERFECTO
```
Vendor: E.S.CUATRO OLIVOS S.L.
NIF: B11368479
Total: 66.83
Tax base: 55.23
Tax rate: 21.0
Tax amount: 11.6
Confidence: 1.0
```

#### Factura noviembre.pdf (OLIVENET) ✅ MEJORADO
```
Extraction method: pdftotext_layout
Vendor: OLIVENET NETWORK S.L.U.
NIF: B93340198
Total: 94.74
Tax base: 78.3
Tax rate: 21.0
Tax amount: 16.44
Date: 05/11/2025
Confidence: 0.9
```

#### Factura_VFR25087570.pdf (VODAFONE) ✅ MEJORADO
```
Extraction method: pdftotext_layout
Vendor: DMI Computer S.A.
NIF: A79522702
Total: 172.12
Tax base: 142.25
Tax rate: 21.0
Tax amount: 29.87
Date: 19/11/25
Confidence: 1.0
```

#### CamScanner 30-08-2025 15.03-página9.pdf - TIMEOUT
- Archivo muy grande (293KB)
- Requiere más tiempo de procesamiento
- Considerar optimización para archivos grandes

---

### 9. PRÓXIMOS PASOS

1. [x] Manejar error std::exception con reintentos - HECHO
2. [x] Probar con facturas de prueba - HECHO (3/4)
3. [ ] Mejorar patrones de extracción para facturas Olivenet y Vodafone
4. [ ] Mejorar format_text_with_layout con clustering de coordenadas X
5. [ ] Optimizar tiempo de procesamiento para archivos grandes

---

## CÓMO RESTAURAR SI SE PIERDE TODO

1. El Dockerfile ya tiene las dependencias (`paddlex[ocr]==3.3.10`, `scikit-learn`)
2. El app.py tiene todos los endpoints y funciones
3. Solo necesitas: `docker-compose build && docker-compose up -d`
4. Los modelos se descargarán automáticamente al primer uso
