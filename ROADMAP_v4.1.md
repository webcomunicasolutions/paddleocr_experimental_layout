# ROADMAP v4.1+ - Próximas Mejoras

**Versión actual:** v4.0.0 ESTABLE
**Fecha:** 2025-12-07

---

## Estado Actual (v4.0 - ESTABLE)

### Funcionalidades completadas:
- OCR con PaddleOCR + diccionario 407 correcciones
- Modo layout con DBSCAN clustering para columnas
- Detección híbrida (pdftotext para vectoriales, OCR para escaneados)
- PP-Structure (tablas HTML con SLANet, layout analysis)
- KIE básico (vendor, NIF, total, tax_base, tax_rate, tax_amount)
- Extracción de customer_name, customer_nif, invoice_number
- LINE_ITEMS básico (descripción + importe)

### Endpoints disponibles:
- `/process` con `format=layout` → Texto perfecto para IA
- `/extract` → JSON estructurado (best-effort)
- `/structure` → Análisis de layout + tablas HTML
- `/ocr` → Endpoint original (compatibilidad n8n)

---

## ROADMAP v4.1 - Optimización de Rendimiento

### 1.1 Manejo de std::exception en PP-Structure
**Problema:** PP-Structure falla intermitentemente con `std::exception`
**Solución:**
- [ ] Pool de pipelines con reinicialización automática
- [ ] Circuit breaker para evitar llamadas repetidas fallidas
- [ ] Fallback a OCR básico si PP-Structure falla

### 1.2 Timeout inteligente para archivos grandes
**Problema:** Archivos grandes (>500KB) pueden tardar mucho
**Solución:**
- [ ] Timeout dinámico basado en tamaño del archivo
- [ ] Procesamiento por páginas para PDFs multipágina
- [ ] Respuesta parcial si se alcanza timeout

### 1.3 Caché de resultados
- [ ] Caché por hash de archivo para evitar reprocesar
- [ ] TTL configurable

---

## ROADMAP v4.2 - LINE_ITEMS Avanzado

### 2.1 Extracción completa de conceptos
**Objetivo:** Reproducir facturas completas con todos los detalles
**Campos a extraer por cada línea:**
- [ ] Descripción del producto/servicio
- [ ] Cantidad
- [ ] Precio unitario
- [ ] Importe (cantidad × precio)
- [ ] IVA aplicado (si es diferente por línea)

### 2.2 Detección de estructura tabular
- [ ] Identificar cabeceras de tabla (Concepto, Cantidad, Precio, Importe)
- [ ] Mapear columnas automáticamente
- [ ] Usar coordenadas X para alinear valores con cabeceras

### 2.3 Formatos de factura específicos
- [ ] Template para facturas de telecomunicaciones (Vodafone, Olivenet)
- [ ] Template para tickets de gasolinera
- [ ] Template para facturas de servicios

---

## ROADMAP v4.3 - Post-procesamiento del Layout

### 3.1 Limpieza de texto
- [ ] Eliminar caracteres basura (artifacts OCR)
- [ ] Unificar espacios y saltos de línea
- [ ] Detectar y corregir palabras cortadas

### 3.2 Normalización de formatos
- [ ] Fechas: convertir a formato ISO (YYYY-MM-DD)
- [ ] Importes: normalizar a formato numérico (1234.56)
- [ ] NIFs: validar formato y dígito de control

### 3.3 Enriquecimiento
- [ ] Detectar idioma del documento
- [ ] Clasificar tipo de documento automáticamente
- [ ] Extraer metadatos (emisor, receptor, período)

---

## Prioridades

| Versión | Funcionalidad | Impacto | Esfuerzo |
|---------|---------------|---------|----------|
| v4.1 | std::exception fix | Alto | Medio |
| v4.1 | Timeout inteligente | Medio | Bajo |
| v4.2 | LINE_ITEMS completo | Alto | Alto |
| v4.3 | Normalización | Medio | Medio |

---

## Notas

- El objetivo principal sigue siendo: **OCR perfecto para pasar a IA**
- KIE es un bonus, no el foco principal
- Cada mejora debe ser incremental y no romper lo existente
- Probar siempre con las facturas de prueba antes de commit
