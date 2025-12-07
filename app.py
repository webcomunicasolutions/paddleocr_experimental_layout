#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PaddlePaddle CPU Document Preprocessor
Prepara documentos para OCR con deteccion de orientacion y correccion

LAZY LOADING: Flask arranca primero, modelos se cargan después
DEBUG MODE: Logging extensivo para diagnosticar problemas de arranque
"""

import os
import sys
import json
import subprocess
import logging
import time
import math
import tempfile
import threading
import signal
import traceback
from pathlib import Path
from flask import Flask, request, jsonify

# Configurar logging ANTES de cualquier otra cosa
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Función para obtener uso de memoria
def get_memory_usage():
    """Retorna uso de memoria en MB"""
    try:
        import resource
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB to MB
        return f"{mem:.1f}MB"
    except:
        try:
            # Alternativa: leer /proc/self/status
            with open('/proc/self/status', 'r') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        mem_kb = int(line.split()[1])
                        return f"{mem_kb/1024:.1f}MB"
        except:
            return "N/A"
    return "N/A"

# Manejador de señales para debug
def signal_handler(signum, frame):
    sig_name = signal.Signals(signum).name
    logger.error(f"[SIGNAL] Recibida señal {sig_name} ({signum})")
    logger.error(f"[SIGNAL] Stack trace:\n{traceback.format_stack(frame)}")
    sys.exit(1)

# Registrar manejadores de señales
for sig in [signal.SIGTERM, signal.SIGINT]:
    signal.signal(sig, signal_handler)

logger.info(f"[STARTUP] ========== INICIANDO PADDLEOCR V3 ==========")
logger.info(f"[STARTUP] Memoria inicial: {get_memory_usage()}")
logger.info(f"[STARTUP] Python: {sys.version}")
logger.info(f"[STARTUP] PID: {os.getpid()}")

# CONFIGURAR DIRECTORIOS PADDLE ANTES DE IMPORTAR
os.environ['PADDLE_HOME'] = '/home/n8n/.paddleocr'
os.environ['PADDLEX_HOME'] = '/home/n8n/.paddlex'
os.environ['HOME'] = '/home/n8n'

logger.info("[STARTUP] Variables de entorno configuradas")
logger.info(f"[STARTUP] PADDLE_HOME: {os.environ.get('PADDLE_HOME')}")
logger.info(f"[STARTUP] PADDLEX_HOME: {os.environ.get('PADDLEX_HOME')}")

# Imports básicos (rápidos)
logger.info("[STARTUP] Importando OpenCV...")
import cv2
logger.info(f"[STARTUP] OpenCV {cv2.__version__} OK - Memoria: {get_memory_usage()}")

logger.info("[STARTUP] Importando NumPy...")
import numpy as np
logger.info(f"[STARTUP] NumPy {np.__version__} OK - Memoria: {get_memory_usage()}")

# Flask arranca PRIMERO
logger.info("[STARTUP] Creando Flask app...")
app = Flask(__name__)
logger.info(f"[STARTUP] Flask app creada - Memoria: {get_memory_usage()}")

# Variables globales para lazy loading
paddle = None
paddleocr = None
DocImgOrientationClassification = None
models_loaded = False
models_loading = False
models_error = None
startup_time = time.time()

def load_models_background():
    """Carga los modelos de PaddleOCR en segundo plano"""
    global paddle, paddleocr, DocImgOrientationClassification, models_loaded, models_loading, models_error

    # Esperar 5 segundos para que Waitress arranque completamente
    logger.info("[MODELS] Esperando 5 segundos antes de cargar modelos...")
    time.sleep(5)

    logger.info(f"[MODELS] Memoria antes de cargar: {get_memory_usage()}")

    models_loading = True
    logger.info("[MODELS] ========== INICIANDO CARGA DE MODELOS ==========")

    try:
        logger.info("[MODELS] Paso 1/5: Importando paddle...")
        import paddle as _paddle
        paddle = _paddle
        logger.info(f"[MODELS] Paddle {paddle.__version__} importado - Memoria: {get_memory_usage()}")

        logger.info("[MODELS] Paso 2/5: Importando paddleocr...")
        import paddleocr as _paddleocr
        paddleocr = _paddleocr
        logger.info(f"[MODELS] PaddleOCR {paddleocr.__version__} importado - Memoria: {get_memory_usage()}")

        logger.info("[MODELS] Paso 3/5: Importando DocImgOrientationClassification...")
        from paddleocr import DocImgOrientationClassification as _DocImgOrientationClassification
        DocImgOrientationClassification = _DocImgOrientationClassification
        logger.info(f"[MODELS] DocImgOrientationClassification importado - Memoria: {get_memory_usage()}")

        # Ahora inicializar los preprocesadores
        logger.info("[MODELS] Paso 4/5: Inicializando DocPreprocessor...")
        init_docpreprocessor()
        logger.info(f"[MODELS] DocPreprocessor OK - Memoria: {get_memory_usage()}")

        logger.info("[MODELS] Paso 5/5: Inicializando OCR...")
        init_ocr()
        logger.info(f"[MODELS] OCR OK - Memoria: {get_memory_usage()}")

        models_loaded = True
        models_loading = False
        logger.info("[MODELS] ========== TODOS LOS MODELOS CARGADOS ==========")
        logger.info(f"[MODELS] Memoria final: {get_memory_usage()}")

    except Exception as e:
        models_error = str(e)
        models_loading = False
        logger.error(f"[MODELS] ERROR cargando modelos: {e}")
        logger.error(f"[MODELS] Traceback:\n{traceback.format_exc()}")

# NO iniciar el hilo aquí - lo haremos después de que Waitress arranque
logger.info("[STARTUP] Carga de modelos diferida hasta después del arranque del servidor")

# Variables configurables desde ENV
OPENCV_CONFIG = {
    'HSV_LOWER_H': int(os.getenv('OPENCV_HSV_LOWER_H', '0')),
    'HSV_LOWER_S': int(os.getenv('OPENCV_HSV_LOWER_S', '0')),
    'HSV_LOWER_V': int(os.getenv('OPENCV_HSV_LOWER_V', '140')),
    'HSV_UPPER_H': int(os.getenv('OPENCV_HSV_UPPER_H', '180')),
    'HSV_UPPER_S': int(os.getenv('OPENCV_HSV_UPPER_S', '60')),
    'HSV_UPPER_V': int(os.getenv('OPENCV_HSV_UPPER_V', '255')),
    'MIN_AREA_PERCENT': float(os.getenv('OPENCV_MIN_AREA_PERCENT', '0.05')),
    'EPSILON_FACTOR': float(os.getenv('OPENCV_EPSILON_FACTOR', '0.01')),
    'ERODE_ITERATIONS': int(os.getenv('OPENCV_ERODE_ITERATIONS', '1')),
    'DILATE_ITERATIONS': int(os.getenv('OPENCV_DILATE_ITERATIONS', '2')),
    'MIN_WIDTH': int(os.getenv('OPENCV_MIN_WIDTH', '300')),
    'MIN_HEIGHT': int(os.getenv('OPENCV_MIN_HEIGHT', '400')),
    'EROSION_PERCENT': float(os.getenv('OPENCV_EROSION_PERCENT', '0.100')),
    'INNER_SCALE_FACTOR': float(os.getenv('OPENCV_INNER_SCALE_FACTOR', '1.12'))
}

ROTATION_CONFIG = {
    'MIN_CONFIDENCE': float(os.getenv('ROTATION_MIN_CONFIDENCE', '0.7')),
    'MIN_SKEW_ANGLE': float(os.getenv('ROTATION_MIN_SKEW_ANGLE', '0.2'))
}

# Configuracion OCR desde variables de entorno
OCR_CONFIG = {
    'text_det_thresh': float(os.getenv('OCR_TEXT_DET_THRESH', '0.1')),
    'text_det_box_thresh': float(os.getenv('OCR_TEXT_DET_BOX_THRESH', '0.4')),
    'text_det_unclip_ratio': float(os.getenv('OCR_TEXT_DET_UNCLIP_RATIO', '1.5')),
    'text_det_limit_side_len': int(os.getenv('OCR_TEXT_DET_LIMIT_SIDE_LEN', '4800')),
    'text_det_limit_type': os.getenv('OCR_TEXT_DET_LIMIT_TYPE', 'max'),
    'text_recognition_batch_size': int(os.getenv('OCR_TEXT_RECOGNITION_BATCH_SIZE', '6')),
    'textline_orientation_batch_size': int(os.getenv('OCR_TEXTLINE_ORIENTATION_BATCH_SIZE', '1'))
}

# Inicializar DocPreprocessor y OCR globalmente
doc_preprocessor = None
ocr_instance = None
ocr_initialized = False
ocr_consecutive_errors = 0  # Contador de errores consecutivos para auto-reinicialización


def init_docpreprocessor():
#    """Verificar versiones de PaddlePaddle e inicializar PP-LCNet_x1_0_doc_ori"""
    """Verificar versiones de PaddlePaddle e inicializar text_image_orientation"""
    global doc_preprocessor

    try:
        # Verificar versiones instaladas
        import paddle
        logger.info(f"[INIT] PaddlePaddle version: {paddle.__version__}")

        import paddleocr
        logger.info(f"[INIT] PaddleOCR version: {paddleocr.__version__}")

        # Verificar si estamos en CPU o GPU
        logger.info(f"[INIT] Paddle device: {paddle.device.get_device()}")
        logger.info(f"[INIT] CUDA available: {paddle.device.cuda.device_count()}")

        logger.info("[INIT] Inicializando DocImgOrientationClassification...")
        from paddleocr import DocImgOrientationClassification
        # Intentar con configuracion especifica para CPU
        doc_preprocessor = DocImgOrientationClassification(
            model_name="PP-LCNet_x1_0_doc_ori",
            device="cpu"
        )
        logger.info("[OK] DocImgOrientationClassification inicializado correctamente")

        return True

    except Exception as e:
        logger.error(f"[ERROR] Error inicializando DocImgOrientationClassification: {e}")
        import traceback
        logger.error(f"[ERROR TRACEBACK] {traceback.format_exc()}")
        doc_preprocessor = None
        return False


def init_ocr(force=False):
    """Inicializar PaddleOCR con configuracion optimizada desde ENV

    Args:
        force: Si es True, reinicializa aunque ya esté inicializado (para recuperación de errores)
    """
    global ocr_instance, ocr_initialized, ocr_consecutive_errors

    if ocr_initialized and not force:
        return True

    # Si es forzado, limpiar estado previo
    if force:
        logger.warning("[OCR REINIT] Forzando reinicialización del OCR por errores consecutivos...")
        ocr_instance = None
        ocr_initialized = False
        ocr_consecutive_errors = 0

    try:
        logger.info("[OCR INIT] ==========================================================================================")
        logger.info("[OCR INIT]                                Inicializando PaddleOCR                                    ")
        logger.info("[OCR INIT] ==========================================================================================")

        # Verificar versiones
        import paddleocr
        import paddle
        from paddleocr import PaddleOCR
        logger.info(f"[OCR INIT] PaddleOCR version: {paddleocr.__version__}")
        logger.info(f"[OCR INIT] PaddlePaddle version: {paddle.__version__}")
        logger.info(f"[OCR INIT] Device: {paddle.device.get_device()}")

        # Leer configuracion desde ENV
        ocr_config = {
            'ocr_version': os.getenv('OCR_VERSION', 'PP-OCRv3'),
            'lang': os.getenv('OCR_LANG', 'es'),
            'use_doc_orientation_classify': os.getenv('OCR_USE_DOC_ORIENTATION', 'false').lower() == 'true',
            'use_doc_unwarping': os.getenv('OCR_USE_DOC_UNWARPING', 'false').lower() == 'true',
            'use_textline_orientation': os.getenv('OCR_USE_TEXTLINE_ORIENTATION', 'false').lower() == 'true',
            'text_det_thresh': float(os.getenv('OCR_TEXT_DET_THRESH', '0.1')),
            'text_det_box_thresh': float(os.getenv('OCR_TEXT_DET_BOX_THRESH', '0.4')),
            'text_det_limit_side_len': int(os.getenv('OCR_TEXT_DET_LIMIT_SIDE_LEN', '960')),
            'text_det_limit_type': os.getenv('OCR_TEXT_DET_LIMIT_TYPE', 'min'),
            'text_recognition_batch_size': int(os.getenv('OCR_TEXT_RECOGNITION_BATCH_SIZE', '6')),
            'text_det_unclip_ratio': float(os.getenv('OCR_TEXT_DET_UNCLIP_RATIO', '1.5')),
        }

        logger.info("[OCR INIT] Configuracion:")
        logger.info(f"[OCR INIT]   Modelo: {ocr_config['ocr_version']}")
        logger.info(f"[OCR INIT]   Idioma: {ocr_config['lang']}")
        logger.info(f"[OCR INIT]   Deteccion - Umbral: {ocr_config['text_det_thresh']}")
        logger.info(f"[OCR INIT]   Deteccion - Umbral cajas: {ocr_config['text_det_box_thresh']}")
        logger.info(f"[OCR INIT]   Deteccion - Limite lado: {ocr_config['text_det_limit_side_len']}px ({ocr_config['text_det_limit_type']})")
        logger.info(f"[OCR INIT]   Reconocimiento - Batch: {ocr_config['text_recognition_batch_size']}")
        logger.info(f"[OCR INIT]   Orientacion documento: {'SI' if ocr_config['use_doc_orientation_classify'] else 'NO'}")
        logger.info(f"[OCR INIT]   Correccion distorsion: {'SI' if ocr_config['use_doc_unwarping'] else 'NO'}")
        logger.info(f"[OCR INIT]   Orientacion lineas: {'SI' if ocr_config['use_textline_orientation'] else 'NO'}")

        logger.info("[OCR INIT] Cargando modelos...")
        ocr_instance = PaddleOCR(**ocr_config)

        ocr_initialized = True
        logger.info("[OCR INIT] ==========================================================================================")
        logger.info("[OCR INIT] PaddleOCR inicializado correctamente")
        logger.info("[OCR INIT] Modelos cargados en memoria")
        logger.info("[OCR INIT] ==========================================================================================")
        return True

    except Exception as e:
        logger.error(f"[OCR INIT ERROR] Error inicializando PaddleOCR: {e}")
        import traceback
        logger.error(f"[OCR INIT ERROR] {traceback.format_exc()}")
        ocr_instance = None
        ocr_initialized = False
        return False

# LAZY LOADING: No forzar inicializacion al inicio
# Los modelos se cargan en segundo plano via load_models_background()
# init_docpreprocessor() y init_ocr() se llaman cuando se necesiten
logger.info("[START] Inicializacion diferida activada - modelos se cargan en segundo plano")


def find_inner_rectangle(contour, image_shape, config):
    """
    Encuentra el cuadrilátero inscrito dentro del contorno usando erosión morfológica
    para eliminar penínsulas, pero preservando la forma trapezoidal si existe.
    Retorna tanto el trapezoide erosionado como el expandido.
    """
    try:
        # ========================================
        # PASO 1: Crear máscara y aplicar erosión
        # ========================================
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)

        min_dimension = min(image_shape[0], image_shape[1])
        target_erosion_pixels = int(min_dimension * config['EROSION_PERCENT'])

        kernel_size = max(5, int(target_erosion_pixels / 3))
        if kernel_size % 2 == 0:
            kernel_size += 1

        iterations = 3
        actual_erosion = kernel_size * iterations
        actual_percent = (actual_erosion / min_dimension) * 100

        logger.info(f"[IMG] [OCV] [BORDER] Erosion: kernel {kernel_size}x{kernel_size}, {iterations} iter = {actual_erosion}px ({actual_percent:.1f}%)")

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        mask_eroded = cv2.erode(mask, kernel, iterations=iterations)

        # ========================================
        # PASO 2: Encontrar contorno de la máscara erosionada
        # ========================================
        eroded_contours, _ = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not eroded_contours:
            logger.warning("[IMG] [OCV] [BORDER] No se encontraron contornos después de la erosión")
            return None, None, None, None, None, None

        largest_eroded = max(eroded_contours, key=cv2.contourArea)

        # ========================================
        # PASO 3: Aproximar a 4 puntos (preservar trapezoide)
        # ========================================
        epsilon = config['EPSILON_FACTOR'] * cv2.arcLength(largest_eroded, True)
        approx = cv2.approxPolyDP(largest_eroded, epsilon, True)

        if len(approx) != 4:
            for eps_mult in [0.02, 0.03, 0.04, 0.05, 0.01, 0.06, 0.07]:
                epsilon = eps_mult * cv2.arcLength(largest_eroded, True)
                approx = cv2.approxPolyDP(largest_eroded, epsilon, True)
                if len(approx) == 4:
                    break

        # ========================================
        # PASO 4: Obtener puntos erosionados (azul)
        # ========================================
        if len(approx) == 4:
            eroded_pts = approx.reshape(4, 2).astype("float32")
        else:
            points = np.array(largest_eroded).reshape(-1, 2)
            if len(points) > 4:
                rect = cv2.minAreaRect(points.astype(np.float32))
                eroded_pts = cv2.boxPoints(rect).astype("float32")
            else:
                return None, None, None, None, None, None

        # Ordenar puntos erosionados
        s = eroded_pts.sum(axis=1)
        diff = np.diff(eroded_pts, axis=1).flatten()
        tl = eroded_pts[np.argmin(s)]
        br = eroded_pts[np.argmax(s)]
        tr = eroded_pts[np.argmin(diff)]
        bl = eroded_pts[np.argmax(diff)]
        eroded_pts = np.array([tl, tr, br, bl], dtype="float32")

        # ========================================
        # PASO 5: Expandir para crear puntos finales (verde)
        # ========================================
        expanded_pts = eroded_pts.copy()

        if 'INNER_SCALE_FACTOR' in config and config['INNER_SCALE_FACTOR'] != 1.0:
            scale_factor = config['INNER_SCALE_FACTOR']

            # Calcular cuánto expandir en píxeles
            # Basado en el perímetro promedio del trapecio
            perimeter = (np.linalg.norm(eroded_pts[1] - eroded_pts[0]) + 
                        np.linalg.norm(eroded_pts[2] - eroded_pts[1]) +
                        np.linalg.norm(eroded_pts[3] - eroded_pts[2]) +
                        np.linalg.norm(eroded_pts[0] - eroded_pts[3]))

            # Expansión uniforme: cantidad de píxeles a expandir
            expansion_pixels = (perimeter / 4) * (scale_factor - 1.0)

            # Expandir cada lado perpendicularmente
            expanded_pts = []
            for i in range(4):
                p1 = eroded_pts[i]
                p2 = eroded_pts[(i + 1) % 4]
                p_prev = eroded_pts[(i - 1) % 4]
                p_next = eroded_pts[(i + 2) % 4]

                # Vector del lado actual
                side_vec = p2 - p1
                side_len = np.linalg.norm(side_vec)
                if side_len > 0:
                    side_unit = side_vec / side_len
                else:
                    side_unit = np.array([1, 0])

                # Vector perpendicular hacia afuera (rotación 90° antihoraria)
                perp = np.array([-side_unit[1], side_unit[0]])

                # Vector del lado anterior
                prev_vec = p1 - p_prev
                prev_len = np.linalg.norm(prev_vec)
                if prev_len > 0:
                    prev_unit = prev_vec / prev_len
                else:
                    prev_unit = np.array([1, 0])

                # Vector perpendicular del lado anterior
                prev_perp = np.array([prev_unit[1], prev_unit[0]])

                # Promedio de las perpendiculares para la esquina
                corner_direction = (perp + prev_perp) / 2
                corner_dir_len = np.linalg.norm(corner_direction)
                if corner_dir_len > 0:
                    corner_direction = corner_direction / corner_dir_len

                # Expandir el punto
                if i == 0 or i == 2:
                    # Para puntos 0 y 2, invertir el signo de la componente X de corner_direction
                    corner_direction[0] = -corner_direction[0]

                new_pt = p1 - corner_direction * expansion_pixels
                expanded_pts.append(new_pt)

            expanded_pts = np.array(expanded_pts, dtype="float32")

            logger.info(f"[IMG] [OCV] [BORDER] Expansión paralela aplicada: {scale_factor:.2f} ({(scale_factor-1)*100:.0f}%)")
            logger.info(f"[IMG] [OCV] [BORDER] Píxeles de expansión: {expansion_pixels:.1f}px")

        # ========================================
        # PASO 6: Calcular métricas
        # ========================================
        width_top = np.linalg.norm(expanded_pts[1] - expanded_pts[0])
        width_bottom = np.linalg.norm(expanded_pts[2] - expanded_pts[3])
        height_left = np.linalg.norm(expanded_pts[3] - expanded_pts[0])
        height_right = np.linalg.norm(expanded_pts[2] - expanded_pts[1])

        width_avg = (width_top + width_bottom) / 2
        height_avg = (height_left + height_right) / 2
        aspect_ratio = width_avg / height_avg if height_avg > 0 else 1
        aspect_factor = np.power(aspect_ratio, 1/25) if aspect_ratio > 0 else 1

        # Retornar ambos conjuntos de puntos: erosionados y expandidos
        return expanded_pts, eroded_pts, width_avg, height_avg, aspect_ratio, aspect_factor

    except Exception as e:
        logger.error(f"[IMG] [OCV] [BORDER] Error en find_inner_rectangle: {e}")
        return None, None, None, None, None, None


def det_borders(image_path, npy_file, config):
    """
    Detectar contorno del papel y guardar puntos con visualización de tres niveles:
    - Rojo: contorno original
    - Azul: erosionado (sin penínsulas)
    - Verde: expandido final
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error("FALLO: No se pudo leer la imagen")
            return False, None

        visualization = image.copy()
        original_area = image.shape[0] * image.shape[1]
        logger.info(f"[IMG] [OCV] [BORDER] Imagen original: {image.shape[1]}x{image.shape[0]} pixels")

        # Convertir a HSV y crear máscara
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        HSV_LOWER = np.array([config['HSV_LOWER_H'], config['HSV_LOWER_S'], config['HSV_LOWER_V']])
        HSV_UPPER = np.array([config['HSV_UPPER_H'], config['HSV_UPPER_S'], config['HSV_UPPER_V']])
        mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

        # Operaciones morfológicas
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.erode(mask, kernel, iterations=config['ERODE_ITERATIONS'])
        mask = cv2.dilate(mask, kernel, iterations=config['DILATE_ITERATIONS'])
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(visualization, contours, -1, (200, 200, 200), 2)
        logger.info(f"[IMG] [OCV] [BORDER] Total contornos encontrados: {len(contours)}")

        if contours:
            largest = max(contours, key=cv2.contourArea)
            detected_area = cv2.contourArea(largest)
            area_percent = (detected_area / original_area) * 100
            logger.info(f"[IMG] [OCV] [BORDER] Contorno mas grande: {area_percent:.1f}% del area total")

            # Dibujar contorno original en amarillo
            cv2.drawContours(visualization, [largest], -1, (0, 255, 255), 3)

            min_area = original_area * config['MIN_AREA_PERCENT']
            if detected_area > min_area:
                # ========================================
                # TRAPEZOIDE ROJO (original)
                # ========================================
                epsilon = config['EPSILON_FACTOR'] * cv2.arcLength(largest, True)
                approx = cv2.approxPolyDP(largest, epsilon, True)

                if len(approx) == 4:
                    outer_pts = approx.reshape(4, 2).astype("float32")
                else:
                    rect = cv2.minAreaRect(largest)
                    outer_pts = cv2.boxPoints(rect).astype("float32")

                # Ordenar y dibujar en rojo
                s = outer_pts.sum(axis=1)
                diff = np.diff(outer_pts, axis=1).flatten()
                tl = outer_pts[np.argmin(s)]
                br = outer_pts[np.argmax(s)]
                tr = outer_pts[np.argmin(diff)]
                bl = outer_pts[np.argmax(diff)]
                outer_pts = np.array([tl, tr, br, bl], dtype="float32")

                outer_pts_int = outer_pts.astype(int)
                cv2.polylines(visualization, [outer_pts_int], True, (0, 0, 255), 2)

                # ========================================
                # TRAPEZOIDES AZUL Y VERDE (erosionado y expandido)
                # ========================================
                expanded_pts, eroded_pts, width_side_in, height_side_in, aspect_ratio_in, aspect_factor_in = find_inner_rectangle(
                    largest, image.shape, config
                )

                if expanded_pts is not None:
                    # Dibujar trapezoide erosionado en AZUL
                    eroded_pts_int = eroded_pts.astype(int)
                    cv2.polylines(visualization, [eroded_pts_int], True, (255, 0, 0), 3)  # Azul

                    # Dibujar trapezoide expandido en VERDE
                    expanded_pts_int = expanded_pts.astype(int)
                    cv2.polylines(visualization, [expanded_pts_int], True, (0, 255, 0), 4)  # Verde

                    # Marcar vértices del verde (final)
                    for i, pt in enumerate(expanded_pts_int):
                        cv2.circle(visualization, tuple(pt), 8, (0, 255, 0), -1)
                        cv2.circle(visualization, tuple(pt), 10, (255, 255, 255), 2)
                        cv2.putText(visualization, str(i), tuple(pt + [15, -10]),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # Calcular areas
                    green_area = cv2.contourArea(expanded_pts)
                    red_area = cv2.contourArea(outer_pts)

                    # Usar el menor
                    if red_area < green_area:
                        pts_final = outer_pts
                        logger.info(f"[IMG] [OCV] [BORDER] Usando trapezoide ROJO (más pequeño): {red_area:.0f} < {green_area:.0f}")
                    else:
                        pts_final = expanded_pts
                        logger.info(f"[IMG] [OCV] [BORDER] Usando trapezoide VERDE (más pequeño): {green_area:.0f} <= {red_area:.0f}")

                    detection_method = "eroded-expanded"

                else:
                    # Fallback
                    logger.warning("[IMG] [OCV] [BORDER] Fallback: usando contorno reducido")
                    center = np.mean(outer_pts, axis=0)
                    pts_final = []
                    for pt in outer_pts:
                        new_pt = center + (pt - center) * 0.9
                        pts_final.append(new_pt)
                    pts_final = np.array(pts_final, dtype=np.float32)

                    pts_int = pts_final.astype(int)
                    cv2.polylines(visualization, [pts_int], True, (0, 255, 0), 4)
                    detection_method = "fallback"

                # Calcular ángulo
                dx = pts_final[1][0] - pts_final[0][0]
                dy = pts_final[1][1] - pts_final[0][1]
                paper_angle = math.degrees(math.atan2(dy, dx))
                if paper_angle < 0:
                    paper_angle += 360

                # Añadir leyenda
                cv2.putText(visualization, "Metodo: " + detection_method, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(visualization, "Rojo: Original", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(visualization, "Azul: Erosionado", (10, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(visualization, "Verde: Final", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(visualization, f"Area: {area_percent:.1f}%", (10, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(visualization, f"Angulo: {paper_angle:.1f} deg", (10, 170),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Guardar visualización
                output_base = npy_file.replace('.npy', '')
                vis_filename = f"{output_base}.png"
                cv2.imwrite(vis_filename, visualization)
                logger.info(f"[IMG] [OCV] [BORDER] Imagen provisional: {vis_filename}")

                # Guardar puntos finales
                np.save(npy_file, pts_final)
                logger.info(f"[IMG] [OCV] [BORDER] Puntos guardados en {npy_file}")

                return True, f"{detection_method}|{area_percent:.1f}%|{paper_angle:.1f}deg"

            else:
                logger.error(f"FALLO: Area muy pequena ({area_percent:.1f}%)")
                return False, None
        else:
            logger.error("FALLO: No se encontraron contornos")
            return False, None

    except Exception as e:
        logger.error(f"FALLO: {e}")
        return False, None


def fix_perspective(image_path, npy_file, perspective_file, config):
    """
    Corregir perspectiva aplicando factor de aspecto para compensar
    la expansión diferencial en dimensiones
    """
    try:
        image = cv2.imread(image_path)
        pts = np.load(npy_file)

        logger.info(f"[IMG] [OCV] [PERSPECTIVE] Aplicando correccion perspectiva")

        # Ordenar puntos
        s = pts.sum(axis=1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]

        src = np.array([tl, tr, br, bl], dtype="float32")

        # Calcular dimensiones base
        width_base = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
        height_base = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))

        # Aplicar compensación por aspecto si está configurado
        if 'ASPECT_COMPENSATION' in config and config['ASPECT_COMPENSATION']:
            aspect_ratio = width_base / height_base if height_base > 0 else 1
            aspect_factor = np.power(aspect_ratio, 1/25)

            # Ajustar dimensiones con el factor de aspecto
            # Nota: Aquí aplicamos la compensación inversa porque ya se aplicó en la expansión
            width = int(width_base / aspect_factor)
            height = int(height_base * aspect_factor)
            
            logger.info(f"[IMG] [OCV] [PERSPECTIVE] Compensación de aspecto aplicada: {aspect_factor:.3f}")
            logger.info(f"[IMG] [OCV] [PERSPECTIVE] Dimensiones: {width_base}x{height_base} -> {width}x{height}")
        else:
            width = width_base
            height = height_base
        
        # Aplicar límites mínimos
        width = max(width, config.get('MIN_WIDTH', 100))
        height = max(height, config.get('MIN_HEIGHT', 100))
        
        dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(src, dst)
        corrected = cv2.warpPerspective(image, M, (width, height), 
                                       flags=cv2.INTER_CUBIC, 
                                       borderMode=cv2.BORDER_REPLICATE)
        
        cv2.imwrite(perspective_file, corrected, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        
        return True, f"{width}x{height}"
        
    except Exception as e:
        logger.error(f"FALLO: {e}")
        return False, None


def fix_orientation(img_path, doc_preprocessor):
    """
    Detectar y corregir orientacion de imagen
    Returns: (success, orientation_degrees, confidence, rotated)
    """
    try:
        if not doc_preprocessor:
            logger.info("[IMG] [PADDLE] [ORIENTATION] Modelo no disponible")
            return False, 0, 0.0, False

        output = doc_preprocessor.predict(img_path, batch_size=1)
        orientation = '0'
        confidence = 0.0

        for res in output:
            result_data = res.res if hasattr(res, 'res') else res
            if isinstance(result_data, dict):
                label_names = result_data.get('label_names', [])
                scores = result_data.get('scores', [])
                if label_names and scores:
                    orientation = label_names[0]
                    confidence = scores[0]

        # Rotar si es necesario
        rotated = False
        if orientation in ['90', '180', '270'] and confidence > ROTATION_CONFIG['MIN_CONFIDENCE']:
            img = cv2.imread(img_path)
            if orientation == '90':
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif orientation == '180':
                img = cv2.rotate(img, cv2.ROTATE_180)
            elif orientation == '270':
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(img_path, img)
            rotated = True

        return True, int(orientation), confidence, rotated

    except Exception as e:
        logger.warning(f"[IMG] [PADDLE] [ORIENTATION] Error detectando orientacion: {e}")
        return False, 0, 0.0, False


def enhance_image_for_ocr(img_path, enhance_level='auto'):
    """
    Mejora la calidad de imagen para OCR en documentos escaneados/fotos de baja calidad.

    Args:
        img_path: Ruta a la imagen
        enhance_level: 'auto', 'light', 'medium', 'strong' o 'none'

    Returns:
        (success, enhanced, details)
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            return False, False, "No se pudo leer la imagen"

        original_img = img.copy()

        # Convertir a escala de grises para análisis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Analizar la calidad de la imagen
        # - Contraste: desviación estándar de los valores de gris
        # - Brillo: media de los valores de gris
        contrast = gray.std()
        brightness = gray.mean()

        logger.info(f"[ENHANCE] Análisis de imagen: contraste={contrast:.1f}, brillo={brightness:.1f}")

        # Determinar nivel de mejora automáticamente
        # Nota: La binarización agresiva puede empeorar resultados en algunos casos
        # Mejor ser conservador y usar 'medium' como máximo para auto
        if enhance_level == 'auto':
            if contrast < 30:  # Muy bajo contraste (imagen muy lavada)
                enhance_level = 'medium'  # No usar strong automáticamente
            elif contrast < 50:  # Bajo contraste
                enhance_level = 'medium'
            elif contrast < 70:  # Contraste moderado
                enhance_level = 'light'
            else:
                enhance_level = 'none'
            logger.info(f"[ENHANCE] Nivel auto-detectado: {enhance_level}")

        if enhance_level == 'none':
            return True, False, "Imagen con buena calidad, no requiere mejora"

        # Aplicar mejoras según el nivel
        enhanced = gray.copy()

        # 1. Eliminación de ruido (siempre útil)
        if enhance_level in ['light', 'medium', 'strong']:
            enhanced = cv2.fastNlMeansDenoising(enhanced, None, h=10, templateWindowSize=7, searchWindowSize=21)
            logger.debug("[ENHANCE] Aplicado: eliminación de ruido")

        # 2. Mejora de contraste con CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if enhance_level in ['medium', 'strong']:
            clip_limit = 2.0 if enhance_level == 'medium' else 3.0
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            enhanced = clahe.apply(enhanced)
            logger.debug(f"[ENHANCE] Aplicado: CLAHE (clipLimit={clip_limit})")

        # 3. Binarización adaptativa para texto muy difuso
        if enhance_level == 'strong':
            # Usamos binarización adaptativa que funciona mejor con iluminación desigual
            enhanced = cv2.adaptiveThreshold(
                enhanced, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=11,  # Tamaño del bloque para calcular umbral local
                C=2  # Constante a restar del umbral calculado
            )
            logger.debug("[ENHANCE] Aplicado: binarización adaptativa")

        # 4. Sharpening (enfoque) para texto borroso
        if enhance_level in ['medium', 'strong']:
            # Kernel de enfoque
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
            # Solo aplicar enfoque si no está binarizado
            if enhance_level != 'strong':
                enhanced = cv2.filter2D(enhanced, -1, kernel)
                logger.debug("[ENHANCE] Aplicado: enfoque (sharpening)")

        # Convertir de vuelta a BGR para guardar
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        # Guardar imagen mejorada (sobrescribe la original)
        cv2.imwrite(img_path, enhanced_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])

        # Analizar mejora
        new_contrast = enhanced.std()
        logger.info(f"[ENHANCE] Imagen mejorada: contraste {contrast:.1f} -> {new_contrast:.1f}")

        return True, True, f"Mejora aplicada: {enhance_level} (contraste: {contrast:.1f} -> {new_contrast:.1f})"

    except Exception as e:
        logger.error(f"[ENHANCE] Error mejorando imagen: {e}")
        return False, False, str(e)


def fix_deskew(img_path):
    """
    Detectar y corregir inclinacion de imagen usando ImageMagick
    Returns: (success, skew_angle, corrected)
    """
    try:
        # Detectar angulo de inclinacion
        result = subprocess.run(['convert', img_path, '-deskew', '45%', '-format', '%[deskew:angle]', 'info:'],
                              capture_output=True, text=True)

        if result.returncode != 0 or not result.stdout.strip():
            logger.warning("[IMG] [CONVERT] [DESKEW] Error detectando inclinacion")
            return False, 0.0, False

        skew_angle = result.stdout.strip()

        try:
            skew_float = float(skew_angle)
            skew_abs = abs(skew_float)

            corrected = False
            if skew_abs > ROTATION_CONFIG['MIN_SKEW_ANGLE']:
                deskewed_path = img_path.replace('.png', '_deskewed.png')
                result = subprocess.run([
                    'convert', img_path,
                    '-background', 'white',
                    '-interpolate', 'bicubic',
                    '-deskew', '45%',
                    '-fuzz', '10%',
                    '+repage',
                    deskewed_path
                ], capture_output=True, text=True)

                if result.returncode == 0 and os.path.exists(deskewed_path):
                    subprocess.run(['mv', deskewed_path, img_path])
                    corrected = True
                else:
                    logger.warning("[IMG] [CONVERT] [DESKEW] Error aplicando correccion")
                    return False, skew_float, False

            return True, skew_float, corrected

        except ValueError:
            logger.warning(f"[IMG] [CONVERT] [DESKEW] No se pudo parsear angulo: {skew_angle}")
            return False, 0.0, False

    except Exception as e:
        logger.warning(f"[IMG] [CONVERT] [DESKEW] Error procesando inclinacion: {e}")
        return False, 0.0, False


def init_pdf_prep(n8nHomeDir, base_name, ext):
    """Preparacion inicial de PDF - desproteger y copiar"""
    try:
        filename = f"{base_name}{ext}"
        in_file = f"{n8nHomeDir}/in/{filename}"
        out_pdf = f"{n8nHomeDir}/ocr/{base_name}_2.0.preocr.pdf"

        logger.info(f"[PDF] Preparando PDF: {in_file}")

        # Leer configuracion del JSON
        json_file = f"{n8nHomeDir}/json/{filename}.json"
        empresaNif = ""

        if os.path.exists(json_file):
            try:
                result = subprocess.run(['jq', '-r', '.empresaNif // ""', json_file], capture_output=True, text=True)
                if result.returncode == 0:
                    empresaNif = result.stdout.strip()
                    logger.info(f"[JSON] empresaNif: {empresaNif}")
            except Exception as e:
                logger.warning(f"[JSON] Error leyendo JSON: {e}")

        # Verificar si esta protegido
        result = subprocess.run(['pdfinfo', in_file], capture_output=True, text=True)

        if 'Incorrect password' in result.stderr and empresaNif:
            logger.info("[PDF] PDF protegido, desprotegiendo...")

            # Desproteger con empresaNif
            tmp_file = f"{in_file}_unlocked.pdf"
            result = subprocess.run([
                'qpdf', '--password=' + empresaNif, '--decrypt',
                in_file, tmp_file
            ], capture_output=True, text=True)

            if result.returncode == 0 and os.path.exists(tmp_file):
                # Mover archivo desprotegido
                subprocess.run(['mv', tmp_file, in_file])
                logger.info("[PDF] PDF desprotegido correctamente")
            else:
                logger.warning("[PDF] No se pudo desproteger PDF")

        # Copiar a directorio OCR
        subprocess.run(['cp', in_file, out_pdf])
        logger.info(f"[PDF] PDF copiado a {out_pdf}")

        return True

    except Exception as e:
        logger.error(f"[PDF ERROR] {e}")
        return False


def init_img_prep(n8nHomeDir, base_name, ext):
    """Preparacion inicial de imagen - perspectiva y crear PDF"""
    try:
        filename = f"{base_name}{ext}"
        in_file = f"{n8nHomeDir}/in/{filename}"
        out_pdf = f"{n8nHomeDir}/ocr/{base_name}_2.0.preocr.pdf"

        logger.info(f"[IMG] Preparando imagen: {in_file}")

        # Rutas para archivos intermedios
        npy_file = f"{n8nHomeDir}/ocr/{base_name}_1.1.borders.npy"
        ocv_img = f"{n8nHomeDir}/ocr/{base_name}_1.2.ocv.png"
        fallback_pdf_file = f"{n8nHomeDir}/ocr/{base_name}_1.4.ocv.pdf"

        # 1.1. Detectar bordes/contorno
        success, detect_result = det_borders(in_file, npy_file, OPENCV_CONFIG)
        if success:
            logger.info(f"[IMG] [OCV] [BORDER] Resultado OK - {detect_result}")
        else:
            logger.warning(f"[IMG] [OCV] [BORDER] Fallo en deteccion de bordes")

        # 1.2. Corregir perspectiva (solo si 1.1. funciono)
        if os.path.exists(npy_file):
            success, perspective_result = fix_perspective(in_file, npy_file, ocv_img, OPENCV_CONFIG)
            if success:
                logger.info(f"[IMG] [OCV] [PERSPECTIVE] Resultado OK - {perspective_result} pixels")
            else:
                logger.warning("[IMG] [OCV] [PERSPECTIVE] Fallo en correccion de perspectiva")

        # 1.3. Crear PDF preocr
        if os.path.exists(ocv_img):
            result = subprocess.run(['convert', ocv_img, '-compress', 'jpeg', '-quality', '75', out_pdf], capture_output=True, text=True)

            if result.returncode == 0:
                # Mostrar resumen
                final_size_result = subprocess.run(['identify', '-format', '%wx%h', ocv_img], capture_output=True, text=True)

                if final_size_result.returncode == 0:
                    final_size = final_size_result.stdout.strip()
                    logger.info(f"[IMG] [PDF] PDF creado con imagen procesada: {final_size} pixels")
            else:
                logger.error("[IMG] [PDF] Fallo al crear PDF con imagen procesada")

        # 1.3.1. Fallback: crear PDF con imagen original si no existe
        if not os.path.exists(out_pdf):
            result = subprocess.run(['convert', in_file, '-compress', 'jpeg', '-quality', '75', out_pdf], capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("[IMG] [PDF] PDF creado con imagen original")
            else:
                logger.error("[IMG] [PDF] Fallo al crear PDF con imagen original")
                return False

        return True

    except Exception as e:
        logger.error(f"[IMG ERROR] {e}")
        return False


def det_scanned(pdf_path, page_num=1):
    """
    Detectar si una pagina especifica es escaneada o vectorial

    Criterios (OR):
    - Es vectorial si tiene fuentes embebidas
    - Es vectorial si NO tiene ninguna imagen >80% del area de pagina

    Returns: True si es escaneada, False si es vectorial
    """
    try:
        import subprocess
        import fitz  # PyMuPDF

        # 1. Verificar fuentes embebidas
        result = subprocess.run(
            ['pdffonts', '-f', str(page_num), '-l', str(page_num), pdf_path], 
            capture_output=True, text=True
        )

        if result.returncode != 0:
            logger.warning(f"[det_scanned] pdffonts fallo en pagina {page_num}")
            return True  # Asumir escaneada si hay error

        # Contar fuentes embebidas
        embedded_fonts = 0
        lines = result.stdout.splitlines()

        for line in lines[2:]:  # Saltar headers
            if line.strip():
                parts = line.split()
                if len(parts) >= 5 and parts[4] == 'yes':  # columna 'emb'
                    embedded_fonts += 1

        # Si hay fuentes embebidas, es vectorial
        if embedded_fonts > 0:
            logger.info(f"[DET_SCANNED] Detectada pagina VECTORIAL ({embedded_fonts} fuentes embebidas)")
            return False

        # 2. Si no hay fuentes embebidas, verificar imagenes con PyMuPDF
        try:
            pdf = fitz.open(pdf_path)

            # Verificar que la pagina existe
            if page_num > len(pdf):
                logger.warning(f"[det_scanned] Pagina {page_num} no existe")
                return False

            page = pdf[page_num - 1]  # PyMuPDF usa indice base 0

            # Obtener area de la pagina
            page_width = page.rect.width
            page_height = page.rect.height
            page_area = page_width * page_height
            threshold_percentage = 80.0

            logger.info(f"[DET_SCANNED] Pagina: {page_width:.0f}x{page_height:.0f} pts")

            # Obtener todas las imagenes de la pagina
            images = page.get_images(full=True)

            if not images:
                # Sin imagenes = probablemente vectorial puro
                logger.info(f"[DET_SCANNED] Detectada pagina VECTORIAL (sin imagenes, {embedded_fonts} fuentes embebidas)")
                pdf.close()
                return False

            # Verificar el tamano de cada imagen en la pagina
            has_large_image = False

            for img_index, img_info in enumerate(images):
                xref = img_info[0]

                # Obtener los rectangulos donde aparece esta imagen
                try:
                    img_rects = page.get_image_rects(xref)

                    for rect in img_rects:
                        # Calcular area de la imagen
                        img_area = rect.width * rect.height
                        percentage = (img_area / page_area) * 100

                        logger.debug(f"[DET_SCANNED] Imagen {img_index}: {rect.width:.0f}x{rect.height:.0f} pts = {percentage:.1f}% del area")

                        if percentage > threshold_percentage:
                            logger.info(f"[DET_SCANNED] Imagen grande detectada: {percentage:.1f}% del area de pagina")
                            has_large_image = True
                            break

                    if has_large_image:
                        break

                except Exception as e:
                    logger.debug(f"[DET_SCANNED] Error obteniendo rectangulos de imagen {img_index}: {e}")
                    continue

            pdf.close()

            # Determinar resultado
            if has_large_image:
                logger.info(f"[DET_SCANNED] Detectada pagina ESCANEADA (imagen >{threshold_percentage:.0f}% del area)")
                return True
            else:
                logger.info(f"[DET_SCANNED] Detectada pagina VECTORIAL (sin imagenes grandes, {embedded_fonts} fuentes embebidas)")
                return False

        except Exception as e:
            logger.warning(f"[DET_SCANNED] Error usando PyMuPDF: {e}")
            # Fallback: si no podemos verificar imagenes, asumir vectorial si no hay fuentes embebidas
            return False

    except Exception as e:
        logger.error(f"[DET_SCANNED] Error en pagina {page_num}: {e}")
        return True  # Asumir escaneada en caso de error


def extract_pdf_images(n8nHomeDir, base_name, in_pdf, out_png, target_dpi=288):
    """
    Extraer imagenes de PDF vectorial y crear PNG con solo imagenes posicionadas

    Args:
        n8nHomeDir: Directorio base de n8n
        base_name: Nombre base del archivo
        in_pdf: Path del PDF de entrada
        out_png: Path del PNG de salida
        target_dpi: DPI objetivo para el PNG de salida (default: 144)
    """
    try:
        import fitz  # PyMuPDF
        from PIL import Image
        import io

        logger.info(f"[EXTRACT_IMAGES] Extrayendo imagenes de: {in_pdf}")
        logger.info(f"[EXTRACT_IMAGES] DPI objetivo: {target_dpi}")

        # Calcular factor de escala respecto a 72 DPI (base de PDF)
        scale_factor = target_dpi / 72.0

        # Abrir PDF con PyMuPDF
        pdf_document = fitz.open(in_pdf)

        # Procesar primera pagina (PDF individual)
        page = pdf_document[0]

        # Obtener dimensiones de la pagina en puntos
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height

        logger.info(f"[EXTRACT_IMAGES] Pagina original: {page_width:.1f}x{page_height:.1f} pts (72 DPI)")

        # Calcular dimensiones escaladas para el canvas
        canvas_width = int(page_width * scale_factor)
        canvas_height = int(page_height * scale_factor)

        logger.info(f"[EXTRACT_IMAGES] Canvas escalado: {canvas_width}x{canvas_height} px ({target_dpi} DPI)")

        # Obtener lista de imagenes en la pagina
        image_list = page.get_images(full=True)

        logger.info(f"[EXTRACT_IMAGES] Imagenes encontradas: {len(image_list)}")

        if not image_list:
            logger.warning(f"[EXTRACT_IMAGES] Sin imagenes en pagina, creando PNG vacio")
            # Crear PNG vacio con dimensiones escaladas
            empty_img = Image.new('RGB', (canvas_width, canvas_height), 'white')
            empty_img.save(out_png, dpi=(target_dpi, target_dpi))
            pdf_document.close()
            return

        # Crear imagen base con dimensiones escaladas (fondo blanco)
        canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')

        # Contador de imagenes procesadas exitosamente
        images_processed = 0

        # Procesar cada imagen
        for img_index, img_info in enumerate(image_list):
            try:
                # img_info contiene: [xref, smask, width, height, bpc, colorspace, alt_colorspace, name, filter]
                xref = img_info[0]

                # Extraer la imagen
                try:
                    img_dict = pdf_document.extract_image(xref)
                    if not img_dict or "image" not in img_dict:
                        logger.warning(f"[EXTRACT_IMAGES] No se pudo extraer imagen {img_index+1} (xref={xref})")
                        continue

                    # Obtener datos de la imagen
                    img_data = img_dict["image"]
                    img_ext = img_dict["ext"]
                    orig_width = img_dict["width"]
                    orig_height = img_dict["height"]

                    # Crear PIL Image desde los bytes
                    pil_img = Image.open(io.BytesIO(img_data))

                    logger.debug(f"[EXTRACT_IMAGES] Imagen {img_index+1} extraida: {orig_width}x{orig_height} {img_ext}")

                except Exception as e:
                    logger.warning(f"[EXTRACT_IMAGES] Error extrayendo imagen {img_index+1}: {e}")
                    continue

                # Obtener las posiciones de esta imagen en la pagina usando get_image_rects
                try:
                    img_rects = page.get_image_rects(xref)

                    if not img_rects:
                        logger.warning(f"[EXTRACT_IMAGES] No se encontraron posiciones para imagen {img_index+1}")
                        continue

                    # Procesar cada instancia de la imagen (puede aparecer varias veces)
                    for inst_index, rect in enumerate(img_rects):
                        # rect es un fitz.Rect con coordenadas en puntos PDF (72 DPI)
                        x0 = rect.x0
                        y0 = rect.y0
                        rect_width = rect.width
                        rect_height = rect.height

                        # Escalar coordenadas y dimensiones segun el DPI objetivo
                        x_pos = int(x0 * scale_factor)
                        y_pos = int(y0 * scale_factor)
                        target_width = int(rect_width * scale_factor)
                        target_height = int(rect_height * scale_factor)

                        logger.debug(f"[EXTRACT_IMAGES] Imagen {img_index+1}.{inst_index+1}:")
                        logger.debug(f"[EXTRACT_IMAGES]   - Original en PDF: {int(rect_width)}x{int(rect_height)} en ({int(x0)}, {int(y0)}) pts")
                        logger.debug(f"[EXTRACT_IMAGES]   - Escalada a {target_dpi} DPI: {target_width}x{target_height} en ({x_pos}, {y_pos}) px")

                        # Validar dimensiones
                        if target_width <= 0 or target_height <= 0:
                            logger.warning(f"[EXTRACT_IMAGES] Dimensiones invalidas: {target_width}x{target_height}")
                            continue

                        # Redimensionar imagen al tamano escalado
                        try:
                            # Usar LANCZOS para mejor calidad al escalar
                            pil_img_resized = pil_img.resize((target_width, target_height), Image.LANCZOS)
                            logger.debug(f"[EXTRACT_IMAGES] Redimensionada de {orig_width}x{orig_height} a {target_width}x{target_height}")
                        except Exception as resize_err:
                            logger.warning(f"[EXTRACT_IMAGES] Error redimensionando: {resize_err}")
                            continue

                        # Verificar que la imagen cabe en el canvas escalado
                        if (x_pos + target_width > canvas_width) or (y_pos + target_height > canvas_height):
                            logger.warning(f"[EXTRACT_IMAGES] Imagen excede limites del canvas, ajustando")
                            # Ajustar si es necesario
                            if x_pos + target_width > canvas_width:
                                crop_width = canvas_width - x_pos
                                if crop_width > 0:
                                    pil_img_resized = pil_img_resized.crop((0, 0, crop_width, target_height))
                                    target_width = crop_width
                            if y_pos + target_height > canvas_height:
                                crop_height = canvas_height - y_pos
                                if crop_height > 0:
                                    pil_img_resized = pil_img_resized.crop((0, 0, target_width, crop_height))
                                    target_height = crop_height

                        # Pegar en canvas escalado
                        # Las coordenadas Y en PyMuPDF ya tienen origen arriba-izquierda (correcto para PIL)
                        canvas.paste(pil_img_resized, (x_pos, y_pos))
                        images_processed += 1

                        logger.debug(f"[EXTRACT_IMAGES] Imagen pegada en canvas en posicion ({x_pos}, {y_pos})")

                except Exception as e:
                    logger.warning(f"[EXTRACT_IMAGES] Error obteniendo posiciones de imagen {img_index+1}: {e}")
                    continue

            except Exception as e:
                logger.warning(f"[EXTRACT_IMAGES] Error procesando imagen {img_index+1}: {e}")
                continue

        # Informar resultado
        if images_processed == 0:
            logger.warning(f"[EXTRACT_IMAGES] No se pudo procesar ninguna imagen, creando PNG vacio")
            canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
        else:
            logger.info(f"[EXTRACT_IMAGES] Imagenes procesadas exitosamente: {images_processed}")

        # Guardar PNG resultante con metadatos DPI
        canvas.save(out_png, dpi=(target_dpi, target_dpi))
        logger.info(f"[EXTRACT_IMAGES] PNG creado: {out_png} a {target_dpi} DPI")

        # Cerrar documento PDF
        pdf_document.close()

    except Exception as e:
        logger.error(f"[EXTRACT_IMAGES ERROR] Error extrayendo imagenes: {e}")
        # Crear PNG vacio como fallback con DPI por defecto
        from PIL import Image
        fallback_width = int(595 * (target_dpi / 72.0))
        fallback_height = int(842 * (target_dpi / 72.0))
        fallback_img = Image.new('RGB', (fallback_width, fallback_height), 'white')
        fallback_img.save(out_png, dpi=(target_dpi, target_dpi))


def create_spdf(n8nHomeDir, base_name, in_pdf, spdf, page_num, ocr_dpi=288):
    """
    Procesar una pagina individual: escaneada o vectorial
    Genera PDF perfecto de una sola pagina

    Returns:
        Tupla (text_lines, confidences, coordinates) con los datos OCR de la página
    """
    logger.info(f"[CREATE_SPDF] Procesando: {in_pdf}")

    page_start_time = time.time()

    # Detectar tipo de pagina
    page_scanned = det_scanned(in_pdf)

    if page_scanned:

        # Extraer a imagen con ocr_dpi
        subprocess.run(['pdftoppm', '-png', '-f', '1', '-l', '1', '-r', str(ocr_dpi), in_pdf, in_pdf.replace('.pdf', '')], check=True)

        # Detectar y corregir orientacion
        in_png = f"{n8nHomeDir}/ocr/{base_name}_2.2.page-{page_num}.png"
        out_png = f"{n8nHomeDir}/ocr/{base_name}_2.3.page-{page_num}.png"
        subprocess.run(['cp', in_pdf.replace('.pdf', '-1.png'), in_png], check=True)
        subprocess.run(['cp', in_png, out_png], check=True)
        logger.info(f"[ORIENTATION] Detectando orientacion pagina {page_num}...")
        success, degrees, conf, rotated = fix_orientation(out_png, doc_preprocessor)

        if success:
            action = " - CORREGIDO" if rotated else ""
            logger.info(f"[ORIENTATION] Pagina {page_num}: {degrees} grados (confianza: {conf:.3f}){action}")

        # Detectar y corregir inclinacion
        in_png = f"{n8nHomeDir}/ocr/{base_name}_2.3.page-{page_num}.png"
        out_png = f"{n8nHomeDir}/ocr/{base_name}_2.4.page-{page_num}.png"
        subprocess.run(['cp', in_png, out_png], check=True)
        logger.info(f"[DESKEW] Detectando inclinacion pagina {page_num}...")
        success, angle, corrected = fix_deskew(out_png)

        if success:
            action = " - CORREGIDO" if corrected else ""
            logger.info(f"[DESKEW] Pagina {page_num}: {angle:.2f} grados{action}")

        # Mejora de imagen desactivada por defecto (añade tiempo sin mejora consistente)
        # Se puede activar manualmente con enhance_image_for_ocr(out_png, 'medium')

    else:
        # Extraer imagenes a PNG temporal
        out_png = f"{n8nHomeDir}/ocr/{base_name}_2.4.page-{page_num}.png"
        extract_pdf_images(n8nHomeDir, base_name, in_pdf, out_png)

    # Ejecutar OCR sobre la imagen extraida y preparada de la pagina
    logger.info(f"[OCR] Ejecutando OCR en pagina {page_num}...")
    ocr_start = time.time()

    # Reintentos para OCR con auto-reinicialización tras errores consecutivos
    global ocr_consecutive_errors
    page_ocr_result = None
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            page_ocr_result = ocr_instance.predict(out_png)
            ocr_time = time.time() - ocr_start
            if page_ocr_result and len(page_ocr_result) > 0:
                texts = page_ocr_result[0].get('rec_texts', [])
                scores = page_ocr_result[0].get('rec_scores', [])
                avg_conf = sum(scores)/len(scores) if scores else 0
                logger.info(f"[OCR] Pagina {page_num}: {len(texts)} bloques detectados")
                logger.info(f"[OCR] Confianza promedio: {avg_conf:.3f}")
                logger.info(f"[OCR] Tiempo OCR: {ocr_time:.2f}s")
                # Éxito: resetear contador de errores consecutivos
                ocr_consecutive_errors = 0
            else:
                logger.warning(f"[OCR] Pagina {page_num}: Sin texto detectado")
                page_ocr_result = None
            break  # Exito, salir del bucle de reintentos

        except Exception as e:
            ocr_consecutive_errors += 1
            logger.error(f"[OCR] Error en pagina {page_num} (intento {attempt}): {e}")
            logger.error(f"[OCR] Errores consecutivos acumulados: {ocr_consecutive_errors}")

            # Si hay muchos errores consecutivos, forzar reinicialización del OCR
            if ocr_consecutive_errors >= 3 and attempt < max_attempts:
                logger.warning(f"[OCR] {ocr_consecutive_errors} errores consecutivos detectados - reinicializando OCR...")
                if init_ocr(force=True):
                    logger.info("[OCR] OCR reinicializado correctamente, reintentando...")
                else:
                    logger.error("[OCR] Fallo en reinicialización del OCR")

            if attempt < max_attempts:
                logger.info(f"[OCR] Esperando 1 segundo antes del siguiente intento...")
                time.sleep(1)
            else:
                logger.error(f"[OCR] Error definitivo tras {max_attempts} intentos")
                page_ocr_result = None

    # Procesar resultado OCR
    if page_ocr_result and len(page_ocr_result) > 0:
        text_lines, confidences, coordinates = parse_paddleocr_result(page_ocr_result[0])
    else:
        text_lines, confidences, coordinates = [], [], []

    # Crear SPDF con texto buscable
    try:
        if page_scanned:
            # Base: imagen procesada
            compose_pdf_ocr(out_png, (text_lines, confidences, coordinates), spdf, True)
        else:
            # Base: PDF vectorial original
            compose_pdf_ocr(in_pdf, (text_lines, confidences, coordinates), spdf, False)

        logger.info(f"[CREATE_SPDF] PDF con OCR guardado en: {spdf}")

    except Exception as e:
        logger.error(f"[CREATE_SPDF] Error creando PDF final: {e}")
        # Fallback: copiar PDF original
        subprocess.run(['cp', in_pdf, spdf], check=True)
        logger.info(f"[CREATE_SPDF] PDF original copiado como fallback: {spdf}")

    page_time = time.time() - page_start_time
    logger.info(f"[CREATE_SPDF] Pagina {page_num} completada en {page_time:.2f}s: {spdf}")

    # Devolver datos OCR para reconstrucción de layout
    return text_lines, confidences, coordinates


def proc_pdf_ocr(n8nHomeDir, base_name, ext):
    """Procesar PDF: detectar orientacion, corregir inclinacion y ejecutar OCR"""
    global doc_preprocessor, ocr_instance

    try:
        in_pdf = f"{n8nHomeDir}/ocr/{base_name}_2.0.preocr.pdf"
        out_pdf = f"{n8nHomeDir}/ocr/{base_name}_3.0.ocr.pdf"
        final_pdf = f"{n8nHomeDir}/pdf/{base_name}{ext}.pdf"

        logger.info("[PROC_PDF_OCR] ==========================================================================================")
        logger.info(f"[PROC_PDF_OCR] Procesando: {in_pdf}")
        logger.info("[PROC_PDF_OCR] ==========================================================================================")

        # Verificar que exista el archivo
        if not os.path.exists(in_pdf):
            logger.error(f"[PROC_PDF_OCR] Archivo no encontrado: {in_pdf}")
            return False, "File not found", None

        # Verificar modelos inicializados
        if not doc_preprocessor:
            logger.info("[PROC_PDF_OCR] Inicializando modelo de orientacion...")
            if not initialize_docpreprocessor():
                logger.warning("[PROC_PDF_OCR] Modelo de orientacion no disponible, continuando sin rotacion")

        if not ocr_instance:
            logger.info("[PROC_PDF_OCR] Inicializando PaddleOCR...")
            if not initialize_ocr():
                logger.error("[PROC_PDF_OCR] No se pudo inicializar PaddleOCR")
                return False, "OCR initialization failed", None

        # Obtener numero de paginas
        result = subprocess.run(['pdfinfo', in_pdf], capture_output=True, text=True)
        pages = 1
        for line in result.stdout.splitlines():
            if "Pages:" in line:
                pages = int(line.split(":")[1].strip())
                break

        # Extraer paginas individuales en /ocr
        subprocess.run(['pdfseparate', in_pdf, f'{n8nHomeDir}/ocr/{base_name}_2.1.page-%d.pdf'], check=True)
        logger.info(f"[PROC_PDF_OCR] Paginas ({pages}): {base_name}_2.1.page-1.pdf - {base_name}_2.1.page-{pages}.pdf")

        # Procesar cada pagina individualmente
        mpdf = []
        total_start_time = time.time()

        # Acumular datos OCR de todas las páginas para modo Layout
        all_text_lines = []
        all_confidences = []
        all_coordinates = []

        for page in range(1, pages + 1):
            page_pdf = f"{n8nHomeDir}/ocr/{base_name}_2.1.page-{page}.pdf"
            spdf = f"{n8nHomeDir}/ocr/{base_name}_2.6.spdf-{page}.pdf"

            logger.info(f"[PROC_PDF_OCR] =================================  Iniciando pagina {page}/{pages}  ===================================")

            # Procesar pagina individual y obtener datos OCR
            page_texts, page_confs, page_coords = create_spdf(n8nHomeDir, base_name, page_pdf, spdf, page)

            # Acumular datos OCR
            all_text_lines.extend(page_texts)
            all_confidences.extend(page_confs)
            all_coordinates.extend(page_coords)

            # Verificar que se creo correctamente
            if os.path.exists(spdf):
                mpdf.append(spdf)
            else:
                logger.error(f"[PROC_PDF_OCR] Error: No se creo {spdf}")
                return False, f"Failed to create page {page}", None

        # Combinar todas las paginas procesadas
        logger.info(f"[PROC_PDF_OCR] Combinando {len(mpdf)} paginas procesadas...")
        subprocess.run(['pdfunite'] + mpdf + [out_pdf], check=True)
        out_size_kb = os.path.getsize(out_pdf) / 1024
        logger.info(f"[PROC_PDF_OCR] PDF combinado creado ({out_size_kb:.0f}kB): {out_pdf}")

        # Generar en ubicacion final
#        subprocess.run(['cp', out_pdf, final_pdf], check=True)
        subprocess.run([
           'gs',
           '-sDEVICE=pdfwrite',
           '-dCompatibilityLevel=1.4',
           '-dNOPAUSE',
           '-dQUIET',
           '-dBATCH',
           '-dAutoRotatePages=/None',
           '-dColorImageDownsampleType=/Bicubic',
           '-dColorImageResolution=288',
           '-dGrayImageDownsampleType=/Bicubic', 
           '-dGrayImageResolution=288',
           '-dOptimize=true',
           '-dCompressPages=true',
           f'-sOutputFile={final_pdf}',
           out_pdf
        ], check=True)
        final_size_kb = os.path.getsize(final_pdf) / 1024
        logger.info(f"[PROC_PDF_OCR] PDF final creado ({final_size_kb:.0f}kB): {final_pdf}")

        # Calcular estadisticas consolidadas
        total_time = time.time() - total_start_time

        # Leer texto consolidado de todas las paginas (opcional)
        # Esto es para compatibilidad con el sistema anterior
        try:
            # Archivo original (antes de OCR) - importante para PDFs vectoriales
            original_pdf = f"{n8nHomeDir}/in/{base_name}{ext}"

            # Extraer texto del PDF final - dos versiones:
            # 1. Sin layout (rápido, texto plano)
            result_plain = subprocess.run(['pdftotext', final_pdf, '-'], capture_output=True, text=True)
            extracted_text_plain = result_plain.stdout if result_plain.returncode == 0 else ""

            # 2. Con layout del PDF ORIGINAL (para PDFs vectoriales)
            # El PDF original conserva mejor el layout que el procesado por OCR
            if os.path.exists(original_pdf):
                result_layout = subprocess.run(['pdftotext', '-layout', original_pdf, '-'], capture_output=True, text=True)
                extracted_text_layout = result_layout.stdout if result_layout.returncode == 0 else ""
                logger.info(f"[PROC_PDF_OCR] Layout extraido del PDF original: {len(extracted_text_layout)} chars")
            else:
                # Fallback: usar PDF procesado
                result_layout = subprocess.run(['pdftotext', '-layout', final_pdf, '-'], capture_output=True, text=True)
                extracted_text_layout = result_layout.stdout if result_layout.returncode == 0 else ""
                logger.info(f"[PROC_PDF_OCR] Layout extraido del PDF procesado (fallback): {len(extracted_text_layout)} chars")

            text_length = len(extracted_text_plain.strip())

            logger.info("[PROC_PDF_OCR] ==========================================================================================")
            logger.info(f"[PROC_PDF_OCR] Proceso completado exitosamente")
            logger.info(f"[PROC_PDF_OCR] Total paginas procesadas: {pages}")
            logger.info(f"[PROC_PDF_OCR] Caracteres de texto final: {text_length}")
            logger.info(f"[PROC_PDF_OCR] Bloques OCR con coordenadas: {len(all_text_lines)}")
            logger.info(f"[PROC_PDF_OCR] Tiempo total: {total_time:.2f}s")
            logger.info("[PROC_PDF_OCR] ==========================================================================================")

            return True, "Success", {
                'text_lines': extracted_text_plain.splitlines() if extracted_text_plain else [],
                'extracted_text_plain': extracted_text_plain,    # Texto sin layout (rápido)
                'extracted_text_layout': extracted_text_layout,  # Texto con layout (espacial)
                'confidences': all_confidences,
                'coordinates': all_coordinates,  # Coordenadas para modo Layout
                'ocr_blocks': all_text_lines,    # Textos originales del OCR
                'total_blocks': len(all_text_lines),
                'pages': pages,
                'processing_time': total_time
            }

        except Exception as text_error:
            logger.warning(f"[PROC_PDF_OCR] Error extrayendo texto final: {text_error}")

            return True, "Success", {
                'text_lines': all_text_lines,
                'confidences': all_confidences,
                'coordinates': all_coordinates,
                'ocr_blocks': all_text_lines,
                'total_blocks': len(all_text_lines),
                'pages': pages,
                'processing_time': total_time
            }

    except Exception as e:
        logger.error(f"[PROC_PDF_OCR ERROR] Error critico: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, str(e), None


def parse_paddleocr_result(ocr_result):
    """Procesar resultado OCR de PaddleOCR v3 (.predict())"""
    text_lines = []
    confidences = []
    coordinates_list = []

    if not ocr_result:
        return text_lines, confidences, coordinates_list

    def convert_poly_to_list(poly):
        """Convertir ndarray o cualquier tipo a lista de Python"""
        try:
            if hasattr(poly, 'tolist'):
                return poly.tolist()
            elif isinstance(poly, (list, tuple)):
                return [convert_poly_to_list(p) if hasattr(p, 'tolist') else p for p in poly]
            return poly
        except:
            return []

    try:
        logger.info("[OCR PROCESS] Procesando resultado OCR...")

        # PaddleOCR v3 devuelve diccionario con rec_texts, rec_scores, etc.
        if isinstance(ocr_result, dict):
            texts = ocr_result.get('rec_texts', [])
            scores = ocr_result.get('rec_scores', [])
            polys = ocr_result.get('rec_polys', [])

            for i, text in enumerate(texts):
                if text and text.strip():
                    text_lines.append(text.strip())
                    conf = scores[i] if i < len(scores) else 0.0
                    confidences.append(float(conf) if hasattr(conf, 'item') else conf)
                    if i < len(polys):
                        coordinates_list.append(convert_poly_to_list(polys[i]))
                    else:
                        coordinates_list.append([])

        # Si viene en lista (multipagina), procesar cada elemento
        elif isinstance(ocr_result, list):
            for page_result in ocr_result:
                if isinstance(page_result, dict):
                    texts = page_result.get('rec_texts', [])
                    scores = page_result.get('rec_scores', [])
                    polys = page_result.get('rec_polys', [])

                    for i, text in enumerate(texts):
                        if text and text.strip():
                            text_lines.append(text.strip())
                            conf = scores[i] if i < len(scores) else 0.0
                            confidences.append(float(conf) if hasattr(conf, 'item') else conf)
                            if i < len(polys):
                                coordinates_list.append(convert_poly_to_list(polys[i]))
                            else:
                                coordinates_list.append([])

        logger.info(f"[OCR OK] Procesado: {len(text_lines)} bloques detectados")

    except Exception as e:
        logger.error(f"[OCR ERROR] Error procesando resultado OCR: {e}")
        import traceback
        logger.error(traceback.format_exc())

    return text_lines, confidences, coordinates_list


def compose_pdf_ocr(base_source, ocr_data, output_spdf, is_scanned):
    """
    Crear PDF de una pagina con OCR superpuesto

    Args:
        base_source: Path a imagen PNG (escaneada) o PDF original (vectorial)
        ocr_data: Tupla (text_lines, confidences, coordinates) del OCR
        output_spdf: Path donde guardar el PDF resultante
        is_scanned: True para paginas escaneadas, False para vectoriales
    """
    try:
        import io
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
        from PIL import Image

        text_lines, confidences, coordinates = ocr_data

        logger.info(f"[COMPOSE_PDF] Creando PDF {'escaneado' if is_scanned else 'vectorial'} - Base: {base_source}")
        logger.info(f"[COMPOSE_PDF] OCR: {len(text_lines)} bloques de texto")

        # DETECCION UNIFICADA DE DPI
        # Para ambos flujos necesitamos saber el DPI del PNG procesado
        if is_scanned:
            # El PNG es directamente base_source
            png_path = base_source
        else:
            # Para vectorial, el PNG esta en una ruta relacionada
            png_path = base_source.replace('.pdf', '.png').replace('_2.1.page-', '_2.4.page-')

        # Detectar DPI del PNG
        try:
            img_for_dpi = Image.open(png_path)
            source_dpi = img_for_dpi.info.get('dpi', (288, 288))
            if isinstance(source_dpi, tuple):
                source_dpi = source_dpi[0]  # Usar DPI X
            logger.info(f"[COMPOSE_PDF] DPI detectado del PNG: {source_dpi}")
        except Exception as e:
            source_dpi = 288  # Valor por defecto
            logger.warning(f"[COMPOSE_PDF] No se pudo detectar DPI ({e}), asumiendo {source_dpi}")

        if is_scanned:
            # FLUJO ESCANEADO: Imagen como base + texto OCR superpuesto

            # Cargar imagen PNG
            image = Image.open(base_source)
            img_width, img_height = image.size

            # Convertir pixeles a puntos PDF usando el DPI detectado
            pdf_width = (img_width * 72) / source_dpi
            pdf_height = (img_height * 72) / source_dpi

            # Crear PDF con imagen de fondo
            buffer = io.BytesIO()
            c = canvas.Canvas(buffer, pagesize=(pdf_width, pdf_height))

            # Convertir imagen a formato compatible para ReportLab
            img_buffer = io.BytesIO()
            if image.mode in ['RGBA', 'P']:
                image = image.convert('RGB')
            image.save(img_buffer, format='JPEG', quality=75, optimize=True)
            img_buffer.seek(0)

            # Dibujar imagen de fondo
            c.drawImage(ImageReader(img_buffer), 0, 0, pdf_width, pdf_height)

            # Superponer texto OCR invisible
            for i, text in enumerate(text_lines):
                if i < len(coordinates) and len(coordinates[i]) > 0:
                    confidence = confidences[i] if i < len(confidences) else 0.0

                    # Filtrar texto con baja confianza
                    if confidence < 0.3:
                        continue

                    try:
                        coords = coordinates[i]

                        # Calcular coordenadas del texto
                        x_coords = [point[0] for point in coords]
                        y_coords = [point[1] for point in coords]

                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)

                        # Convertir coordenadas de imagen a PDF
                        x_pdf = (x_min / img_width) * pdf_width
                        y_pdf = pdf_height - (y_max / img_height) * pdf_height
                        height_pdf = ((y_max - y_min) / img_height) * pdf_height

                        # Calcular tamano de fuente
                        font_size = max(6, min(height_pdf * 0.8, 20))

                        # Dibujar texto invisible para busqueda
                        c.setFillColorRGB(1, 1, 1, alpha=0.01)  # Casi transparente
                        c.setFont("Helvetica", font_size)
                        c.drawString(x_pdf, y_pdf, text)

                    except Exception as e:
                        logger.debug(f"[COMPOSE_PDF] Error posicionando texto '{text}': {e}")
                        continue

            c.save()

        else:
            # FLUJO VECTORIAL: PDF original como base + texto OCR de imagenes

            import PyPDF2
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter

            # Leer PDF original
            pdf_file = open(base_source, 'rb')
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            original_page = pdf_reader.pages[0]

            # Obtener dimensiones de la pagina original
            media_box = original_page.mediabox
            page_width = float(media_box.width)
            page_height = float(media_box.height)

            # Calcular factor de escala: del DPI del PNG a 72 DPI del PDF
            scale_factor = 72.0 / source_dpi
            logger.info(f"[COMPOSE_PDF] Factor de escala: {scale_factor:.3f} ({source_dpi} DPI -> 72 DPI)")

            # Crear PDF temporal con solo texto OCR
            ocr_buffer = io.BytesIO()
            c = canvas.Canvas(ocr_buffer, pagesize=(page_width, page_height))

            # Solo superponer texto OCR (de imagenes extraidas)
            for i, text in enumerate(text_lines):
                if i < len(coordinates) and len(coordinates[i]) > 0:
                    confidence = confidences[i] if i < len(confidences) else 0.0

                    if confidence < 0.3:
                        continue

                    try:
                        coords = coordinates[i]

                        # Las coordenadas del OCR vienen del PNG a source_dpi
                        # Necesitamos escalarlas a 72 DPI para el PDF
                        x_coords = [point[0] for point in coords]
                        y_coords = [point[1] for point in coords]

                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)

                        # APLICAR FACTOR DE ESCALA a las coordenadas
                        x_pdf = x_min * scale_factor
                        y_pdf = page_height - (y_max * scale_factor)
                        height_pdf = (y_max - y_min) * scale_factor

                        font_size = max(6, min(height_pdf * 0.8, 20))

                        logger.debug(f"[COMPOSE_PDF] Texto '{text[:20]}...': orig({x_min:.0f},{y_min:.0f}) -> pdf({x_pdf:.0f},{y_pdf:.0f})")

                        # Dibujar texto invisible
                        c.setFillColorRGB(1, 1, 1, alpha=0.01)
                        c.setFont("Helvetica", font_size)
                        c.drawString(x_pdf, y_pdf, text)

                    except Exception as e:
                        logger.debug(f"[COMPOSE_PDF] Error posicionando texto vectorial '{text}': {e}")
                        continue

            # Asegurar que siempre hay una pagina aunque no haya texto
            if len(text_lines) == 0:
                c.showPage()  # Crear pagina vacia

            c.save()
            ocr_buffer.seek(0)

            # Combinar PDF original con capa OCR
            from PyPDF2 import PdfWriter

            pdf_writer = PdfWriter()

            # Leer capa OCR
            ocr_pdf = PyPDF2.PdfReader(ocr_buffer)
            ocr_page = ocr_pdf.pages[0]

            # Superponer OCR sobre pagina original
            original_page.merge_page(ocr_page)
            pdf_writer.add_page(original_page)

            # Cerrar archivo
            pdf_file.close()

            # Guardar resultado
            buffer = io.BytesIO()
            pdf_writer.write(buffer)

        # Guardar PDF final
        buffer.seek(0)
        with open(output_spdf, 'wb') as f:
            f.write(buffer.getvalue())

    except Exception as e:
        logger.error(f"[COMPOSE_PDF ERROR] Error creando PDF: {e}")
        import traceback
        logger.error(traceback.format_exc())

        # Fallback: copiar archivo base
        try:
            if is_scanned:
                # Crear PDF simple con la imagen
                from PIL import Image
                image = Image.open(base_source)
                image.save(output_spdf, "PDF", resolution=288.0)
                logger.info(f"[COMPOSE_PDF] Fallback: PDF simple creado desde imagen")
            else:
                # Copiar PDF original
                subprocess.run(['cp', base_source, output_spdf], check=True)
                logger.info(f"[COMPOSE_PDF] Fallback: PDF original copiado")
        except Exception as fallback_error:
            logger.error(f"[COMPOSE_PDF] Error en fallback: {fallback_error}")
            raise


@app.route('/health')
def health():
    """Health check - responde inmediatamente para evitar reinicios"""
    global models_loaded, models_loading, models_error, startup_time

    uptime = time.time() - startup_time
    memory = get_memory_usage()

    # Siempre responde healthy para que EasyPanel no reinicie
    # El estado real se muestra en los campos adicionales
    return jsonify({
        'status': 'healthy',
        'uptime_seconds': round(uptime, 1),
        'memory_usage': memory,
        'models_loaded': models_loaded,
        'models_loading': models_loading,
        'models_error': models_error,
        'preprocessor_ready': doc_preprocessor is not None if models_loaded else False,
        'ocr_ready': ocr_initialized if models_loaded else False,
        'opencv_config': OPENCV_CONFIG,
        'rotation_config': ROTATION_CONFIG
    })


@app.route('/ocr', methods=['POST'])
def ocr():
    """Endpoint OCR - procesa documento completo con orientacion y OCR"""
    global doc_preprocessor, ocr_instance, ocr_initialized, ROTATION_CONFIG
    start_time = time.time()

    try:
        # 1. VALIDACION Y SETUP
        filename_param = request.form.get('filename')
        if not filename_param:
            return jsonify({'error': 'filename required'}), 400

        # Extraer paths y configuracion
        if filename_param.startswith('/'):
            full_path = filename_param
            filename = Path(full_path).name
            n8nHomeDir = str(Path(full_path).parent.parent)
        else:
            filename = filename_param
            n8nHomeDir = request.form.get('n8nHomeDir', '/home/n8n')

        base_name = Path(filename).stem
        ext = Path(filename).suffix.lower()

        logger.info("")
        logger.info("[OCR] ==========================================================================================")
        logger.info(f"[OCR] Procesando: {n8nHomeDir}/in/{filename}")
        logger.info("[OCR] ==========================================================================================")

        # Actualizar MIN_SKEW_ANGLE si se pasa como parametro
        min_angle_param = request.form.get('min_angle')
        if min_angle_param:
            try:
                ROTATION_CONFIG['MIN_SKEW_ANGLE'] = float(min_angle_param)
                logger.info(f"[OCR] MIN_SKEW_ANGLE actualizado a: {ROTATION_CONFIG['MIN_SKEW_ANGLE']}")
            except ValueError:
                logger.warning(f"[OCR] Valor invalido para min_angle: {min_angle_param}")
        else:
            ROTATION_CONFIG['MIN_SKEW_ANGLE'] = float(os.getenv('ROTATION_MIN_SKEW_ANGLE', '0.2'))

        # VERIFICAR Y CARGAR MODELOS SI ES NECESARIO
        if not doc_preprocessor:
            logger.info("[OCR] Modelo de orientacion no cargado, inicializando...")
            if not init_docpreprocessor():
                logger.warning("[OCR] No se pudo cargar modelo de orientacion")

        if not ocr_instance:
            logger.info("[OCR] Modelo OCR no cargado, inicializando...")
            if not init_ocr():
                return jsonify({'error': 'OCR initialization failed'}), 503

        # Verificar que realmente funcionan los modelos
        try:
            # Test rápido para verificar que OCR responde
            test_result = ocr_instance.predict.__name__
        except:
            logger.warning("[OCR] OCR instance no responde, reinicializando...")
            ocr_instance = None
            if not init_ocr():
                return jsonify({'error': 'OCR reinitialization failed'}), 503

        # Crear directorios necesarios
        os.makedirs(f"{n8nHomeDir}/ocr", exist_ok=True)
        os.makedirs(f"{n8nHomeDir}/pdf", exist_ok=True)

        # Verificar que existe archivo de entrada
        in_file = f"{n8nHomeDir}/in/{filename}"
        if not os.path.exists(in_file):
            return jsonify({'error': f'File not found: {in_file}'}), 404

        # PREPARAR ARCHIVO RECIBIDO
        if ext == '.pdf':
            # PREPARACION PDF
            if not init_pdf_prep(n8nHomeDir, base_name, ext):
                return jsonify({'error': 'PDF preparation failed'}), 500
        else:
            # PREPARACION IMAGEN
            if not init_img_prep(n8nHomeDir, base_name, ext):
                return jsonify({'error': 'Image preparation failed'}), 500

        # 3. PROCESAMIENTO OCR (orientacion + OCR integrado)
        logger.info("[OCR] Ejecutando procesamiento OCR completo...")
        success, message, ocr_data = proc_pdf_ocr(n8nHomeDir, base_name, ext)

        if not success:
            logger.error(f"[OCR] Error en procesamiento: {message}")
            return jsonify({'error': message}), 500

        # 4. PREPARAR RESPUESTA
        end_time = time.time()
        duration = end_time - start_time

        # Extraer datos del OCR
        text_lines = ocr_data.get('text_lines', [])
        confidences = ocr_data.get('confidences', [])
        coordinates = ocr_data.get('coordinates', [])  # Coordenadas
        ocr_blocks = ocr_data.get('ocr_blocks', [])    # Bloques OCR originales
        total_blocks = ocr_data.get('total_blocks', 0)
        pages = ocr_data.get('pages', 1)
        # Textos extraídos con/sin layout
        extracted_text_plain = ocr_data.get('extracted_text_plain', '')
        extracted_text_layout = ocr_data.get('extracted_text_layout', '')

        # Calcular estadisticas
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Usar texto plano como default (compatible con versiones anteriores)
        full_text = extracted_text_plain if extracted_text_plain else '\n'.join(text_lines)

        logger.info("[OCR] ==========================================================================================")
        logger.info(f"[OCR STATS] Documento procesado correctamente - Paginas: {pages} - Tiempo {duration:.2f}s")
        logger.info("[OCR] ==========================================================================================")

        return jsonify({
            'success': True,
            'in_file': filename,
            'pdf_file': f"{base_name}.pdf",
            'extracted_text': full_text,              # Texto plano (compatible)
            'extracted_text_plain': extracted_text_plain,   # Sin layout (rápido)
            'extracted_text_layout': extracted_text_layout, # Con layout (espacial)
            'ocr_blocks': ocr_blocks,      # Bloques OCR para modo Layout
            'coordinates': coordinates,     # Coordenadas para modo Layout
            'stats': {
                'total_pages': pages,
                'total_blocks': total_blocks,
                'avg_confidence': round(avg_confidence, 3),
                'processing_time': round(duration, 2)
            }
        })

    except Exception as e:
        logger.error(f"[OCR ERROR] Error en endpoint OCR: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# ============================================================================
# CAPA API REST AÑADIDA (WebComunica API Layer)
# Añadido para proporcionar API REST profesional sobre proyecto de Paco
# ============================================================================

# Estadísticas del servidor (para nuevos endpoints)
server_stats = {
    'startup_time': time.time(),
    'total_requests': 0,
    'successful_requests': 0,
    'failed_requests': 0,
    'total_processing_time': 0.0
}

# Historial de OCR para testing (últimos 50 resultados)
ocr_history = []
MAX_HISTORY = 50

# ============================================================================
# SISTEMA DE DICCIONARIOS OCR (PERSISTENTE)
# ============================================================================
# - Diccionario BASE: Términos fiscales españoles, ciudades, errores comunes
# - Diccionario PERSONALIZADO: Errores detectados por el usuario
# - Patrones REGEX: Correcciones de formato (precios, fechas, etc.)
# ============================================================================

import re

# Ruta para los diccionarios persistentes
DICTIONARY_BASE_PATH = '/app/dictionaries'
DICTIONARY_BASE_FILE = f'{DICTIONARY_BASE_PATH}/base_corrections.json'
DICTIONARY_CUSTOM_FILE = f'{DICTIONARY_BASE_PATH}/custom_corrections.json'
DICTIONARY_REGEX_FILE = f'{DICTIONARY_BASE_PATH}/regex_patterns.json'

# Diccionario BASE (integrado - se usa si no existe el archivo JSON)
OCR_CORRECTIONS_BASE = {
    # ============================================================================
    # DICCIONARIO DE CORRECCIONES OCR - ESPAÑOL
    # Basado en confusiones comunes de caracteres OCR:
    # - 1 <-> l <-> I <-> |  (uno, ele minúscula, i mayúscula, barra)
    # - 0 <-> O <-> o        (cero, o mayúscula, o minúscula)
    # - rn <-> m             (erre-ene vs eme)
    # - c <-> e              (ce vs e)
    # - 5 <-> S              (cinco vs ese)
    # - 8 <-> B              (ocho vs be)
    # - F <-> E <-> P        (mayúsculas similares)
    # ============================================================================

    # === CIUDADES ESPAÑOLAS (con tildes) ===
    'Cag1z': 'Cádiz', 'Cadlz': 'Cádiz', 'Cad1z': 'Cádiz', 'Gad1z': 'Cádiz',
    'Gad17': 'Cádiz', 'CAD1Z': 'CÁDIZ', 'CADLZ': 'CÁDIZ', 'Cadiz': 'Cádiz',
    'Cádlz': 'Cádiz', 'CÁD1Z': 'CÁDIZ', 'CADIZ': 'CÁDIZ',
    'MALAGA': 'MÁLAGA', 'Malaga': 'Málaga', 'MA1AGA': 'MÁLAGA', 'MÄLAGA': 'MÁLAGA',
    'CORDOBA': 'CÓRDOBA', 'Cordoba': 'Córdoba', 'CORDOEA': 'CÓRDOBA', 'C0RDOBA': 'CÓRDOBA',
    'TEREZ': 'JEREZ', 'JERÉZ': 'JEREZ', 'Jeréz': 'Jerez', 'JEREX': 'JEREZ', '1EREZ': 'JEREZ',
    'SEVLLLA': 'SEVILLA', 'SEV1LLA': 'SEVILLA', 'SEVI11A': 'SEVILLA', 'SEVlLLA': 'SEVILLA',
    'ALMERIA': 'ALMERÍA', 'A1MERIA': 'ALMERÍA', 'ALMER1A': 'ALMERÍA', 'ALMERlA': 'ALMERÍA',
    'JAEN': 'JAÉN', '1AEN': 'JAÉN', 'JAÉN': 'JAÉN',
    'HUELVA': 'HUELVA', 'HUE1VA': 'HUELVA', 'HUÉLVA': 'HUELVA',
    'GRANADA': 'GRANADA', '6RANADA': 'GRANADA', 'GRANAD4': 'GRANADA',
    'MADRID': 'MADRID', 'MADR1D': 'MADRID', 'MADRIO': 'MADRID',
    'BARCELONA': 'BARCELONA', 'BARCEL0NA': 'BARCELONA', '8ARCELONA': 'BARCELONA',
    'VALENCIA': 'VALENCIA', 'VA1ENCIA': 'VALENCIA', 'VALENC1A': 'VALENCIA',
    'BILBAO': 'BILBAO', 'B1LBAO': 'BILBAO', 'BILBA0': 'BILBAO',
    'ZARAGOZA': 'ZARAGOZA', 'ZARAG0ZA': 'ZARAGOZA', '2ARAGOZA': 'ZARAGOZA',

    # === TÉRMINOS FISCALES ===
    'XIUA': '%IVA', 'X1VA': '%IVA', 'XTVA': '%IVA', 'ZTUA': '%IVA', '%1VA': '%IVA',
    '1VA': 'IVA', '|VA': 'IVA', 'lVA': 'IVA', 'IVA': 'IVA', '1V4': 'IVA', 'IV4': 'IVA',
    'N.1.F': 'N.I.F', 'N1F': 'NIF', 'N.l.F': 'N.I.F', 'N1F:': 'NIF:', 'NlF': 'NIF',
    'N.1.F.': 'N.I.F.', 'N.l.F.': 'N.I.F.', 'N|F': 'NIF', 'NIF': 'NIF',
    'C.1.F': 'C.I.F', 'C1F': 'CIF', 'C.l.F': 'C.I.F', 'ClF': 'CIF', 'C|F': 'CIF',
    'C.1.F.': 'C.I.F.', 'C.l.F.': 'C.I.F.', 'CIF': 'CIF',
    'lRPF': 'IRPF', '1RPF': 'IRPF', 'IRPE': 'IRPF', '1RPE': 'IRPF',
    'lotal': 'Total', 'Tota1': 'Total', 'TOTA1': 'TOTAL', '7otal': 'Total',
    'T0TAL': 'TOTAL', 'T0tal': 'Total', 'Tota|': 'Total', 'TÓTAL': 'TOTAL',
    'TotalcFactura': 'Total Factura', 'Tota1 Factura': 'Total Factura',
    'Tota1Factura': 'Total Factura', 'TOTA1 FACTURA': 'TOTAL FACTURA',
    'Frecio': 'Precio', 'Prec1o': 'Precio', 'Frec1o': 'Precio', 'FREC1O': 'PRECIO',
    'Prec|o': 'Precio', 'PREC1O': 'PRECIO', 'PR3CIO': 'PRECIO', 'PRECI0': 'PRECIO',
    'Inponible': 'Imponible', 'Inconible': 'Imponible', '1mponible': 'Imponible',
    'lmponible': 'Imponible', 'Irnponible': 'Imponible', '|mponible': 'Imponible',
    'Pase Imponible': 'Base Imponible', 'Pase Inponible': 'Base Imponible',
    'Rase Imponible': 'Base Imponible', 'Rase Inponible': 'Base Imponible',
    'Rase Inconible': 'Base Imponible', 'BASE 1MPONIBLE': 'BASE IMPONIBLE',
    'BASE lMPONIBLE': 'BASE IMPONIBLE', '8ASE IMPONIBLE': 'BASE IMPONIBLE',
    'Inporte': 'Importe', 'Imeorte': 'Importe', 'lmporte': 'Importe', '1mporte': 'Importe',
    'Irnporte': 'Importe', '|mporte': 'Importe', 'IMFORTE': 'IMPORTE', '1MPORTE': 'IMPORTE',
    'Subtota1': 'Subtotal', 'SUBTOTA1': 'SUBTOTAL', 'SU8TOTAL': 'SUBTOTAL',
    'Sub7otal': 'Subtotal', 'SUBTOTAI': 'SUBTOTAL', 'SUBTOTA|': 'SUBTOTAL',
    'Descuento': 'Descuento', 'Oescuento': 'Descuento', 'Oescuerito': 'Descuento',
    'DESCUENT0': 'DESCUENTO', 'D3SCUENTO': 'DESCUENTO', 'OESCUENTO': 'DESCUENTO',
    'Cuota': 'Cuota', 'Cu0ta': 'Cuota', 'CUOTA': 'CUOTA', 'CU0TA': 'CUOTA',
    'Exento': 'Exento', 'Exent0': 'Exento', '3xento': 'Exento',

    # === CONCEPTOS COMUNES FACTURAS/TICKETS ===
    'GASOLEO': 'GASÓLEO', 'GASOLEOA': 'GASÓLEO A', 'GASOLED': 'GASÓLEO',
    'GAS0LEO': 'GASÓLEO', '6ASOLEO': 'GASÓLEO', 'GASOLE0': 'GASÓLEO',
    'GASOLINA': 'GASOLINA', 'GASOL1NA': 'GASOLINA', 'GAS0LINA': 'GASOLINA',
    '6ASOLINA': 'GASOLINA', 'GASO1INA': 'GASOLINA', 'GASOLlNA': 'GASOLINA',
    'DIESEL': 'DIÉSEL', 'D1ESEL': 'DIÉSEL', 'DIESE1': 'DIÉSEL', 'DlESEL': 'DIÉSEL',
    'Factura': 'Factura', 'EACTURA': 'FACTURA', 'Eactura': 'Factura',
    'FACTURA': 'FACTURA', 'F4CTURA': 'FACTURA', 'EACTÜRA': 'FACTURA',
    'Recibo': 'Recibo', 'Rec1bo': 'Recibo', 'RECI8O': 'RECIBO', 'REClBO': 'RECIBO',
    'Ticket': 'Ticket', 'T1cket': 'Ticket', 'Tlcket': 'Ticket', 'TICKET': 'TICKET',
    'Albaran': 'Albarán', 'Albarán': 'Albarán', 'A1baran': 'Albarán', 'ALBARAN': 'ALBARÁN',
    'ELECTR1CIDAD': 'ELECTRICIDAD', 'E1ectricidad': 'Electricidad',
    'ELECTRIC1DAD': 'ELECTRICIDAD', '3LECTRICIDAD': 'ELECTRICIDAD', 'ELECTRlCIDAD': 'ELECTRICIDAD',
    'Serv1cio': 'Servicio', 'SERV1CIO': 'SERVICIO', 'SERVlCIO': 'SERVICIO',
    'Servlcio': 'Servicio', '5ERVICIO': 'SERVICIO',
    'Articulo': 'Artículo', 'Art1culo': 'Artículo', 'ARTICUL0': 'ARTÍCULO',
    'Producto': 'Producto', 'Pr0ducto': 'Producto', 'PR0DUCTO': 'PRODUCTO',
    'Concepto': 'Concepto', 'C0ncepto': 'Concepto', 'Concep7o': 'Concepto',
    'Descripcion': 'Descripción', 'Descr1pcion': 'Descripción', 'Descripc1on': 'Descripción',

    # === UNIDADES DE MEDIDA ===
    '1itros': 'litros', 'L1TROS': 'LITROS', '1itro': 'litro', 'llTROS': 'LITROS',
    'litros': 'litros', 'LlTROS': 'LITROS', '1ITROS': 'LITROS', 'L1TR0S': 'LITROS',
    'Un1dades': 'Unidades', 'UN1DADES': 'UNIDADES', 'Unldades': 'Unidades',
    'UNIDADE5': 'UNIDADES', 'UN1DADE5': 'UNIDADES',
    'Cant1dad': 'Cantidad', 'CANT1DAD': 'CANTIDAD', 'Cantldad': 'Cantidad',
    'CANTID4D': 'CANTIDAD', 'CÄNTIDAD': 'CANTIDAD',
    'Kilo': 'Kilo', 'K1lo': 'Kilo', 'Ki1o': 'Kilo', 'Kl1o': 'Kilo',
    'Kilos': 'Kilos', 'K1los': 'Kilos', 'Ki1os': 'Kilos',
    'Gramos': 'Gramos', '6ramos': 'Gramos', 'Grarn0s': 'Gramos',

    # === FECHAS Y TIEMPO ===
    'Fecha': 'Fecha', 'EECHA': 'FECHA', 'Eecha': 'Fecha', 'F3CHA': 'FECHA',
    'FECH4': 'FECHA', 'E3CHA': 'FECHA', 'Fecba': 'Fecha',
    'Venc1miento': 'Vencimiento', 'VENC1MIENTO': 'VENCIMIENTO', 'Venclmiento': 'Vencimiento',
    'Vencirniento': 'Vencimiento', 'VENCIM1ENTO': 'VENCIMIENTO',
    'Em1sión': 'Emisión', 'EM1SIÓN': 'EMISIÓN', 'Emlsión': 'Emisión',
    'Emlsion': 'Emisión', 'Ernisión': 'Emisión',
    'Caducidad': 'Caducidad', 'Caduc1dad': 'Caducidad', 'CADUCIDAD': 'CADUCIDAD',

    # === DATOS DE EMPRESA/CLIENTE ===
    'NoFactura': 'Nº Factura', 'N°Factura': 'Nº Factura', 'N.Factura': 'Nº Factura',
    'No.Factura': 'Nº Factura', 'NºFactura': 'Nº Factura', 'N° Factura': 'Nº Factura',
    'clte.': 'cliente', 'Clte.': 'Cliente', 'C1te.': 'Cliente', 'Clte': 'Cliente',
    'Cliente': 'Cliente', 'Cl1ente': 'Cliente', 'C1iente': 'Cliente', 'CL1ENTE': 'CLIENTE',
    'Direccion': 'Dirección', 'Dlrección': 'Dirección', 'D1rección': 'Dirección',
    'DIRECC1ON': 'DIRECCIÓN', 'DIRECCION': 'DIRECCIÓN',
    'Telefono': 'Teléfono', 'Te1éfono': 'Teléfono', 'Teléfon0': 'Teléfono',
    'TELEF0NO': 'TELÉFONO', 'TEL3FONO': 'TELÉFONO',
    'Domicilio': 'Domicilio', 'Dom1cilio': 'Domicilio', 'Domlcilio': 'Domicilio',
    'donicilio': 'domicilio', 'DONICILIO': 'DOMICILIO', 'DONIC1LIO': 'DOMICILIO',
    'Correo': 'Correo', 'C0rreo': 'Correo', 'Corre0': 'Correo',
    'Email': 'Email', 'Emai1': 'Email', 'Ema1l': 'Email', '3mail': 'Email',
    'adninistracion': 'administración', 'adrninistración': 'administración',
    'adm1nistracion': 'administración', 'ADMINISTRACION': 'ADMINISTRACIÓN',

    # === CÓDIGOS POSTALES Y CALLES ===
    'C401Z': 'CÁDIZ', '0401Z': 'CÁDIZ', 'CAD1Z': 'CÁDIZ',
    '11405C401Z': '11405 CÁDIZ', '11510C401Z': '11510 CÁDIZ',
    'tra.': 'Ctra.', 'ctra.': 'Ctra.', 'Ctra': 'Ctra.', 'CTRA': 'CTRA.',
    'Carretera': 'Carretera', 'Carr3tera': 'Carretera', 'Carret3ra': 'Carretera',
    'Avda': 'Avda.', 'AVDA': 'AVDA.', 'Avenida': 'Avenida', 'Aven1da': 'Avenida',
    'Calle': 'Calle', 'Ca1le': 'Calle', 'Cal1e': 'Calle', 'CALLE': 'CALLE',
    'Plaza': 'Plaza', 'P1aza': 'Plaza', 'Plaz4': 'Plaza', 'PLAZA': 'PLAZA',

    # === ESTACIONES DE SERVICIO ===
    'ESERVICtOS': 'E.S. SERVICIOS', 'ESERVIC1OS': 'E.S. SERVICIOS',
    'E.S.': 'E.S.', 'ES.': 'E.S.', 'E5.': 'E.S.', 'E.5.': 'E.S.',
    'ESTAC1ON': 'ESTACIÓN', 'ESTACION': 'ESTACIÓN', 'ESTAC10N': 'ESTACIÓN',
    'Estación': 'Estación', 'Estac1ón': 'Estación', 'Estacion': 'Estación',
    'Surtidor': 'Surtidor', 'Surt1dor': 'Surtidor', 'SURTID0R': 'SURTIDOR',
    'Manguera': 'Manguera', 'Mangue|ra': 'Manguera', 'M4nguera': 'Manguera',
    'Olivag.': 'Olivos',

    # === FORMAS DE PAGO ===
    'Efectivo': 'Efectivo', 'Efect1vo': 'Efectivo', '3fectivo': 'Efectivo',
    'EFECTIV0': 'EFECTIVO', 'EFECTIVO': 'EFECTIVO',
    'Tarjeta': 'Tarjeta', 'Tarj3ta': 'Tarjeta', 'Tarje7a': 'Tarjeta', 'TARJ3TA': 'TARJETA',
    'Credito': 'Crédito', 'Créd1to': 'Crédito', 'Crédlto': 'Crédito',
    'Debito': 'Débito', 'Déb1to': 'Débito', 'Déblto': 'Débito',
    'Transferencia': 'Transferencia', 'Transfer3ncia': 'Transferencia',
    'Pago': 'Pago', 'Pag0': 'Pago', 'PAG0': 'PAGO',
    'Cambio': 'Cambio', 'Camb1o': 'Cambio', 'Carnbio': 'Cambio',
    'Entregado': 'Entregado', 'Entregad0': 'Entregado', '3ntregado': 'Entregado',

    # === NÚMEROS CONFUNDIDOS CON LETRAS ===
    'l0': '10', '1O': '10', 'lO': '10',
    '2O': '20', '2o': '20',
    '3O': '30', '3o': '30',
    '4O': '40', '4o': '40',
    '5O': '50', '5o': '50',

    # === SÍMBOLOS Y PUNTUACIÓN ===
    'EUR': 'EUR', '3UR': 'EUR', 'EÜR': 'EUR',
    'EUROS': 'EUROS', '3UROS': 'EUROS', 'EUR0S': 'EUROS',

    # === PALABRAS COMUNES CON ERRORES FRECUENTES ===
    'rnismo': 'mismo', 'misrno': 'mismo', 'm1smo': 'mismo',
    'rnás': 'más', 'mas': 'más', 'rnenos': 'menos', 'men0s': 'menos',
    'corno': 'como', 'corn0': 'como', 'c0mo': 'como',
    'rnuy': 'muy', 'mny': 'muy',
    'sólo': 'sólo', 'so1o': 'sólo', 's0lo': 'sólo',
    'número': 'número', 'númer0': 'número', 'núrnero': 'número', 'nurnero': 'número',
    'según': 'según', 'segün': 'según', '5egún': 'según',
    'también': 'también', 'tamb1én': 'también', 'tarnbién': 'también',
    'información': 'información', 'informac1ón': 'información', 'inforrnación': 'información',
}

# Diccionario PERSONALIZADO (se carga desde archivo)
OCR_CORRECTIONS_CUSTOM = {}

# Diccionario combinado (se actualiza al cargar)
OCR_CORRECTIONS = {}

# Patrones regex para correcciones de formato
OCR_REGEX_CORRECTIONS = [
    # Comas en precios: 55:23 -> 55,23 (dos dígitos después de :)
    (re.compile(r'(\d+):(\d{2})(?!\d)'), r'\1,\2'),
    # Comas en precios con 3 dígitos: 1:009 -> 1,009
    (re.compile(r'(\d+):(\d{3})(?!\d)'), r'\1,\2'),
    # Puntos como separador decimal en contexto de precio: 11.60 -> 11,60 (solo si parece precio)
    # No aplicar porque el punto es correcto en muchos contextos

    # IMPORTANTE: NO usar \s para espacios porque incluye \n y destruye el layout
    # Usar [ \t] (solo espacios y tabs horizontales) o simplemente ' ' (espacio)
    # DESACTIVADO para preservar el layout de pdftotext:
    # (re.compile(r'[ \t]{3,}'), '  '),  # Solo espacios horizontales, no newlines
]


def load_dictionaries():
    """
    Carga los diccionarios desde archivos JSON.
    Si no existen, crea los archivos con los valores por defecto.
    """
    global OCR_CORRECTIONS, OCR_CORRECTIONS_CUSTOM, OCR_CORRECTIONS_BASE

    try:
        os.makedirs(DICTIONARY_BASE_PATH, exist_ok=True)

        # Cargar diccionario base
        if os.path.exists(DICTIONARY_BASE_FILE):
            with open(DICTIONARY_BASE_FILE, 'r', encoding='utf-8') as f:
                loaded_base = json.load(f)
                OCR_CORRECTIONS_BASE.update(loaded_base)
                logger.info(f"[DICTIONARY] Cargado diccionario base: {len(loaded_base)} entradas")
        else:
            # Guardar diccionario base por defecto
            with open(DICTIONARY_BASE_FILE, 'w', encoding='utf-8') as f:
                json.dump(OCR_CORRECTIONS_BASE, f, ensure_ascii=False, indent=2)
            logger.info(f"[DICTIONARY] Creado diccionario base con {len(OCR_CORRECTIONS_BASE)} entradas")

        # Cargar diccionario personalizado
        if os.path.exists(DICTIONARY_CUSTOM_FILE):
            with open(DICTIONARY_CUSTOM_FILE, 'r', encoding='utf-8') as f:
                OCR_CORRECTIONS_CUSTOM = json.load(f)
                logger.info(f"[DICTIONARY] Cargado diccionario personalizado: {len(OCR_CORRECTIONS_CUSTOM)} entradas")
        else:
            OCR_CORRECTIONS_CUSTOM = {}
            with open(DICTIONARY_CUSTOM_FILE, 'w', encoding='utf-8') as f:
                json.dump(OCR_CORRECTIONS_CUSTOM, f, ensure_ascii=False, indent=2)
            logger.info(f"[DICTIONARY] Creado diccionario personalizado vacío")

        # Combinar diccionarios (personalizado tiene prioridad)
        OCR_CORRECTIONS = {**OCR_CORRECTIONS_BASE, **OCR_CORRECTIONS_CUSTOM}
        logger.info(f"[DICTIONARY] Total de correcciones activas: {len(OCR_CORRECTIONS)}")

    except Exception as e:
        logger.error(f"[DICTIONARY] Error cargando diccionarios: {e}")
        # Usar diccionario en memoria como fallback
        OCR_CORRECTIONS = OCR_CORRECTIONS_BASE.copy()


def save_custom_dictionary():
    """Guarda el diccionario personalizado a archivo JSON"""
    global OCR_CORRECTIONS, OCR_CORRECTIONS_CUSTOM
    try:
        os.makedirs(DICTIONARY_BASE_PATH, exist_ok=True)
        with open(DICTIONARY_CUSTOM_FILE, 'w', encoding='utf-8') as f:
            json.dump(OCR_CORRECTIONS_CUSTOM, f, ensure_ascii=False, indent=2)
        # Actualizar diccionario combinado
        OCR_CORRECTIONS = {**OCR_CORRECTIONS_BASE, **OCR_CORRECTIONS_CUSTOM}
        logger.info(f"[DICTIONARY] Guardado diccionario personalizado: {len(OCR_CORRECTIONS_CUSTOM)} entradas")
        return True
    except Exception as e:
        logger.error(f"[DICTIONARY] Error guardando diccionario: {e}")
        return False


def add_correction(wrong, correct, dictionary='custom'):
    """Añade una corrección al diccionario"""
    global OCR_CORRECTIONS_CUSTOM, OCR_CORRECTIONS_BASE
    if dictionary == 'custom':
        OCR_CORRECTIONS_CUSTOM[wrong] = correct
        return save_custom_dictionary()
    elif dictionary == 'base':
        OCR_CORRECTIONS_BASE[wrong] = correct
        try:
            with open(DICTIONARY_BASE_FILE, 'w', encoding='utf-8') as f:
                json.dump(OCR_CORRECTIONS_BASE, f, ensure_ascii=False, indent=2)
            load_dictionaries()
            return True
        except Exception as e:
            logger.error(f"[DICTIONARY] Error guardando diccionario base: {e}")
            return False
    return False


def remove_correction(wrong, dictionary='custom'):
    """Elimina una corrección del diccionario"""
    global OCR_CORRECTIONS_CUSTOM, OCR_CORRECTIONS_BASE
    if dictionary == 'custom' and wrong in OCR_CORRECTIONS_CUSTOM:
        del OCR_CORRECTIONS_CUSTOM[wrong]
        return save_custom_dictionary()
    elif dictionary == 'base' and wrong in OCR_CORRECTIONS_BASE:
        del OCR_CORRECTIONS_BASE[wrong]
        try:
            with open(DICTIONARY_BASE_FILE, 'w', encoding='utf-8') as f:
                json.dump(OCR_CORRECTIONS_BASE, f, ensure_ascii=False, indent=2)
            load_dictionaries()
            return True
        except Exception as e:
            logger.error(f"[DICTIONARY] Error guardando diccionario base: {e}")
            return False
    return False


# Cargar diccionarios al iniciar
load_dictionaries()


def apply_ocr_corrections(text):
    """
    Aplica correcciones de diccionario al texto extraído por OCR.

    Args:
        text: Texto extraído del OCR

    Returns:
        Texto con correcciones aplicadas
    """
    if not text:
        return text

    corrected = text
    corrections_applied = 0

    # Aplicar correcciones de diccionario
    for wrong, correct in OCR_CORRECTIONS.items():
        if wrong in corrected:
            corrected = corrected.replace(wrong, correct)
            corrections_applied += 1

    # Aplicar correcciones regex
    for pattern, replacement in OCR_REGEX_CORRECTIONS:
        if pattern.search(corrected):
            corrected = pattern.sub(replacement, corrected)
            corrections_applied += 1

    if corrections_applied > 0:
        logger.info(f"[OCR CORRECTIONS] Aplicadas {corrections_applied} correcciones de diccionario")

    return corrected


# ============================================================================
# FUNCIÓN LAYOUT: Reconstrucción espacial usando coordenadas de bounding boxes
# ============================================================================
# Esta función usa las coordenadas X,Y de cada bloque de texto detectado
# para reconstruir la estructura visual del documento, similar a LLMWhisperer.
# MEJORADO v4.0: Usa DBSCAN para detectar columnas automáticamente.
# ============================================================================

def detect_columns_dbscan(x_positions, eps_factor=0.05):
    """
    Detecta columnas usando DBSCAN clustering en las posiciones X.

    Args:
        x_positions: Lista de coordenadas X (x_min de cada bloque)
        eps_factor: Factor para calcular eps relativo al ancho del documento

    Returns:
        Lista de tuplas (x_inicio, x_fin) para cada columna detectada
    """
    if not x_positions or len(x_positions) < 2:
        return [(0, 1000)]  # Una sola columna por defecto

    try:
        from sklearn.cluster import DBSCAN
        import numpy as np

        # Preparar datos para DBSCAN
        X = np.array(x_positions).reshape(-1, 1)

        # Calcular eps basado en el rango de X
        x_range = max(x_positions) - min(x_positions)
        eps = max(x_range * eps_factor, 20)  # Mínimo 20px

        # Aplicar DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=2).fit(X)
        labels = clustering.labels_

        # Agrupar posiciones por cluster
        columns = {}
        for i, label in enumerate(labels):
            if label == -1:  # Ruido - asignar al cluster más cercano
                continue
            if label not in columns:
                columns[label] = []
            columns[label].append(x_positions[i])

        # Si no hay clusters válidos, retornar una columna
        if not columns:
            return [(min(x_positions), max(x_positions))]

        # Calcular rangos de cada columna
        column_ranges = []
        for label, positions in columns.items():
            x_start = min(positions)
            x_end = max(positions) + 100  # Añadir margen para el texto
            column_ranges.append((x_start, x_end))

        # Ordenar columnas de izquierda a derecha
        column_ranges.sort(key=lambda c: c[0])

        logger.info(f"[LAYOUT-DBSCAN] Detectadas {len(column_ranges)} columnas: {column_ranges}")

        return column_ranges

    except ImportError:
        logger.warning("[LAYOUT-DBSCAN] scikit-learn no disponible, usando método simple")
        return [(min(x_positions), max(x_positions))]
    except Exception as e:
        logger.warning(f"[LAYOUT-DBSCAN] Error: {e}, usando método simple")
        return [(min(x_positions), max(x_positions))]


def format_text_with_layout(text_blocks, coordinates, page_width=200, use_dbscan=True):
    """
    Reconstruye la estructura espacial del documento usando coordenadas.
    MEJORADO v4.0: Usa DBSCAN para detectar columnas automáticamente.

    Args:
        text_blocks: Lista de textos detectados
        coordinates: Lista de polígonos/bboxes [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        page_width: Ancho de caracteres para la salida (default 200 para facturas)
        use_dbscan: Si True, usa DBSCAN para detectar columnas (default True)

    Returns:
        Texto formateado manteniendo la estructura espacial con columnas detectadas
    """
    if not text_blocks or not coordinates:
        return '\n'.join(text_blocks) if text_blocks else ''

    logger.info(f"[LAYOUT] Procesando {len(text_blocks)} bloques con {len(coordinates)} coordenadas")

    # Crear lista de bloques con sus coordenadas
    blocks = []
    for i, text in enumerate(text_blocks):
        if i < len(coordinates) and coordinates[i]:
            poly = coordinates[i]
            # Extraer bounding box del polígono
            try:
                if isinstance(poly, (list, tuple)) and len(poly) >= 4:
                    if isinstance(poly[0], (list, tuple)):
                        # poly es [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                        xs = [float(p[0]) for p in poly]
                        ys = [float(p[1]) for p in poly]
                    else:
                        # poly es [x1, y1, x2, y2, x3, y3, x4, y4] o similar
                        xs = [float(poly[j]) for j in range(0, min(len(poly), 8), 2)]
                        ys = [float(poly[j]) for j in range(1, min(len(poly), 8), 2)]

                    if xs and ys:
                        x_min = min(xs)
                        y_min = min(ys)
                        x_max = max(xs)
                        y_max = max(ys)
                        y_center = (y_min + y_max) / 2
                        height = y_max - y_min
                        width = x_max - x_min

                        blocks.append({
                            'text': text,
                            'x_min': x_min,
                            'x_max': x_max,
                            'y_min': y_min,
                            'y_max': y_max,
                            'y_center': y_center,
                            'height': height,
                            'width': width
                        })
                        continue
            except (IndexError, TypeError, ValueError) as e:
                logger.debug(f"[LAYOUT] Error parsing poly {i}: {e}")

            # Fallback si falla el parsing
            blocks.append({
                'text': text,
                'x_min': 0,
                'x_max': 100,
                'y_min': i * 30,
                'y_max': i * 30 + 20,
                'y_center': i * 30 + 10,
                'height': 20,
                'width': 100
            })
        else:
            # Sin coordenadas, posición por defecto
            blocks.append({
                'text': text,
                'x_min': 0,
                'x_max': 100,
                'y_min': i * 30,
                'y_max': i * 30 + 20,
                'y_center': i * 30 + 10,
                'height': 20,
                'width': 100
            })

    if not blocks:
        return '\n'.join(text_blocks)

    # Calcular tolerancia dinámica basada en la altura promedio de los bloques
    avg_height = sum(b['height'] for b in blocks) / len(blocks) if blocks else 20
    ROW_TOLERANCE = avg_height * 0.7  # 70% de la altura promedio (más permisivo)

    logger.info(f"[LAYOUT] Altura promedio: {avg_height:.1f}px, tolerancia fila: {ROW_TOLERANCE:.1f}px")

    # Obtener dimensiones del documento
    all_x = [b['x_min'] for b in blocks] + [b['x_max'] for b in blocks]
    doc_width = max(all_x) - min(all_x) if all_x else 1
    x_offset = min(all_x) if all_x else 0

    logger.info(f"[LAYOUT] Documento: ancho={doc_width:.1f}px, offset_x={x_offset:.1f}px")

    # =========================================================================
    # DBSCAN: Detectar columnas automáticamente
    # =========================================================================
    columns = []
    if use_dbscan:
        x_positions = [b['x_min'] for b in blocks]
        columns = detect_columns_dbscan(x_positions)
        logger.info(f"[LAYOUT] DBSCAN detectó {len(columns)} columnas")

    # Función para asignar bloque a columna
    def get_column_index(x_min):
        if not columns:
            return 0
        for i, (col_start, col_end) in enumerate(columns):
            if col_start - 50 <= x_min <= col_end + 50:  # Margen de tolerancia
                return i
        # Si no encaja, asignar a la columna más cercana
        distances = [(i, abs(x_min - (c[0] + c[1])/2)) for i, c in enumerate(columns)]
        return min(distances, key=lambda x: x[1])[0]

    # Ordenar por Y primero
    blocks_sorted = sorted(blocks, key=lambda b: b['y_center'])

    # Agrupar bloques por filas (Y similar = misma fila)
    rows = []
    current_row = [blocks_sorted[0]]

    for block in blocks_sorted[1:]:
        # Calcular Y promedio de la fila actual
        row_y_avg = sum(b['y_center'] for b in current_row) / len(current_row)

        # Si está en la misma fila (Y similar)
        if abs(block['y_center'] - row_y_avg) < ROW_TOLERANCE:
            current_row.append(block)
        else:
            # Nueva fila
            rows.append(current_row)
            current_row = [block]
    rows.append(current_row)

    logger.info(f"[LAYOUT] Agrupados en {len(rows)} filas")

    # Calcular ancho promedio de caracteres para espaciado más preciso
    total_text_len = sum(len(b['text']) for b in blocks)
    total_text_width = sum(b['width'] for b in blocks)
    char_width_px = total_text_width / total_text_len if total_text_len > 0 else 10

    logger.info(f"[LAYOUT] Ancho promedio por caracter: {char_width_px:.1f}px")

    # =========================================================================
    # Construir salida con columnas detectadas por DBSCAN
    # =========================================================================
    output_lines = []

    # Calcular ancho de cada columna en caracteres
    num_columns = len(columns) if columns else 1
    col_char_width = page_width // num_columns if num_columns > 0 else page_width

    for row in rows:
        # Ordenar bloques de la fila por X (izquierda a derecha)
        row_sorted = sorted(row, key=lambda b: b['x_min'])

        if use_dbscan and len(columns) > 1:
            # Modo DBSCAN: Posicionar bloques en sus columnas correspondientes
            line = [' '] * page_width

            for block in row_sorted:
                col_idx = get_column_index(block['x_min'])

                # Calcular posición dentro de la columna
                if col_idx < len(columns):
                    col_start_px, col_end_px = columns[col_idx]
                    col_width_px = col_end_px - col_start_px

                    # Posición relativa dentro de la columna
                    rel_pos = (block['x_min'] - col_start_px) / col_width_px if col_width_px > 0 else 0
                    rel_pos = max(0, min(1, rel_pos))

                    # Posición en caracteres
                    char_start = int(col_idx * col_char_width + rel_pos * col_char_width * 0.8)
                    char_start = min(char_start, page_width - len(block['text']) - 1)
                    char_start = max(0, char_start)

                    # Insertar texto
                    text = block['text']
                    for i, char in enumerate(text):
                        pos = char_start + i
                        if pos < page_width:
                            line[pos] = char

            output_lines.append(''.join(line).rstrip())

        else:
            # Modo simple: gaps entre bloques
            line_parts = []
            prev_block_end = None

            for block in row_sorted:
                if prev_block_end is not None:
                    # Calcular gap entre el bloque anterior y este
                    gap_px = block['x_min'] - prev_block_end

                    # Convertir gap a espacios (basado en ancho de caracter estimado)
                    if gap_px > char_width_px * 8:
                        spaces = '    '  # 4 espacios para columnas
                    elif gap_px > char_width_px * 3:
                        spaces = '   '
                    elif gap_px > char_width_px * 1.5:
                        spaces = '  '
                    elif gap_px > char_width_px * 0.5:
                        spaces = ' '
                    else:
                        spaces = ''

                    line_parts.append(spaces)

                line_parts.append(block['text'])
                prev_block_end = block['x_max']

            output_lines.append(''.join(line_parts).strip())

    return '\n'.join(output_lines)

@app.route('/')
def dashboard():
    """Dashboard web interactivo con pestañas"""
    uptime = int(time.time() - server_stats['startup_time'])
    success_rate = (server_stats['successful_requests'] / server_stats['total_requests'] * 100) if server_stats['total_requests'] > 0 else 0
    avg_time = (server_stats['total_processing_time'] / server_stats['successful_requests']) if server_stats['successful_requests'] > 0 else 0

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>PaddleOCR Fusion v4 - Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }}
        .container {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        h1 {{ color: #667eea; border-bottom: 3px solid #667eea; padding-bottom: 10px; margin-top: 0; }}
        h2 {{ color: #764ba2; margin-top: 20px; border-left: 4px solid #764ba2; padding-left: 15px; }}

        /* Tabs */
        .tabs {{ display: flex; border-bottom: 2px solid #667eea; margin-bottom: 20px; flex-wrap: wrap; }}
        .tab {{ padding: 12px 24px; cursor: pointer; border: none; background: #f0f0f0; margin-right: 5px; border-radius: 8px 8px 0 0; font-size: 1em; transition: all 0.3s; }}
        .tab:hover {{ background: #e0e0e0; }}
        .tab.active {{ background: #667eea; color: white; }}
        .tab-content {{ display: none; animation: fadeIn 0.3s; }}
        .tab-content.active {{ display: block; }}
        @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}

        .status-box {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin: 20px 0; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin: 20px 0; }}
        .stat-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #667eea; text-align: center; }}
        .stat-value {{ font-size: 1.8em; font-weight: bold; color: #667eea; }}
        .stat-label {{ color: #666; font-size: 0.85em; }}
        .endpoint-list {{ background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; }}
        .endpoint {{ padding: 10px; margin: 5px 0; background: white; border-left: 4px solid #764ba2; border-radius: 4px; }}
        .badge {{ display: inline-block; padding: 3px 8px; border-radius: 3px; font-size: 0.8em; font-weight: bold; margin-right: 10px; }}
        .badge-get {{ background: #28a745; color: white; }}
        .badge-post {{ background: #007bff; color: white; }}
        .badge-new {{ background: #ffc107; color: #333; }}
        .badge-original {{ background: #6c757d; color: white; }}
        code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-family: 'Courier New', monospace; }}
        .feature-badge {{ display: inline-block; padding: 5px 10px; margin: 3px; border-radius: 5px; font-size: 0.85em; }}
        .badge-success {{ background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }}

        /* OCR Form */
        .ocr-form {{ background: #f8f9fa; padding: 25px; border-radius: 10px; margin: 20px 0; }}
        .form-group {{ margin-bottom: 20px; }}
        .form-group label {{ display: block; margin-bottom: 8px; font-weight: bold; color: #333; }}
        .form-group input[type="file"] {{ width: 100%; padding: 15px; border: 2px dashed #667eea; border-radius: 8px; background: white; cursor: pointer; }}
        .format-selector {{ background: white; padding: 15px; border-radius: 8px; margin: 15px 0; }}
        .format-selector label {{ display: inline-block; margin-right: 20px; cursor: pointer; padding: 8px 15px; border-radius: 5px; transition: all 0.2s; }}
        .format-selector input[type="radio"] {{ margin-right: 8px; }}
        .format-selector label:hover {{ background: #e9ecef; }}
        .btn {{ padding: 12px 30px; border: none; border-radius: 8px; cursor: pointer; font-size: 1em; font-weight: bold; transition: all 0.3s; }}
        .btn-primary {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }}
        .btn-primary:hover {{ transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102,126,234,0.4); }}
        .btn-primary:disabled {{ opacity: 0.6; cursor: not-allowed; transform: none; }}
        .btn-danger {{ background: #dc3545; color: white; }}
        .btn-danger:hover {{ background: #c82333; }}

        /* Results */
        .result-box {{ margin-top: 20px; padding: 20px; border-radius: 10px; display: none; }}
        .result-box.success {{ background: #d4edda; border: 1px solid #c3e6cb; }}
        .result-box.error {{ background: #f8d7da; border: 1px solid #f5c6cb; }}
        .result-text {{ background: white; padding: 15px; border-radius: 8px; margin-top: 15px; max-height: 400px; overflow-y: auto; white-space: pre-wrap; font-family: 'Courier New', monospace; font-size: 0.9em; line-height: 1.5; }}
        .result-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: 10px; margin-top: 15px; }}
        .result-stat {{ background: white; padding: 10px; border-radius: 5px; text-align: center; }}

        /* History */
        .history-list {{ max-height: 600px; overflow-y: auto; }}
        .history-item {{ background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #667eea; }}
        .history-item.failed {{ border-left-color: #dc3545; }}
        .history-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
        .history-filename {{ font-weight: bold; color: #333; }}
        .history-time {{ color: #666; font-size: 0.85em; }}
        .history-meta {{ display: flex; gap: 15px; font-size: 0.9em; color: #666; flex-wrap: wrap; }}
        .history-text {{ background: white; padding: 10px; border-radius: 5px; margin-top: 10px; max-height: 150px; overflow-y: auto; font-family: 'Courier New', monospace; font-size: 0.85em; white-space: pre-wrap; }}
        .history-controls {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }}
        .empty-history {{ text-align: center; padding: 40px; color: #666; }}

        /* Loading */
        .loading {{ display: none; text-align: center; padding: 30px; }}
        .spinner {{ border: 4px solid #f3f3f3; border-top: 4px solid #667eea; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 15px; }}
        @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}

        pre {{ background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>PaddleOCR Fusion v4</h1>

        <!-- Tabs -->
        <div class="tabs">
            <button class="tab active" onclick="showTab('dashboard')">Dashboard</button>
            <button class="tab" onclick="showTab('test')">Probar OCR</button>
            <button class="tab" onclick="showTab('dictionary')">Diccionario</button>
            <button class="tab" onclick="showTab('improve')">Mejorar Diccionario</button>
            <button class="tab" onclick="showTab('config')">Configuracion</button>
            <button class="tab" onclick="showTab('history')">Historial</button>
            <button class="tab" onclick="showTab('docs')">Documentacion</button>
        </div>

        <!-- Tab: Dashboard -->
        <div id="dashboard" class="tab-content active">
            <div class="status-box">
                <h3 style="margin-top:0;">Estado del Servidor</h3>
                <p><strong>Estado:</strong> {'Operativo' if (doc_preprocessor and ocr_initialized) else 'Inicializando...'}</p>
                <p><strong>Preprocesador:</strong> {'Listo' if doc_preprocessor else 'Cargando...'}</p>
                <p><strong>OCR:</strong> {'Listo' if ocr_initialized else 'Cargando...'}</p>
                <p><strong>Uptime:</strong> {uptime//3600}h {(uptime%3600)//60}m {uptime%60}s</p>
            </div>

            <h2>Estadisticas</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{server_stats['total_requests']}</div>
                    <div class="stat-label">Total Requests</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{server_stats['successful_requests']}</div>
                    <div class="stat-label">Exitosos</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{success_rate:.1f}%</div>
                    <div class="stat-label">Tasa Exito</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{avg_time:.2f}s</div>
                    <div class="stat-label">Tiempo Prom.</div>
                </div>
            </div>

            <h2>Caracteristicas</h2>
            <div>
                <span class="feature-badge badge-success">PaddleOCR 3.x</span>
                <span class="feature-badge badge-success">Preprocesamiento OpenCV</span>
                <span class="feature-badge badge-success">Correccion Perspectiva</span>
                <span class="feature-badge badge-success">Correccion Orientacion</span>
                <span class="feature-badge badge-success">Correccion Inclinacion</span>
                <span class="feature-badge badge-success">Multi-pagina</span>
                <span class="feature-badge badge-success">Integracion n8n</span>
                <span class="feature-badge badge-success">API REST</span>
            </div>

            <h2>Endpoints API</h2>
            <div class="endpoint-list">
                <div class="endpoint"><span class="badge badge-get">GET</span><code>/</code> - Dashboard web</div>
                <div class="endpoint"><span class="badge badge-get">GET</span><code>/health</code> - Health check</div>
                <div class="endpoint"><span class="badge badge-get">GET</span><code>/stats</code> - Estadisticas JSON</div>
                <div class="endpoint"><span class="badge badge-post">POST</span><span class="badge badge-new">NUEVO</span><code>/process</code> - OCR via API REST</div>
                <div class="endpoint"><span class="badge badge-post">POST</span><code>/analyze</code> - Analisis detallado</div>
                <div class="endpoint"><span class="badge badge-post">POST</span><span class="badge badge-original">n8n</span><code>/ocr</code> - Endpoint original</div>
            </div>
        </div>

        <!-- Tab: Test OCR -->
        <div id="test" class="tab-content">
            <h2>Probar OCR</h2>
            <p>Sube un archivo PDF o imagen para extraer el texto.</p>

            <div class="ocr-form">
                <form id="ocrForm" enctype="multipart/form-data">
                    <div class="form-group">
                        <label>Archivo (PDF, PNG, JPG, JPEG, BMP, TIFF):</label>
                        <input type="file" id="fileInput" name="file" accept=".pdf,.png,.jpg,.jpeg,.bmp,.tiff,.tif" required>
                    </div>

                    <!-- LAYOUT EXPERIMENTAL: Usa coordenadas de bounding boxes -->
                    <div class="format-selector">
                        <strong>Formato de salida:</strong><br><br>
                        <label><input type="radio" name="format" value="normal" checked> Normal (texto plano)</label>
                        <label><input type="radio" name="format" value="layout"> Layout (mantiene estructura espacial)</label>
                    </div>

                    <button type="submit" class="btn btn-primary" id="submitBtn">Procesar OCR</button>
                </form>
            </div>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Procesando documento...</p>
                <p style="font-size:0.9em;color:#666;">Esto puede tomar unos segundos</p>
            </div>

            <div class="result-box" id="resultBox">
                <h3 id="resultTitle"></h3>
                <div class="result-stats" id="resultStats"></div>
                <div class="result-text" id="resultText"></div>
            </div>
        </div>

        <!-- Tab: Dictionary -->
        <div id="dictionary" class="tab-content">
            <h2>Diccionario de Correcciones OCR</h2>
            <p>Gestiona las correcciones automaticas para errores comunes del OCR en facturas y tickets.</p>

            <div class="stats-grid" id="dictStats">
                <div class="stat-card">
                    <div class="stat-value" id="baseCount">{len(OCR_CORRECTIONS_BASE)}</div>
                    <div class="stat-label">Base</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="customCount">{len(OCR_CORRECTIONS_CUSTOM)}</div>
                    <div class="stat-label">Personalizadas</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="totalCount">{len(OCR_CORRECTIONS)}</div>
                    <div class="stat-label">Total Activas</div>
                </div>
            </div>

            <h3>Añadir Nueva Correccion</h3>
            <div class="ocr-form" style="padding:20px;">
                <div style="display:grid;grid-template-columns:1fr 1fr auto auto;gap:10px;align-items:end;">
                    <div>
                        <label style="display:block;margin-bottom:5px;">Error OCR (ej: Cad1z):</label>
                        <input type="text" id="wrongText" placeholder="Texto con error" style="width:100%;padding:10px;border:1px solid #ddd;border-radius:5px;">
                    </div>
                    <div>
                        <label style="display:block;margin-bottom:5px;">Correccion (ej: Cadiz):</label>
                        <input type="text" id="correctText" placeholder="Texto correcto" style="width:100%;padding:10px;border:1px solid #ddd;border-radius:5px;">
                    </div>
                    <div>
                        <label style="display:block;margin-bottom:5px;">Tipo:</label>
                        <select id="dictType" style="padding:10px;border:1px solid #ddd;border-radius:5px;">
                            <option value="custom">Personalizado</option>
                            <option value="base">Base</option>
                        </select>
                    </div>
                    <button onclick="addCorrection()" class="btn btn-primary" style="padding:10px 20px;">Añadir</button>
                </div>
                <div id="addResult" style="margin-top:10px;display:none;"></div>
            </div>

            <h3>Probar Correcciones</h3>
            <div class="ocr-form" style="padding:20px;">
                <div class="form-group">
                    <label>Texto con errores (pega aqui el texto del OCR):</label>
                    <textarea id="testText" rows="4" style="width:100%;padding:10px;border:1px solid #ddd;border-radius:5px;font-family:monospace;" placeholder="Ej: Cad1z, N.1.F: 12345678A, TOTA1: 55:23 EUR"></textarea>
                </div>
                <button onclick="testCorrections()" class="btn btn-primary">Probar Correcciones</button>
                <div id="testResult" style="margin-top:15px;display:none;">
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:15px;">
                        <div>
                            <strong>Original:</strong>
                            <pre id="testOriginal" style="background:#f8d7da;padding:10px;border-radius:5px;max-height:200px;overflow:auto;"></pre>
                        </div>
                        <div>
                            <strong>Corregido:</strong>
                            <pre id="testCorrected" style="background:#d4edda;padding:10px;border-radius:5px;max-height:200px;overflow:auto;"></pre>
                        </div>
                    </div>
                    <div id="testApplied" style="margin-top:10px;"></div>
                </div>
            </div>

            <h3>Analizar Documento</h3>
            <div class="ocr-form" style="padding:20px;">
                <p style="color:#666;">Sube un documento para analizar y detectar posibles errores OCR.</p>
                <div class="form-group">
                    <input type="file" id="analyzeFile" accept=".pdf,.png,.jpg,.jpeg,.bmp,.tiff,.tif">
                </div>
                <button onclick="analyzeDocument()" class="btn btn-primary" id="analyzeBtn">Analizar</button>
                <div id="analyzeLoading" style="display:none;text-align:center;padding:20px;">
                    <div class="spinner"></div>
                    <p>Analizando documento...</p>
                </div>
                <div id="analyzeResult" style="margin-top:15px;display:none;"></div>
            </div>

            <h3>Diccionario Base (solo lectura)</h3>
            <div style="max-height:300px;overflow-y:auto;background:#f8f9fa;padding:15px;border-radius:8px;margin-bottom:20px;">
                <table style="width:100%;border-collapse:collapse;">
                    <thead><tr style="background:#667eea;color:white;"><th style="padding:8px;text-align:left;">Error</th><th style="padding:8px;text-align:left;">Correccion</th></tr></thead>
                    <tbody id="baseDict">
                        {"".join(f'<tr><td style="padding:6px;border-bottom:1px solid #ddd;font-family:monospace;">{wrong}</td><td style="padding:6px;border-bottom:1px solid #ddd;">{correct}</td></tr>' for wrong, correct in list(OCR_CORRECTIONS_BASE.items())[:50])}
                        <tr><td colspan="2" style="padding:10px;text-align:center;color:#666;">... y {max(0, len(OCR_CORRECTIONS_BASE)-50)} mas</td></tr>
                    </tbody>
                </table>
            </div>

            <h3>Diccionario Personalizado</h3>
            <div style="max-height:300px;overflow-y:auto;background:#f8f9fa;padding:15px;border-radius:8px;">
                <div id="customDictEmpty" style="{{'display:none' if OCR_CORRECTIONS_CUSTOM else 'block'}};text-align:center;padding:20px;color:#666;">
                    <p>No hay correcciones personalizadas todavia.</p>
                    <p>Añade correcciones arriba para empezar.</p>
                </div>
                <table id="customDictTable" style="width:100%;border-collapse:collapse;{{'display:table' if OCR_CORRECTIONS_CUSTOM else 'display:none'}}">
                    <thead><tr style="background:#764ba2;color:white;"><th style="padding:8px;text-align:left;">Error</th><th style="padding:8px;text-align:left;">Correccion</th><th style="padding:8px;width:80px;">Accion</th></tr></thead>
                    <tbody id="customDict">
                        {"".join('<tr id="row_' + wrong + '"><td style="padding:6px;border-bottom:1px solid #ddd;font-family:monospace;">' + wrong + '</td><td style="padding:6px;border-bottom:1px solid #ddd;">' + correct + '</td><td style="padding:6px;border-bottom:1px solid #ddd;"><button onclick="removeCorrection(' + "'" + wrong + "'" + ')" class="btn btn-danger" style="padding:5px 10px;font-size:0.8em;">Eliminar</button></td></tr>' for wrong, correct in OCR_CORRECTIONS_CUSTOM.items())}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Tab: Mejorar Diccionario con IA -->
        <div id="improve" class="tab-content">
            <h2>Mejorar Diccionario con IA</h2>
            <p style="color:#666;margin-bottom:20px;">Sube un documento, ejecuta OCR, y usa Gemini Vision para detectar y corregir errores automaticamente.</p>

            <div id="improveApiKeyWarning" style="background:#fff3cd;border:1px solid #ffc107;padding:15px;border-radius:8px;margin-bottom:20px;display:none;">
                <strong>Atencion:</strong> Necesitas configurar una API Key de Gemini en la pestana "Configuracion" para usar esta funcion.
                <button onclick="showTab('config')" class="btn btn-primary" style="margin-left:15px;padding:8px 15px;">Ir a Configuracion</button>
            </div>

            <!-- Paso 1: Subir documento -->
            <div class="ocr-form">
                <h3>Paso 1: Subir Documento</h3>
                <div class="form-group">
                    <label>Seleccionar PDF o Imagen:</label>
                    <input type="file" id="improveFile" accept=".pdf,.png,.jpg,.jpeg,.tiff,.bmp">
                </div>
                <button class="btn btn-primary" onclick="runImproveOCR()">Ejecutar OCR</button>
            </div>

            <!-- Paso 2: Resultado OCR -->
            <div id="improveOcrResult" style="display:none;margin-top:20px;">
                <h3>Paso 2: Resultado OCR</h3>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
                    <div>
                        <h4>Texto Original (OCR)</h4>
                        <textarea id="improveOcrText" style="width:100%;height:300px;font-family:monospace;padding:10px;border:1px solid #ddd;border-radius:8px;" readonly></textarea>
                    </div>
                    <div>
                        <h4>Texto Corregido (IA)</h4>
                        <textarea id="improveCorrectedText" style="width:100%;height:300px;font-family:monospace;padding:10px;border:1px solid #ddd;border-radius:8px;" readonly placeholder="Haz clic en 'Corregir con IA' para ver el resultado..."></textarea>
                    </div>
                </div>
                <div style="margin-top:15px;text-align:center;">
                    <button class="btn btn-primary" onclick="correctWithAI()" id="correctAIBtn">Corregir con Gemini Vision</button>
                    <span id="improveLoading" style="display:none;margin-left:15px;">
                        <span class="spinner" style="display:inline-block;width:20px;height:20px;border-width:2px;vertical-align:middle;"></span>
                        Procesando con IA...
                    </span>
                </div>
            </div>

            <!-- Paso 3: Correcciones detectadas -->
            <div id="improveCorrections" style="display:none;margin-top:20px;">
                <h3>Paso 3: Nuevas Correcciones Detectadas</h3>
                <p style="color:#666;">Selecciona las correcciones que quieres anadir al diccionario:</p>
                <div id="correctionsList" style="background:#f8f9fa;padding:15px;border-radius:8px;max-height:400px;overflow-y:auto;">
                    <!-- Se llena dinamicamente -->
                </div>
                <div style="margin-top:15px;">
                    <button class="btn btn-primary" onclick="addSelectedCorrections()">Anadir Seleccionadas al Diccionario</button>
                    <span id="addedCount" style="margin-left:15px;color:#28a745;display:none;"></span>
                </div>
            </div>

            <!-- Seccion: Importar diccionarios -->
            <div style="margin-top:40px;padding-top:20px;border-top:2px solid #eee;">
                <h3>Importar Diccionarios Externos</h3>
                <p style="color:#666;">Importa diccionarios de correcciones desde archivos JSON o URLs.</p>

                <div class="form-group" style="margin-top:15px;">
                    <label>Subir archivo JSON:</label>
                    <input type="file" id="importDictFile" accept=".json">
                </div>
                <button class="btn btn-primary" onclick="importDictionaryFile()" style="margin-bottom:20px;">Importar desde Archivo</button>

                <div class="form-group">
                    <label>O importar desde URL:</label>
                    <div style="display:flex;gap:10px;">
                        <input type="text" id="importDictUrl" placeholder="https://ejemplo.com/diccionario.json" style="flex:1;padding:10px;border:1px solid #ddd;border-radius:5px;">
                        <button class="btn btn-primary" onclick="importDictionaryUrl()">Importar</button>
                    </div>
                </div>

                <div id="importResult" style="display:none;margin-top:15px;padding:15px;border-radius:8px;"></div>
            </div>
        </div>

        <!-- Tab: Configuracion -->
        <div id="config" class="tab-content">
            <h2>Configuracion</h2>

            <div style="background:#f8f9fa;padding:25px;border-radius:10px;margin:20px 0;">
                <h3 style="margin-top:0;">API Keys para Correccion con IA</h3>
                <p style="color:#666;">Configura tu API Key de Gemini para usar la funcion de correccion automatica con IA.</p>

                <div class="form-group" style="margin-top:20px;">
                    <label>Gemini API Key:</label>
                    <div style="display:flex;gap:10px;">
                        <input type="password" id="geminiApiKey" placeholder="AIza..." style="flex:1;padding:12px;border:1px solid #ddd;border-radius:5px;font-family:monospace;">
                        <button class="btn" onclick="toggleApiKeyVisibility()" style="background:#6c757d;color:white;">Mostrar</button>
                        <button class="btn btn-primary" onclick="saveApiKey()">Guardar</button>
                    </div>
                </div>

                <div id="apiKeyStatus" style="margin-top:15px;padding:12px;border-radius:5px;display:none;"></div>

                <div style="margin-top:20px;padding:15px;background:#e8f4f8;border-radius:8px;border-left:4px solid #17a2b8;">
                    <strong>Como obtener una API Key de Gemini:</strong>
                    <ol style="margin:10px 0 0 20px;color:#666;">
                        <li>Ve a <a href="https://aistudio.google.com/apikey" target="_blank" style="color:#667eea;">Google AI Studio</a></li>
                        <li>Inicia sesion con tu cuenta de Google</li>
                        <li>Haz clic en "Create API Key"</li>
                        <li>Copia la clave y pegala aqui</li>
                    </ol>
                    <p style="margin-top:10px;font-size:0.9em;color:#666;"><strong>Nota:</strong> La API de Gemini tiene un tier gratuito generoso. La clave se guarda localmente en el servidor.</p>
                </div>
            </div>

            <div style="background:#f8f9fa;padding:25px;border-radius:10px;margin:20px 0;">
                <h3 style="margin-top:0;">Estado de la Configuracion</h3>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value" id="configApiKeyStatus">-</div>
                        <div class="stat-label">API Key Gemini</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="configDictCount">-</div>
                        <div class="stat-label">Correcciones en Diccionario</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="configCustomCount">-</div>
                        <div class="stat-label">Correcciones Personalizadas</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Tab: History -->
        <div id="history" class="tab-content">
            <h2>Historial de OCR</h2>
            <div class="history-controls">
                <span id="historyCount">Cargando...</span>
                <button class="btn btn-danger" onclick="clearHistory()">Limpiar Historial</button>
            </div>
            <div class="history-list" id="historyList">
                <div class="empty-history"><p>Cargando historial...</p></div>
            </div>
        </div>

        <!-- Tab: Docs -->
        <div id="docs" class="tab-content">
            <h2>Documentacion</h2>

            <h3>Formatos de salida:</h3>
            <div style="background:#e8f4f8;padding:15px;border-radius:8px;margin:15px 0;border-left:4px solid #17a2b8;">
                <p><strong>Normal:</strong> Texto plano extraido directamente del OCR. Funciona bien para la mayoria de documentos.</p>
                <p><strong>Layout:</strong> <span style="color:#dc3545;">(Proximamente)</span> Actualmente devuelve el mismo resultado que Normal. Se implementara con PP-Structure para reconstruir la estructura espacial real usando coordenadas de bounding boxes.</p>
            </div>

            <h3>API REST - Endpoint /process:</h3>
            <pre>
# Formato Normal (texto plano)
curl -X POST http://localhost:8503/process \\
  -F "file=@documento.pdf" \\
  -F "format=normal"

# Formato Layout (estructura espacial - recomendado para facturas)
curl -X POST http://localhost:8503/process \\
  -F "file=@documento.pdf" \\
  -F "format=layout"

# Con estadisticas detalladas
curl -X POST http://localhost:8503/process \\
  -F "file=@documento.pdf" \\
  -F "format=layout" \\
  -F "detailed=true"
            </pre>

            <h3>Respuesta JSON:</h3>
            <pre>
{{
  "success": true,
  "text": "Texto extraido...",
  "stats": {{"total_pages": 1, "total_blocks": 15}},
  "processing_time": 1.234,
  "format": "layout",
  "timestamp": 1234567890.123
}}
            </pre>

            <h3>Otros endpoints:</h3>
            <pre>
# Analisis ultra-detallado
curl -X POST http://localhost:8503/analyze \\
  -F "file=@documento.pdf"

# Estadisticas del servidor
curl http://localhost:8503/stats

# Historial de procesamientos
curl http://localhost:8503/api/history

# Limpiar historial
curl -X DELETE http://localhost:8503/api/history
            </pre>

            <h3>Integracion n8n (endpoint original):</h3>
            <pre>
# Usar endpoint /ocr (compatibilidad total con n8n)
curl -X POST http://localhost:8503/ocr \\
  -F "filename=/home/n8n/in/documento.pdf"
            </pre>
            <p style="background:#d4edda;padding:10px;border-radius:5px;border-left:4px solid #28a745;">
                El endpoint <code>/ocr</code> original de Paco sigue funcionando igual para mantener compatibilidad con workflows de n8n existentes.
            </p>

            <h3>Uso en n8n (HTTP Request node):</h3>
            <pre>
URL: http://paddleocr:8503/process
Method: POST
Body Content Type: Multipart Form Data
Body Parameters:
  - file: {{{{$binary.data}}}}  (Binary Data)
  - format: layout
            </pre>

            <h3>Enlaces utiles:</h3>
            <ul>
                <li><a href="https://github.com/PaddlePaddle/PaddleOCR" target="_blank">PaddleOCR Official Docs</a></li>
                <li><strong>README.md</strong> - Documentacion completa</li>
                <li><strong>CLAUDE.md</strong> - Guia de desarrollo</li>
            </ul>

            <h3>Requisitos de CPU:</h3>
            <p style="background:#fff3cd;padding:15px;border-radius:8px;border-left:4px solid #ffc107;">
                <strong>IMPORTANTE:</strong> Este proyecto requiere CPU con soporte AVX/AVX2.<br>
                Procesadores Intel 4ta generacion o posterior, o AMD Zen o posterior.<br>
                NO funciona en VPS con CPU virtualizada basica (Common KVM processor).
            </p>
        </div>

        <p style="text-align:center; color:#999; margin-top:30px;"><strong>PaddleOCR Fusion v4.0.0</strong> - Inteligencia Documental</p>
    </div>

    <script>
        function showTab(tabId) {{
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            event.target.classList.add('active');
            if (tabId === 'history') loadHistory();
        }}

        document.getElementById('ocrForm').addEventListener('submit', async function(e) {{
            e.preventDefault();
            const fileInput = document.getElementById('fileInput');
            const format = document.querySelector('input[name="format"]:checked').value;
            const submitBtn = document.getElementById('submitBtn');
            const loading = document.getElementById('loading');
            const resultBox = document.getElementById('resultBox');

            if (!fileInput.files[0]) {{ alert('Selecciona un archivo'); return; }}

            submitBtn.disabled = true;
            loading.style.display = 'block';
            resultBox.style.display = 'none';

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            formData.append('format', format);
            formData.append('detailed', 'true');

            try {{
                const response = await fetch('/process', {{ method: 'POST', body: formData }});
                const data = await response.json();

                loading.style.display = 'none';
                submitBtn.disabled = false;
                resultBox.style.display = 'block';

                if (data.success) {{
                    resultBox.className = 'result-box success';
                    document.getElementById('resultTitle').textContent = 'OCR Completado';
                    const stats = data.stats || {{}};
                    document.getElementById('resultStats').innerHTML = `
                        <div class="result-stat"><strong>${{data.processing_time}}s</strong><br>Tiempo</div>
                        <div class="result-stat"><strong>${{stats.total_pages || 1}}</strong><br>Paginas</div>
                        <div class="result-stat"><strong>${{stats.total_blocks || 0}}</strong><br>Bloques</div>
                        <div class="result-stat"><strong>${{((stats.avg_confidence || 0) * 100).toFixed(1)}}%</strong><br>Confianza</div>
                        <div class="result-stat"><strong>${{(data.text || '').length}}</strong><br>Caracteres</div>
                        <div class="result-stat"><strong>${{format}}</strong><br>Formato</div>
                    `;
                    document.getElementById('resultText').textContent = data.text || 'Sin texto extraido';
                }} else {{
                    resultBox.className = 'result-box error';
                    document.getElementById('resultTitle').textContent = 'Error';
                    document.getElementById('resultStats').innerHTML = '';
                    document.getElementById('resultText').textContent = data.error || 'Error desconocido';
                }}
                loadHistory();
            }} catch (error) {{
                loading.style.display = 'none';
                submitBtn.disabled = false;
                resultBox.style.display = 'block';
                resultBox.className = 'result-box error';
                document.getElementById('resultTitle').textContent = 'Error de conexion';
                document.getElementById('resultStats').innerHTML = '';
                document.getElementById('resultText').textContent = error.message;
            }}
        }});

        async function loadHistory() {{
            try {{
                const response = await fetch('/api/history');
                const data = await response.json();
                const historyList = document.getElementById('historyList');
                const historyCount = document.getElementById('historyCount');

                historyCount.textContent = `${{data.total}} de ${{data.max_history}} registros`;

                if (data.history.length === 0) {{
                    historyList.innerHTML = '<div class="empty-history"><p>No hay registros en el historial</p><p style="color:#999;">Los resultados de OCR apareceran aqui</p></div>';
                    return;
                }}

                historyList.innerHTML = data.history.map(item => `
                    <div class="history-item ${{item.success ? '' : 'failed'}}">
                        <div class="history-header">
                            <span class="history-filename">${{item.filename}}</span>
                            <span class="history-time">${{item.timestamp}}</span>
                        </div>
                        <div class="history-meta">
                            <span>Tiempo: ${{item.processing_time}}s</span>
                            <span>Paginas: ${{item.pages || 1}}</span>
                            <span>Caracteres: ${{item.chars || 0}}</span>
                            <span>Formato: ${{item.format || 'normal'}}</span>
                        </div>
                        ${{item.text ? `<div class="history-text">${{item.text}}</div>` : ''}}
                    </div>
                `).join('');
            }} catch (error) {{
                document.getElementById('historyList').innerHTML = '<div class="empty-history"><p>Error cargando historial</p></div>';
            }}
        }}

        async function clearHistory() {{
            if (!confirm('Seguro que quieres limpiar el historial?')) return;
            try {{
                await fetch('/api/history/clear', {{ method: 'POST' }});
                loadHistory();
            }} catch (error) {{
                alert('Error limpiando historial');
            }}
        }}

        // ========== DICTIONARY FUNCTIONS ==========

        async function addCorrection() {{
            const wrong = document.getElementById('wrongText').value.trim();
            const correct = document.getElementById('correctText').value.trim();
            const dictType = document.getElementById('dictType').value;
            const resultDiv = document.getElementById('addResult');

            if (!wrong || !correct) {{
                resultDiv.innerHTML = '<span style="color:red;">Error: Completa ambos campos</span>';
                resultDiv.style.display = 'block';
                return;
            }}

            try {{
                const response = await fetch('/api/dictionary/add', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{ wrong, correct, dictionary: dictType }})
                }});
                const data = await response.json();

                if (data.success) {{
                    resultDiv.innerHTML = `<span style="color:green;">✓ ${{data.message}}</span>`;
                    document.getElementById('wrongText').value = '';
                    document.getElementById('correctText').value = '';
                    updateDictStats(data.stats);
                    // Recargar pagina para actualizar tablas
                    setTimeout(() => location.reload(), 1000);
                }} else {{
                    resultDiv.innerHTML = `<span style="color:red;">✗ ${{data.error}}</span>`;
                }}
                resultDiv.style.display = 'block';
            }} catch (error) {{
                resultDiv.innerHTML = `<span style="color:red;">Error: ${{error.message}}</span>`;
                resultDiv.style.display = 'block';
            }}
        }}

        async function removeCorrection(wrong) {{
            if (!confirm(`Eliminar correccion para "${{wrong}}"?`)) return;

            try {{
                const response = await fetch('/api/dictionary/remove', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{ wrong, dictionary: 'custom' }})
                }});
                const data = await response.json();

                if (data.success) {{
                    document.getElementById(`row_${{wrong}}`).remove();
                    updateDictStats(data.stats);
                }} else {{
                    alert('Error: ' + data.error);
                }}
            }} catch (error) {{
                alert('Error: ' + error.message);
            }}
        }}

        async function testCorrections() {{
            const text = document.getElementById('testText').value;
            const resultDiv = document.getElementById('testResult');

            if (!text) {{
                alert('Introduce texto para probar');
                return;
            }}

            try {{
                const response = await fetch('/api/dictionary/test', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{ text }})
                }});
                const data = await response.json();

                if (data.success) {{
                    document.getElementById('testOriginal').textContent = data.original;
                    document.getElementById('testCorrected').textContent = data.corrected;

                    let appliedHtml = '';
                    if (data.corrections_applied.length > 0) {{
                        appliedHtml = '<strong>Correcciones aplicadas:</strong><ul>';
                        data.corrections_applied.forEach(c => {{
                            if (c.wrong) {{
                                appliedHtml += `<li><code>${{c.wrong}}</code> → <code>${{c.correct}}</code> (${{c.count}}x)</li>`;
                            }} else {{
                                appliedHtml += `<li>Patron: <code>${{c.pattern}}</code> (${{c.count}}x)</li>`;
                            }}
                        }});
                        appliedHtml += '</ul>';
                    }} else {{
                        appliedHtml = '<span style="color:#666;">No se aplicaron correcciones</span>';
                    }}
                    document.getElementById('testApplied').innerHTML = appliedHtml;
                    resultDiv.style.display = 'block';
                }}
            }} catch (error) {{
                alert('Error: ' + error.message);
            }}
        }}

        async function analyzeDocument() {{
            const fileInput = document.getElementById('analyzeFile');
            const loading = document.getElementById('analyzeLoading');
            const resultDiv = document.getElementById('analyzeResult');
            const btn = document.getElementById('analyzeBtn');

            if (!fileInput.files[0]) {{
                alert('Selecciona un archivo');
                return;
            }}

            btn.disabled = true;
            loading.style.display = 'block';
            resultDiv.style.display = 'none';

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            try {{
                const response = await fetch('/api/dictionary/analyze', {{
                    method: 'POST',
                    body: formData
                }});
                const data = await response.json();

                loading.style.display = 'none';
                btn.disabled = false;

                if (data.success) {{
                    let html = `<p><strong>Tiempo:</strong> ${{data.processing_time}}s | <strong>Bloques:</strong> ${{data.blocks_count}}</p>`;

                    if (data.potential_errors.length > 0) {{
                        html += `<h4 style="color:#dc3545;">Posibles errores detectados: ${{data.potential_errors_count}}</h4>`;
                        html += '<table style="width:100%;border-collapse:collapse;background:white;">';
                        html += '<thead><tr style="background:#f8d7da;"><th style="padding:8px;">Texto</th><th style="padding:8px;">Razon</th><th style="padding:8px;">Accion</th></tr></thead><tbody>';
                        data.potential_errors.forEach(e => {{
                            html += '<tr><td style="padding:6px;border:1px solid #ddd;font-family:monospace;">' + e.text + '</td><td style="padding:6px;border:1px solid #ddd;">' + e.reason + '</td><td style="padding:6px;border:1px solid #ddd;"><button onclick="document.getElementById(\\'wrongText\\').value=\\'' + e.text + '\\';document.getElementById(\\'wrongText\\').focus();" class="btn btn-primary" style="padding:3px 8px;font-size:0.8em;">Añadir correccion</button></td></tr>';
                        }});
                        html += '</tbody></table>';
                    }} else {{
                        html += '<p style="color:green;">✓ No se detectaron errores sospechosos</p>';
                    }}

                    html += '<h4>Texto OCR (corregido):</h4>';
                    html += `<pre style="background:white;padding:10px;max-height:200px;overflow:auto;">${{data.ocr_text_corrected}}</pre>`;

                    resultDiv.innerHTML = html;
                    resultDiv.style.display = 'block';
                }} else {{
                    resultDiv.innerHTML = `<p style="color:red;">Error: ${{data.error}}</p>`;
                    resultDiv.style.display = 'block';
                }}
            }} catch (error) {{
                loading.style.display = 'none';
                btn.disabled = false;
                resultDiv.innerHTML = `<p style="color:red;">Error: ${{error.message}}</p>`;
                resultDiv.style.display = 'block';
            }}
        }}

        function updateDictStats(stats) {{
            document.getElementById('baseCount').textContent = stats.base_count;
            document.getElementById('customCount').textContent = stats.custom_count;
            document.getElementById('totalCount').textContent = stats.total_count;
        }}

        // ========== API KEY / CONFIG FUNCTIONS ==========

        let currentImproveImagePath = null;

        async function checkApiKeyStatus() {{
            try {{
                const response = await fetch('/api/config/apikey');
                const data = await response.json();

                // Actualizar estado en tab config
                const statusEl = document.getElementById('configApiKeyStatus');
                if (statusEl) {{
                    statusEl.textContent = data.configured ? 'Configurada' : 'No configurada';
                    statusEl.style.color = data.configured ? '#28a745' : '#dc3545';
                }}

                // Mostrar/ocultar warning en tab improve
                const warningEl = document.getElementById('improveApiKeyWarning');
                if (warningEl) {{
                    warningEl.style.display = data.configured ? 'none' : 'block';
                }}

                // Actualizar stats del diccionario
                const dictResponse = await fetch('/api/dictionary');
                const dictData = await dictResponse.json();
                if (document.getElementById('configDictCount')) {{
                    document.getElementById('configDictCount').textContent = Object.keys(dictData.base || {{}}).length;
                }}
                if (document.getElementById('configCustomCount')) {{
                    document.getElementById('configCustomCount').textContent = Object.keys(dictData.custom || {{}}).length;
                }}

                return data.configured;
            }} catch (error) {{
                console.error('Error checking API key:', error);
                return false;
            }}
        }}

        function toggleApiKeyVisibility() {{
            const input = document.getElementById('geminiApiKey');
            const btn = event.target;
            if (input.type === 'password') {{
                input.type = 'text';
                btn.textContent = 'Ocultar';
            }} else {{
                input.type = 'password';
                btn.textContent = 'Mostrar';
            }}
        }}

        async function saveApiKey() {{
            const apiKey = document.getElementById('geminiApiKey').value.trim();
            const statusDiv = document.getElementById('apiKeyStatus');

            if (!apiKey) {{
                statusDiv.innerHTML = '<span style="color:red;">Error: Introduce una API Key</span>';
                statusDiv.style.display = 'block';
                statusDiv.style.background = '#f8d7da';
                return;
            }}

            try {{
                const response = await fetch('/api/config/apikey', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{ api_key: apiKey }})
                }});
                const data = await response.json();

                if (data.success) {{
                    statusDiv.innerHTML = '<span style="color:green;">✓ API Key guardada correctamente</span>';
                    statusDiv.style.background = '#d4edda';
                    document.getElementById('geminiApiKey').value = '';
                    checkApiKeyStatus();
                }} else {{
                    statusDiv.innerHTML = `<span style="color:red;">✗ ${{data.error}}</span>`;
                    statusDiv.style.background = '#f8d7da';
                }}
                statusDiv.style.display = 'block';
            }} catch (error) {{
                statusDiv.innerHTML = `<span style="color:red;">Error: ${{error.message}}</span>`;
                statusDiv.style.display = 'block';
                statusDiv.style.background = '#f8d7da';
            }}
        }}

        // ========== IMPROVE DICTIONARY FUNCTIONS ==========

        async function runImproveOCR() {{
            const fileInput = document.getElementById('improveFile');
            if (!fileInput.files[0]) {{
                alert('Selecciona un archivo primero');
                return;
            }}

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            formData.append('format', 'normal');
            formData.append('return_image_path', 'true');

            try {{
                document.getElementById('improveOcrResult').style.display = 'none';
                document.getElementById('improveCorrections').style.display = 'none';

                const response = await fetch('/process', {{ method: 'POST', body: formData }});
                const data = await response.json();

                if (data.success) {{
                    document.getElementById('improveOcrText').value = data.text || '';
                    document.getElementById('improveCorrectedText').value = '';
                    document.getElementById('improveOcrResult').style.display = 'block';
                    currentImproveImagePath = data.image_path || null;
                }} else {{
                    alert('Error en OCR: ' + (data.error || 'Error desconocido'));
                }}
            }} catch (error) {{
                alert('Error: ' + error.message);
            }}
        }}

        async function correctWithAI() {{
            const hasKey = await checkApiKeyStatus();
            if (!hasKey) {{
                alert('Configura una API Key de Gemini primero');
                showTab('config');
                return;
            }}

            const ocrText = document.getElementById('improveOcrText').value;
            if (!ocrText) {{
                alert('Primero ejecuta el OCR');
                return;
            }}

            const fileInput = document.getElementById('improveFile');
            if (!fileInput.files[0]) {{
                alert('El archivo original ya no esta disponible. Vuelve a subirlo.');
                return;
            }}

            document.getElementById('improveLoading').style.display = 'inline';
            document.getElementById('correctAIBtn').disabled = true;

            try {{
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                formData.append('ocr_text', ocrText);

                const response = await fetch('/api/dictionary/improve', {{
                    method: 'POST',
                    body: formData
                }});
                const data = await response.json();

                document.getElementById('improveLoading').style.display = 'none';
                document.getElementById('correctAIBtn').disabled = false;

                if (data.success) {{
                    document.getElementById('improveCorrectedText').value = data.corrected_text || '';

                    // Mostrar correcciones detectadas
                    if (data.corrections && data.corrections.length > 0) {{
                        displayCorrections(data.corrections);
                    }} else {{
                        document.getElementById('correctionsList').innerHTML = '<p style="color:#666;">No se detectaron nuevas correcciones.</p>';
                        document.getElementById('improveCorrections').style.display = 'block';
                    }}
                }} else {{
                    alert('Error: ' + (data.error || 'Error desconocido'));
                }}
            }} catch (error) {{
                document.getElementById('improveLoading').style.display = 'none';
                document.getElementById('correctAIBtn').disabled = false;
                alert('Error: ' + error.message);
            }}
        }}

        function displayCorrections(corrections) {{
            const container = document.getElementById('correctionsList');
            let html = '';

            corrections.forEach((c, idx) => {{
                const existsClass = c.exists ? 'style="opacity:0.5;"' : '';
                const existsNote = c.exists ? '<span style="color:#666;font-size:0.9em;"> (ya existe)</span>' : '';
                const checked = c.exists ? '' : 'checked';

                html += `
                    <div style="padding:10px;margin:5px 0;background:white;border-radius:5px;display:flex;align-items:center;gap:15px;" ${{existsClass}}>
                        <input type="checkbox" id="correction_${{idx}}" data-wrong="${{c.wrong}}" data-correct="${{c.correct}}" ${{checked}} ${{c.exists ? 'disabled' : ''}}>
                        <code style="background:#f4f4f4;padding:3px 8px;border-radius:3px;">${{c.wrong}}</code>
                        <span>→</span>
                        <code style="background:#d4edda;padding:3px 8px;border-radius:3px;">${{c.correct}}</code>
                        ${{existsNote}}
                    </div>
                `;
            }});

            container.innerHTML = html || '<p style="color:#666;">No se detectaron nuevas correcciones.</p>';
            document.getElementById('improveCorrections').style.display = 'block';
        }}

        async function addSelectedCorrections() {{
            const checkboxes = document.querySelectorAll('#correctionsList input[type="checkbox"]:checked:not([disabled])');
            let added = 0;

            for (const cb of checkboxes) {{
                const wrong = cb.dataset.wrong;
                const correct = cb.dataset.correct;

                try {{
                    await fetch('/api/dictionary/add', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{ wrong, correct, dictionary: 'custom' }})
                    }});
                    added++;
                    cb.checked = false;
                    cb.disabled = true;
                    cb.parentElement.style.opacity = '0.5';
                }} catch (e) {{
                    console.error('Error adding correction:', e);
                }}
            }}

            const countEl = document.getElementById('addedCount');
            countEl.textContent = `✓ ${{added}} correcciones anadidas`;
            countEl.style.display = 'inline';
            setTimeout(() => countEl.style.display = 'none', 3000);
        }}

        // ========== IMPORT DICTIONARY FUNCTIONS ==========

        async function importDictionaryFile() {{
            const fileInput = document.getElementById('importDictFile');
            if (!fileInput.files[0]) {{
                alert('Selecciona un archivo JSON');
                return;
            }}

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            try {{
                const response = await fetch('/api/dictionary/import', {{
                    method: 'POST',
                    body: formData
                }});
                const data = await response.json();
                showImportResult(data);
            }} catch (error) {{
                showImportResult({{ success: false, error: error.message }});
            }}
        }}

        async function importDictionaryUrl() {{
            const url = document.getElementById('importDictUrl').value.trim();
            if (!url) {{
                alert('Introduce una URL');
                return;
            }}

            try {{
                const response = await fetch('/api/dictionary/import', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{ url }})
                }});
                const data = await response.json();
                showImportResult(data);
            }} catch (error) {{
                showImportResult({{ success: false, error: error.message }});
            }}
        }}

        function showImportResult(data) {{
            const resultDiv = document.getElementById('importResult');
            if (data.success) {{
                resultDiv.innerHTML = `<span style="color:green;">✓ ${{data.message || 'Importado correctamente'}} (${{data.imported || 0}} correcciones)</span>`;
                resultDiv.style.background = '#d4edda';
            }} else {{
                resultDiv.innerHTML = `<span style="color:red;">✗ ${{data.error || 'Error desconocido'}}</span>`;
                resultDiv.style.background = '#f8d7da';
            }}
            resultDiv.style.display = 'block';
        }}

        // Comprobar estado API key al cargar
        document.addEventListener('DOMContentLoaded', function() {{
            checkApiKeyStatus();
        }});
    </script>
</body>
</html>
    """
    return html


@app.route('/stats')
def stats():
    """Estadísticas detalladas del servidor"""
    uptime = int(time.time() - server_stats['startup_time'])
    success_rate = (server_stats['successful_requests'] / server_stats['total_requests'] * 100) if server_stats['total_requests'] > 0 else 0
    avg_time = (server_stats['total_processing_time'] / server_stats['successful_requests']) if server_stats['successful_requests'] > 0 else 0

    return jsonify({
        'status': 'healthy' if (doc_preprocessor and ocr_initialized) else 'initializing',
        'uptime_seconds': uptime,
        'uptime_formatted': f"{uptime//3600}h {(uptime%3600)//60}m {uptime%60}s",
        'preprocessor_ready': doc_preprocessor is not None,
        'ocr_ready': ocr_initialized,
        'statistics': {
            'total_requests': server_stats['total_requests'],
            'successful_requests': server_stats['successful_requests'],
            'failed_requests': server_stats['failed_requests'],
            'success_rate': round(success_rate, 2),
            'total_processing_time': round(server_stats['total_processing_time'], 2),
            'avg_processing_time': round(avg_time, 3)
        },
        'configuration': {
            'opencv_config': OPENCV_CONFIG,
            'rotation_config': ROTATION_CONFIG
        },
        'version': '4.0.0-fusion',
        'base_project': 'paddlepaddle_paco',
        'api_layer': 'webcomunica'
    })


@app.route('/process', methods=['POST'])
def process():
    """
    Endpoint REST estándar - Wrapper sobre el endpoint /ocr de Paco
    Acepta archivos multipart en lugar de rutas en disco
    """
    global server_stats, ocr_history
    start_time = time.time()
    server_stats['total_requests'] += 1

    temp_file_path = None

    try:
        # Validar archivo
        if 'file' not in request.files:
            server_stats['failed_requests'] += 1
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            server_stats['failed_requests'] += 1
            return jsonify({'error': 'Empty filename'}), 400

        # Validar extensión
        ext = Path(file.filename).suffix.lower()
        if ext not in ['.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
            server_stats['failed_requests'] += 1
            return jsonify({'error': f'Unsupported file format: {ext}'}), 400

        # Obtener parámetros
        language = request.form.get('language', 'es')
        detailed = request.form.get('detailed', 'false').lower() == 'true'
        output_format = request.form.get('format', 'normal')  # normal o layout
        original_filename = file.filename  # Guardar nombre original para historial

        # Guardar archivo temporal en /home/n8n/in para compatibilidad con /ocr
        n8nHomeDir = '/home/n8n'
        os.makedirs(f"{n8nHomeDir}/in", exist_ok=True)
        os.makedirs(f"{n8nHomeDir}/ocr", exist_ok=True)
        os.makedirs(f"{n8nHomeDir}/pdf", exist_ok=True)

        temp_filename = f"temp_{int(time.time())}_{file.filename}"
        temp_file_path = f"{n8nHomeDir}/in/{temp_filename}"
        file.save(temp_file_path)

        logger.info(f"[PROCESS] Archivo guardado temporalmente: {temp_file_path}")

        # Llamar al endpoint /ocr internamente usando la lógica de Paco
        # Creamos un request simulado
        with app.test_request_context(
            '/ocr',
            method='POST',
            data={'filename': temp_file_path}
        ):
            response = ocr()

            # Extraer datos de la respuesta
            if isinstance(response, tuple):
                response_data, status_code = response
            else:
                response_data = response
                status_code = 200

            response_json = response_data.get_json()

        # Limpiar archivo temporal
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            # Limpiar archivos intermedios generados
            base_name = Path(temp_filename).stem
            for pattern in [f"{n8nHomeDir}/ocr/{base_name}*", f"{n8nHomeDir}/pdf/{base_name}*"]:
                import glob
                for f in glob.glob(pattern):
                    try:
                        os.remove(f)
                    except:
                        pass
        except Exception as e:
            logger.warning(f"[PROCESS] Error limpiando archivos temporales: {e}")

        # Preparar respuesta en formato REST estándar
        processing_time = time.time() - start_time
        server_stats['total_processing_time'] += processing_time

        if response_json.get('success'):
            server_stats['successful_requests'] += 1

            # Obtener ambas versiones del texto
            extracted_text_plain = response_json.get('extracted_text_plain', response_json.get('extracted_text', ''))
            extracted_text_layout = response_json.get('extracted_text_layout', '')
            ocr_blocks = response_json.get('ocr_blocks', [])
            coordinates = response_json.get('coordinates', [])
            stats = response_json.get('stats', {})

            # Aplicar formato según selección del usuario
            if output_format == 'layout':
                # Layout: Estrategia híbrida inteligente
                #
                # PDFs vectoriales (facturas digitales): pdftotext -layout funciona mejor
                # PDFs escaneados: usar coordenadas OCR para reconstrucción espacial
                #
                # Heurística: si pdftotext -layout tiene mucho más contenido que los
                # bloques OCR, el PDF es probablemente vectorial y pdftotext es mejor

                pdftotext_chars = len(extracted_text_layout.strip()) if extracted_text_layout else 0
                ocr_blocks_count = len(ocr_blocks) if ocr_blocks else 0

                # Umbral: pdftotext es preferido si tiene contenido sustancial
                # y hay pocos bloques OCR (señal de PDF vectorial mal segmentado)
                use_pdftotext = pdftotext_chars > 500 and (ocr_blocks_count < 50 or pdftotext_chars > ocr_blocks_count * 50)

                if use_pdftotext and extracted_text_layout:
                    # PRIORIDAD 1: pdftotext -layout para PDFs vectoriales
                    formatted_text = extracted_text_layout
                    logger.info(f"[PROCESS] Modo Layout (pdftotext - PDF vectorial) - {pdftotext_chars} chars, {ocr_blocks_count} bloques OCR")
                elif ocr_blocks and coordinates and len(coordinates) > 0:
                    # PRIORIDAD 2: Coordenadas OCR para PDFs escaneados
                    formatted_text = format_text_with_layout(ocr_blocks, coordinates, page_width=120)
                    logger.info(f"[PROCESS] Modo Layout (coordenadas OCR - PDF escaneado) - {len(ocr_blocks)} bloques, {len(coordinates)} coords")
                elif extracted_text_layout:
                    # Fallback: pdftotext si no hay coordenadas
                    formatted_text = extracted_text_layout
                    logger.info(f"[PROCESS] Modo Layout (pdftotext fallback) - {len(formatted_text)} chars")
                else:
                    formatted_text = extracted_text_plain
                    logger.info(f"[PROCESS] Modo Layout fallback a texto plano")
            else:
                # Normal: texto plano sin estructura espacial (más rápido de procesar)
                formatted_text = extracted_text_plain
                logger.info(f"[PROCESS] Modo Normal - {len(formatted_text)} chars texto plano")

            # Aplicar correcciones del diccionario OCR
            formatted_text = apply_ocr_corrections(formatted_text)

            result = {
                'success': True,
                'text': formatted_text,
                'stats': stats,
                'processing_time': round(processing_time, 3),
                'format': output_format,
                'timestamp': time.time()
            }

            if detailed:
                result['detailed_stats'] = {
                    'total_pages': stats.get('total_pages', 1),
                    'total_blocks': stats.get('total_blocks', 0),
                    'avg_confidence': stats.get('avg_confidence', 0.0)
                }

            # Guardar en historial (con el texto formateado según el modo seleccionado)
            history_entry = {
                'filename': original_filename,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'processing_time': round(processing_time, 3),
                'success': True,
                'text': formatted_text[:2000] + ('...' if len(formatted_text) > 2000 else ''),
                'chars': len(formatted_text),
                'pages': stats.get('total_pages', 1),
                'format': output_format
            }
            ocr_history.insert(0, history_entry)
            if len(ocr_history) > MAX_HISTORY:
                ocr_history.pop()

            return jsonify(result)
        else:
            server_stats['failed_requests'] += 1

            # Guardar error en historial
            history_entry = {
                'filename': original_filename,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'processing_time': round(processing_time, 3),
                'success': False,
                'text': response_json.get('error', 'Unknown error'),
                'chars': 0,
                'pages': 0,
                'format': output_format
            }
            ocr_history.insert(0, history_entry)
            if len(ocr_history) > MAX_HISTORY:
                ocr_history.pop()

            return jsonify({
                'success': False,
                'error': response_json.get('error', 'Unknown error'),
                'processing_time': round(processing_time, 3)
            }), 500

    except Exception as e:
        server_stats['failed_requests'] += 1
        processing_time = time.time() - start_time
        server_stats['total_processing_time'] += processing_time

        logger.error(f"[PROCESS ERROR] {e}")
        import traceback
        logger.error(traceback.format_exc())

        # Limpiar archivo temporal en caso de error
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass

        return jsonify({
            'success': False,
            'error': str(e),
            'processing_time': round(processing_time, 3)
        }), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Análisis ultra-detallado con visualización
    Similar a /process pero con salida formateada para debugging
    """
    global server_stats
    start_time = time.time()
    server_stats['total_requests'] += 1

    temp_file_path = None

    try:
        # Validar archivo
        if 'file' not in request.files:
            server_stats['failed_requests'] += 1
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            server_stats['failed_requests'] += 1
            return jsonify({'error': 'Empty filename'}), 400

        # Guardar archivo temporal
        n8nHomeDir = '/home/n8n'
        os.makedirs(f"{n8nHomeDir}/in", exist_ok=True)

        temp_filename = f"analyze_{int(time.time())}_{file.filename}"
        temp_file_path = f"{n8nHomeDir}/in/{temp_filename}"
        file.save(temp_file_path)

        # Llamar al endpoint /ocr
        with app.test_request_context(
            '/ocr',
            method='POST',
            data={'filename': temp_file_path}
        ):
            response = ocr()
            if isinstance(response, tuple):
                response_data, status_code = response
            else:
                response_data = response
                status_code = 200

            response_json = response_data.get_json()

        # Limpiar archivos temporales
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            base_name = Path(temp_filename).stem
            import glob
            for pattern in [f"{n8nHomeDir}/ocr/{base_name}*", f"{n8nHomeDir}/pdf/{base_name}*"]:
                for f in glob.glob(pattern):
                    try:
                        os.remove(f)
                    except:
                        pass
        except Exception as e:
            logger.warning(f"[ANALYZE] Error limpiando archivos: {e}")

        processing_time = time.time() - start_time
        server_stats['total_processing_time'] += processing_time

        if response_json.get('success'):
            server_stats['successful_requests'] += 1

            # Formatear texto para visualización
            text = response_json.get('extracted_text', '')
            stats = response_json.get('stats', {})

            ultra_analysis = f"""
╔══════════════════════════════════════════════════════════════╗
║              ANÁLISIS ULTRA-DETALLADO - FUSION v3            ║
╚══════════════════════════════════════════════════════════════╝

📊 ESTADÍSTICAS:
   • Total de Páginas: {stats.get('total_pages', 1)}
   • Total de Bloques: {stats.get('total_blocks', 0)}
   • Confianza Promedio: {stats.get('avg_confidence', 0)*100:.1f}%
   • Tiempo de Procesamiento: {processing_time:.2f}s

📄 TEXTO EXTRAÍDO:
{text}

════════════════════════════════════════════════════════════════
"""

            return jsonify({
                'success': True,
                'ultra_analysis': ultra_analysis,
                'stats': stats,
                'processing_time': round(processing_time, 3),
                'configuration': 'PaddleOCR 3.x + Preprocesamiento Completo (Paco)'
            })
        else:
            server_stats['failed_requests'] += 1
            return jsonify({
                'success': False,
                'error': response_json.get('error', 'Unknown error')
            }), 500

    except Exception as e:
        server_stats['failed_requests'] += 1
        processing_time = time.time() - start_time
        server_stats['total_processing_time'] += processing_time

        logger.error(f"[ANALYZE ERROR] {e}")
        import traceback
        logger.error(traceback.format_exc())

        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass

        return jsonify({
            'success': False,
            'error': str(e),
            'processing_time': round(processing_time, 3)
        }), 500


@app.route('/api/history')
def api_history():
    """Endpoint para obtener el historial de OCR"""
    return jsonify({
        'history': ocr_history,
        'total': len(ocr_history),
        'max_history': MAX_HISTORY
    })


@app.route('/api/history/clear', methods=['POST'])
def api_history_clear():
    """Endpoint para limpiar el historial de OCR"""
    global ocr_history
    ocr_history = []
    return jsonify({
        'success': True,
        'message': 'Historial limpiado'
    })


# ============================================================================
# API DE DICCIONARIOS
# ============================================================================

@app.route('/api/dictionary')
def api_dictionary():
    """Obtiene todos los diccionarios"""
    return jsonify({
        'success': True,
        'base': OCR_CORRECTIONS_BASE,
        'custom': OCR_CORRECTIONS_CUSTOM,
        'combined': OCR_CORRECTIONS,
        'stats': {
            'base_count': len(OCR_CORRECTIONS_BASE),
            'custom_count': len(OCR_CORRECTIONS_CUSTOM),
            'total_count': len(OCR_CORRECTIONS)
        }
    })


@app.route('/api/dictionary/add', methods=['POST'])
def api_dictionary_add():
    """Añade una corrección al diccionario"""
    data = request.get_json() or {}
    wrong = data.get('wrong', '').strip()
    correct = data.get('correct', '').strip()
    dictionary = data.get('dictionary', 'custom')

    if not wrong or not correct:
        return jsonify({'success': False, 'error': 'Faltan campos wrong/correct'}), 400

    if wrong == correct:
        return jsonify({'success': False, 'error': 'El error y la corrección no pueden ser iguales'}), 400

    success = add_correction(wrong, correct, dictionary)
    return jsonify({
        'success': success,
        'message': f'Añadida corrección: "{wrong}" → "{correct}"' if success else 'Error al guardar',
        'stats': {
            'base_count': len(OCR_CORRECTIONS_BASE),
            'custom_count': len(OCR_CORRECTIONS_CUSTOM),
            'total_count': len(OCR_CORRECTIONS)
        }
    })


@app.route('/api/dictionary/remove', methods=['POST'])
def api_dictionary_remove():
    """Elimina una corrección del diccionario"""
    data = request.get_json() or {}
    wrong = data.get('wrong', '').strip()
    dictionary = data.get('dictionary', 'custom')

    if not wrong:
        return jsonify({'success': False, 'error': 'Falta campo wrong'}), 400

    success = remove_correction(wrong, dictionary)
    return jsonify({
        'success': success,
        'message': f'Eliminada corrección: "{wrong}"' if success else 'No se encontró la corrección',
        'stats': {
            'base_count': len(OCR_CORRECTIONS_BASE),
            'custom_count': len(OCR_CORRECTIONS_CUSTOM),
            'total_count': len(OCR_CORRECTIONS)
        }
    })


@app.route('/api/dictionary/reload', methods=['POST'])
def api_dictionary_reload():
    """Recarga los diccionarios desde archivos"""
    load_dictionaries()
    return jsonify({
        'success': True,
        'message': 'Diccionarios recargados',
        'stats': {
            'base_count': len(OCR_CORRECTIONS_BASE),
            'custom_count': len(OCR_CORRECTIONS_CUSTOM),
            'total_count': len(OCR_CORRECTIONS)
        }
    })


@app.route('/api/dictionary/test', methods=['POST'])
def api_dictionary_test():
    """Prueba las correcciones de diccionario sobre un texto"""
    data = request.get_json() or {}
    text = data.get('text', '')

    if not text:
        return jsonify({'success': False, 'error': 'Falta campo text'}), 400

    # Aplicar correcciones
    corrected = text
    applied = []

    for wrong, correct in OCR_CORRECTIONS.items():
        if wrong in corrected:
            count = corrected.count(wrong)
            corrected = corrected.replace(wrong, correct)
            applied.append({'wrong': wrong, 'correct': correct, 'count': count})

    for pattern, replacement in OCR_REGEX_CORRECTIONS:
        matches = pattern.findall(corrected)
        if matches:
            corrected = pattern.sub(replacement, corrected)
            applied.append({'pattern': pattern.pattern, 'replacement': replacement, 'count': len(matches)})

    return jsonify({
        'success': True,
        'original': text,
        'corrected': corrected,
        'corrections_applied': applied,
        'total_corrections': len(applied)
    })


@app.route('/api/dictionary/analyze', methods=['POST'])
def api_dictionary_analyze():
    """
    Analiza una imagen/PDF con OCR y sugiere correcciones.
    Opcionalmente puede usar Vision AI para comparar.
    """
    global server_stats
    start_time = time.time()
    temp_file_path = None

    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        # Parámetro para API de Vision (opcional)
        vision_api_key = request.form.get('vision_api_key', '')
        vision_provider = request.form.get('vision_provider', 'none')  # none, openai, gemini, anthropic

        # Guardar archivo temporal
        n8nHomeDir = '/home/n8n'
        os.makedirs(f"{n8nHomeDir}/in", exist_ok=True)

        temp_filename = f"analyze_dict_{int(time.time())}_{file.filename}"
        temp_file_path = f"{n8nHomeDir}/in/{temp_filename}"
        file.save(temp_file_path)

        # Procesar con OCR
        with app.test_request_context('/ocr', method='POST', data={'filename': temp_file_path}):
            response = ocr()
            if isinstance(response, tuple):
                response_data, status_code = response
            else:
                response_data = response
            response_json = response_data.get_json()

        # Limpiar archivo temporal
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        if not response_json.get('success'):
            return jsonify({'success': False, 'error': response_json.get('error', 'OCR failed')}), 500

        ocr_text = response_json.get('extracted_text_plain', '')
        ocr_blocks = response_json.get('ocr_blocks', [])

        # Análisis de texto para detectar posibles errores
        potential_errors = []

        # Patrones comunes de errores OCR
        error_patterns = [
            (r'\b\d[a-zA-Z]\b', 'Número+letra sospechoso'),
            (r'\b[a-zA-Z]\d[a-zA-Z]\b', 'Letra+número+letra'),
            (r'\b[A-Z]{2,}\d[A-Z]*\b', 'Mayúsculas con número'),
            (r'[|l1]VA\b', 'Posible IVA mal leído'),
            (r'\bN[.|]?[1Il][.|]?F\b', 'Posible NIF mal leído'),
            (r'\bC[.|]?[1Il][.|]?F\b', 'Posible CIF mal leído'),
            (r'\d+:\d{2}\b', 'Posible precio con : en vez de ,'),
        ]

        for pattern, description in error_patterns:
            matches = re.findall(pattern, ocr_text)
            for match in matches:
                if match not in OCR_CORRECTIONS:  # Solo si no está ya corregido
                    potential_errors.append({
                        'text': match,
                        'reason': description,
                        'pattern': pattern
                    })

        # Vision AI comparison (si se proporciona API key)
        vision_result = None
        if vision_api_key and vision_provider != 'none':
            vision_result = {'status': 'not_implemented', 'message': 'Vision AI comparison coming soon'}
            # TODO: Implementar llamada a API de Vision (OpenAI, Gemini, Anthropic)

        processing_time = time.time() - start_time

        return jsonify({
            'success': True,
            'ocr_text': ocr_text,
            'ocr_text_corrected': apply_ocr_corrections(ocr_text),
            'blocks_count': len(ocr_blocks),
            'potential_errors': potential_errors,
            'potential_errors_count': len(potential_errors),
            'vision_result': vision_result,
            'processing_time': round(processing_time, 3)
        })

    except Exception as e:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass
        logger.error(f"[DICTIONARY ANALYZE ERROR] {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# API DE CONFIGURACIÓN - API Keys
# ============================================================================

# Directorio para configuración persistente
CONFIG_DIR = '/app/config'
API_KEYS_FILE = os.path.join(CONFIG_DIR, 'api_keys.json')

def load_api_keys():
    """Carga las API keys guardadas"""
    try:
        if os.path.exists(API_KEYS_FILE):
            with open(API_KEYS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"[CONFIG] Error cargando API keys: {e}")
    return {}

def save_api_keys(keys_data):
    """Guarda las API keys"""
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(API_KEYS_FILE, 'w') as f:
            json.dump(keys_data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"[CONFIG] Error guardando API keys: {e}")
        return False

def get_gemini_api_key():
    """Obtiene la API key de Gemini guardada"""
    keys = load_api_keys()
    return keys.get('gemini_api_key', '')


@app.route('/api/config/apikey', methods=['GET', 'POST'])
def api_config_apikey():
    """
    GET: Verifica si hay API key configurada (sin revelar la key)
    POST: Guarda una nueva API key
    """
    if request.method == 'GET':
        keys = load_api_keys()
        has_key = bool(keys.get('gemini_api_key', ''))
        return jsonify({
            'configured': has_key,
            'configured_at': keys.get('configured_at', None),
            'provider': 'gemini'
        })

    elif request.method == 'POST':
        try:
            data = request.get_json()
            if not data:
                return jsonify({'success': False, 'error': 'No data provided'}), 400

            api_key = data.get('api_key', '').strip()
            if not api_key:
                return jsonify({'success': False, 'error': 'API key is required'}), 400

            # Validación básica del formato de API key de Gemini
            if not api_key.startswith('AIza'):
                return jsonify({
                    'success': False,
                    'error': 'Invalid Gemini API key format. Should start with "AIza..."'
                }), 400

            # Guardar la API key
            keys_data = load_api_keys()
            keys_data['gemini_api_key'] = api_key
            keys_data['configured_at'] = datetime.now().isoformat()
            keys_data['provider'] = 'gemini'

            if save_api_keys(keys_data):
                logger.info("[CONFIG] Gemini API key guardada correctamente")
                return jsonify({
                    'success': True,
                    'message': 'API key saved successfully'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to save API key'
                }), 500

        except Exception as e:
            logger.error(f"[CONFIG API KEY ERROR] {e}")
            return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/config/apikey/test', methods=['POST'])
def api_config_apikey_test():
    """
    Prueba si la API key de Gemini funciona
    """
    try:
        api_key = get_gemini_api_key()
        if not api_key:
            return jsonify({
                'success': False,
                'error': 'No API key configured'
            }), 400

        # Intentar importar google-generativeai
        try:
            import google.generativeai as genai
        except ImportError:
            return jsonify({
                'success': False,
                'error': 'google-generativeai not installed. Rebuild Docker image.'
            }), 500

        # Configurar y probar
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Say 'OK' if you can read this.")

        return jsonify({
            'success': True,
            'message': 'API key is valid and working',
            'response': response.text[:100] if response.text else 'OK'
        })

    except Exception as e:
        error_msg = str(e)
        if 'API_KEY_INVALID' in error_msg or 'invalid' in error_msg.lower():
            return jsonify({
                'success': False,
                'error': 'API key is invalid. Please check and try again.'
            }), 400
        logger.error(f"[CONFIG API KEY TEST ERROR] {e}")
        return jsonify({'success': False, 'error': error_msg}), 500


# ============================================================================
# API DE MEJORA DE DICCIONARIO CON GEMINI VISION
# ============================================================================

def correct_text_with_gemini(image_path, ocr_text):
    """
    Usa Gemini Vision para corregir el texto OCR comparándolo con la imagen.

    Args:
        image_path: Ruta a la imagen del documento
        ocr_text: Texto extraído por OCR

    Returns:
        dict con texto corregido y correcciones detectadas
    """
    api_key = get_gemini_api_key()
    if not api_key:
        return {'success': False, 'error': 'No API key configured'}

    try:
        import google.generativeai as genai
        from PIL import Image
    except ImportError as e:
        return {'success': False, 'error': f'Missing dependency: {e}'}

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Cargar imagen
        image = Image.open(image_path)

        # Prompt para corrección OCR
        prompt = f"""Eres un corrector de OCR especializado en documentos en español (facturas, tickets, albaranes).

Te doy una imagen de un documento y el texto que extrajo el OCR. Tu tarea es:
1. Comparar el texto OCR con lo que ves en la imagen
2. Corregir SOLO los errores de reconocimiento de caracteres
3. NO cambies el formato ni añadas información que no esté en la imagen
4. Presta especial atención a:
   - Tildes y acentos (á, é, í, ó, ú, ñ)
   - Confusiones comunes: 0/O, 1/l/I, 5/S, 8/B, rn/m
   - Precios con : en lugar de , (55:23 → 55,23)
   - Términos fiscales: IVA, NIF, CIF, Total, Base
   - Ciudades españolas con tildes

TEXTO OCR A CORREGIR:
{ocr_text}

Responde SOLO con el texto corregido, manteniendo exactamente el mismo formato y estructura que el original."""

        # Llamar a Gemini Vision
        response = model.generate_content([prompt, image])
        corrected_text = response.text.strip()

        # Extraer diferencias (correcciones)
        corrections = extract_text_differences(ocr_text, corrected_text)

        return {
            'success': True,
            'original_text': ocr_text,
            'corrected_text': corrected_text,
            'corrections': corrections,
            'corrections_count': len(corrections)
        }

    except Exception as e:
        logger.error(f"[GEMINI VISION ERROR] {e}")
        return {'success': False, 'error': str(e)}


def extract_text_differences(original, corrected):
    """
    Extrae las diferencias entre el texto original y el corregido.
    Retorna lista de correcciones {wrong, correct, exists}.
    """
    corrections = []

    # Dividir en palabras
    original_words = original.split()
    corrected_words = corrected.split()

    # Usar difflib para encontrar diferencias
    import difflib

    matcher = difflib.SequenceMatcher(None, original_words, corrected_words)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            # Palabras cambiadas
            for orig, corr in zip(original_words[i1:i2], corrected_words[j1:j2]):
                if orig != corr and len(orig) > 1:  # Ignorar caracteres sueltos
                    # Verificar si ya existe en el diccionario
                    exists = orig in OCR_CORRECTIONS or orig.upper() in OCR_CORRECTIONS
                    corrections.append({
                        'wrong': orig,
                        'correct': corr,
                        'exists': exists
                    })

    # Eliminar duplicados
    seen = set()
    unique_corrections = []
    for c in corrections:
        key = (c['wrong'], c['correct'])
        if key not in seen:
            seen.add(key)
            unique_corrections.append(c)

    return unique_corrections


@app.route('/api/dictionary/improve', methods=['POST'])
def api_dictionary_improve():
    """
    Procesa un documento con OCR y luego lo corrige con Gemini Vision.
    Extrae las diferencias como correcciones sugeridas.

    Pasos:
    1. Recibe PDF/imagen
    2. Ejecuta OCR con PaddleOCR
    3. Envía imagen + texto OCR a Gemini Vision
    4. Gemini devuelve texto corregido
    5. Compara y extrae diferencias
    6. Retorna lista de correcciones sugeridas
    """
    global server_stats
    start_time = time.time()
    temp_file_path = None
    temp_image_path = None

    try:
        # Verificar API key
        if not get_gemini_api_key():
            return jsonify({
                'success': False,
                'error': 'No Gemini API key configured. Go to Configuration tab to set it up.'
            }), 400

        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        # Guardar archivo temporal
        n8nHomeDir = '/home/n8n'
        os.makedirs(f"{n8nHomeDir}/in", exist_ok=True)

        temp_filename = f"improve_{int(time.time())}_{file.filename}"
        temp_file_path = f"{n8nHomeDir}/in/{temp_filename}"
        file.save(temp_file_path)

        # 1. Procesar con OCR
        logger.info(f"[IMPROVE] Procesando OCR: {temp_filename}")
        with app.test_request_context('/ocr', method='POST', data={'filename': temp_file_path}):
            response = ocr()
            if isinstance(response, tuple):
                response_data, status_code = response
            else:
                response_data = response
            response_json = response_data.get_json()

        if not response_json.get('success'):
            return jsonify({
                'success': False,
                'error': response_json.get('error', 'OCR failed')
            }), 500

        ocr_text = response_json.get('extracted_text_plain', '')

        # 2. Preparar imagen para Gemini Vision
        # Si es PDF, convertir primera página a imagen
        if temp_file_path.lower().endswith('.pdf'):
            import subprocess
            temp_image_path = temp_file_path.replace('.pdf', '_page1.png')
            # Usar pdftoppm para convertir PDF a imagen
            try:
                subprocess.run([
                    'pdftoppm', '-png', '-f', '1', '-l', '1', '-r', '150',
                    temp_file_path, temp_file_path.replace('.pdf', '')
                ], check=True, capture_output=True)
                # pdftoppm añade -1 al nombre
                expected_path = temp_file_path.replace('.pdf', '-1.png')
                if os.path.exists(expected_path):
                    temp_image_path = expected_path
            except Exception as pdf_err:
                logger.warning(f"[IMPROVE] Error convirtiendo PDF: {pdf_err}")
                # Intentar usar el PDF directamente si Gemini lo soporta
                temp_image_path = temp_file_path
        else:
            temp_image_path = temp_file_path

        # 3. Corregir con Gemini Vision
        logger.info(f"[IMPROVE] Enviando a Gemini Vision...")
        vision_result = correct_text_with_gemini(temp_image_path, ocr_text)

        # Limpiar archivos temporales
        for path in [temp_file_path, temp_image_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass

        # También limpiar posibles archivos de conversión PDF
        for suffix in ['-1.png', '-2.png', '_page1.png']:
            check_path = temp_file_path.replace('.pdf', suffix) if temp_file_path else None
            if check_path and os.path.exists(check_path):
                try:
                    os.remove(check_path)
                except:
                    pass

        if not vision_result.get('success'):
            return jsonify(vision_result), 500

        processing_time = time.time() - start_time

        return jsonify({
            'success': True,
            'ocr_text': ocr_text,
            'corrected_text': vision_result.get('corrected_text', ''),
            'corrections': vision_result.get('corrections', []),
            'corrections_count': vision_result.get('corrections_count', 0),
            'processing_time': round(processing_time, 3)
        })

    except Exception as e:
        # Limpiar archivos temporales
        for path in [temp_file_path, temp_image_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass

        logger.error(f"[DICTIONARY IMPROVE ERROR] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# API DE IMPORTACIÓN DE DICCIONARIOS
# ============================================================================

# Diccionarios externos predefinidos
EXTERNAL_DICTIONARIES = {
    'spanish_cities': {
        'name': 'Ciudades Españolas',
        'description': 'Ciudades y provincias de España con tildes correctas',
        'url': None,  # Diccionario embebido
        'corrections': {
            # Capitales de provincia
            'Cadiz': 'Cádiz', 'CADIZ': 'CÁDIZ', 'cadiz': 'cádiz',
            'Cordoba': 'Córdoba', 'CORDOBA': 'CÓRDOBA', 'cordoba': 'córdoba',
            'Malaga': 'Málaga', 'MALAGA': 'MÁLAGA', 'malaga': 'málaga',
            'Almeria': 'Almería', 'ALMERIA': 'ALMERÍA', 'almeria': 'almería',
            'Jaen': 'Jaén', 'JAEN': 'JAÉN', 'jaen': 'jaén',
            'Leon': 'León', 'LEON': 'LEÓN', 'leon': 'león',
            'Avila': 'Ávila', 'AVILA': 'ÁVILA', 'avila': 'ávila',
            'Caceres': 'Cáceres', 'CACERES': 'CÁCERES', 'caceres': 'cáceres',
            'Castellon': 'Castellón', 'CASTELLON': 'CASTELLÓN',
            'Gijon': 'Gijón', 'GIJON': 'GIJÓN', 'gijon': 'gijón',
            'Logrono': 'Logroño', 'LOGRONO': 'LOGROÑO', 'logrono': 'logroño',
            'San Sebastian': 'San Sebastián', 'SAN SEBASTIAN': 'SAN SEBASTIÁN',
            'Merida': 'Mérida', 'MERIDA': 'MÉRIDA', 'merida': 'mérida',
            # Comunidades autónomas
            'Aragon': 'Aragón', 'ARAGON': 'ARAGÓN',
            'Cataluna': 'Cataluña', 'CATALUNA': 'CATALUÑA',
            'Andalucia': 'Andalucía', 'ANDALUCIA': 'ANDALUCÍA',
            'Pais Vasco': 'País Vasco', 'PAIS VASCO': 'PAÍS VASCO',
            # Otras ciudades
            'Mostoles': 'Móstoles', 'Alcorcon': 'Alcorcón',
            'Torrejon': 'Torrejón', 'Leganes': 'Leganés',
            'Getafe': 'Getafe', 'Alcala': 'Alcalá',
        }
    },
    'fiscal_terms': {
        'name': 'Términos Fiscales',
        'description': 'IVA, NIF, CIF y términos de facturación',
        'url': None,
        'corrections': {
            # IVA
            '1VA': 'IVA', 'lVA': 'IVA', '|VA': 'IVA',
            'I.V.A': 'IVA', 'I.V.A.': 'IVA',
            # NIF/CIF
            'N1F': 'NIF', 'NlF': 'NIF', 'N.1.F': 'NIF', 'N.I.F': 'NIF',
            'C1F': 'CIF', 'ClF': 'CIF', 'C.1.F': 'CIF', 'C.I.F': 'CIF',
            # IRPF
            '1RPF': 'IRPF', 'lRPF': 'IRPF', '|RPF': 'IRPF',
            # Otros términos
            'TOTA1': 'TOTAL', 'T0TAL': 'TOTAL', 'TOTAI': 'TOTAL',
            'lMPORTE': 'IMPORTE', '1MPORTE': 'IMPORTE',
            'SUBTOTA1': 'SUBTOTAL', 'SUBT0TAL': 'SUBTOTAL',
            'DESCUENT0': 'DESCUENTO', 'DESCUENT': 'DESCUENTO',
            'FACTURA': 'FACTURA', 'FACT': 'FACTURA',
            'FECHA': 'FECHA', 'FECH': 'FECHA',
            'CANTIDAD': 'CANTIDAD', 'CANT': 'CANTIDAD',
            'PREC10': 'PRECIO', 'PRECI0': 'PRECIO',
            # Base imponible
            'BASE 1MPONIBLE': 'BASE IMPONIBLE',
            'BASE IMPON1BLE': 'BASE IMPONIBLE',
            'BASE IMP0NIBLE': 'BASE IMPONIBLE',
        }
    },
    'common_ocr_errors': {
        'name': 'Errores OCR Comunes',
        'description': 'Confusiones típicas de reconocimiento de caracteres',
        'url': None,
        'corrections': {
            # Números por letras
            '0': 'O',  # Solo en contextos específicos
            '1': 'l',  # Solo en contextos específicos
            # Palabras comunes mal leídas
            'cornprar': 'comprar', 'cornpra': 'compra',
            'nurnero': 'número', 'nurneros': 'números',
            'tarnbien': 'también', 'tarnbién': 'también',
            'siernpre': 'siempre',
            'tiernpo': 'tiempo',
            'ejernplo': 'ejemplo',
            # rn -> m
            'inforrnacion': 'información',
            'inforrnación': 'información',
            'forrnulario': 'formulario',
            'norrnativa': 'normativa',
            # Tildes comunes
            'informacion': 'información',
            'direccion': 'dirección',
            'telefono': 'teléfono',
            'numero': 'número',
            'articulo': 'artículo',
            'metodo': 'método',
            'pagina': 'página',
            'codigo': 'código',
            'automatico': 'automático',
            'electronico': 'electrónico',
        }
    },
    'products_services': {
        'name': 'Productos y Servicios',
        'description': 'Productos, combustibles y servicios comunes',
        'url': None,
        'corrections': {
            # Combustibles
            'GASOLEO': 'GASÓLEO', 'Gasoleo': 'Gasóleo', 'gasoleo': 'gasóleo',
            'GASO1L': 'GASOIL', 'GAS0IL': 'GASOIL',
            'GASO1EO': 'GASÓLEO', 'GAS0LEO': 'GASÓLEO',
            'DIES3L': 'DIESEL', 'D1ESEL': 'DIESEL', 'DIESE1': 'DIESEL',
            # Gasolina
            'GASO1INA': 'GASOLINA', 'GAS0LINA': 'GASOLINA',
            # Electricidad
            'E1ECTRICIDAD': 'ELECTRICIDAD', 'ELECTRIC1DAD': 'ELECTRICIDAD',
            'ELECTR1CIDAD': 'ELECTRICIDAD',
            # Agua
            'SUMIN1STRO': 'SUMINISTRO', 'SUMINISTR0': 'SUMINISTRO',
            # Servicios
            'SERV1CIO': 'SERVICIO', 'SERVIC1O': 'SERVICIO',
            'MANTEN1MIENTO': 'MANTENIMIENTO', 'MANTENIM1ENTO': 'MANTENIMIENTO',
            'REPARAC1ON': 'REPARACIÓN', 'REPARAC1ÓN': 'REPARACIÓN',
        }
    }
}


@app.route('/api/dictionary/import', methods=['POST'])
def api_dictionary_import():
    """
    Importa diccionarios desde:
    - Diccionarios predefinidos (por nombre)
    - URL de archivo JSON
    - Archivo JSON subido

    El JSON debe tener formato: {"wrong": "correct", ...}
    """
    try:
        # Caso 1: Archivo subido
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'Empty filename'}), 400

            try:
                content = file.read().decode('utf-8')
                corrections = json.loads(content)

                if not isinstance(corrections, dict):
                    return jsonify({
                        'success': False,
                        'error': 'Invalid format. Expected {"wrong": "correct", ...}'
                    }), 400

                # Añadir al diccionario custom
                added = 0
                for wrong, correct in corrections.items():
                    if wrong not in OCR_CORRECTIONS:
                        OCR_CORRECTIONS[wrong] = correct
                        OCR_CORRECTIONS_CUSTOM[wrong] = correct
                        added += 1

                # Guardar diccionario
                save_custom_dictionary()

                return jsonify({
                    'success': True,
                    'message': f'Imported from file: {file.filename}',
                    'imported': added,
                    'total_in_file': len(corrections),
                    'skipped': len(corrections) - added
                })

            except json.JSONDecodeError:
                return jsonify({
                    'success': False,
                    'error': 'Invalid JSON file'
                }), 400

        # Caso 2: JSON body con URL o nombre de diccionario
        data = request.get_json() or {}

        # Importar desde URL
        if 'url' in data:
            url = data['url']
            try:
                import urllib.request
                with urllib.request.urlopen(url, timeout=10) as response:
                    content = response.read().decode('utf-8')
                    corrections = json.loads(content)

                    if not isinstance(corrections, dict):
                        return jsonify({
                            'success': False,
                            'error': 'Invalid format in URL. Expected {"wrong": "correct", ...}'
                        }), 400

                    added = 0
                    for wrong, correct in corrections.items():
                        if wrong not in OCR_CORRECTIONS:
                            OCR_CORRECTIONS[wrong] = correct
                            OCR_CORRECTIONS_CUSTOM[wrong] = correct
                            added += 1

                    save_custom_dictionary()

                    return jsonify({
                        'success': True,
                        'message': f'Imported from URL',
                        'imported': added,
                        'total_in_url': len(corrections)
                    })

            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'Error fetching URL: {str(e)}'
                }), 400

        # Importar diccionario predefinido
        if 'dictionary' in data:
            dict_name = data['dictionary']

            if dict_name not in EXTERNAL_DICTIONARIES:
                return jsonify({
                    'success': False,
                    'error': f'Unknown dictionary: {dict_name}',
                    'available': list(EXTERNAL_DICTIONARIES.keys())
                }), 400

            ext_dict = EXTERNAL_DICTIONARIES[dict_name]
            corrections = ext_dict.get('corrections', {})

            added = 0
            for wrong, correct in corrections.items():
                if wrong not in OCR_CORRECTIONS:
                    OCR_CORRECTIONS[wrong] = correct
                    OCR_CORRECTIONS_CUSTOM[wrong] = correct
                    added += 1

            save_custom_dictionary()

            return jsonify({
                'success': True,
                'message': f"Imported: {ext_dict['name']}",
                'description': ext_dict.get('description', ''),
                'imported': added,
                'total_in_dictionary': len(corrections),
                'skipped': len(corrections) - added
            })

        return jsonify({
            'success': False,
            'error': 'Provide file, url, or dictionary name'
        }), 400

    except Exception as e:
        logger.error(f"[DICTIONARY IMPORT ERROR] {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/dictionary/available')
def api_dictionary_available():
    """
    Lista los diccionarios externos disponibles para importar
    """
    available = []
    for key, value in EXTERNAL_DICTIONARIES.items():
        available.append({
            'id': key,
            'name': value['name'],
            'description': value.get('description', ''),
            'count': len(value.get('corrections', {}))
        })

    return jsonify({
        'success': True,
        'dictionaries': available
    })


# ============================================================================
# PP-STRUCTURE: ANÁLISIS AVANZADO DE DOCUMENTOS
# Layout Analysis + Table Recognition (SLANet)
# ============================================================================

# Variables globales para pipelines PP-Structure (lazy loading)
pp_table_pipeline = None
pp_layout_pipeline = None
pp_structure_initialized = False

# Circuit Breaker para PP-Structure
# Evita llamadas repetidas cuando el servicio está fallando
pp_structure_circuit_breaker = {
    'failures': 0,           # Contador de fallos consecutivos
    'last_failure': 0,       # Timestamp del último fallo
    'state': 'closed',       # closed (OK), open (bloqueado), half-open (probando)
    'threshold': 3,          # Fallos antes de abrir el circuito
    'reset_timeout': 60,     # Segundos antes de intentar de nuevo
}

def check_circuit_breaker():
    """
    Verifica si el circuit breaker permite llamadas a PP-Structure.
    Returns: True si se puede llamar, False si está bloqueado
    """
    global pp_structure_circuit_breaker
    cb = pp_structure_circuit_breaker

    if cb['state'] == 'closed':
        return True

    if cb['state'] == 'open':
        # Verificar si ya pasó el timeout para intentar de nuevo
        import time
        if time.time() - cb['last_failure'] > cb['reset_timeout']:
            cb['state'] = 'half-open'
            logger.info("[CIRCUIT-BREAKER] Estado: half-open, probando PP-Structure...")
            return True
        return False

    if cb['state'] == 'half-open':
        return True

    return False

def record_circuit_success():
    """Registra un éxito, resetea el circuit breaker"""
    global pp_structure_circuit_breaker
    pp_structure_circuit_breaker['failures'] = 0
    pp_structure_circuit_breaker['state'] = 'closed'

def record_circuit_failure():
    """Registra un fallo, puede abrir el circuit breaker"""
    global pp_structure_circuit_breaker
    import time
    cb = pp_structure_circuit_breaker

    cb['failures'] += 1
    cb['last_failure'] = time.time()

    if cb['state'] == 'half-open':
        # Si falla en half-open, volver a abrir
        cb['state'] = 'open'
        logger.warning("[CIRCUIT-BREAKER] Fallo en half-open, circuito ABIERTO")
    elif cb['failures'] >= cb['threshold']:
        cb['state'] = 'open'
        logger.warning(f"[CIRCUIT-BREAKER] {cb['failures']} fallos consecutivos, circuito ABIERTO")

def init_pp_structure_pipelines(force_reinit=False):
    """
    Inicializa los pipelines de PP-Structure bajo demanda.

    Args:
        force_reinit: Si True, fuerza reinicialización aunque ya estén cargados
    """
    global pp_table_pipeline, pp_layout_pipeline, pp_structure_initialized

    if pp_structure_initialized and not force_reinit:
        return True

    try:
        from paddlex import create_pipeline

        if force_reinit:
            logger.info("[PP-STRUCTURE] Forzando reinicialización de pipelines...")
            pp_structure_initialized = False
            pp_table_pipeline = None
            pp_layout_pipeline = None

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
        pp_structure_initialized = False
        return False


def run_pp_structure_with_retry(pipeline, file_path, max_retries=2, timeout_seconds=120):
    """
    Ejecuta un pipeline de PP-Structure con manejo de errores std::exception.

    Incluye:
    - Circuit breaker para evitar llamadas cuando el servicio falla repetidamente
    - Reintentos con reinicialización de pipelines
    - Timeout configurable

    Args:
        pipeline: El pipeline a ejecutar ('table' o 'layout')
        file_path: Ruta al archivo a procesar
        max_retries: Número máximo de reintentos
        timeout_seconds: Timeout máximo en segundos

    Returns:
        Lista de resultados o None si falla
    """
    global pp_table_pipeline, pp_layout_pipeline
    import threading
    import os

    # Verificar circuit breaker primero
    if not check_circuit_breaker():
        logger.warning("[PP-STRUCTURE] Circuit breaker ABIERTO, saltando PP-Structure")
        return None

    # Calcular timeout dinámico basado en tamaño del archivo
    try:
        file_size = os.path.getsize(file_path)
        # Base: 60s + 30s por cada 500KB
        dynamic_timeout = min(60 + (file_size // 500000) * 30, timeout_seconds)
        logger.info(f"[PP-STRUCTURE] Archivo: {file_size/1024:.1f}KB, timeout: {dynamic_timeout}s")
    except:
        dynamic_timeout = timeout_seconds

    result_container = {'result': None, 'error': None}

    def run_pipeline():
        try:
            if pipeline == 'table':
                pp = pp_table_pipeline
            else:
                pp = pp_layout_pipeline

            if pp is None:
                logger.warning(f"[PP-STRUCTURE] Pipeline {pipeline} no inicializado, inicializando...")
                if not init_pp_structure_pipelines(force_reinit=True):
                    result_container['error'] = "No se pudo inicializar"
                    return
                pp = pp_table_pipeline if pipeline == 'table' else pp_layout_pipeline

            result_container['result'] = list(pp.predict(file_path))
        except Exception as e:
            result_container['error'] = str(e)

    for attempt in range(max_retries + 1):
        result_container = {'result': None, 'error': None}

        # Ejecutar en thread con timeout
        thread = threading.Thread(target=run_pipeline)
        thread.start()
        thread.join(timeout=dynamic_timeout)

        if thread.is_alive():
            logger.error(f"[PP-STRUCTURE] Timeout después de {dynamic_timeout}s")
            record_circuit_failure()
            # No podemos matar el thread, pero al menos no esperamos más
            return None

        if result_container['result'] is not None:
            record_circuit_success()
            return result_container['result']

        if result_container['error']:
            error_str = result_container['error']

            # Detectar errores de tipo std::exception u otros errores de C++
            is_cpp_error = 'std::exception' in error_str or 'Segmentation' in error_str

            if is_cpp_error and attempt < max_retries:
                logger.warning(f"[PP-STRUCTURE] Error C++ detectado (intento {attempt + 1}/{max_retries + 1}): {error_str}")
                logger.info("[PP-STRUCTURE] Reinicializando pipelines...")

                # Forzar reinicialización
                if init_pp_structure_pipelines(force_reinit=True):
                    logger.info("[PP-STRUCTURE] Pipelines reinicializados, reintentando...")
                    continue
                else:
                    logger.error("[PP-STRUCTURE] No se pudieron reinicializar los pipelines")
                    record_circuit_failure()
                    return None
            else:
                logger.error(f"[PP-STRUCTURE] Error fatal (intento {attempt + 1}): {error_str}")
                record_circuit_failure()
                return None

    record_circuit_failure()
    return None


@app.route('/structure', methods=['POST'])
def structure():
    """
    Endpoint avanzado de análisis estructural de documentos.
    Usa PP-Structure para:
    - Detectar regiones del documento (tablas, texto, imágenes)
    - Extraer tablas como HTML estructurado
    - Proporcionar coordenadas de cada región

    Parámetros:
        file: Archivo PDF o imagen
        extract_tables: bool - Extraer tablas como HTML (default: true)
        extract_layout: bool - Detectar regiones del layout (default: true)

    Retorna JSON estructurado con:
        - layout_regions: Lista de regiones detectadas
        - tables: Lista de tablas con HTML y datos
        - ocr_text: Texto completo extraído
        - metadata: Información del documento
    """
    global server_stats
    start_time = time.time()
    server_stats['total_requests'] += 1

    temp_file_path = None

    try:
        # Validar archivo
        if 'file' not in request.files:
            server_stats['failed_requests'] += 1
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            server_stats['failed_requests'] += 1
            return jsonify({'error': 'Empty filename'}), 400

        # Validar extensión
        ext = Path(file.filename).suffix.lower()
        if ext not in ['.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
            server_stats['failed_requests'] += 1
            return jsonify({'error': f'Unsupported file format: {ext}'}), 400

        # Obtener parámetros
        extract_tables = request.form.get('extract_tables', 'true').lower() == 'true'
        extract_layout = request.form.get('extract_layout', 'true').lower() == 'true'

        # Inicializar pipelines si es necesario
        if not init_pp_structure_pipelines():
            server_stats['failed_requests'] += 1
            return jsonify({'error': 'PP-Structure pipelines not available'}), 503

        # Guardar archivo temporal
        n8nHomeDir = '/home/n8n'
        os.makedirs(f"{n8nHomeDir}/in", exist_ok=True)

        temp_filename = f"struct_{int(time.time())}_{file.filename}"
        temp_file_path = f"{n8nHomeDir}/in/{temp_filename}"
        file.save(temp_file_path)

        logger.info(f"[STRUCTURE] Procesando: {temp_filename}")

        # Resultados
        result = {
            'success': True,
            'filename': file.filename,
            'pages': [],
            'summary': {
                'total_pages': 0,
                'total_tables': 0,
                'total_regions': 0,
                'region_types': {}
            }
        }

        # Procesar con layout_parsing (detecta regiones + tablas)
        if extract_layout or extract_tables:
            try:
                # Usar función con reintentos para manejar std::exception
                pipeline_type = 'table' if extract_tables else 'layout'
                pipeline_results = run_pp_structure_with_retry(pipeline_type, temp_file_path)

                if pipeline_results is None:
                    server_stats['failed_requests'] += 1
                    return jsonify({
                        'error': 'PP-Structure pipeline failed after retries',
                        'suggestion': 'Try again or use /process endpoint instead'
                    }), 503

                for page_idx, page_result in enumerate(pipeline_results):
                    page_data = {
                        'page_number': page_idx + 1,
                        'regions': [],
                        'tables': [],
                        'ocr_texts': []
                    }

                    res = page_result.json.get('res', {})

                    # Extraer regiones del layout
                    layout_res = res.get('layout_det_res', {})
                    if 'boxes' in layout_res:
                        for box in layout_res['boxes']:
                            region = {
                                'type': box.get('label', 'unknown'),
                                'confidence': round(box.get('score', 0), 3),
                                'bbox': box.get('coordinate', [])
                            }
                            page_data['regions'].append(region)

                            # Contar tipos de región
                            rtype = region['type']
                            if rtype not in result['summary']['region_types']:
                                result['summary']['region_types'][rtype] = 0
                            result['summary']['region_types'][rtype] += 1

                    # Extraer OCR global
                    ocr_res = res.get('overall_ocr_res', {})
                    if 'rec_texts' in ocr_res:
                        page_data['ocr_texts'] = ocr_res['rec_texts']

                    # Extraer tablas con HTML
                    if extract_tables:
                        table_list = res.get('table_res_list', [])
                        for table_idx, table in enumerate(table_list):
                            table_data = {
                                'table_number': table_idx + 1,
                                'html': table.get('pred_html', ''),
                                'cell_count': len(table.get('cell_box_list', [])),
                                'cells': []
                            }

                            # Extraer info de celdas
                            cell_boxes = table.get('cell_box_list', [])
                            ocr_pred = table.get('table_ocr_pred', [])

                            for i, cell_box in enumerate(cell_boxes[:50]):  # Limitar a 50 celdas
                                cell = {
                                    'bbox': cell_box,
                                    'text': ''
                                }
                                # El texto de la celda se puede extraer del HTML
                                table_data['cells'].append(cell)

                            page_data['tables'].append(table_data)
                            result['summary']['total_tables'] += 1

                    result['pages'].append(page_data)
                    result['summary']['total_regions'] += len(page_data['regions'])

                result['summary']['total_pages'] = len(result['pages'])

            except Exception as e:
                logger.error(f"[STRUCTURE] Error en PP-Structure: {e}")
                import traceback
                logger.error(traceback.format_exc())
                result['pp_structure_error'] = str(e)

        # Calcular tiempo de procesamiento
        processing_time = time.time() - start_time
        result['processing_time'] = round(processing_time, 2)

        server_stats['successful_requests'] += 1
        server_stats['total_processing_time'] += processing_time

        logger.info(f"[STRUCTURE] Completado en {processing_time:.2f}s - {result['summary']['total_tables']} tablas, {result['summary']['total_regions']} regiones")

        return jsonify(result)

    except Exception as e:
        server_stats['failed_requests'] += 1
        logger.error(f"[STRUCTURE ERROR] {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

    finally:
        # Limpiar archivo temporal
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass


@app.route('/extract', methods=['POST'])
def extract():
    """
    Endpoint de extracción inteligente para facturas y tickets.
    Combina PP-Structure con heurísticas para extraer campos clave.

    Parámetros:
        file: Archivo PDF o imagen
        document_type: Tipo de documento (invoice, ticket, receipt) - auto-detecta si no se especifica

    Retorna JSON estructurado con:
        - document_type: Tipo detectado
        - fields: Campos extraídos (vendor, date, total, etc.)
        - tables: Tablas detectadas con datos estructurados
        - raw_text: Texto completo
        - confidence: Nivel de confianza de la extracción
    """
    global server_stats
    start_time = time.time()
    server_stats['total_requests'] += 1

    temp_file_path = None

    try:
        # Validar archivo
        if 'file' not in request.files:
            server_stats['failed_requests'] += 1
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            server_stats['failed_requests'] += 1
            return jsonify({'error': 'Empty filename'}), 400

        ext = Path(file.filename).suffix.lower()
        if ext not in ['.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
            server_stats['failed_requests'] += 1
            return jsonify({'error': f'Unsupported file format: {ext}'}), 400

        document_type = request.form.get('document_type', 'auto')

        # Guardar archivo temporal
        n8nHomeDir = '/home/n8n'
        os.makedirs(f"{n8nHomeDir}/in", exist_ok=True)

        temp_filename = f"extract_{int(time.time())}_{file.filename}"
        temp_file_path = f"{n8nHomeDir}/in/{temp_filename}"
        file.save(temp_file_path)

        logger.info(f"[EXTRACT] Procesando: {temp_filename}, tipo: {document_type}")

        raw_text = ""
        ocr_blocks = []
        extraction_method = "ocr"

        # Para PDFs, intentar primero con pdftotext -layout (mejor para vectoriales)
        if ext == '.pdf':
            try:
                import subprocess
                # Intentar pdftotext con layout
                result = subprocess.run(
                    ['pdftotext', '-layout', temp_file_path, '-'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0 and result.stdout.strip():
                    pdftotext_output = result.stdout.strip()
                    # Verificar que no sea solo espacios/saltos de línea
                    if len(pdftotext_output.replace('\n', '').replace(' ', '')) > 50:
                        raw_text = pdftotext_output
                        extraction_method = "pdftotext_layout"
                        logger.info(f"[EXTRACT] Usando pdftotext -layout ({len(raw_text)} chars)")
            except Exception as e:
                logger.warning(f"[EXTRACT] pdftotext falló, usando OCR: {e}")

        # Si pdftotext no funcionó (PDF escaneado o imagen), usar OCR
        if not raw_text:
            with app.test_request_context():
                ocr_response = None
                try:
                    from flask import g
                    with app.test_request_context(
                        '/ocr',
                        method='POST',
                        data={'filename': temp_file_path}
                    ):
                        ocr_response = ocr()
                except Exception as e:
                    logger.error(f"[EXTRACT] Error llamando OCR interno: {e}")

            if ocr_response and hasattr(ocr_response, 'get_json'):
                ocr_data = ocr_response.get_json()
                if ocr_data:
                    raw_text = ocr_data.get('extracted_text', '')
                    ocr_blocks = ocr_data.get('ocr_blocks', [])
                    extraction_method = "ocr"
                    logger.info(f"[EXTRACT] Usando OCR ({len(raw_text)} chars)")

        # Aplicar correcciones de diccionario
        if raw_text:
            raw_text = apply_ocr_corrections(raw_text)

        # Extraer campos usando heurísticas
        fields = extract_invoice_fields(raw_text)

        # Aplicar normalización v4.3 (pasamos raw_text para detectar moneda)
        fields = normalize_fields(fields, raw_text=raw_text)

        # Auto-detectar tipo de documento
        if document_type == 'auto':
            document_type = detect_document_type(raw_text, fields)

        # Intentar obtener tablas con PP-Structure
        tables = []
        if init_pp_structure_pipelines():
            try:
                table_results = list(pp_table_pipeline.predict(temp_file_path))
                for page_result in table_results:
                    res = page_result.json.get('res', {})
                    for table in res.get('table_res_list', []):
                        tables.append({
                            'html': table.get('pred_html', ''),
                            'cell_count': len(table.get('cell_box_list', []))
                        })
            except Exception as e:
                logger.warning(f"[EXTRACT] Error extrayendo tablas: {e}")

        processing_time = time.time() - start_time

        result = {
            'success': True,
            'document_type': document_type,
            'extraction_method': extraction_method,
            'fields': fields,
            'tables': tables,
            'raw_text': raw_text,
            'processing_time': round(processing_time, 2),
            'confidence': calculate_extraction_confidence(fields)
        }

        server_stats['successful_requests'] += 1
        server_stats['total_processing_time'] += processing_time

        logger.info(f"[EXTRACT] Completado en {processing_time:.2f}s - tipo: {document_type}, confianza: {result['confidence']}")

        return jsonify(result)

    except Exception as e:
        server_stats['failed_requests'] += 1
        logger.error(f"[EXTRACT ERROR] {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass


def extract_invoice_fields(text):
    """
    Extrae campos clave de una factura/ticket usando regex y heurísticas.
    MEJORADO v4.0: Incluye LINE_ITEMS, CUSTOMER_NAME, CUSTOMER_NIF
    """
    import re

    fields = {
        'vendor': None,
        'vendor_nif': None,
        'customer_name': None,
        'customer_nif': None,
        'invoice_number': None,
        'date': None,
        'total': None,
        'tax_base': None,
        'tax_rate': None,
        'tax_amount': None,
        'payment_method': None,
        'line_items': []  # Lista de conceptos/productos
    }

    if not text:
        return fields

    text_upper = text.upper()
    lines = text.split('\n')

    # NIF/CIF (patrón español) - formato: letra + 8 dígitos o 8 dígitos + letra
    nif_patterns = [
        r'(?:N\.?I\.?F\.?|C\.?I\.?F\.?)[:\s]*([A-Z]\d{8})',  # B12345678
        r'(?:N\.?I\.?F\.?|C\.?I\.?F\.?)[:\s]*(\d{8}[A-Z])',  # 12345678A
        r'(?:N\.?I\.?F\.?|C\.?I\.?F\.?)[:\s]*([A-Z]?\d{7,8}[A-Z]?)',  # Genérico
    ]
    for pattern in nif_patterns:
        nif_match = re.search(pattern, text_upper)
        if nif_match:
            nif = nif_match.group(1)
            # Corregir confusiones OCR comunes en NIFs
            nif = nif.replace('R', 'B').replace('I', '1').replace('O', '0')
            fields['vendor_nif'] = nif
            break

    # Número de factura - Mejorado para múltiples formatos
    # Estrategia 1: Buscar patrones específicos de proveedores conocidos PRIMERO
    # (Estos son más confiables que los patrones genéricos)
    specific_patterns = [
        # Vodafone/DMI: VFR seguido de 8 dígitos
        r'\b(VFR\d{8})\b',
        # Pedidos: VP seguido de 8 dígitos
        r'\b(VP\d{8})\b',
        # Olivenet: ON2025-584267 (prefijo+año+guión+secuencial)
        r'\b(ON\d{4}-\d+)\b',
        # Formato año+secuencial: 2025003047
        r'\b(20\d{2}\d{6,})\b',
        # Genérico: FAC, INV seguido de números (con límite de palabra)
        r'\b((?:FAC|INV)\d{6,})\b',
    ]
    for pattern in specific_patterns:
        match = re.search(pattern, text_upper)
        if match:
            fields['invoice_number'] = match.group(1)
            break

    # Estrategia 2: Patrones inline con etiquetas explícitas
    if not fields.get('invoice_number'):
        invoice_patterns_inline = [
            # Nº de factura: ON2025-584267 (con 'de' opcional, acepta guiones)
            r'N[°º]?\s*(?:DE\s+)?FACTURA[:\s]+([A-Z0-9\-]+)',
            # Nº Factura: 12345678 o FACTURA Nº: 12345678
            r'(?:FACTURA\s*N[°º]?)[:\s]+([A-Z]{0,3}\d{6,}[A-Z0-9]*)',
            # INVOICE: INV12345
            r'\bINVOICE[:\s#]+([A-Z0-9\-/]+)',
            # REF. FACTURA: (con límite de palabra para evitar 'reflejados')
            r'\bREF\.?\s*(?:FACTURA)?[:\s]+([A-Z0-9\-/]+)',
        ]
        for pattern in invoice_patterns_inline:
            match = re.search(pattern, text_upper)
            if match:
                invoice_num = match.group(1).strip()
                # Validar que parece un número de factura (mínimo 4 caracteres, debe tener dígitos)
                if len(invoice_num) >= 4 and any(c.isdigit() for c in invoice_num):
                    fields['invoice_number'] = invoice_num
                    break

    # Estrategia 3: Formato multilínea (etiqueta en una línea, valor en la siguiente)
    # Común en facturas Vodafone/DMI: "Nº Factura\nVFR25087570"
    if not fields.get('invoice_number'):
        for i, line in enumerate(lines):
            line_upper = line.upper().strip()
            # Buscar líneas que contienen "Nº Factura" al inicio
            if re.match(r'^N[°º]?\s*FACTURA', line_upper):
                # Buscar en las siguientes 3 líneas un código alfanumérico
                for j in range(1, 4):
                    if i + j < len(lines):
                        next_line = lines[i + j].strip()
                        # Buscar código tipo VFR25087570, 2025003047, etc.
                        code_match = re.match(r'^([A-Z]{0,4}\d{6,}[A-Z0-9]*)$', next_line.upper())
                        if code_match:
                            fields['invoice_number'] = code_match.group(1)
                            break
                if fields.get('invoice_number'):
                    break

    # Fecha
    date_patterns = [
        r'(?:FECHA)[:\s]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
        r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
        r'(\d{1,2}\s+(?:ENE|FEB|MAR|ABR|MAY|JUN|JUL|AGO|SEP|OCT|NOV|DIC)[A-Z]*\s+\d{2,4})',
    ]
    for pattern in date_patterns:
        match = re.search(pattern, text_upper)
        if match:
            fields['date'] = match.group(1).strip()
            break

    # Total - buscar en múltiples formatos
    # Formato 1: "TOTAL FACTURA: 94.74 €" o "TOTAL: 94.74"
    # Formato 2: "Total Factura                           94.74 €" (espacios largos)
    total_patterns = [
        # Formato con espacios largos (PDFs vectoriales con layout)
        r'TOTAL\s+FACTURA\s+(\d+[.,]\d{2})\s*€?',
        r'TOTAL\s+A\s+PAGAR\s+(\d+[.,]\d{2})\s*€?',
        # Formato tradicional con : o espacios cortos
        r'(?:TOTAL\s*(?:FACTURA|A\s*PAGAR)?)[:\s]+(\d+[.,]\d{2})\s*€?',
        r'(?:IMPORTE\s*TOTAL)[:\s]+(\d+[.,]\d{2})\s*€?',
        # Genérico TOTAL seguido de número
        r'TOTAL[:\s]+(\d+[.,]\d{2})\s*€?',
    ]
    for pattern in total_patterns:
        match = re.search(pattern, text_upper)
        if match:
            total_str = match.group(1).replace(',', '.')
            try:
                fields['total'] = float(total_str)
            except:
                pass
            break

    # Si no encontramos total, buscar línea por línea
    if not fields.get('total'):
        for line in lines:
            line_upper = line.upper().strip()
            if 'TOTAL' in line_upper and 'FACTURA' in line_upper:
                # Buscar número en la línea
                amounts = re.findall(r'(\d+[.,]\d{2})\s*€?', line)
                if amounts:
                    try:
                        fields['total'] = float(amounts[-1].replace(',', '.'))
                    except:
                        pass
                    break

    # Base imponible - múltiples formatos
    base_patterns = [
        # Formato con espacios largos
        r'(?:TOTAL\s*)?(?:\(?\s*BASE\s*IMPONIBLE\s*\)?)\s+(\d+[.,]\d{2})\s*€?',
        # Formato tradicional
        r'(?:BASE\s*(?:IMPONIBLE)?)[:\s]+(\d+[.,]\d{2})\s*€?',
        r'(?:SUBTOTAL)[:\s]+(\d+[.,]\d{2})\s*€?',
    ]
    for pattern in base_patterns:
        match = re.search(pattern, text_upper)
        if match:
            base_str = match.group(1).replace(',', '.')
            try:
                fields['tax_base'] = float(base_str)
            except:
                pass
            break

    # Si no encontramos base, buscar línea por línea
    if not fields.get('tax_base'):
        for line in lines:
            line_upper = line.upper().strip()
            if 'BASE' in line_upper and 'IMPONIBLE' in line_upper:
                amounts = re.findall(r'(\d+[.,]\d{2})\s*€?', line)
                if amounts:
                    try:
                        fields['tax_base'] = float(amounts[-1].replace(',', '.'))
                    except:
                        pass
                    break

    # =========================================================================
    # IVA - Extracción inteligente usando múltiples estrategias
    # =========================================================================

    # Estrategia 0: Buscar línea "IVA (21%)" con espacios largos y número al final
    # Formato: "IVA (21%)                                                     16.44 €"
    for line in lines:
        line_upper = line.upper().strip()
        # Buscar línea que contenga IVA y porcentaje
        iva_line_match = re.search(r'IVA\s*\(?\s*(\d{1,2})\s*%\s*\)?', line_upper)
        if iva_line_match:
            # Extraer la tasa
            if not fields.get('tax_rate'):
                try:
                    fields['tax_rate'] = float(iva_line_match.group(1))
                except:
                    pass
            # Buscar el importe al final de la línea
            amounts = re.findall(r'(\d+[.,]\d{2})\s*€?$', line_upper)
            if amounts and not fields.get('tax_amount'):
                try:
                    fields['tax_amount'] = float(amounts[-1].replace(',', '.'))
                except:
                    pass
            # Si encontramos ambos, salir
            if fields.get('tax_rate') and fields.get('tax_amount'):
                break

    # Estrategia 1: Buscar patrón "IVA (21%)" o similar seguido de importe
    if not fields.get('tax_amount'):
        iva_with_amount = re.search(r'IVA\s*\(?(\d{1,2})\s*%?\)?[:\s]+(\d+[.,]\d{2})', text_upper)
        if iva_with_amount:
            try:
                if not fields.get('tax_rate'):
                    fields['tax_rate'] = float(iva_with_amount.group(1))
                fields['tax_amount'] = float(iva_with_amount.group(2).replace(',', '.'))
            except:
                pass

    # Estrategia 2: Buscar etiquetas en líneas separadas (formato típico de tickets)
    if not fields.get('tax_amount'):
        # Buscar líneas con etiquetas clave
        text_lines = text.upper().split('\n')
        for i, line in enumerate(text_lines):
            line = line.strip()

            # Buscar "%IVA" seguido de número en siguiente(s) línea(s)
            if '%IVA' in line or ('IVA' in line and '(' not in line):
                # Buscar número en la misma línea después de IVA
                iva_same_line = re.search(r'(?:%?\s*IVA)[:\s]*(\d+[.,]\d{2})', line)
                if iva_same_line:
                    try:
                        val = float(iva_same_line.group(1).replace(',', '.'))
                        # Si es un porcentaje (4-21), es la tasa
                        if 4 <= val <= 21:
                            if not fields.get('tax_rate'):
                                fields['tax_rate'] = val
                        elif not fields.get('tax_amount'):
                            fields['tax_amount'] = val
                    except:
                        pass

                # Buscar en líneas siguientes
                for j in range(i+1, min(i+4, len(text_lines))):
                    next_line = text_lines[j].strip()
                    # Si es solo un número, podría ser la tasa de IVA
                    num_match = re.match(r'^(\d+)[.,](\d{2})$', next_line)
                    if num_match:
                        try:
                            val = float(f"{num_match.group(1)}.{num_match.group(2)}")
                            if 4 <= val <= 21 and not fields.get('tax_rate'):
                                fields['tax_rate'] = val
                            break
                        except:
                            pass

            # Buscar "CUOTA" seguido de número (esto es el tax_amount)
            if 'CUOTA' in line:
                cuota_same_line = re.search(r'CUOTA[:\s]*(\d+[.,]\d{2})', line)
                if cuota_same_line:
                    try:
                        fields['tax_amount'] = float(cuota_same_line.group(1).replace(',', '.'))
                    except:
                        pass
                else:
                    # Buscar en siguiente línea
                    for j in range(i+1, min(i+3, len(text_lines))):
                        next_line = text_lines[j].strip()
                        num_match = re.match(r'^(\d+)[.,](\d{2})$', next_line)
                        if num_match:
                            try:
                                fields['tax_amount'] = float(f"{num_match.group(1)}.{num_match.group(2)}")
                                break
                            except:
                                pass

    # Estrategia 3: Verificación cruzada usando base imponible
    # Si tenemos base y total, podemos deducir el IVA
    if fields.get('tax_base') and fields.get('total'):
        calculated_tax = round(fields['total'] - fields['tax_base'], 2)

        # Si no tenemos tax_amount o difiere mucho, usar el calculado
        if not fields.get('tax_amount'):
            fields['tax_amount'] = calculated_tax

        # Si no tenemos tax_rate, deducirlo
        if not fields.get('tax_rate') and fields['tax_base'] > 0:
            calculated_rate = round(calculated_tax / fields['tax_base'] * 100, 0)
            if calculated_rate in [4, 10, 21]:  # Tasas estándar de IVA en España
                fields['tax_rate'] = calculated_rate

    # Estrategia 4: Si tenemos base y tasa pero no monto, calcularlo
    if fields.get('tax_base') and fields.get('tax_rate') and not fields.get('tax_amount'):
        fields['tax_amount'] = round(fields['tax_base'] * fields['tax_rate'] / 100, 2)

    # Estrategia 5: Validación final - verificar coherencia de valores
    if fields.get('tax_base') and fields.get('tax_rate') and fields.get('tax_amount'):
        expected_tax = round(fields['tax_base'] * fields['tax_rate'] / 100, 2)
        # Si el tax_amount está muy lejos del esperado, recalcular
        if abs(fields['tax_amount'] - expected_tax) > 1:
            logger.warning(f"[KIE] tax_amount {fields['tax_amount']} difiere del esperado {expected_tax}")
            # Mantener el extraído pero loguear la discrepancia

    # Vendor (primera línea no vacía que parezca nombre de empresa)
    for line in lines[:5]:
        line = line.strip()
        if len(line) > 5 and not any(x in line.upper() for x in ['FACTURA', 'NIF', 'CIF', 'FECHA', 'TOTAL']):
            if re.search(r'S\.?L\.?U?\.?|S\.?A\.?|SOCIEDAD|EMPRESA', line.upper()):
                fields['vendor'] = line
                break
            elif not fields['vendor'] and len(line) > 10:
                fields['vendor'] = line

    # =========================================================================
    # CUSTOMER: Extraer nombre y NIF del cliente
    # =========================================================================
    # Buscar sección de cliente (después de "Cliente", "Datos del cliente", etc.)
    customer_section_start = -1
    for i, line in enumerate(lines):
        line_upper = line.upper().strip()
        if any(x in line_upper for x in ['CLIENTE', 'DATOS CLIENTE', 'NOMBRE:', 'REF. CLIENTE']):
            customer_section_start = i
            break

    if customer_section_start >= 0:
        # Buscar nombre del cliente (línea después de "Nombre:" o primera línea con contenido)
        for i in range(customer_section_start, min(customer_section_start + 10, len(lines))):
            line = lines[i].strip()
            line_upper = line.upper()

            # Buscar "Nombre: Juan Jose..."
            name_match = re.search(r'NOMBRE[:\s]*(.+)', line_upper)
            if name_match:
                fields['customer_name'] = name_match.group(1).strip().title()
                continue

            # Buscar NIF del cliente (diferente al del vendor)
            if 'N.I.F' in line_upper or 'NIF' in line_upper:
                nif_match = re.search(r'N\.?I\.?F\.?[:\s]*(\d{8}[A-Z])', line_upper)
                if nif_match:
                    customer_nif = nif_match.group(1)
                    # Verificar que no sea el mismo que el vendor
                    if customer_nif != fields.get('vendor_nif'):
                        fields['customer_nif'] = customer_nif

    # Buscar segundo NIF en el documento (probablemente el cliente)
    if not fields.get('customer_nif'):
        all_nifs = re.findall(r'N\.?I\.?F\.?[:\s]*(\d{8}[A-Z]|\d{8}[A-Z]|[A-Z]\d{8})', text_upper)
        for nif in all_nifs:
            nif_clean = nif.replace('I', '1').replace('O', '0')
            if nif_clean != fields.get('vendor_nif') and not fields.get('customer_nif'):
                fields['customer_nif'] = nif_clean

    # =========================================================================
    # LINE_ITEMS v4.2: Extraer conceptos/productos de la factura
    # Soporta múltiples formatos:
    # - Olivenet: Concepto + Importe
    # - Vodafone/DMI: Código + Descripción + Cantidad + Precio + Importe
    # - Tickets: Producto + Importe
    # =========================================================================
    line_items = []

    # Keywords para ignorar (líneas que no son productos)
    skip_keywords = [
        'TOTAL', 'BASE IMPONIBLE', 'IVA', 'CUOTA', 'SUBTOTAL', 'NIF', 'CIF',
        'FACTURA', 'FECHA', 'CLIENTE', 'DOMICILIO', 'POBLACIÓN', 'PROVINCIA',
        'FORMA DE PAGO', 'CUENTA', 'CONFORME', 'ESCANEADO', 'DERECHOS',
        'TÉRMINOS', 'CONDICIONES', 'REGISTRO', 'INSCRIPCIÓN', 'PORTES',
        'REPERCUTIDO', 'SUPLIDOS', 'COBRO TARJETA', 'GARANTÍA', 'PART NUMBER',
        'CODIGO EAN', 'IMPORTE DTO', 'IMPORTE RE', 'IMPORTE IVA',
        'DETALLE POR PRODUCTOS', 'CONCEPTO', 'DESCRIPCIÓN', 'CANTIDAD',
        'PRECIO VENTA', 'VENCIMIENTO', 'COMERCIAL', 'PEDIDO', 'PROVEEDOR'
    ]

    # Detectar zona de productos buscando cabeceras típicas
    in_products_zone = False
    products_zone_start = -1

    for i, line in enumerate(lines):
        line_upper = line.upper().strip()
        # Detectar inicio de zona de productos
        if any(header in line_upper for header in ['DETALLE POR PRODUCTOS', 'DESCRIPCIÓN', 'CONCEPTO']):
            if 'CANTIDAD' in line_upper or 'IMPORTE' in line_upper or i < len(lines) - 1:
                in_products_zone = True
                products_zone_start = i
                continue

    # Si encontramos zona de productos, procesar desde ahí
    start_line = products_zone_start + 1 if products_zone_start >= 0 else 0

    for i in range(start_line, len(lines)):
        line = lines[i]
        line_stripped = line.strip()
        if not line_stripped:
            continue

        line_upper = line_stripped.upper()

        # Salir de zona de productos si encontramos totales
        if any(kw in line_upper for kw in ['SUBTOTAL', 'BASE IMPONIBLE', 'TOTAL FACTURA']):
            break

        # Ignorar líneas de metadatos
        if any(kw in line_upper for kw in skip_keywords):
            continue

        # Ignorar líneas que son solo información adicional
        if line_stripped.startswith('(') or line_stripped.startswith('Total de días'):
            continue

        # =====================================================================
        # PATRÓN 1: Vodafone/DMI - Código + Descripción + Cantidad + Precio + Importe
        # Ejemplo: "IM12137317  IMPRESORA BROTHER...  1  137,00  137,00"
        # =====================================================================
        vodafone_match = re.search(
            r'^([A-Z]{2}\d+|\d+)\s+(.+?)\s+(\d+)\s+(\d+[.,]\d{2})\s+(\d+[.,]\d{2})\s*€?$',
            line_stripped
        )
        if vodafone_match:
            code = vodafone_match.group(1).strip()
            description = vodafone_match.group(2).strip()
            quantity = int(vodafone_match.group(3))
            unit_price = float(vodafone_match.group(4).replace(',', '.'))
            total_price = float(vodafone_match.group(5).replace(',', '.'))

            if len(description) > 3:
                line_items.append({
                    'code': code,
                    'description': description,
                    'quantity': quantity,
                    'unit_price': unit_price,
                    'amount': total_price
                })
                continue

        # =====================================================================
        # PATRÓN 2: Cantidad + Descripción + Precio + Importe (sin código)
        # Ejemplo: "1  IMPRESORA BROTHER  137,00  137,00"
        # =====================================================================
        qty_first_match = re.search(
            r'^(\d+)\s+(.+?)\s+(\d+[.,]\d{2})\s+(\d+[.,]\d{2})\s*€?$',
            line_stripped
        )
        if qty_first_match:
            quantity = int(qty_first_match.group(1))
            description = qty_first_match.group(2).strip()
            unit_price = float(qty_first_match.group(3).replace(',', '.'))
            total_price = float(qty_first_match.group(4).replace(',', '.'))

            if len(description) > 3 and quantity <= 1000:  # Evitar confundir códigos con cantidades
                line_items.append({
                    'description': description,
                    'quantity': quantity,
                    'unit_price': unit_price,
                    'amount': total_price
                })
                continue

        # =====================================================================
        # PATRÓN 3: Olivenet - Concepto + Importe (sin cantidad)
        # Ejemplo: "Static IP (01/10/2025 - 31/10/2025)    15.0000 €"
        # =====================================================================
        olivenet_match = re.search(
            r'^(.+?)\s{2,}(-?\d+[.,]\d{2,4})\s*€?$',
            line_stripped
        )
        if olivenet_match:
            description = olivenet_match.group(1).strip()
            amount_str = olivenet_match.group(2).replace(',', '.')
            amount = float(amount_str)

            # Filtrar descripciones inválidas
            if len(description) > 5 and not description.replace('.', '').replace(',', '').replace('-', '').isdigit():
                # No añadir si es el total o la base
                if abs(amount) != fields.get('total') and abs(amount) != fields.get('tax_base'):
                    # Detectar si es un descuento
                    is_discount = amount < 0 or 'DESCUENTO' in description.upper()
                    line_items.append({
                        'description': description,
                        'amount': amount,
                        'is_discount': is_discount
                    })
                    continue

        # =====================================================================
        # PATRÓN 4: Simple - Texto + Importe al final
        # Ejemplo: "GASÓLEO A    64.83"
        # =====================================================================
        simple_match = re.search(
            r'^(.+?)\s+(\d+[.,]\d{2})\s*€?$',
            line_stripped
        )
        if simple_match:
            description = simple_match.group(1).strip()
            amount_str = simple_match.group(2).replace(',', '.')
            amount = float(amount_str)

            # Validaciones
            if len(description) > 3:
                if not description.replace('.', '').replace(',', '').replace(' ', '').isdigit():
                    if amount != fields.get('total') and amount != fields.get('tax_base'):
                        if amount != fields.get('tax_amount'):
                            line_items.append({
                                'description': description,
                                'amount': amount
                            })

    # Eliminar duplicados manteniendo orden
    seen = set()
    unique_items = []
    for item in line_items:
        # Crear key única basada en descripción y amount
        key = (item.get('description', ''), item.get('amount', 0))
        if key not in seen:
            seen.add(key)
            unique_items.append(item)

    # Validar coherencia: si hay cantidad y unit_price, verificar que coincida con amount
    for item in unique_items:
        if 'quantity' in item and 'unit_price' in item and 'amount' in item:
            expected = round(item['quantity'] * item['unit_price'], 2)
            actual = round(item['amount'], 2)
            item['amount_validated'] = abs(expected - actual) < 0.02

    fields['line_items'] = unique_items[:30]  # Limitar a 30 items máximo

    return fields


# =========================================================================
# v4.3 - FUNCIONES DE NORMALIZACIÓN Y VALIDACIÓN
# =========================================================================

def normalize_date(date_str):
    """
    Normaliza cualquier formato de fecha español a DD/MM/YYYY.

    Formatos soportados:
    - DD-MM-YY / DD/MM/YY → DD/MM/YYYY (asume 20XX para años < 50, 19XX para >= 50)
    - DD-MM-YYYY / DD/MM/YYYY → DD/MM/YYYY
    - DD MMM YYYY (ej: "05 NOV 2025") → DD/MM/YYYY

    Returns:
        dict: {'original': str, 'normalized': str, 'valid': bool}
    """
    import re

    if not date_str:
        return {'original': None, 'normalized': None, 'valid': False}

    date_str = date_str.strip().upper()
    result = {'original': date_str, 'normalized': None, 'valid': False}

    # Meses en español
    months_es = {
        'ENE': '01', 'ENERO': '01',
        'FEB': '02', 'FEBRERO': '02',
        'MAR': '03', 'MARZO': '03',
        'ABR': '04', 'ABRIL': '04',
        'MAY': '05', 'MAYO': '05',
        'JUN': '06', 'JUNIO': '06',
        'JUL': '07', 'JULIO': '07',
        'AGO': '08', 'AGOSTO': '08',
        'SEP': '09', 'SEPT': '09', 'SEPTIEMBRE': '09',
        'OCT': '10', 'OCTUBRE': '10',
        'NOV': '11', 'NOVIEMBRE': '11',
        'DIC': '12', 'DICIEMBRE': '12'
    }

    try:
        # Formato 1: DD-MM-YY o DD/MM/YY (año corto)
        match = re.match(r'^(\d{1,2})[-/](\d{1,2})[-/](\d{2})$', date_str)
        if match:
            day, month, year = match.groups()
            year_int = int(year)
            # Asumir 2000-2049 para 00-49, 1950-1999 para 50-99
            full_year = 2000 + year_int if year_int < 50 else 1900 + year_int
            result['normalized'] = f"{day.zfill(2)}/{month.zfill(2)}/{full_year}"
            result['valid'] = True
            return result

        # Formato 2: DD-MM-YYYY o DD/MM/YYYY (año completo)
        match = re.match(r'^(\d{1,2})[-/](\d{1,2})[-/](\d{4})$', date_str)
        if match:
            day, month, year = match.groups()
            result['normalized'] = f"{day.zfill(2)}/{month.zfill(2)}/{year}"
            result['valid'] = True
            return result

        # Formato 3: DD MMM YYYY (ej: "05 NOV 2025")
        match = re.match(r'^(\d{1,2})\s+([A-Z]+)\s+(\d{4})$', date_str)
        if match:
            day, month_name, year = match.groups()
            # Buscar el mes en el diccionario
            for key, value in months_es.items():
                if month_name.startswith(key):
                    result['normalized'] = f"{day.zfill(2)}/{value}/{year}"
                    result['valid'] = True
                    return result

        # Formato 4: DD de MMM de YYYY (ej: "05 de noviembre de 2025")
        match = re.match(r'^(\d{1,2})\s+(?:DE\s+)?([A-Z]+)\s+(?:DE\s+)?(\d{4})$', date_str)
        if match:
            day, month_name, year = match.groups()
            for key, value in months_es.items():
                if month_name.startswith(key):
                    result['normalized'] = f"{day.zfill(2)}/{value}/{year}"
                    result['valid'] = True
                    return result

    except Exception as e:
        pass

    return result


def validate_spanish_nif(nif):
    """
    Valida un NIF/CIF español verificando el formato y dígito de control.

    Tipos soportados:
    - NIF persona física: 8 dígitos + letra (ej: 12345678Z)
    - NIE: X/Y/Z + 7 dígitos + letra (ej: X1234567L)
    - CIF empresa: letra + 8 dígitos (ej: B12345678) - la letra final puede ser número o letra

    Returns:
        dict: {'original': str, 'formatted': str, 'valid': bool, 'type': str, 'control_check': bool}
    """
    import re

    if not nif:
        return {'original': None, 'formatted': None, 'valid': False, 'type': None, 'control_check': False}

    nif = nif.strip().upper()
    result = {'original': nif, 'formatted': nif, 'valid': False, 'type': None, 'control_check': False}

    # Limpiar caracteres no alfanuméricos
    nif_clean = re.sub(r'[^A-Z0-9]', '', nif)

    # Tabla de letras para NIF persona física
    nif_letters = 'TRWAGMYFPDXBNJZSQVHLCKE'

    # NIF persona física: 8 dígitos + letra
    if re.match(r'^\d{8}[A-Z]$', nif_clean):
        result['type'] = 'NIF'
        result['formatted'] = nif_clean
        digits = int(nif_clean[:8])
        expected_letter = nif_letters[digits % 23]
        actual_letter = nif_clean[8]
        result['control_check'] = (expected_letter == actual_letter)
        result['valid'] = result['control_check']
        return result

    # NIE (extranjeros): X/Y/Z + 7 dígitos + letra
    if re.match(r'^[XYZ]\d{7}[A-Z]$', nif_clean):
        result['type'] = 'NIE'
        result['formatted'] = nif_clean
        # Convertir primera letra a número: X=0, Y=1, Z=2
        nie_map = {'X': '0', 'Y': '1', 'Z': '2'}
        nie_number = int(nie_map[nif_clean[0]] + nif_clean[1:8])
        expected_letter = nif_letters[nie_number % 23]
        actual_letter = nif_clean[8]
        result['control_check'] = (expected_letter == actual_letter)
        result['valid'] = result['control_check']
        return result

    # CIF empresa: letra + 7 dígitos + control (letra o número)
    if re.match(r'^[ABCDEFGHJKLMNPQRSUVW]\d{7}[A-J0-9]$', nif_clean):
        result['type'] = 'CIF'
        result['formatted'] = nif_clean
        result['valid'] = True  # Formato válido

        # Cálculo del dígito de control del CIF (algoritmo estándar)
        try:
            tipo = nif_clean[0]
            digits = nif_clean[1:8]
            control = nif_clean[8]

            # Suma de pares
            suma_pares = sum(int(digits[i]) for i in [1, 3, 5])

            # Suma de impares (multiplicar por 2 y sumar dígitos)
            suma_impares = 0
            for i in [0, 2, 4, 6]:
                doble = int(digits[i]) * 2
                suma_impares += doble if doble < 10 else doble - 9

            total = suma_pares + suma_impares
            control_digit = (10 - (total % 10)) % 10
            control_letter = 'JABCDEFGHI'[control_digit]

            # Según el tipo de entidad, el control puede ser letra o número
            # Tipos que usan letra: K, P, Q, S (y N, W a veces)
            # Tipos que usan número: A, B (resto)
            if control.isdigit():
                result['control_check'] = (int(control) == control_digit)
            else:
                result['control_check'] = (control == control_letter)

        except Exception:
            pass

        return result

    # Si tiene formato de CIF pero el control no es correcto, aún lo marcamos como CIF
    if re.match(r'^[A-Z]\d{8}$', nif_clean):
        result['type'] = 'CIF'
        result['formatted'] = nif_clean
        result['valid'] = True  # Formato válido pero sin verificar control
        return result

    return result


def normalize_amount(amount, default_currency='EUR'):
    """
    Normaliza un importe a formato estándar float con 2 decimales y detecta moneda.

    Formatos soportados:
    - "94,74 €" → 94.74 EUR
    - "94.74 EUR" → 94.74 EUR
    - "$150.00" → 150.00 USD
    - "£75.50" → 75.50 GBP
    - "1.234,56" → 1234.56 (formato español con miles)
    - "-12.3967 €" → -12.40 EUR (redondeo a 2 decimales)

    Returns:
        dict: {'original': any, 'normalized': float, 'formatted': str, 'currency': str, 'valid': bool}
    """
    import re

    if amount is None:
        return {'original': None, 'normalized': None, 'formatted': None, 'currency': None, 'valid': False}

    # Si ya es float, normalizar con moneda por defecto
    if isinstance(amount, (int, float)):
        normalized = round(float(amount), 2)
        return {
            'original': amount,
            'normalized': normalized,
            'formatted': f"{normalized:.2f} {default_currency}",
            'currency': default_currency,
            'valid': True
        }

    result = {'original': amount, 'normalized': None, 'formatted': None, 'currency': None, 'valid': False}

    try:
        amount_str = str(amount).strip()

        # Detectar moneda antes de limpiar
        currency = default_currency
        if '€' in amount_str or 'EUR' in amount_str.upper():
            currency = 'EUR'
        elif '$' in amount_str or 'USD' in amount_str.upper():
            currency = 'USD'
        elif '£' in amount_str or 'GBP' in amount_str.upper():
            currency = 'GBP'
        elif 'CHF' in amount_str.upper():
            currency = 'CHF'
        elif 'MXN' in amount_str.upper():
            currency = 'MXN'

        # Eliminar símbolos de moneda y espacios
        amount_clean = re.sub(r'[€$£\s]', '', amount_str)
        amount_clean = re.sub(r'\b(EUR|USD|GBP|CHF|MXN)\b', '', amount_clean, flags=re.IGNORECASE).strip()

        # Detectar si hay separador de miles (formato español: 1.234,56)
        if re.match(r'^-?\d{1,3}(\.\d{3})+,\d+$', amount_clean):
            # Formato español con miles: quitar puntos de miles, cambiar coma por punto
            amount_clean = amount_clean.replace('.', '').replace(',', '.')
        else:
            # Formato simple: solo cambiar coma por punto si es decimal
            amount_clean = amount_clean.replace(',', '.')

        normalized = round(float(amount_clean), 2)
        result['normalized'] = normalized
        result['formatted'] = f"{normalized:.2f} {currency}"
        result['currency'] = currency
        result['valid'] = True

    except Exception:
        pass

    return result


def detect_currency(text):
    """
    Detecta la moneda predominante en el texto del documento.

    Returns:
        str: Código de moneda (EUR, USD, GBP, CHF, MXN)
    """
    import re

    if not text:
        return 'EUR'

    text_upper = text.upper()

    # Contar ocurrencias de cada símbolo/código de moneda
    currency_counts = {
        'EUR': len(re.findall(r'€|EUR\b', text_upper)),
        'USD': len(re.findall(r'\$|USD\b', text_upper)),
        'GBP': len(re.findall(r'£|GBP\b', text_upper)),
        'CHF': len(re.findall(r'CHF\b', text_upper)),
        'MXN': len(re.findall(r'MXN\b', text_upper)),
    }

    # Devolver la moneda con más ocurrencias, o EUR por defecto
    max_currency = max(currency_counts, key=currency_counts.get)
    if currency_counts[max_currency] > 0:
        return max_currency

    return 'EUR'


def normalize_fields(fields, raw_text=None):
    """
    Aplica normalización v4.3 a todos los campos extraídos.

    Normaliza:
    - date → formato DD/MM/YYYY
    - vendor_nif, customer_nif → validación y formato
    - total, tax_base, tax_amount → formato numérico estándar con moneda
    - line_items.amount → formato numérico estándar con moneda

    Returns:
        dict: campos originales más '_normalized' con datos normalizados
    """
    if not fields:
        return fields

    # Detectar moneda del documento
    document_currency = detect_currency(raw_text) if raw_text else 'EUR'

    # Crear copia para no modificar original
    normalized = dict(fields)

    # Añadir sección de normalización
    normalized['_normalized'] = {
        'version': '4.3',
        'date': None,
        'vendor_nif': None,
        'customer_nif': None,
        'currency': document_currency,
        'amounts': {}
    }

    # Normalizar fecha
    if fields.get('date'):
        date_norm = normalize_date(fields['date'])
        normalized['_normalized']['date'] = date_norm
        if date_norm['valid']:
            normalized['date_normalized'] = date_norm['normalized']

    # Validar NIFs
    if fields.get('vendor_nif'):
        nif_norm = validate_spanish_nif(fields['vendor_nif'])
        normalized['_normalized']['vendor_nif'] = nif_norm
        if nif_norm['valid']:
            normalized['vendor_nif_valid'] = nif_norm['control_check']
            normalized['vendor_nif_type'] = nif_norm['type']

    if fields.get('customer_nif'):
        nif_norm = validate_spanish_nif(fields['customer_nif'])
        normalized['_normalized']['customer_nif'] = nif_norm
        if nif_norm['valid']:
            normalized['customer_nif_valid'] = nif_norm['control_check']
            normalized['customer_nif_type'] = nif_norm['type']

    # Normalizar importes principales con la moneda del documento
    amount_fields = ['total', 'tax_base', 'tax_amount']
    for field_name in amount_fields:
        if fields.get(field_name) is not None:
            amount_norm = normalize_amount(fields[field_name], default_currency=document_currency)
            normalized['_normalized']['amounts'][field_name] = amount_norm
            # Añadir campo formateado con moneda al nivel superior
            if amount_norm['valid']:
                normalized[f'{field_name}_formatted'] = amount_norm['formatted']

    # Añadir moneda detectada al nivel superior
    normalized['currency'] = document_currency

    # Normalizar importes en line_items
    if fields.get('line_items'):
        for item in normalized.get('line_items', []):
            if 'amount' in item:
                amount_norm = normalize_amount(item['amount'], default_currency=document_currency)
                if amount_norm['valid']:
                    item['amount_normalized'] = amount_norm['normalized']
                    item['amount_formatted'] = amount_norm['formatted']
            if 'unit_price' in item:
                price_norm = normalize_amount(item['unit_price'], default_currency=document_currency)
                if price_norm['valid']:
                    item['unit_price_normalized'] = price_norm['normalized']
                    item['unit_price_formatted'] = price_norm['formatted']

    return normalized


def detect_document_type(text, fields):
    """Detecta el tipo de documento basándose en el contenido"""
    text_upper = text.upper()

    if 'FACTURA' in text_upper or fields.get('invoice_number'):
        return 'invoice'
    elif 'TICKET' in text_upper or 'RECIBO' in text_upper:
        return 'receipt'
    elif 'ALBARÁN' in text_upper or 'ALBARAN' in text_upper:
        return 'delivery_note'
    elif fields.get('total') and fields.get('vendor_nif'):
        return 'invoice'
    else:
        return 'unknown'


def calculate_extraction_confidence(fields):
    """Calcula un score de confianza basado en campos extraídos"""
    required_fields = ['vendor', 'total', 'date']
    optional_fields = ['vendor_nif', 'invoice_number', 'tax_base', 'tax_rate']

    score = 0
    max_score = len(required_fields) * 2 + len(optional_fields)

    for field in required_fields:
        if fields.get(field):
            score += 2

    for field in optional_fields:
        if fields.get(field):
            score += 1

    confidence = round(score / max_score, 2)

    if confidence >= 0.8:
        return {'score': confidence, 'level': 'high'}
    elif confidence >= 0.5:
        return {'score': confidence, 'level': 'medium'}
    else:
        return {'score': confidence, 'level': 'low'}


# ============================================================================
# FIN DE CAPA API REST AÑADIDA
# ============================================================================

def start_model_loading():
    """Inicia la carga de modelos en segundo plano"""
    logger.info("[STARTUP] Iniciando hilo de carga de modelos...")
    model_thread = threading.Thread(target=load_models_background, daemon=True)
    model_thread.start()
    logger.info("[STARTUP] Hilo de carga de modelos iniciado")

if __name__ == '__main__':
    port = int(os.getenv('FLASK_PORT', '8503'))
    logger.info("")
    logger.info("=" * 60)
    logger.info("[START] PADDLEOCR V3 FUSION - WEBCOMUNICA")
    logger.info("=" * 60)
    logger.info(f"[START] Puerto: {port}")
    logger.info(f"[START] Memoria: {get_memory_usage()}")
    logger.info(f"[START] PID: {os.getpid()}")
    logger.info("[START] Proyecto base: paddlepaddle_paco")
    logger.info("[START] Capa API: webcomunica REST layer")

    # Cargar modelos en segundo plano al inicio
    # El servidor arranca inmediatamente mientras los modelos cargan en background
    start_model_loading()

    # Detectar si estamos en produccion
    if os.getenv('FLASK_ENV') == 'production':
        from waitress import serve
        logger.info("")
        logger.info("[READY] *** SERVIDOR WAITRESS INICIANDO ***")
        logger.info(f"[READY] URL: http://0.0.0.0:{port}")
        logger.info("[READY] Health check: /health")
        logger.info("[READY] Dashboard: /")
        logger.info("[READY] Modelos: cargando en segundo plano...")
        logger.info("")
        # Waitress bloqueará aquí - el servidor estará corriendo
        serve(app, host='0.0.0.0', port=port, threads=4)
    else:
        logger.info("[READY] Iniciando servidor Flask (desarrollo)")
        app.run(host='0.0.0.0', port=port, debug=False)

    # Este código solo se ejecuta si el servidor termina
    logger.info("[SHUTDOWN] Servidor terminado")

