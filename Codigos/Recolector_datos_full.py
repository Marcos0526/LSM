"""
collect_dataset_v2.py
=====================
Modos:
  Estático   → guarda 1 fila en hand_landmarks_static.csv (75 nodos x 3 coords = 225 cols)
  Movimiento → graba una secuencia fluida a tiempo real → la guarda como .npy con shape (30, 75, 3)
               y detiene la grabación automáticamente al llegar a los 30 frames.
               + registra 1 fila en hand_landmarks_motion.csv

Controles:
  [L]   → Ingresar / cambiar etiqueta
  [M]   → Toggle modo  (Estático / Movimiento)
  [R]   → Toggle grabación continua (en Movimiento se detiene solo al llegar a 30)
  [ESC] → Salir
"""

import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time

# ── Rutas y Directorios
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
DATASET_PATH = os.path.join(BASE_DIR, "Dataset")                            
SEQ_DIR      = os.path.join(DATASET_PATH, "sequences")                      
STATIC_CSV   = os.path.join(DATASET_PATH, "hand_landmarks_static.csv")
MOTION_CSV   = os.path.join(DATASET_PATH, "hand_landmarks_motion.csv")

os.makedirs(DATASET_PATH, exist_ok=True)  
os.makedirs(SEQ_DIR, exist_ok=True)      

# Retraso solo para el modo estático (1 captura por segundo para evitar ráfagas masivas)
RECORD_FPS    = 1
RECORD_DELAY  = 1.0 / RECORD_FPS

# ── Extracción y Normalización Espacial A Prueba de Futuro
def extract_keypoints(results):
    """
    Extrae la pose completa (33 puntos), mano izquierda (21 puntos) y mano derecha (21 puntos).
    Si MediaPipe pierde el tracking de alguna parte, rellena con ceros para no alterar la estructura.
    Retorna una matriz ordenada de forma (75, 3).
    """
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    
    return np.concatenate([pose, lh, rh])

def normalize_landmarks(keypoints):
    """
    Toma la matriz (75, 3) y traslada el origen (0,0,0) al centro de los hombros 
    (puntos 11 y 12 de la pose). Esto permite que el modelo entienda la trayectoria de las manos
    en el espacio cuando escales a palabras completas.
    """
    lm = np.copy(keypoints)
    
    # Intentar centrar en el pecho (punto medio entre hombro izquierdo [11] y derecho [12])
    if np.any(lm[11]) and np.any(lm[12]):
        origin = (lm[11] + lm[12]) / 2.0
    elif np.any(lm[0]): 
        origin = lm[0]  # Respaldo: usar la nariz si no ve los hombros
    else:
        origin = np.zeros(3)

    lm = lm - origin
    
    # Escalar para mantener los valores en un rango manejable (-1 a 1 relativo)
    mx = np.max(np.abs(lm))
    if mx != 0:
        lm = lm / mx
        
    return lm

# ── Manejo de Archivos CSV Estáticos (225 columnas de datos)
STATIC_HEADER = ["label"] + [f"node_{i}_{ax}" for i in range(75) for ax in ("x", "y", "z")]

def init_static_csv():
    if not os.path.exists(STATIC_CSV):
        with open(STATIC_CSV, "w", newline="") as f:
            csv.writer(f).writerow(STATIC_HEADER)
        print(f"[INFO] CSV estático creado: {STATIC_CSV}")

def save_static(label, vector_mat):
    # Convertimos la matriz (75, 3) en un vector plano (225,) para guardarlo en una sola fila
    flat_vector = vector_mat.flatten()
    with open(STATIC_CSV, "a", newline="") as f:
        csv.writer(f).writerow([label] + flat_vector.tolist())

# ── Manejo de Archivos CSV y NPY Dinámicos (Matriz de Movimiento)
MOTION_HEADER = ["label", "ruta_npy", "num_frames"]

def init_motion_csv():
    if not os.path.exists(MOTION_CSV):
        with open(MOTION_CSV, "w", newline="") as f:
            csv.writer(f).writerow(MOTION_HEADER)
        print(f"[INFO] CSV movimiento creado: {MOTION_CSV}")

def save_motion(label, frames_list):
    if len(frames_list) == 0:
        print("[WARN] Secuencia vacía, no se guardó.")
        return

    # Generar un nombre único basado en fecha, hora y milisegundos
    ts       = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time()*1000)%1000:03d}"
    filename = f"{label}_{ts}.npy"
    npy_path = os.path.join(SEQ_DIR, filename)

    # Convertir la lista de 30 matrices en un tensor NumPy tridimensional (30, 75, 3)
    seq = np.array(frames_list, dtype=np.float32)
    np.save(npy_path, seq)

    with open(MOTION_CSV, "a", newline="") as f:
        csv.writer(f).writerow([label, npy_path, len(frames_list)])

    print(f"[SAVE] Secuencia '{label}' guardada exitosamente con {len(frames_list)} frames.")

# ── Captura de Entrada de Texto por OpenCV
def capture_label(current_label):
    typed = list(current_label)
    win   = "Etiqueta"
    while True:
        canvas = np.zeros((90, 380, 3), dtype=np.uint8)
        cv2.putText(canvas, "Etiqueta  [ENTER=OK  ESC=Cancelar]",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(canvas, "> " + "".join(typed) + "_",
                    (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 220, 100), 2)
        cv2.imshow(win, canvas)
        key = cv2.waitKey(0) & 0xFF
        if key == 13: # ENTER
            cv2.destroyWindow(win)
            result = "".join(typed).strip()
            return result if result else current_label
        elif key == 27: # ESC
            cv2.destroyWindow(win)
            return current_label
        elif key == 8 and typed: # BACKSPACE
            typed.pop()
        elif 32 <= key <= 126:
            typed.append(chr(key))

# ── Bucle de Ejecución Principal
def main():
    init_static_csv()
    init_motion_csv()

    mp_holistic = mp.solutions.holistic
    mp_drawing  = mp.solutions.drawing_utils
    
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    cap = cv2.VideoCapture(0)

    current_label   = ""
    mode            = "estatico"    # "estatico" | "movimiento"
    recording       = False
    last_save_time  = 0.0
    session_count   = 0
    motion_buffer   = []            # Almacén de frames para la ráfaga actual

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = holistic.process(frame_rgb)
        
        # Dibujar malla de seguimiento en pantalla
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Captura y lógica de grabación
        if results.pose_landmarks or results.left_hand_landmarks or results.right_hand_landmarks:
            raw_keypoints = extract_keypoints(results)
            norm_mat = normalize_landmarks(raw_keypoints)

            now = time.time()
            if recording and current_label:
                if mode == "estatico":
                    # Modo Estático: Aplica retraso para no saturar el CSV
                    if (now - last_save_time >= RECORD_DELAY):
                        save_static(current_label, norm_mat)
                        session_count += 1
                        last_save_time = now
                else:
                    # Modo Movimiento: Graba directo a los FPS nativos de tu cámara
                    motion_buffer.append(norm_mat)
                    
                    # Cortafuegos automático al alcanzar la ventana temporal exacta
                    if len(motion_buffer) == 30:
                        save_motion(current_label, motion_buffer)
                        session_count += 1
                        motion_buffer = []
                        recording = False  # Auto-stop instantáneo

        # ── Interfaz HUD en Pantalla
        h, w = frame.shape[:2]
        font  = cv2.FONT_HERSHEY_SIMPLEX

        # Indicador de Modo actual
        mode_color = (60, 220, 100) if mode == "estatico" else (50, 160, 240)
        cv2.putText(frame, f"Modo: {mode.upper()}", (10, 28), font, 0.7, mode_color, 2)

        # Estado de la Grabación (REC / ESPERA)
        rec_color = (50, 50, 220) if recording else (80, 80, 80)
        cv2.circle(frame, (w - 120, 22), 8, rec_color, -1)
        cv2.putText(frame, "REC" if recording else "ESPERA", (w - 108, 28), font, 0.65, rec_color, 2)

        # Estado del buffer dinámico
        if mode == "movimiento" and recording:
            cv2.putText(frame, f"Frames: {len(motion_buffer)}/30", (w - 160, 55), font, 0.6, (50, 160, 240), 2)

        # Nombre de la Seña activa
        lbl_display = current_label or "sin etiqueta"
        lbl_color   = (60, 220, 100) if current_label else (30, 210, 230)
        cv2.putText(frame, f"Sena: {lbl_display}", (10, h - 15), font, 0.85, lbl_color, 2)

        # Contador de muestras guardadas en esta sesión
        cv2.putText(frame, f"Guardados: {session_count}", (10, 58), font, 0.6, (200, 200, 200), 1)

        # Guía visual de comandos rápidos
        controls = ["[L] Etiqueta", "[M] Modo", "[R] Iniciar Grab.", "[ESC] Salir"]
        for i, ctrl in enumerate(controls):
            cv2.putText(frame, ctrl, (10, 88 + i * 22), font, 0.48, (140, 140, 140), 1)

        cv2.imshow("Recoleccion de Dataset (Holistic)", frame)

        # ── Escucha del Teclado
        key = cv2.waitKey(1) & 0xFF

        if key == 27:   # Tecla ESC
            break

        elif key in (ord('l'), ord('L')):
            if recording:              
                recording = False
                motion_buffer = []
            current_label = capture_label(current_label)
            print(f"[INFO] Nueva etiqueta asignada: '{current_label}'")

        elif key in (ord('m'), ord('M')):
            if recording:
                print("[WARN] Detén la grabación activa antes de conmutar el modo.")
            else:
                mode = "movimiento" if mode == "estatico" else "estatico"
                motion_buffer = []
                print(f"[INFO] Cambio de modo: {mode.upper()}")

        elif key in (ord('r'), ord('R')):
            if not current_label:
                print("[WARN] Asigna una etiqueta primero utilizando la tecla [L]")
            elif not recording:
                recording = True
                last_save_time = 0.0
                motion_buffer = []      
                print(f"[INFO] Grabación iniciada para la seña: '{current_label}'")
            else:
                # Cancelación manual por parte del usuario si decide soltar la tecla antes
                recording = False
                motion_buffer = []
                print(f"[INFO] Grabación cancelada por el usuario.")

    cap.release()
    cv2.destroyAllWindows()
    holistic.close()
    print(f"\n[FIN] Finalizado. Se guardaron {session_count} muestras válidas en la carpeta Dataset.")

if __name__ == "__main__":
    main()