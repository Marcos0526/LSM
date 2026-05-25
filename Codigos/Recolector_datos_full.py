"""
Recolector_datos_full.py
========================
Modos:
  Movimiento → Bucle automático: Cuenta regresiva (5s) -> Graba (4s a 30fps) -> Guarda (.npy) -> Repite.
               Shape final: (120, 79, 3).
  Estático   → Guarda 1 fila plana (237 columnas).

Controles:
  [L]   → Ingresar / cambiar etiqueta
  [M]   → Cambiar modo (Estático / Movimiento)
  [R]   → Iniciar / Detener ciclo de grabación
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

# ── Parámetros de Grabación
RECORD_FPS             = 10
RECORD_DELAY           = 1.0 / RECORD_FPS
MAX_STATIC_SAMPLES     = 100

# Parámetros para Movimiento (GCN)
MOTION_DURATION_SEC    = 4.0   
MOTION_FPS             = 30    
MOTION_FRAMES_TARGET   = int(MOTION_DURATION_SEC * MOTION_FPS) # 120 frames
MOTION_SAMPLE_INTERVAL = 1.0 / MOTION_FPS  
COUNTDOWN_SEC          = 5.0   # Descanso entre tomas

# ── Índices de Landmarks (FaceMesh)
LEFT_EYEBROW_IDS  = [70, 63, 105, 66]
RIGHT_EYEBROW_IDS = [300, 293, 334, 296]
MOUTH_IDS         = [13, 14, 61, 291] # Apertura + Comisuras

# ── Extracción y Normalización
def extract_keypoints(results):
    """
    Extrae la pose superior (25), manos (21 c/u), cejas (8) y boca (4).
    Retorna matriz (79, 3).
    """
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark[:25]]) \
           if results.pose_landmarks else np.zeros((25, 3))
           
    lh   = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) \
           if results.left_hand_landmarks else np.zeros((21, 3))
           
    rh   = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) \
           if results.right_hand_landmarks else np.zeros((21, 3))

    if results.face_landmarks:
        left_brow  = np.array([[results.face_landmarks.landmark[i].x,
                                results.face_landmarks.landmark[i].y,
                                results.face_landmarks.landmark[i].z] for i in LEFT_EYEBROW_IDS])
        
        right_brow = np.array([[results.face_landmarks.landmark[i].x,
                                results.face_landmarks.landmark[i].y,
                                results.face_landmarks.landmark[i].z] for i in RIGHT_EYEBROW_IDS])
        
        mouth      = np.array([[results.face_landmarks.landmark[i].x,
                                results.face_landmarks.landmark[i].y,
                                results.face_landmarks.landmark[i].z] for i in MOUTH_IDS])
    else:
        left_brow  = np.zeros((4, 3))
        right_brow = np.zeros((4, 3))
        mouth      = np.zeros((4, 3))

    return np.concatenate([pose, lh, rh, left_brow, right_brow, mouth])

def normalize_landmarks(keypoints):
    lm = np.copy(keypoints)
    if np.any(lm[11]) and np.any(lm[12]):
        origin = (lm[11] + lm[12]) / 2.0
    elif np.any(lm[0]): 
        origin = lm[0] 
    else:
        origin = np.zeros(3)

    lm = lm - origin
    mx = np.max(np.abs(lm))
    if mx != 0:
        lm = lm / mx
    return lm

# ── Manejo de CSVs
STATIC_HEADER = ["label"] + [f"node_{i}_{ax}" for i in range(79) for ax in ("x", "y", "z")]

def init_csvs():
    if not os.path.exists(STATIC_CSV):
        with open(STATIC_CSV, "w", newline="") as f:
            csv.writer(f).writerow(STATIC_HEADER)
    
    if not os.path.exists(MOTION_CSV):
        with open(MOTION_CSV, "w", newline="") as f:
            csv.writer(f).writerow(["label", "ruta_npy", "num_frames"])

def save_static(label, vector_mat):
    with open(STATIC_CSV, "a", newline="") as f:
        csv.writer(f).writerow([label] + vector_mat.flatten().tolist())

def save_motion(label, frames_list):
    if len(frames_list) == 0: return
    ts       = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time()*1000)%1000:03d}"
    filename = f"{label}_{ts}.npy"
    npy_path = os.path.join(SEQ_DIR, filename)

    seq = np.array(frames_list, dtype=np.float32)
    np.save(npy_path, seq)

    with open(MOTION_CSV, "a", newline="") as f:
        csv.writer(f).writerow([label, npy_path, len(frames_list)])
    print(f"[SAVE] '{label}' guardada: {len(frames_list)} frames.")

# ── Interfaz de Etiqueta
def capture_label(current_label):
    typed = list(current_label)
    win   = "Etiqueta"
    while True:
        canvas = np.zeros((90, 380, 3), dtype=np.uint8)
        cv2.putText(canvas, "Etiqueta  [ENTER=OK  ESC=Cancelar]", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(canvas, "> " + "".join(typed) + "_", (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 220, 100), 2)
        cv2.imshow(win, canvas)
        key = cv2.waitKey(0) & 0xFF
        if key == 13: 
            cv2.destroyWindow(win)
            res = "".join(typed).strip()
            return res if res else current_label
        elif key == 27: 
            cv2.destroyWindow(win)
            return current_label
        elif key == 8 and typed: typed.pop()
        elif 32 <= key <= 126: typed.append(chr(key))

# ── Bucle Principal
def main():
    init_csvs()
    mp_holistic = mp.solutions.holistic
    mp_drawing  = mp.solutions.drawing_utils
    holistic    = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    cap         = cv2.VideoCapture(0)

    current_label        = ""
    mode                 = "movimiento"    
    recording            = False
    is_counting_down     = False
    countdown_start      = 0.0
    
    session_count        = 0
    motion_buffer        = []            
    static_session_count = 0
    
    motion_start_time    = 0.0
    last_motion_sample   = 0.0
    last_save_time       = 0.0

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1) # Efecto espejo opcional, útil en señas
        h, w  = frame.shape[:2]
        font  = cv2.FONT_HERSHEY_SIMPLEX

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = holistic.process(frame_rgb)
        
        # ── Dibujo de landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        if results.face_landmarks:
            for idx in LEFT_EYEBROW_IDS + RIGHT_EYEBROW_IDS + MOUTH_IDS:
                lm = results.face_landmarks.landmark[idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1) 

        # ── Lógica de Extracción y Grabación
        if results.pose_landmarks or results.left_hand_landmarks or results.right_hand_landmarks:
            norm_mat = normalize_landmarks(extract_keypoints(results))
            now      = time.time()

            # LÓGICA DE CUENTA REGRESIVA
            if is_counting_down:
                elapsed_cd = now - countdown_start
                remaining  = COUNTDOWN_SEC - elapsed_cd
                
                if remaining > 0:
                    segundos_int = int(np.ceil(remaining))
                    texto_cuenta = f"Comenzando en {segundos_int}s..."
                    # Centrado en pantalla
                    text_size = cv2.getTextSize(texto_cuenta, font, 1.5, 4)[0]
                    cv2.putText(frame, texto_cuenta, ((w - text_size[0]) // 2, h // 2), font, 1.5, (0, 165, 255), 4)
                else:
                    is_counting_down   = False
                    recording          = True
                    motion_buffer      = []
                    motion_start_time  = time.time()
                    last_motion_sample = time.time()
                    print(f"[INFO] Grabando muestra {session_count + 1}...")

            # LÓGICA DE GRABACIÓN
            elif recording and current_label:
                if mode == "estatico":
                    if (now - last_save_time >= RECORD_DELAY):
                        save_static(current_label, norm_mat)
                        session_count        += 1
                        static_session_count += 1   
                        last_save_time        = now
                        if static_session_count >= MAX_STATIC_SAMPLES:
                            recording            = False
                            static_session_count = 0    

                else: # Modo Movimiento
                    # Añadir frame respetando el intervalo para que no se grabe demasiado rápido
                    if (now - last_motion_sample) >= MOTION_SAMPLE_INTERVAL:
                        motion_buffer.append(norm_mat)
                        last_motion_sample = now

                    # EL CAMBIO CRUCIAL: Detener SOLO cuando tengamos la cantidad exacta de frames
                    if len(motion_buffer) >= MOTION_FRAMES_TARGET:
                        
                        # Cortafuegos extra: Aseguramos que sea exactamente la cantidad pedida
                        # por si un hilo de procesamiento metió un frame de más
                        buffer_exacto = motion_buffer[:MOTION_FRAMES_TARGET]
                        
                        save_motion(current_label, buffer_exacto)
                        session_count += 1
                        recording      = False  
                        
                        # Reiniciar ciclo para la siguiente toma
                        is_counting_down = True
                        countdown_start  = time.time()

        # ── HUD (Interfaz en pantalla)
        # Modo
        mode_color = (60, 220, 100) if mode == "estatico" else (50, 160, 240)
        cv2.putText(frame, f"Modo: {mode.upper()}", (10, 28), font, 0.7, mode_color, 2)

        # Estado REC
        rec_color = (50, 50, 220) if recording else (80, 80, 80)
        cv2.circle(frame, (w - 120, 22), 8, rec_color, -1)
        cv2.putText(frame, "REC" if recording else "ESPERA", (w - 108, 28), font, 0.65, rec_color, 2)

        # HUD Movimiento: Temporizador X s / 4 s
        if mode == "movimiento" and recording:
            elapsed_disp = time.time() - motion_start_time
            texto_segundos = f"Grabando: {int(elapsed_disp)}s / {int(MOTION_DURATION_SEC)}s"
            
            # Dibujar arriba al centro
            tsize = cv2.getTextSize(texto_segundos, font, 1.0, 3)[0]
            cv2.putText(frame, texto_segundos, ((w - tsize[0]) // 2, 50), font, 1.0, (50, 50, 255), 3)
            
            # Mantener contador de frames a la derecha
            cv2.putText(frame, f"Frames: {len(motion_buffer)}/{MOTION_FRAMES_TARGET}", 
                        (w - 200, 85), font, 0.6, (50, 160, 240), 2)

        # HUD Estático
        if mode == "estatico" and recording:
            cv2.putText(frame, f"Instancias: {static_session_count}/{MAX_STATIC_SAMPLES}", (w - 230, 55), font, 0.6, (60, 220, 100), 2)  

        # Etiqueta y Guardados
        lbl_color = (60, 220, 100) if current_label else (30, 210, 230)
        cv2.putText(frame, f"Sena: {current_label or 'sin etiqueta'}", (10, h - 15), font, 0.85, lbl_color, 2)
        cv2.putText(frame, f"Guardados: {session_count}", (10, 58), font, 0.6, (200, 200, 200), 1)

        # Controles
        for i, ctrl in enumerate(["[L] Etiqueta", "[M] Modo", "[R] Iniciar/Stop", "[ESC] Salir"]):
            cv2.putText(frame, ctrl, (10, 88 + i * 22), font, 0.48, (140, 140, 140), 1)

        cv2.imshow("Recolector Full", frame)

        # ── Eventos de Teclado
        key = cv2.waitKey(1) & 0xFF
        if key == 27: break
        
        elif key in (ord('l'), ord('L')):
            recording        = False
            is_counting_down = False
            motion_buffer    = []
            current_label    = capture_label(current_label)

        elif key in (ord('m'), ord('M')):
            if not recording and not is_counting_down:
                mode = "estatico" if mode == "movimiento" else "movimiento"
                
        elif key in (ord('r'), ord('R')):
            if not current_label:
                print("[WARN] Asigna etiqueta con [L]")
            elif not recording and not is_counting_down:
                if mode == "movimiento":
                    is_counting_down = True
                    countdown_start  = time.time()
                else:
                    recording            = True
                    last_save_time       = 0.0
                    static_session_count = 0            
            else:
                recording        = False
                is_counting_down = False
                motion_buffer    = []
                print("[INFO] Ciclo cancelado.")

    cap.release()
    cv2.destroyAllWindows()
    holistic.close()
    print(f"\n[FIN] {session_count} muestras guardadas.")

if __name__ == "__main__":
    main()