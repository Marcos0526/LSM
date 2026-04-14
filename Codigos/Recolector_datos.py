"""
collect_dataset_v2.py
=====================
Modos:
  Estático   → guarda 1 fila en hand_landmarks_static.csv  (igual que antes)
  Movimiento → graba secuencia → la guarda como .npy
               + registra 1 fila en hand_landmarks_motion.csv

Controles:
  [L]   → Ingresar / cambiar etiqueta
  [M]   → Toggle modo  (Estático / Movimiento)
  [R]   → Toggle grabación continua
  [ESC] → Salir
"""

import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time

# ── Rutas
BASE_DIR      = "/home/marcos/LSM/Dataset"
STATIC_CSV    = os.path.join(BASE_DIR, "hand_landmarks_static.csv")
MOTION_CSV    = os.path.join(BASE_DIR, "hand_landmarks_motion.csv")
SEQ_DIR       = os.path.join(BASE_DIR, "sequences")   # carpeta para .npy
os.makedirs(SEQ_DIR, exist_ok=True)

RECORD_FPS    = 10
RECORD_DELAY  = 1.0 / RECORD_FPS

# ── Normalización 
def normalize_landmarks(landmarks):
    lm = np.array(landmarks).reshape(21, 3)
    lm = lm - lm[0]                        # restar muñeca
    mx = np.max(np.abs(lm))
    if mx != 0:
        lm = lm / mx
    return lm.flatten()                     # (63,) 21 puntos × 3 coordenadas (x, y, z) = 63 valores

# ── CSV estático
STATIC_HEADER = ["label"] + [
    f"{ax}{i}" for i in range(21) for ax in ("x", "y", "z")
]

def init_static_csv():
    if not os.path.exists(STATIC_CSV):
        with open(STATIC_CSV, "w", newline="") as f:
            csv.writer(f).writerow(STATIC_HEADER)
        print(f"[INFO] CSV estático creado: {STATIC_CSV}")
    else:
        print(f"[INFO] Usando CSV estático existente: {STATIC_CSV}")

def save_static(label, vector):
    with open(STATIC_CSV, "a", newline="") as f:
        csv.writer(f).writerow([label] + vector.tolist())

# ── CSV de movimiento  (1 fila = 1 grabación completa)
MOTION_HEADER = ["label", "ruta_npy", "num_frames"]

def init_motion_csv():
    if not os.path.exists(MOTION_CSV):
        with open(MOTION_CSV, "w", newline="") as f:
            csv.writer(f).writerow(MOTION_HEADER)
        print(f"[INFO] CSV movimiento creado: {MOTION_CSV}")
    else:
        print(f"[INFO] Usando CSV movimiento existente: {MOTION_CSV}")

def save_motion(label, frames_list):
    """
    frames_list : lista de arrays (63,) — uno por frame capturado
    Guarda un .npy y registra 1 fila en el CSV.
    """
    if len(frames_list) == 0:
        print("[WARN] Secuencia vacía, no se guardó.")
        return

    # Nombre único: label_YYYYMMDD_HHMMSS_MS.npy
    ts       = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time()*1000)%1000:03d}"
    filename = f"{label}_{ts}.npy"
    npy_path = os.path.join(SEQ_DIR, filename)

    seq = np.array(frames_list, dtype=np.float32)  # shape: (T, 63)
    np.save(npy_path, seq)

    with open(MOTION_CSV, "a", newline="") as f:
        csv.writer(f).writerow([label, npy_path, len(frames_list)])

    print(f"[SAVE] Secuencia '{label}' → {len(frames_list)} frames → {filename}")

# ── Captura de etiqueta
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
        if key == 13:
            cv2.destroyWindow(win)
            result = "".join(typed).strip()
            return result if result else current_label
        elif key == 27:
            cv2.destroyWindow(win)
            return current_label
        elif key == 8 and typed:
            typed.pop()
        elif 32 <= key <= 126:
            typed.append(chr(key))

# ── Main 
def main():
    init_static_csv()
    init_motion_csv()

    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    cap = cv2.VideoCapture(0)

    current_label   = ""
    mode            = "estatico"    # "estatico" | "movimiento"
    recording       = False
    last_save_time  = 0.0
    session_count   = 0
    motion_buffer   = []            # frames acumulados de la secuencia actual

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = hands.process(frame_rgb)
        norm_vec  = None

        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                lm_flat = []
                for lm in hand_lm.landmark:
                    lm_flat += [lm.x, lm.y, lm.z]
                norm_vec = normalize_landmarks(lm_flat)

                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS
                )

                now = time.time()
                if recording and current_label and (now - last_save_time >= RECORD_DELAY):
                    if mode == "estatico":
                        save_static(current_label, norm_vec)
                        session_count += 1
                    else:
                        motion_buffer.append(norm_vec)
                    last_save_time = now

        # ── HUD 
        h, w = frame.shape[:2]
        font  = cv2.FONT_HERSHEY_SIMPLEX

        # Modo
        mode_color = (60, 220, 100) if mode == "estatico" else (50, 160, 240)
        cv2.putText(frame, f"Modo: {mode.upper()}", (10, 28), font, 0.7, mode_color, 2)

        # REC
        rec_color = (50, 50, 220) if recording else (80, 80, 80)
        cv2.circle(frame, (w - 120, 22), 8, rec_color, -1)
        cv2.putText(frame, "REC" if recording else "ESPERA",
                    (w - 108, 28), font, 0.65, rec_color, 2)

        # Buffer movimiento
        if mode == "movimiento" and recording:
            cv2.putText(frame, f"Frames: {len(motion_buffer)}",
                        (w - 160, 55), font, 0.6, (50, 160, 240), 2)

        # Etiqueta
        lbl_display = current_label or "sin etiqueta"
        lbl_color   = (60, 220, 100) if current_label else (30, 210, 230)
        cv2.putText(frame, f"Sena: {lbl_display}", (10, h - 15), font, 0.85, lbl_color, 2)

        # Contador
        cv2.putText(frame, f"Guardados: {session_count}", (10, 58), font, 0.6, (200, 200, 200), 1)

        # Controles
        controls = ["[L] Etiqueta", "[M] Modo", "[R] Rec. continua", "[ESC] Salir"]
        for i, ctrl in enumerate(controls):
            cv2.putText(frame, ctrl, (10, 88 + i * 22), font, 0.48, (140, 140, 140), 1)

        cv2.imshow("Recoleccion de Dataset", frame)

        # ── Teclado 
        key = cv2.waitKey(1) & 0xFF

        if key == 27:   # ESC — también guarda secuencia pendiente
            if mode == "movimiento" and motion_buffer:
                save_motion(current_label, motion_buffer)
                motion_buffer = []
            break

        elif key in (ord('l'), ord('L')):
            if recording:              # detener antes de cambiar etiqueta
                recording = False
                if mode == "movimiento" and motion_buffer:
                    save_motion(current_label, motion_buffer)
                    motion_buffer = []
            current_label = capture_label(current_label)
            print(f"[INFO] Etiqueta: '{current_label}'")

        elif key in (ord('m'), ord('M')):
            if recording:
                print("[WARN] Detén la grabación antes de cambiar de modo.")
            else:
                mode = "movimiento" if mode == "estatico" else "estatico"
                motion_buffer = []
                print(f"[INFO] Modo: {mode.upper()}")

        elif key in (ord('r'), ord('R')):
            if not current_label:
                print("[WARN] Ingresa una etiqueta primero con [L]")
            elif not recording:
                recording = True
                last_save_time = 0.0
                if mode == "movimiento":
                    motion_buffer = []      # nuevo buffer para esta grabación
                print(f"[INFO] Grabacion INICIADA — '{current_label}' [{mode}]")
            else:
                recording = False
                if mode == "movimiento" and motion_buffer:
                    save_motion(current_label, motion_buffer)
                    session_count += 1
                    motion_buffer = []
                print(f"[INFO] Grabacion DETENIDA — '{current_label}'")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print(f"\n[FIN] {session_count} muestras en '{BASE_DIR}'")

if __name__ == "__main__":
    main()
