import cv2
import torch
import numpy as np
import mediapipe as mp
from tgcn_model import GCN_muti_att
from configs import Config
# from mi_dataset_csv import MiDatasetLSM # (Ya no es estrictamente necesario aquí si no extraes data)
import os
import time

directorio_actual = os.getcwd()
proyecto_raiz = os.path.dirname(directorio_actual)
subset = 'asl100'
config_file = os.path.join(directorio_actual, 'configs', '{}.ini'.format(subset))
configs = Config(config_file)

# FORZAMOS las dimensiones exactas de tu recolector
num_samples = 120  # 4 segundos a 30fps
n_nodes = 79       # 25 (pose) + 21 (MI) + 21 (MD) + 4 (ceja_izq) + 4 (ceja_der) + 4 (boca)
n_dims = 3    

hidden_size = configs.hidden_size
drop_p = configs.drop_p
num_stages = configs.num_stages
 
pose_data_root = os.path.join(proyecto_raiz, 'Codigos/Dataset')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ruta_pesos = 'checkpoints/asl100/best_model.pth'

# --- MODIFICACIÓN 1: Carga del Checkpoint Completo ---
# NOTA: Quitamos weights_only=True porque ahora cargamos un diccionario que contiene strings (las clases)
print("Cargando checkpoint...")
checkpoint = torch.load(ruta_pesos, map_location=device, weights_only=False)

# Extraemos las clases que guardamos al entrenar
clases_guardadas = checkpoint['classes']
num_class = len(clases_guardadas) 

# Instanciamos el modelo pasándole las clases_guardadas
model = GCN_muti_att(input_feature=num_samples * n_dims, 
                     hidden_feature=hidden_size,
                     num_class=num_class, 
                     p_dropout=drop_p, 
                     num_stage=num_stages,
                     num_nodes=n_nodes,
                     classes=clases_guardadas) # <-- Pasamos las clases aquí

# Cargamos los pesos (state_dict) desde el diccionario
model.load_state_dict(checkpoint['state_dict'])

model.eval()
model.to(device)
print(f"✅ Modelo cargado correctamente con {num_class} clases: {clases_guardadas}")

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# --- Índices exactos de tu recolector ---
LEFT_EYEBROW_IDS  = [70, 63, 105, 66]
RIGHT_EYEBROW_IDS = [300, 293, 334, 296]
MOUTH_IDS         = [13, 14, 61, 291]

def extract_keypoints(results):
    """
    Extrae exactamente los 79 nodos como lo hace el recolector de datos.
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
    """
    Normalización idéntica a la del recolector (sin flatten extra).
    """
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

cap = cv2.VideoCapture(0)
print("Iniciando inferencia... Presiona ESC para salir.")

buffer = []
estado = "Recolectando..."
ultima_prediccion = ""
tiempo_pausa = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1) # Espejo para que sea intuitivo al hacer las señas
    h, w = frame.shape[:2]
    
    if estado == "Pausa":
        if time.time() - tiempo_pausa > 3.0: 
            estado = "Recolectando..."
            buffer = []
        else:
            cv2.putText(frame, "Pausa...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Sena: {ultima_prediccion}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.imshow("Inferencia LSM", frame)
            if cv2.waitKey(1) & 0xFF == 27: break
            continue 

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = holistic.process(rgb)
    
    if res.pose_landmarks or res.left_hand_landmarks or res.right_hand_landmarks:
        # Extraemos los 79 nodos reales
        raw_keypoints = extract_keypoints(res)
        
        # --- DIBUJADO DE PUNTOS (Puntos exactos que ve la IA, sin líneas) ---
        for pt in raw_keypoints:
            if np.any(pt): # Si el punto no es [0,0,0] (detectado)
                cx, cy = int(pt[0] * w), int(pt[1] * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
        # -------------------------------------------------------------------
        
        # Normalizamos y guardamos en el buffer
        norm_vector = normalize_landmarks(raw_keypoints)
        buffer.append(norm_vector)
        estado = f"Recolectando: {len(buffer)}/{num_samples}"
        
        if len(buffer) >= num_samples:
            # Reestructuramos al formato requerido por la red GCN
            # Buffer actual: (120, 79, 3)
            data = np.array(buffer).reshape(num_samples, n_nodes, n_dims)
            data = data.transpose(1, 0, 2).reshape(n_nodes, num_samples * n_dims)
            tensor = torch.FloatTensor(data).unsqueeze(0).to(device)
            
            with torch.no_grad():
                out = model(tensor)
                # --- MODIFICACIÓN 2: Obtener la predicción de texto directamente ---
                # get_predicted_classes devuelve una lista (ej. ["hola"]), extraemos el índice [0]
                ultima_prediccion = model.get_predicted_classes(out)[0]
            
            estado = "Pausa"
            tiempo_pausa = time.time()

    # HUD
    cv2.putText(frame, estado, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Ultima: {ultima_prediccion}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    cv2.imshow("Inferencia LSM", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
holistic.close()