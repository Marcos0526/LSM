import cv2
import torch
import numpy as np
import mediapipe as mp
from tgcn_model import GCN_muti_att
from configs import Config
from mi_dataset_csv import MiDatasetLSM 
import os
import time

directorio_actual = os.getcwd()
proyecto_raiz = os.path.dirname(directorio_actual)
subset = 'asl100'
config_file = os.path.join(directorio_actual, 'configs', '{}.ini'.format(subset))
configs = Config(config_file)
log_interval = configs.log_interval
num_samples = configs.num_samples
hidden_size = configs.hidden_size
drop_p = configs.drop_p
num_stages = configs.num_stages
num_class = 5 #['j', 'k', 'q', 'x', 'z']
clases = ['j', 'k', 'q', 'x', 'z']
n_nodes = 83  
n_dims = 3    
pose_data_root = os.path.join(proyecto_raiz, 'Codigos/Dataset')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = MiDatasetLSM(root_dir=os.path.join(pose_data_root, 'train'), 
                                 num_samples=num_samples, 
                                 num_nodes=n_nodes, 
                                 num_dims=n_dims)

model = GCN_muti_att(input_feature=num_samples * n_dims, 
                         hidden_feature=hidden_size,
                         num_class=num_class, 
                         p_dropout=drop_p, 
                         num_stage=num_stages,
                         num_nodes=n_nodes).cuda()
print(train_dataset.classes_)
ruta_pesos = 'checkpoints/asl100/best_model.pth'
model.load_state_dict(torch.load(ruta_pesos,weights_only=True))

# Esto desactiva capas como Dropout o BatchNorm que no deben actuar igual al inferir
model.eval()

model.to(device)
print("✅ Modelo cargado correctamente.")

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# --- Utilidades de dibujo de MediaPipe ---
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
# ------------------------------------------------

# Función de extracción
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    
    # Índices de cejas
    LEFT_EYEBROW_IDS = [70, 63, 105, 66]
    RIGHT_EYEBROW_IDS = [300, 293, 334, 296]
    
    if results.face_landmarks:
        left_brow = np.array([[results.face_landmarks.landmark[i].x, results.face_landmarks.landmark[i].y, results.face_landmarks.landmark[i].z] for i in LEFT_EYEBROW_IDS])
        right_brow = np.array([[results.face_landmarks.landmark[i].x, results.face_landmarks.landmark[i].y, results.face_landmarks.landmark[i].z] for i in RIGHT_EYEBROW_IDS])
    else:
        left_brow = np.zeros((4, 3))
        right_brow = np.zeros((4, 3))
        
    return np.concatenate([pose, lh, rh, left_brow, right_brow])

# Función normalizadora usando las variables globales (n_nodes, etc)
def normalize_landmarks(keypoints):
    lm = np.copy(keypoints)
    # Centrado en pecho (hombros)
    if np.any(lm[11]) and np.any(lm[12]):
        origin = (lm[11] + lm[12]) / 2.0
    else:
        origin = lm[0] if np.any(lm[0]) else np.zeros(3)
    lm = lm - origin
    mx = np.max(np.abs(lm))
    return (lm / mx if mx != 0 else lm).flatten()


cap = cv2.VideoCapture(0)

print("Iniciando inferencia... Presiona ESC para salir.")

buffer = []
estado = "Recolectando..."
ultima_prediccion = ""
tiempo_pausa = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # Si estamos en pausa, solo dibujamos la última predicción y vaciamos el buffer de cámara
    if estado == "Pausa":
        if time.time() - tiempo_pausa > 3.0: # Pasó 3 segundos
            estado = "Recolectando..."
            buffer = []
        else:
            cv2.putText(frame, "Pausa...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Senia: {ultima_prediccion}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.imshow("Inferencia LSM", frame)
            if cv2.waitKey(1) & 0xFF == 27: break
            continue # Saltamos el procesamiento de MediaPipe

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = holistic.process(rgb)
    
    # --- Dibujar los landmarks (grafos) sobre el frame original ---
    
    # 1. Dibujar el cuerpo (Pose)
    mp_drawing.draw_landmarks(
        frame,
        res.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )
    
    # 2. Dibujar mano izquierda
    mp_drawing.draw_landmarks(
        frame,
        res.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
    )
    
    # 3. Dibujar mano derecha
    mp_drawing.draw_landmarks(
        frame,
        res.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
    )
    
    # Opcional: Dibujar el rostro (descomenta las siguientes líneas si quieres ver la malla facial)
    if res.face_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            res.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
        )
    #---------------------------------------------------------------------

    if res.pose_landmarks or res.left_hand_landmarks or res.right_hand_landmarks:
        raw_keypoints = extract_keypoints(res)
        norm_vector = normalize_landmarks(raw_keypoints)
        buffer.append(norm_vector)
        estado = f"Recolectando: {len(buffer)}/{num_samples}"
        
        if len(buffer) >= num_samples:
            
            data = np.array(buffer).reshape(num_samples, n_nodes, n_dims)
            data = data.transpose(1, 0, 2).reshape(n_nodes, num_samples * n_dims)
            tensor = torch.FloatTensor(data).unsqueeze(0).to(device)
            
            with torch.no_grad():
                out = model(tensor)
                idx = torch.argmax(out, dim=1).item()
                ultima_prediccion = clases[idx] 
            
            # Cambiamos a estado de pausa y registramos el tiempo
            estado = "Pausa"
            tiempo_pausa = time.time()

    # Dibujar UI durante la recolección
    cv2.putText(frame, estado, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Ultima: {ultima_prediccion}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    cv2.imshow("Inferencia LSM", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
holistic.close()