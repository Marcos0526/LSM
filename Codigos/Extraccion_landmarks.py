# Normalizacion y extracción de landmarks de la mano utilizando MediaPipe
import cv2
import mediapipe as mp
import numpy as np


def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(21, 3)

    wrist = landmarks[0]
    landmarks = landmarks - wrist  # centrar en muñeca

    max_value = np.max(np.abs(landmarks))

    if max_value != 0:
        landmarks = landmarks / max_value

    return landmarks.flatten()


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmark_list = []

            for lm in hand_landmarks.landmark:
                landmark_list.append(lm.x)
                landmark_list.append(lm.y)
                landmark_list.append(lm.z)

            # Normalizar
            normalized_vector = normalize_landmarks(landmark_list)

            print(normalized_vector)
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
