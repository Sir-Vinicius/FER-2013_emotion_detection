import cv2
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import StandardScaler
import pickle

# Carregar o modelo LDA treinado
model_path = '/home/viniciuss/Documents/projetos/FER-2013_emotion_detection/lda_model.pkl'
with open(model_path, 'rb') as file:
    lda = pickle.load(file)

# Inicializar MediaPipe Face Mesh
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Definir pontos de interesse
points_of_interest = [list(range(10, 14)),  # Lábios
                     list(range(468, 478)),  # Olho esquerdo
                     list(range(249, 259)),  # Olho direito
                     [21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251],  # Linhas da testa
                     [71, 68, 104, 69, 108, 151, 337, 299, 333, 298, 301, 9],  # Linhas da testa
                     [143, 111, 117, 118, 119, 120, 121, 128, 245],  # Linhas rosto parte direita
                     [214, 207, 205, 36, 142, 126, 217, 174],  # Linhas rosto parte direita
                     [372, 340, 346, 347, 348, 349, 350, 357, 465],  # Linhas rosto parte esquerda
                     [434, 427, 425, 266, 371, 355, 437, 399]  # Linhas rosto parte esquerda
                    ]

# Função para extrair pontos de interesse
def extract_points_of_interest(landmarks):
    selected_points = []
    for group in points_of_interest:
        for idx in group:
            selected_points.extend([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])
    return np.array(selected_points).reshape(1, -1)

# Mapeamento de emoções
emotion_to_num = {'angry': 0, 'sad': 1, 'surprise': 2, 'happy': 3}

# Inicializar a webcam
cap = cv2.VideoCapture(0)

# Verificar se a webcam está aberta
if not cap.isOpened():
    print("Erro ao abrir a webcam.")
    exit()

while True:
    # Capturar um frame da webcam
    ret, frame = cap.read()

    if not ret:
        print("Erro ao capturar o frame.")
        break

    # Processar o frame para detectar emoções e landmarks
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(image)

    # Desenhar os landmarks na imagem
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx in range(len(face_landmarks.landmark)):
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

    # Processar o frame para detectar emoções
    landmarks = extract_points_of_interest(results.multi_face_landmarks[0].landmark)
    if landmarks.shape[0] > 0:
        scaler = StandardScaler()
        landmarks_scaled = scaler.fit_transform(landmarks)
        try:
            emotion = lda.predict(landmarks_scaled)
            emotion_name = list(emotion_to_num.keys())[emotion[0]]
            print(f"Detected emotion: {emotion_name}")
        except ValueError as e:
            print(f"Erro na predição: {str(e)}")

    # Exibir o frame na janela
    cv2.imshow('Webcam', image)

    # Aguardar a tecla 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a webcam e fechar a janela
cap.release()
cv2.destroyAllWindows()
