from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import pickle
import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import StandardScaler

# Carregar o modelo LDA treinado
model_path = '/home/viniciuss/Documents/projetos/FER-2013_emotion_detection/lda_model.pkl'
with open(model_path, 'rb') as file:
    lda = pickle.load(file)

# Inicializar o FastAPI
app = FastAPI()

# Montar o diretório de arquivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

# Inicializar MediaPipe Face Mesh
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5)

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

# # Função para extrair pontos de interesse
def extract_points_of_interest(landmarks):
    selected_points = []
    for group in points_of_interest:
        for idx in group:
            selected_points.extend([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])
    return np.array(selected_points).reshape(1, -1)

# Mapeamento de emoções
emotion_to_num = {'angry': 0, 'sad': 1, 'surprise': 2, 'happy': 3}

# Endpoint Index
@app.get("/", response_class=HTMLResponse)
async def read_root():
    file_path = os.path.join(os.getcwd(), "static", "pages", "index.html")
    with open(file_path) as f:
        return HTMLResponse(content=f.read(), status_code=200)

# Endpoint para upload de imagem
@app.post("/predict-image/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    # Processar a imagem para detectar landmarks
    results = mp_face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = extract_points_of_interest(face_landmarks.landmark)
            #landmarks = np.array([[lmk.x, lmk.y, lmk.z] for lmk in face_landmarks.landmark]).flatten()

            print("Shape dos dados de entrada:", landmarks.shape)  # Verifica o shape dos dados
            print("Dados de entrada:\n", landmarks)  # Imprime os dados para verificação

            # Verificar se landmarks foram detectados
        if landmarks.shape[0] > 0:
            # Normalizar os landmarks
            #landmarks = landmarks.reshape(1, -1)  # Transforma em formato de linha para compatibilidade com StandardScaler
            scaler = StandardScaler()
            landmarks_scaled = scaler.fit_transform(landmarks)
            
            #Normalizar os landmarks
            scaler = StandardScaler()
            landmarks_scaled = scaler.fit_transform(landmarks)
            print("Landmarks scaled:", landmarks_scaled)
            
            # Fazer a predição com o modelo LDA
            try:
                emotion = lda.predict(landmarks_scaled)
                print(emotion)
                # Retornar a emoção prevista
                emotion_name = list(emotion_to_num.keys())[emotion[0]]
                return {"emotion": emotion_name}
            except ValueError as e:
                return {"error": f"Erro na predição: {str(e)}"}
    return {"error": "No face detected"}

# Endpoint WebSocket
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_bytes()
            npimg = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            results = mp_face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = extract_points_of_interest(face_landmarks.landmark)
                    if landmarks.shape[0] > 0:
                        scaler = StandardScaler()
                        landmarks_scaled = scaler.fit_transform(landmarks)
                        try:
                            emotion = lda.predict(landmarks_scaled)
                            emotion_name = list(emotion_to_num.keys())[emotion[0]]
                            await websocket.send_text(emotion_name)
                        except ValueError as e:
                            await websocket.send_text(f"Erro na predição: {str(e)}")
                    else:
                        await websocket.send_text("No face detected")
            else:
                await websocket.send_text("No face detected")
    except Exception as e:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
