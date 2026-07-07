import cv2
import face_recognition
import pickle
import time
import csv
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
LOGS_DIR = BASE_DIR / "logs"

LOGS_DIR.mkdir(exist_ok=True)

ENCODING_PATH = DATASET_DIR / "guilherme_encoding_lab01.pkl"
LOG_PATH = LOGS_DIR / "autenticacao_lab01_log.csv"


TOLERANCIA = 0.45
CAMERA_INDEX = 1  # OBS Virtual Camera. Para webcam real, use 0.

if not ENCODING_PATH.exists():
    print("[ERRO] Encoding não encontrado. Execute primeiro cadastro_face.py")
    exit()

with open(ENCODING_PATH, "rb") as f:
    dados = pickle.load(f)

nome_usuario = dados["nome"]
encoding_cadastrado = dados["encoding"]

if not LOG_PATH.exists():
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["data_hora", "usuario", "resultado", "distancia", "tolerancia", "fps", "observacao"])

print("[INFO] Sistema de autenticação iniciado.")
print("[INFO] Pressione Q para sair.")
print(f"[INFO] Tolerância: {TOLERANCIA}")

camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    print("[ERRO] Não foi possível acessar a webcam.")
    exit()

ultimo_tempo = time.time()

while True:
    ret, frame = camera.read()

    if not ret:
        print("[ERRO] Falha ao capturar frame.")
        break

    agora = time.time()
    fps = 1 / (agora - ultimo_tempo)
    ultimo_tempo = agora

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(frame_rgb)
    face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

    resultado_texto = "SEM FACE"
    cor = (0, 255, 255)
    distancia_final = None

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        distancia = face_recognition.face_distance([encoding_cadastrado], face_encoding)[0]
        distancia_final = float(distancia)

        if distancia <= TOLERANCIA:
            resultado_texto = f"ACEITO - {nome_usuario}"
            cor = (0, 255, 0)
        else:
            resultado_texto = "REJEITADO"
            cor = (0, 0, 255)

        cv2.rectangle(frame, (left, top), (right, bottom), cor, 2)

        cv2.putText(frame, f"Distancia: {distancia:.3f}",
                    (left, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)

        break

    cv2.putText(frame, resultado_texto,
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor, 2)

    cv2.putText(frame, f"FPS: {fps:.1f}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Autenticacao Facial - LAB01", frame)

    if distancia_final is not None:
        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                nome_usuario,
                resultado_texto,
                round(distancia_final, 4),
                TOLERANCIA,
                round(fps, 2),
                "execucao_tempo_real"
            ])

    tecla = cv2.waitKey(1) & 0xFF

    if tecla == ord("q"):
        print("[INFO] Encerrando autenticação...")
        break

camera.release()
cv2.destroyAllWindows()

print("[INFO] Autenticação finalizada.")
print(f"[INFO] Log salvo em: {LOG_PATH}")