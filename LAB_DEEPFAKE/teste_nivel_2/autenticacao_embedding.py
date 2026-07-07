import csv
import datetime
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


# -----------------------------
# Configuracoes
# -----------------------------

TOLERANCIA = 8
CAMERA_INDEX = 1
WINDOW_NAME = "Autenticacao Facial - LAB02"

BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "dataset" / "guilherme_embedding_lab02.pkl"
LOG_PATH = BASE_DIR / "logs" / "autenticacao_embedding_log.csv"


def criar_log():
    LOG_PATH.parent.mkdir(exist_ok=True)

    if LOG_PATH.exists():
        return

    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "data_hora",
            "usuario",
            "resultado",
            "distancia",
            "tolerancia",
            "fps"
        ])


def carregar_embedding():
    if not DATASET_PATH.exists():
        print("[ERRO] Embedding nao encontrado.")
        print("[INFO] Execute cadastro_embedding.py primeiro.")
        raise FileNotFoundError(DATASET_PATH)

    with open(DATASET_PATH, "rb") as f:
        dados = pickle.load(f)

    return dados["nome"], dados["embedding"]


def registrar_log(nome_usuario, resultado, distancia, fps):
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.datetime.now(),
            nome_usuario,
            resultado,
            round(distancia, 4),
            TOLERANCIA,
            round(fps, 2)
        ])


def desenhar_resultado(frame, face, resultado, distancia, cor):
    x1, y1, x2, y2 = face.bbox.astype(int)

    cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
    cv2.putText(
        frame,
        resultado,
        (x1, y1 - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        cor,
        2
    )
    cv2.putText(
        frame,
        f"Distancia: {distancia:.3f}",
        (x1, y2 + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        cor,
        2
    )


def autenticar_face(face, nome_usuario, embedding_cadastrado):
    distancia = np.linalg.norm(embedding_cadastrado - face.embedding)

    if distancia <= TOLERANCIA:
        return f"ACEITO - {nome_usuario}", distancia, (0, 255, 0)

    return "REJEITADO", distancia, (0, 0, 255)


def deve_encerrar():
    tecla = cv2.waitKey(10) & 0xFF

    if tecla in (ord("q"), ord("Q"), 27):
        print("[INFO] Encerrando autenticacao...")
        return True

    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        print("[INFO] Janela fechada. Encerrando autenticacao...")
        return True

    return False


def main():
    criar_log()
    nome_usuario, embedding_cadastrado = carregar_embedding()

    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))

    camera = cv2.VideoCapture(CAMERA_INDEX)

    print("[INFO] Sistema de autenticacao iniciado.")
    print("[INFO] Pressione Q ou ESC na janela da camera para sair.")
    print(f"[INFO] Camera: {CAMERA_INDEX}")
    print(f"[INFO] Tolerancia: {TOLERANCIA}")

    if not camera.isOpened():
        print("[ERRO] Nao foi possivel acessar webcam.")
        return

    ultimo_tempo = time.time()

    try:
        while True:
            ret, frame = camera.read()

            if not ret:
                print("[ERRO] Falha ao capturar frame.")
                break

            tempo_atual = time.time()
            delta_tempo = tempo_atual - ultimo_tempo
            fps = 1 / delta_tempo if delta_tempo > 0 else 0
            ultimo_tempo = tempo_atual

            faces = app.get(frame)
            resultado = "SEM FACE"
            distancia = 0

            if faces:
                resultado, distancia, cor = autenticar_face(
                    faces[0],
                    nome_usuario,
                    embedding_cadastrado
                )
                desenhar_resultado(frame, faces[0], resultado, distancia, cor)

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2
            )

            cv2.imshow(WINDOW_NAME, frame)
            registrar_log(nome_usuario, resultado, distancia, fps)

            if deve_encerrar():
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("[INFO] Webcam liberada e janelas fechadas.")


if __name__ == "__main__":
    main()
