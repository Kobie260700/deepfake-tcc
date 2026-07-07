import csv
import datetime
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


# 0 = webcam fisica; 1 = OBS Virtual Camera / DeepLiveCam.
CAMERA_INDEX = 1
WINDOW_NAME = "Autenticacao Facial com Liveness Passivo - LAB03"

# Threshold biometrico provisório. Deve ser calibrado no experimento do LAB03.
THRESHOLD_IDENTIDADE = 20.210

# Parametros iniciais da prova de vida passiva.
DURACAO_COLETA_SEGUNDOS = 4.0
MIN_FRAMES_VALIDOS = 8
MIN_MOVIMENTO_LANDMARKS = 0.0015
MAX_MOVIMENTO_LANDMARKS = 0.0800
MIN_VARIACAO_TEMPORAL = 1.0

BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "dataset" / "guilherme_embedding_lab03.pkl"
LOG_PATH = BASE_DIR / "logs" / "autenticacao_liveness_lab03.csv"


def criar_log():
    LOG_PATH.parent.mkdir(exist_ok=True)

    if LOG_PATH.exists():
        return

    with open(LOG_PATH, "w", newline="", encoding="utf-8") as arquivo:
        writer = csv.writer(arquivo)
        writer.writerow([
            "data_hora",
            "usuario",
            "fonte_camera",
            "distancia_media",
            "threshold_identidade",
            "movimento_landmarks",
            "variacao_temporal",
            "frames_validos",
            "identidade_valida",
            "liveness_valido",
            "resultado"
        ])


def carregar_cadastro():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Cadastro não encontrado: {DATASET_PATH}. "
            "Execute cadastro_embedding_lab03.py primeiro."
        )

    with open(DATASET_PATH, "rb") as arquivo:
        dados = pickle.load(arquivo)

    return dados["nome"], np.asarray(dados["embedding"], dtype=np.float32)


def normalizar_landmarks(face):
    x1, y1, x2, y2 = face.bbox.astype(float)
    largura = max(x2 - x1, 1.0)
    altura = max(y2 - y1, 1.0)
    pontos = np.asarray(face.kps, dtype=np.float32)
    pontos[:, 0] = (pontos[:, 0] - x1) / largura
    pontos[:, 1] = (pontos[:, 1] - y1) / altura
    return pontos


def recortar_rosto_cinza(frame, face):
    altura_frame, largura_frame = frame.shape[:2]
    x1, y1, x2, y2 = face.bbox.astype(int)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(largura_frame, x2)
    y2 = min(altura_frame, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    recorte = frame[y1:y2, x1:x2]
    cinza = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    return cv2.resize(cinza, (128, 128))


def calcular_metricas(amostras):
    distancias = [amostra["distancia"] for amostra in amostras]
    landmarks = [amostra["landmarks"] for amostra in amostras]
    recortes = [amostra["recorte"] for amostra in amostras]

    movimentos = [
        float(np.mean(np.linalg.norm(atual - anterior, axis=1)))
        for anterior, atual in zip(landmarks, landmarks[1:])
    ]

    variacoes = [
        float(np.mean(cv2.absdiff(anterior, atual)))
        for anterior, atual in zip(recortes, recortes[1:])
    ]

    return {
        "distancia_media": float(np.mean(distancias)),
        "movimento_landmarks": float(np.mean(movimentos)) if movimentos else 0.0,
        "variacao_temporal": float(np.mean(variacoes)) if variacoes else 0.0,
        "frames_validos": len(amostras)
    }


def avaliar(metricas):
    identidade_valida = metricas["distancia_media"] <= THRESHOLD_IDENTIDADE
    movimento = metricas["movimento_landmarks"]
    liveness_valido = (
        MIN_MOVIMENTO_LANDMARKS <= movimento <= MAX_MOVIMENTO_LANDMARKS
        and metricas["variacao_temporal"] >= MIN_VARIACAO_TEMPORAL
        and metricas["frames_validos"] >= MIN_FRAMES_VALIDOS
    )

    if identidade_valida and liveness_valido:
        resultado = "ACEITO"
    elif not identidade_valida:
        resultado = "REJEITADO_IDENTIDADE"
    else:
        resultado = "REJEITADO_LIVENESS"

    return identidade_valida, liveness_valido, resultado


def registrar_resultado(nome, metricas, identidade_valida, liveness_valido, resultado):
    fonte = "webcam_fisica" if CAMERA_INDEX == 0 else "camera_virtual"

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as arquivo:
        writer = csv.writer(arquivo)
        writer.writerow([
            datetime.datetime.now().isoformat(sep=" "),
            nome,
            fonte,
            round(metricas["distancia_media"], 4),
            THRESHOLD_IDENTIDADE,
            round(metricas["movimento_landmarks"], 6),
            round(metricas["variacao_temporal"], 4),
            metricas["frames_validos"],
            identidade_valida,
            liveness_valido,
            resultado
        ])


def desenhar_status(frame, mensagem, cor, linha=40):
    cv2.putText(
        frame,
        mensagem,
        (20, linha),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        cor,
        2
    )


def main():
    criar_log()
    nome_usuario, embedding_cadastrado = carregar_cadastro()

    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("[ERRO] Não foi possível acessar a câmera.")
        return

    print("[INFO] LAB03 iniciado.")
    print("[INFO] ESPAÇO inicia uma tentativa controlada de liveness.")
    print("[INFO] Q ou ESC encerra.")

    coletando = False
    inicio_coleta = 0.0
    amostras = []
    mensagem_final = "Pressione ESPACO para iniciar o teste"
    cor_final = (255, 255, 0)
    metricas_finais = None
    liveness_final = None
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

            if len(faces) == 1:
                face = faces[0]
                x1, y1, x2, y2 = face.bbox.astype(int)
                cor_contorno = (0, 255, 255) if coletando else cor_final
                cv2.rectangle(frame, (x1, y1), (x2, y2), cor_contorno, 2)

                if coletando:
                    recorte = recortar_rosto_cinza(frame, face)

                    if recorte is not None:
                        amostras.append({
                            "distancia": float(np.linalg.norm(
                                embedding_cadastrado - face.embedding
                            )),
                            "landmarks": normalizar_landmarks(face),
                            "recorte": recorte
                        })
            elif coletando:
                mensagem_final = "Mantenha exatamente um rosto visivel"

            if coletando:
                decorrido = time.time() - inicio_coleta
                restante = max(0.0, DURACAO_COLETA_SEGUNDOS - decorrido)
                desenhar_status(frame, f"Coletando: {restante:.1f}s", (0, 255, 255))
                desenhar_status(frame, f"Frames validos: {len(amostras)}", (0, 255, 255), 72)

                if decorrido >= DURACAO_COLETA_SEGUNDOS:
                    coletando = False

                    if len(amostras) < MIN_FRAMES_VALIDOS:
                        mensagem_final = "TESTE INVALIDO - poucos frames validos"
                        cor_final = (0, 0, 255)
                        metricas_finais = None
                        liveness_final = None
                    else:
                        metricas = calcular_metricas(amostras)
                        identidade_valida, liveness_valido, resultado = avaliar(metricas)
                        registrar_resultado(
                            nome_usuario,
                            metricas,
                            identidade_valida,
                            liveness_valido,
                            resultado
                        )
                        mensagem_final = resultado
                        cor_final = (0, 255, 0) if resultado == "ACEITO" else (0, 0, 255)
                        metricas_finais = metricas
                        liveness_final = liveness_valido

                        print("\n===== RESULTADO LAB03 =====")
                        print(f"Distância média: {metricas['distancia_media']:.4f}")
                        print(f"Movimento landmarks: {metricas['movimento_landmarks']:.6f}")
                        print(f"Variação temporal: {metricas['variacao_temporal']:.4f}")
                        print(f"Frames válidos: {metricas['frames_validos']}")
                        print(f"Identidade válida: {identidade_valida}")
                        print(f"Liveness válido: {liveness_valido}")
                        print(f"Resultado: {resultado}")
            else:
                desenhar_status(frame, mensagem_final, cor_final, 42)

                if metricas_finais is not None:
                    desenhar_status(
                        frame,
                        f"Distancia media: {metricas_finais['distancia_media']:.3f}",
                        cor_final,
                        74
                    )
                    status_liveness = "VALIDO" if liveness_final else "INVALIDO"
                    desenhar_status(
                        frame,
                        f"Liveness: {status_liveness}",
                        cor_final,
                        106
                    )

            desenhar_status(frame, f"FPS: {fps:.1f}", (255, 255, 0), 138)

            cv2.imshow(WINDOW_NAME, frame)
            tecla = cv2.waitKey(10) & 0xFF

            if tecla in (ord("q"), ord("Q"), 27):
                break

            if tecla == 32 and not coletando:
                if len(faces) != 1:
                    mensagem_final = "TESTE INVALIDO - posicione um unico rosto"
                    cor_final = (0, 0, 255)
                else:
                    amostras = []
                    metricas_finais = None
                    liveness_final = None
                    inicio_coleta = time.time()
                    coletando = True
                    mensagem_final = "Coleta iniciada"
                    cor_final = (0, 255, 255)

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print(f"[INFO] Log: {LOG_PATH}")


if __name__ == "__main__":
    main()
