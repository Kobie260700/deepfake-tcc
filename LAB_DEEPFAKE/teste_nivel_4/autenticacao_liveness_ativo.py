import csv
import datetime
import pickle
import random
import time
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


# 0 = webcam física; 1 = OBS Virtual Camera / DeepLiveCam.
CAMERA_INDEX = 1
WINDOW_NAME = "Autenticacao Facial com Liveness Ativo - LAB04"

# Herdado da calibração experimental do LAB02.
THRESHOLD_IDENTIDADE = 20.210

# Parâmetros iniciais do challenge-response.
ACOES_POR_TENTATIVA = 4
TEMPO_LIMITE_POR_ACAO = 6.0
FRAMES_ESTAVEIS = 3
LIMITE_CENTRO = 0.08
LIMITE_LATERAL = 0.13
FATOR_EAR_OLHO_FECHADO = 0.72
FRAMES_CALIBRACAO_OLHO_ABERTO = 3
FRAMES_OLHO_FECHADO = 1
FRAMES_OLHO_ABERTO = 2

BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "dataset" / "guilherme_embedding_lab04.pkl"
LOG_PATH = BASE_DIR / "logs" / "autenticacao_liveness_ativo_lab04.csv"

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
            "sequencia_desafios",
            "acoes_concluidas",
            "tempos_resposta_segundos",
            "tempo_total_segundos",
            "distancia_media",
            "threshold_identidade",
            "identidade_valida",
            "challenge_valido",
            "resultado"
        ])


def carregar_cadastro():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Cadastro não encontrado: {DATASET_PATH}. "
            "Execute cadastro_embedding_lab04.py primeiro."
        )

    with open(DATASET_PATH, "rb") as arquivo:
        dados = pickle.load(arquivo)

    return dados["nome"], np.asarray(dados["embedding"], dtype=np.float32)


def sortear_sequencia():
    primeira = random.SystemRandom().choice(("ESQUERDA", "DIREITA"))
    ultima = "DIREITA" if primeira == "ESQUERDA" else "ESQUERDA"
    return [primeira, "CENTRO", "PISQUE", ultima]


def calcular_deslocamento_horizontal(face):
    pontos = np.asarray(face.kps, dtype=np.float32)
    olho_esquerdo, olho_direito, nariz = pontos[0], pontos[1], pontos[2]
    centro_olhos_x = (olho_esquerdo[0] + olho_direito[0]) / 2
    distancia_olhos = max(abs(olho_direito[0] - olho_esquerdo[0]), 1.0)
    return float((nariz[0] - centro_olhos_x) / distancia_olhos)


def classificar_orientacao(deslocamento):
    if deslocamento <= -LIMITE_LATERAL:
        return "DIREITA"
    if deslocamento >= LIMITE_LATERAL:
        return "ESQUERDA"
    if abs(deslocamento) <= LIMITE_CENTRO:
        return "CENTRO"
    return "TRANSICAO"


def obter_landmarks_68(face):
    pontos = getattr(face, "landmark_3d_68", None)

    if pontos is None:
        return None

    return np.asarray(pontos, dtype=np.float32)[:, :2]


def calcular_ear(olho):
    vertical_1 = np.linalg.norm(olho[1] - olho[5])
    vertical_2 = np.linalg.norm(olho[2] - olho[4])
    horizontal = max(np.linalg.norm(olho[0] - olho[3]), 1e-6)
    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def calcular_ear_medio(landmarks):
    olho_esquerdo = landmarks[36:42]
    olho_direito = landmarks[42:48]
    return (calcular_ear(olho_esquerdo) + calcular_ear(olho_direito)) / 2


def desenhar_contorno_facial(frame, landmarks, cor=(0, 255, 0)):
    if landmarks is None:
        return

    pontos = np.round(landmarks).astype(np.int32)
    segmentos_abertos = (
        range(0, 17),
        range(17, 22),
        range(22, 27),
        range(27, 31),
        range(31, 36)
    )
    segmentos_fechados = (
        range(36, 42),
        range(42, 48),
        range(48, 60),
        range(60, 68)
    )

    for indices in segmentos_abertos:
        cv2.polylines(frame, [pontos[list(indices)]], False, cor, 1, cv2.LINE_AA)

    for indices in segmentos_fechados:
        cv2.polylines(frame, [pontos[list(indices)]], True, cor, 1, cv2.LINE_AA)


def desenhar_texto(frame, texto, linha, cor=(255, 255, 255), escala=0.7):
    cv2.putText(
        frame,
        texto,
        (20, linha),
        cv2.FONT_HERSHEY_SIMPLEX,
        escala,
        cor,
        2
    )


def registrar_resultado(
    nome,
    sequencia,
    concluidas,
    tempos,
    distancia_media,
    identidade_valida,
    challenge_valido,
    resultado
):
    fonte = "webcam_fisica" if CAMERA_INDEX == 0 else "camera_virtual"

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as arquivo:
        writer = csv.writer(arquivo)
        writer.writerow([
            datetime.datetime.now().isoformat(sep=" "),
            nome,
            fonte,
            " > ".join(sequencia),
            " > ".join(concluidas),
            " | ".join(f"{tempo:.3f}" for tempo in tempos),
            round(sum(tempos), 3),
            round(distancia_media, 4) if distancia_media is not None else "",
            THRESHOLD_IDENTIDADE,
            identidade_valida,
            challenge_valido,
            resultado
        ])


def main():
    criar_log()
    nome_usuario, embedding_cadastrado = carregar_cadastro()

    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("[ERRO] Não foi possível acessar a câmera.")
        return

    print("[INFO] LAB04 iniciado.")
    print("[INFO] ESPAÇO inicia uma tentativa challenge-response.")
    print("[INFO] Q ou ESC encerra.")

    ativo = False
    sequencia = []
    indice_acao = 0
    frames_corretos = 0
    inicio_acao = 0.0
    acoes_concluidas = []
    tempos_resposta = []
    distancias = []
    mensagem = "Pressione ESPACO para iniciar"
    cor_resultado = (255, 255, 0)
    distancia_final = None
    challenge_final = None
    ultimo_tempo = time.time()
    blink_estado = "AGUARDANDO_ABERTO"
    blink_frames_fechados = 0
    blink_frames_abertos = 0
    amostras_ear_aberto = []
    ear_aberto_referencia = None

    try:
        while True:
            ret, frame = camera.read()

            if not ret:
                print("[ERRO] Falha ao capturar frame.")
                break

            agora = time.time()
            delta = agora - ultimo_tempo
            fps = 1 / delta if delta > 0 else 0
            ultimo_tempo = agora

            faces = app.get(frame)
            orientacao = "SEM FACE"
            deslocamento = 0.0
            ear_atual = None

            if len(faces) == 1:
                face = faces[0]
                distancia_atual = float(np.linalg.norm(
                    embedding_cadastrado - face.embedding
                ))
                deslocamento = calcular_deslocamento_horizontal(face)
                orientacao = classificar_orientacao(deslocamento)
                landmarks_68 = obter_landmarks_68(face)
                desenhar_contorno_facial(frame, landmarks_68)

                if landmarks_68 is not None:
                    ear_atual = calcular_ear_medio(landmarks_68)

                if ativo:
                    distancias.append(distancia_atual)
                    acao_atual = sequencia[indice_acao]

                    if acao_atual == "PISQUE":
                        piscada_concluida = False

                        if ear_atual is not None:
                            if ear_aberto_referencia is None:
                                amostras_ear_aberto.append(ear_atual)

                                if len(amostras_ear_aberto) >= FRAMES_CALIBRACAO_OLHO_ABERTO:
                                    ear_aberto_referencia = float(np.median(amostras_ear_aberto))

                            threshold_ear = (
                                ear_aberto_referencia * FATOR_EAR_OLHO_FECHADO
                                if ear_aberto_referencia is not None
                                else None
                            )
                            olho_fechado = (
                                threshold_ear is not None
                                and ear_atual < threshold_ear
                            )

                            if blink_estado == "AGUARDANDO_ABERTO":
                                if ear_aberto_referencia is not None and not olho_fechado:
                                    blink_frames_abertos += 1
                                else:
                                    blink_frames_abertos = 0

                                if blink_frames_abertos >= FRAMES_OLHO_ABERTO:
                                    blink_estado = "AGUARDANDO_FECHAR"
                                    blink_frames_fechados = 0

                            elif blink_estado == "AGUARDANDO_FECHAR":
                                if olho_fechado:
                                    blink_frames_fechados += 1
                                else:
                                    blink_frames_fechados = 0

                                if blink_frames_fechados >= FRAMES_OLHO_FECHADO:
                                    blink_estado = "AGUARDANDO_REABRIR"
                                    blink_frames_abertos = 0

                            elif blink_estado == "AGUARDANDO_REABRIR":
                                if not olho_fechado:
                                    blink_frames_abertos += 1
                                else:
                                    blink_frames_abertos = 0

                                if blink_frames_abertos >= FRAMES_OLHO_ABERTO:
                                    piscada_concluida = True

                        frames_corretos = FRAMES_ESTAVEIS if piscada_concluida else 0
                    elif orientacao == acao_atual:
                        frames_corretos += 1
                    else:
                        frames_corretos = 0

                    if frames_corretos >= FRAMES_ESTAVEIS:
                        tempo_resposta = agora - inicio_acao
                        acoes_concluidas.append(acao_atual)
                        tempos_resposta.append(tempo_resposta)
                        indice_acao += 1
                        frames_corretos = 0

                        if indice_acao >= len(sequencia):
                            ativo = False
                            distancia_final = float(np.mean(distancias))
                            identidade_valida = distancia_final <= THRESHOLD_IDENTIDADE
                            challenge_final = True
                            resultado = "ACEITO" if identidade_valida else "REJEITADO_IDENTIDADE"
                            mensagem = resultado
                            cor_resultado = (0, 255, 0) if resultado == "ACEITO" else (0, 0, 255)
                            registrar_resultado(
                                nome_usuario,
                                sequencia,
                                acoes_concluidas,
                                tempos_resposta,
                                distancia_final,
                                identidade_valida,
                                True,
                                resultado
                            )
                            print(f"[RESULTADO] {resultado}")
                        else:
                            inicio_acao = agora
                            blink_estado = "AGUARDANDO_ABERTO"
                            blink_frames_fechados = 0
                            blink_frames_abertos = 0
                            amostras_ear_aberto = []
                            ear_aberto_referencia = None

            if ativo:
                acao_atual = sequencia[indice_acao]
                restante = TEMPO_LIMITE_POR_ACAO - (agora - inicio_acao)

                if restante <= 0:
                    ativo = False
                    distancia_final = float(np.mean(distancias)) if distancias else None
                    identidade_valida = (
                        distancia_final is not None
                        and distancia_final <= THRESHOLD_IDENTIDADE
                    )
                    challenge_final = False
                    mensagem = "REJEITADO_CHALLENGE"
                    cor_resultado = (0, 0, 255)
                    registrar_resultado(
                        nome_usuario,
                        sequencia,
                        acoes_concluidas,
                        tempos_resposta,
                        distancia_final,
                        identidade_valida,
                        False,
                        mensagem
                    )
                else:
                    desenhar_texto(frame, f"ACAO: {acao_atual}", 38, (0, 255, 255), 0.9)
                    desenhar_texto(
                        frame,
                        f"Progresso: {indice_acao + 1}/{len(sequencia)}",
                        72,
                        (0, 255, 255)
                    )
                    desenhar_texto(frame, f"Tempo: {restante:.1f}s", 104, (0, 255, 255))
                    if acao_atual == "PISQUE" and ear_atual is not None:
                        desenhar_texto(frame, f"EAR: {ear_atual:.3f}", 136, (0, 255, 255))
                        if ear_aberto_referencia is None:
                            desenhar_texto(frame, "Mantenha os olhos abertos", 168, (0, 255, 255))
                        else:
                            threshold_ear = ear_aberto_referencia * FATOR_EAR_OLHO_FECHADO
                            desenhar_texto(
                                frame,
                                f"Feche e abra os olhos (limite: {threshold_ear:.3f})",
                                168,
                                (0, 255, 255)
                            )
            else:
                desenhar_texto(frame, mensagem, 38, cor_resultado, 0.9)

                if distancia_final is not None:
                    desenhar_texto(
                        frame,
                        f"Distancia media: {distancia_final:.3f}",
                        72,
                        cor_resultado
                    )
                if challenge_final is not None:
                    status = "VALIDO" if challenge_final else "INVALIDO"
                    desenhar_texto(frame, f"Challenge: {status}", 104, cor_resultado)

            desenhar_texto(frame, f"Orientacao: {orientacao}", 202, (255, 255, 0))
            desenhar_texto(frame, f"FPS: {fps:.1f}", 234, (255, 255, 0))

            cv2.imshow(WINDOW_NAME, frame)
            tecla = cv2.waitKey(10) & 0xFF

            if tecla in (ord("q"), ord("Q"), 27):
                break

            if tecla == 32 and not ativo:
                if len(faces) != 1:
                    mensagem = "TESTE INVALIDO - posicione um unico rosto"
                    cor_resultado = (0, 0, 255)
                else:
                    sequencia = sortear_sequencia()
                    indice_acao = 0
                    frames_corretos = 0
                    inicio_acao = time.time()
                    acoes_concluidas = []
                    tempos_resposta = []
                    distancias = []
                    distancia_final = None
                    challenge_final = None
                    blink_estado = "AGUARDANDO_ABERTO"
                    blink_frames_fechados = 0
                    blink_frames_abertos = 0
                    amostras_ear_aberto = []
                    ear_aberto_referencia = None
                    ativo = True
                    mensagem = "Challenge iniciado"
                    cor_resultado = (0, 255, 255)
                    print(f"[DESAFIO] {' > '.join(sequencia)}")

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print(f"[INFO] Log: {LOG_PATH}")


if __name__ == "__main__":
    main()
