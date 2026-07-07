import datetime
import pickle
from pathlib import Path

import cv2
from insightface.app import FaceAnalysis


NOME_USUARIO = "guilherme"
CAMERA_INDEX = 0
WINDOW_NAME = "Cadastro Facial - LAB03"

BASE_DIR = Path(__file__).parent
CAPTURAS_DIR = BASE_DIR / "capturas"
DATASET_DIR = BASE_DIR / "dataset"
EMBEDDING_PATH = DATASET_DIR / f"{NOME_USUARIO}_embedding_lab03.pkl"


def main():
    CAPTURAS_DIR.mkdir(exist_ok=True)
    DATASET_DIR.mkdir(exist_ok=True)

    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))

    camera = cv2.VideoCapture(CAMERA_INDEX)

    print("[INFO] Cadastro facial do LAB03 iniciado.")
    print("[INFO] Pressione ESPAÇO com exatamente um rosto visível.")
    print("[INFO] Pressione Q ou ESC para sair.")

    if not camera.isOpened():
        print("[ERRO] Não foi possível acessar a webcam.")
        return

    try:
        while True:
            ret, frame = camera.read()

            if not ret:
                print("[ERRO] Falha ao capturar frame.")
                break

            faces = app.get(frame)

            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"Faces detectadas: {len(faces)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            cv2.imshow(WINDOW_NAME, frame)

            tecla = cv2.waitKey(10) & 0xFF

            if tecla in (ord("q"), ord("Q"), 27):
                break

            if tecla == 32:
                if len(faces) != 1:
                    print("[ERRO] Capture com exatamente um rosto visível.")
                    continue

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                imagem_path = CAPTURAS_DIR / f"{NOME_USUARIO}_lab03_{timestamp}.jpg"

                cv2.imwrite(str(imagem_path), frame)

                with open(EMBEDDING_PATH, "wb") as arquivo:
                    pickle.dump(
                        {
                            "nome": NOME_USUARIO,
                            "embedding": faces[0].embedding,
                            "data_cadastro": timestamp,
                            "imagem": str(imagem_path),
                            "modelo": "InsightFace buffalo_l",
                            "det_size": (640, 640)
                        },
                        arquivo
                    )

                print("[SUCESSO] Cadastro do LAB03 concluído.")
                print(f"[INFO] Embedding: {EMBEDDING_PATH}")
                print(f"[INFO] Imagem: {imagem_path}")
                break

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
