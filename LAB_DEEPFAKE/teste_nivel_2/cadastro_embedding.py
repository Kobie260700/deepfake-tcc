import cv2
import pickle
import datetime
from pathlib import Path
from insightface.app import FaceAnalysis

BASE_DIR = Path(__file__).parent
CAPTURAS_DIR = BASE_DIR / "capturas"
DATASET_DIR = BASE_DIR / "dataset"

CAPTURAS_DIR.mkdir(exist_ok=True)
DATASET_DIR.mkdir(exist_ok=True)

NOME_USUARIO = "guilherme"

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

camera = cv2.VideoCapture(0)

print("[INFO] Iniciando webcam...")
print("[INFO] Pressione ESPAÇO para capturar embedding.")
print("[INFO] Pressione Q para sair.")

if not camera.isOpened():
    print("[ERRO] Não foi possível acessar a webcam.")
    exit()

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

    cv2.imshow("Cadastro Facial - LAB02", frame)

    tecla = cv2.waitKey(1) & 0xFF

    if tecla == ord("q"):
        print("[INFO] Saindo...")
        break

    if tecla == 32:
        if len(faces) != 1:
            print("[ERRO] Capture com exatamente 1 rosto visível.")
            continue

        face = faces[0]
        embedding = face.embedding

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        imagem_path = CAPTURAS_DIR / f"{NOME_USUARIO}_lab02_{timestamp}.jpg"
        embedding_path = DATASET_DIR / f"{NOME_USUARIO}_embedding_lab02.pkl"

        cv2.imwrite(str(imagem_path), frame)

        with open(embedding_path, "wb") as f:
            pickle.dump(
                {
                    "nome": NOME_USUARIO,
                    "embedding": embedding,
                    "data_cadastro": timestamp,
                    "imagem": str(imagem_path)
                },
                f
            )

        print("[SUCESSO] Embedding cadastrado.")
        print(f"[INFO] Imagem salva em: {imagem_path}")
        print(f"[INFO] Embedding salvo em: {embedding_path}")
        break

camera.release()
cv2.destroyAllWindows()