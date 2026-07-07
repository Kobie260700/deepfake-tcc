import cv2
import face_recognition
import pickle
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
CAPTURAS_DIR = BASE_DIR / "capturas"

DATASET_DIR.mkdir(exist_ok=True)
CAPTURAS_DIR.mkdir(exist_ok=True)

NOME_USUARIO = "guilherme"
CAMERA_INDEX = 0

print("[INFO] Iniciando webcam...")
camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    print("[ERRO] Não foi possível acessar a webcam.")
    exit()

print("[INFO] Pressione ESPAÇO para capturar o rosto.")
print("[INFO] Pressione Q para sair.")

while True:
    ret, frame = camera.read()

    if not ret:
        print("[ERRO] Falha ao capturar frame.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(frame_rgb)

    for top, right, bottom, left in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.putText(frame, f"Faces detectadas: {len(face_locations)}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Cadastro Facial - LAB01", frame)

    tecla = cv2.waitKey(1) & 0xFF

    if tecla == ord("q"):
        print("[INFO] Saindo...")
        break

    if tecla == 32:
        if len(face_locations) != 1:
            print("[ERRO] Capture com exatamente 1 rosto visível.")
            continue

        encodings = face_recognition.face_encodings(frame_rgb, face_locations)

        if len(encodings) == 0:
            print("[ERRO] Não foi possível gerar encoding facial.")
            continue

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        img_path = CAPTURAS_DIR / f"{NOME_USUARIO}_lab01_{timestamp}.jpg"
        encoding_path = DATASET_DIR / f"{NOME_USUARIO}_encoding_lab01.pkl"

        cv2.imwrite(str(img_path), frame)

        with open(encoding_path, "wb") as f:
            pickle.dump({
                "nome": NOME_USUARIO,
                "encoding": encodings[0],
                "data_cadastro": timestamp,
                "imagem": str(img_path)
            }, f)

        print("[SUCESSO] Rosto cadastrado.")
        print(f"[INFO] Imagem salva em: {img_path}")
        print(f"[INFO] Encoding salvo em: {encoding_path}")
        break

camera.release()
cv2.destroyAllWindows()