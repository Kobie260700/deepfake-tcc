# LAB03 - Autenticacao facial com liveness passivo

O LAB03 combina duas verificacoes:

1. Identidade facial por embedding profundo com InsightFace `buffalo_l`.
2. Prova de vida passiva por variacao temporal da imagem e movimento dos landmarks faciais.

O teste nao solicita uma acao especifica. Durante uma janela de quatro segundos, o sistema observa se existe variacao facial natural suficiente e registra uma unica tentativa no CSV.

## Cadastro

```cmd
cd /d C:\Users\guisi\DeepFake\LAB_DEEPFAKE\teste_nivel_3
..\teste_nivel_2\venv\Scripts\python.exe cadastro_embedding_lab03.py
```

## Autenticacao

```cmd
..\teste_nivel_2\venv\Scripts\python.exe autenticacao_liveness_passivo.py
```

- `CAMERA_INDEX = 0`: webcam fisica.
- `CAMERA_INDEX = 1`: OBS Virtual Camera / DeepLiveCam.
- `ESPACO`: inicia uma tentativa controlada.
- `Q` ou `ESC`: encerra.

Os parametros de liveness sao iniciais e devem ser calibrados com os testes reais e DeepLiveCam antes da conclusao experimental.
