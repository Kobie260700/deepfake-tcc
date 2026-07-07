# LAB04 - Liveness ativo (Challenge-Response)

O LAB04 combina a identidade facial herdada do LAB02 com uma prova de vida ativa. A cada tentativa, o sistema sorteia uma sequência imprevisível de quatro ações:

- virar a cabeça para a esquerda;
- virar a cabeça para a direita;
- retornar ao centro.
- piscar os olhos.

A piscada é detectada pelo Eye Aspect Ratio (EAR), calculado a partir dos landmarks das pálpebras fornecidos pelo modelo de 68 pontos do InsightFace. O ciclo completo olho aberto, fechado e novamente aberto deve ser detectado.

A autenticação é aceita somente quando a identidade estiver abaixo do threshold `20.210` e todos os desafios forem concluídos dentro do tempo limite.

## Cadastro

```cmd
cd /d C:\Users\guisi\DeepFake\LAB_DEEPFAKE\teste_nivel_4
..\teste_nivel_2\venv\Scripts\python.exe cadastro_embedding_lab04.py
```

## Autenticação

```cmd
..\teste_nivel_2\venv\Scripts\python.exe autenticacao_liveness_ativo.py
```

- `CAMERA_INDEX = 0`: webcam física.
- `CAMERA_INDEX = 1`: OBS Virtual Camera / DeepLiveCam.
- `ESPAÇO`: inicia uma tentativa.
- `Q` ou `ESC`: encerra.

O retângulo facial foi substituído por contornos verdes da mandíbula, sobrancelhas, olhos, nariz e boca, obtidos pelos landmarks faciais do InsightFace.

Os limites de orientação e de EAR são parâmetros iniciais e devem ser validados com testes reais antes das 20 execuções experimentais.
