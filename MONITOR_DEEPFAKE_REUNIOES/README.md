# Monitor de Verificacao de Reunioes e Entrevistas

Aplicacao desktop criada como prova de conceito para apoiar a deteccao de
personificacao por deepfake em reunioes, entrevistas remotas e chamadas de
video. O monitor funciona em paralelo a plataformas como Google Meet, Zoom,
Microsoft Teams e Discord, sem depender de extensao ou integracao oficial com
essas plataformas.

A ideia central e transformar uma chamada comum, que normalmente nao possui
autenticacao biometrica continua, em um processo de observacao periodica da
identidade facial, dos sinais de manipulacao e da qualidade da captura.

## Objetivo da ferramenta

O sistema foi desenvolvido a partir dos quatro LABs de autenticacao facial. Nos
LABs, a deepfake testada nao conseguiu superar a verificacao biometrica baseada
em embeddings, mas conseguiu apresentar movimentos naturais e responder a
alguns desafios de liveness. Isso motivou a construcao de uma ferramenta para o
cenario real de risco: chamadas de video em que a pessoa nao passa por uma
autenticacao facial formal.

O monitor nao produz uma acusacao automatica de fraude. Ele gera um parecer de
risco baseado em multiplas evidencias, para apoiar a revisao do recrutador ou
responsavel pela reuniao.

## Fluxo de uso

1. O recrutador cadastra um candidato a partir de um video real de referencia,
   preferencialmente com 10 a 15 segundos.
2. O sistema extrai amostras faciais validas, embeddings e uma baseline do
   detector Effort.
3. O recrutador escolhe o candidato ativo na tela principal.
4. Ao iniciar a chamada, o recrutador seleciona a janela da videoconferencia ou
   uma regiao especifica do participante.
5. A cada periodo de analise, o monitor coleta amostras faciais e calcula as
   metricas de identidade, manipulacao, liveness, qualidade e risco.
6. O recrutador encerra a sessao quando desejar.
7. A aplicacao salva a conclusao geral e o historico dos periodos analisados.

## Arquitetura resumida

```text
Video de referencia do candidato
        |
        |-- validacao de qualidade
        |-- deteccao facial
        |-- embeddings InsightFace
        |-- baseline Effort
        v
Perfil local do candidato

Janela ou regiao da chamada
        |
        |-- captura de tela
        |-- deteccao do rosto principal
        |-- avaliacao de qualidade
        |-- verificacao de identidade
        |-- liveness passivo
        |-- Effort + DFDC
        |-- fusao de risco
        v
Conclusao por periodo e conclusao final
```

## Ferramentas e modelos utilizados

### Interface e captura

- `Tkinter`: interface desktop principal, dialogos e telas de selecao.
- `ctypes` com APIs do Windows: listagem, ativacao e leitura da area das
  janelas abertas.
- `Pillow ImageGrab`: captura local da janela ou regiao selecionada.
- `OpenCV`: manipulacao dos frames, recortes faciais, nitidez, brilho e
  diferencas temporais.
- `NumPy`: calculos vetoriais, distancias, medianas, medias e normalizacoes.

### Deteccao e identidade facial

- `InsightFace buffalo_l`: deteccao facial, embeddings biometricos e landmarks.
- Embeddings faciais: vetores numericos usados para comparar a face observada
  com a referencia cadastrada.
- Landmarks faciais: pontos usados para liveness passivo, movimento e
  alinhamento facial.

### Detectores de manipulacao facial

- `Effort CLIP-L/14 (ICML 2025)`: detector principal de manipulacao facial.
- `DFDC EfficientNet-B7 (selimsef)`: detector auxiliar e fallback em ONNX.
- `ONNX Runtime`: execucao local do modelo DFDC exportado para ONNX.
- Processo isolado do Effort: o modelo pesado roda em um ambiente separado para
  reduzir conflito com o monitor principal.

Arquivos atualmente esperados em `models/`:

- `dfdc_efficientnet_b7.onnx`
- `detector.json`
- `effort_faceforensics.pth`
- `effort.json`

O Effort e usado em contexto academico, pois o modelo indicado no projeto esta
sob licenca `CC BY-NC 4.0`. O DFDC EfficientNet-B7 usado como fallback vem do
repositorio `selimsef/dfdc_deepfake_challenge`, sob licenca MIT.

## Metricas utilizadas

### 1. Qualidade da captura

Registrada em `metricas.csv`.

- `face_detectada`: indica se uma face foi encontrada no quadro.
- `qualidade_adequada`: indica se a amostra passou nos filtros minimos.
- `confianca_deteccao`: confianca retornada pelo detector facial.
- `tamanho_face`: menor dimensao da caixa facial.
- `brilho`: media de intensidade do recorte facial.
- `nitidez`: variancia do Laplaciano no recorte facial.
- `largura_frame` e `altura_frame`: dimensoes da area analisada.

Parametros principais em `config/monitor.json`:

```json
{
  "minimum_face_size": 80,
  "minimum_detection_score": 0.5,
  "minimum_brightness": 35.0,
  "maximum_brightness": 220.0,
  "monitor_minimum_sharpness": 4.0
}
```

Essas metricas nao indicam deepfake diretamente. Elas informam se o quadro tem
qualidade suficiente para que as demais analises sejam confiaveis.

### 2. Identidade facial

Registrada em `avaliacoes.csv`.

- `distancia_identidade`: distancia L2 entre o embedding observado e a
  referencia mediana do candidato.
- `identidade_compativel`: resultado booleano da comparacao.
- `identity_distance_threshold`: limite usado para aceitar ou rejeitar a
  identidade. No projeto atual, o valor configurado e `20.21`, herdado da
  calibracao experimental dos LABs.

Formula geral:

```text
identidade_compativel = distancia_identidade <= 20.21
```

O risco de identidade e calculado proporcionalmente:

```text
risco_identidade = min(distancia_identidade / 20.21, 1) * 100
```

Se a distancia ultrapassa o threshold, o risco instantaneo nao pode ficar abaixo
de 75, mesmo que as outras camadas apresentem valores menores.

### 3. Manipulacao facial

Registrada em `avaliacoes.csv`.

- `probabilidade_deepfake`: maior probabilidade disponivel entre os detectores.
- `modelo_detector`: modelos usados no periodo.
- `probabilidade_effort`: saida absoluta do Effort.
- `baseline_effort`: valor medido no video real de referencia do candidato.
- `delta_effort`: diferenca entre chamada e referencia.
- `probabilidade_calibrada`: valor usado pelo motor de risco.

O Effort nao e interpretado somente pela porcentagem absoluta. Durante o
cadastro por video, a propria referencia real do candidato gera uma baseline.
Na chamada, o sistema calcula:

```text
delta_effort = probabilidade_effort_atual - baseline_effort
```

Configuracao atual:

```json
{
  "effort_delta_warning": 0.015,
  "effort_delta_alert": 0.03
}
```

Ou seja:

- delta abaixo de 1,5 ponto percentual: pouca evidencia pelo Effort;
- delta entre 1,5 e 3 pontos percentuais: zona de atencao;
- delta acima de 3 pontos percentuais: forte aumento relativo em relacao ao
  video real de referencia.

Quando a baseline Effort existe, a probabilidade calibrada considera esse
delta. Quando a baseline nao existe, o sistema usa o DFDC como fallback.

### 4. Liveness passivo

Registrado em `avaliacoes.csv`.

- `variacao_temporal`: diferenca media entre recortes faciais consecutivos.
- `movimento_landmarks`: deslocamento medio dos landmarks faciais normalizados.
- `liveness_valido`: indica se houve movimento facial dentro da faixa esperada.
- `confianca_analise`: confianca derivada da qualidade da janela analisada.

Parametros atuais:

```json
{
  "minimum_temporal_variation": 1.0,
  "minimum_landmark_movement": 0.0005,
  "maximum_landmark_movement": 0.08
}
```

O liveness passivo nao prova autenticidade. Ele apenas indica que houve
atividade facial observavel. Os LABs mostraram que uma deepfake em tempo real
pode reproduzir movimentos naturais do operador.

### 5. Risco instantaneo e acumulado

Registrado em `avaliacoes.csv`.

- `risco_instantaneo`: risco do periodo atual.
- `risco_acumulado`: media ponderada dos ultimos periodos, dando mais peso aos
  periodos recentes.
- `nivel_risco`: `BAIXO`, `MODERADO` ou `ALTO`.
- `anomalias_consecutivas`: quantidade de periodos recentes com risco de
  atencao.

Quando o detector de manipulacao esta disponivel, a fusao usa:

```text
risco = 45% identidade + 40% manipulacao + 15% liveness
```

Quando nao ha detector disponivel:

```text
risco = 70% identidade + 30% liveness
```

A qualidade da amostra ajusta a confianca do risco:

```text
risco_final = risco * (0.65 + 0.35 * confianca_qualidade)
```

Limiar atual:

```json
{
  "risk_alert_threshold": 65.0,
  "risk_high_threshold": 80.0
}
```

Interpretacao:

- abaixo de 65: `BAIXO`;
- risco acumulado a partir de 65: `MODERADO`;
- risco acumulado a partir de 80 ou tres anomalias consecutivas: `ALTO`.

## Periodos de analise

A configuracao atual usa:

```json
{
  "sample_interval_seconds": 30.0,
  "analysis_window_samples": 8,
  "target_capture_fps": 12.0
}
```

Na pratica, o monitor observa a chamada continuamente, seleciona amostras
faciais validas e produz uma conclusao parcial por periodo. Periodos com poucos
quadros faciais validos ficam inconclusivos, em vez de gerar uma acusacao.

## Cadastro por video de referencia

O cadastro recomendado e um video curto enviado previamente pelo candidato. O
video deve conter:

- apenas um rosto;
- boa iluminacao;
- rosto proximo da camera;
- visao frontal e leves movimentos laterais;
- piscada e movimento natural;
- ausencia de filtros ou edicoes aparentes.

O sistema extrai embeddings de varios quadros e calcula a baseline do detector
de manipulacao. O video original nao e copiado para o perfil. O perfil armazena:

- `embeddings.npy`: embeddings faciais de referencia;
- `profile.json`: metadados do cadastro;
- quantidade de amostras validas;
- metodo de cadastro;
- probabilidade de manipulacao observada no video de referencia;
- baseline Effort, quando disponivel.

Tambem e possivel cadastrar por fotografias:

```powershell
..\LAB_DEEPFAKE\teste_nivel_2\venv\Scripts\python.exe register_identity.py --name Guilherme --images foto1.jpg foto2.jpg foto3.jpg
```

Para calibrar o Effort, o cadastro por video e preferivel.

## Execucao

Ambiente usado nos LABs:

```powershell
cd C:\Users\guisi\DeepFake\MONITOR_DEEPFAKE_REUNIOES
..\LAB_DEEPFAKE\teste_nivel_2\venv\Scripts\python.exe app.py
```

Monitor direto pela linha de comando:

```powershell
..\LAB_DEEPFAKE\teste_nivel_2\venv\Scripts\python.exe run_monitor.py
```

Cadastro por video:

```powershell
..\LAB_DEEPFAKE\teste_nivel_2\venv\Scripts\python.exe register_identity.py --name Guilherme --video C:\caminho\candidato.mp4
```

Selecionar manualmente a regiao do participante:

```powershell
..\LAB_DEEPFAKE\teste_nivel_2\venv\Scripts\python.exe run_monitor.py --manual-region
```

Repetir experimento com regiao fixa:

```powershell
..\LAB_DEEPFAKE\teste_nivel_2\venv\Scripts\python.exe run_monitor.py --region 100 100 1280 720
```

Os valores de `--region` representam:

```text
LEFT TOP WIDTH HEIGHT
```

## Selecionar janela ou regiao

O Windows identifica janelas, nao abas individuais do navegador. Portanto, no
Chrome, a aba do Meet deve estar ativa dentro da janela selecionada.

Modos disponiveis:

- `Acompanhar a janela inteira`: indicado quando o participante aparece grande
  e a janela nao possui muitos elementos interferindo.
- `Marcar somente o candidato`: indicado quando ha varios participantes ou
  quando o rosto ocupa uma pequena parte da tela.

O candidato nao precisa seguir comandos do monitor. A ferramenta apenas observa
o video da chamada. O recrutador encerra a sessao pelo painel ou com `F9`.

## Saidas geradas

Cada sessao cria uma pasta em:

```text
outputs/sessao_AAAAMMDD_HHMMSS/
```

Arquivos principais:

- `session.json`: informacoes da sessao e do perfil usado.
- `metricas.csv`: qualidade das capturas e dimensoes da area analisada.
- `embeddings.jsonl`: embeddings faciais observados, quando habilitado.
- `avaliacoes.csv`: identidade, detector, liveness, risco e nivel de atencao.
- `periodos.jsonl`: periodos completos ou inconclusivos.
- `conclusao.json`: parecer final da sessao.

Nenhum frame ou video da chamada e salvo por padrao.

## Avaliacao das sessoes

Abrir historico pela interface principal ou executar:

```powershell
..\LAB_DEEPFAKE\teste_nivel_2\venv\Scripts\python.exe evaluate_sessions.py --open-dialog
```

A avaliacao resume:

- porcentagem de periodos com identidade compativel;
- porcentagem de liveness valido;
- media de probabilidade de manipulacao;
- media calibrada quando existe baseline;
- maior risco acumulado observado;
- quantidade de periodos moderados ou altos;
- conclusao automatica da sessao.

## Instalacao do detector DFDC

O instalador baixa a implementacao do DFDC e exporta o EfficientNet-B7 para
ONNX:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\setup_dfdc_detector.ps1
```

O Effort e configurado separadamente em `models/effort.json` e usa os arquivos
externos indicados em `docs/EFFORT.md`.

## Testes

```powershell
..\LAB_DEEPFAKE\teste_nivel_2\venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Limitacoes

- A ferramenta e uma prova de conceito academica.
- O detector fornece probabilidade, nao prova definitiva de fraude.
- O Effort precisa de baseline por candidato para ser interpretado com mais
  seguranca.
- Compressao, iluminacao, baixa nitidez e rosto pequeno podem tornar periodos
  inconclusivos.
- Liveness passivo confirma atividade facial, mas nao autenticidade.
- O documento segurado pelo candidato no video nao e validado juridicamente.
- A decisao final deve permanecer com o operador humano.

## Contribuicao do projeto

A contribuicao nao esta em treinar um detector do zero. O projeto integra
modelos e metricas ja consolidados em um fluxo aplicado a videoconferencias:

- cadastro previo por video;
- verificacao continua;
- comparacao de identidade por embeddings;
- detector moderno Effort com baseline individual;
- fallback DFDC em ONNX;
- liveness passivo;
- filtros de qualidade;
- fusao temporal de risco;
- conclusao final auditavel por sessao.
