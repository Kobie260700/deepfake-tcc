# DeepFake - Laboratorios e Monitor de Reunioes

Repositorio academico para estudar ataques de deepfake em autenticacao facial e
propor uma contramedida aplicada a reunioes e entrevistas remotas.

O projeto foi desenvolvido em etapas:

1. **LAB01:** reconhecimento facial convencional com `face_recognition`.
2. **LAB02:** embeddings profundos com `InsightFace buffalo_l`.
3. **LAB03:** identidade facial combinada com liveness passivo.
4. **LAB04:** identidade facial combinada com challenge-response ativo.
5. **Monitor de reunioes:** aplicacao desktop para verificacao continua em
   Meet, Zoom, Teams, Discord ou outras plataformas de videoconferencia.
6. **Validacao morfologica:** extracao de proporcoes faciais para orientar a
   criacao controlada de uma identidade sintetica experimental.

## Estrutura

```text
DeepFake/
|-- LAB_DEEPFAKE/                    # Laboratorios incrementais de autenticacao
|-- MONITOR_DEEPFAKE_REUNIOES/        # Ferramenta final de mitigacao
|-- VALIDACAO_SEMELHANCA_MORFOLOGICA/ # Extracao de metricas morfologicas
|-- Deep-Live-Cam/                    # Ferramenta usada no cenario ofensivo
|-- Comandos/                         # Anotacoes de execucao
|-- ROTEIRO_4_LABS_E_MITIGACAO.md     # Sintese tecnica dos LABs e mitigacao
`-- ANALISE_INTEGRADA_PROJETO_TCC_APRESENTACAO.md
```

## Monitor de reunioes

A ferramenta principal fica em:

```text
MONITOR_DEEPFAKE_REUNIOES/
```

Ela executa em paralelo a uma chamada de video e combina:

- verificacao de identidade por embeddings faciais;
- detector Effort CLIP-L/14 com baseline por candidato;
- detector DFDC EfficientNet-B7 como fallback;
- liveness passivo por variacao temporal e landmarks;
- filtros de qualidade de imagem;
- risco instantaneo e acumulado por sessao.

Leia o README especifico:

```text
MONITOR_DEEPFAKE_REUNIOES/README.md
```

## O que nao esta versionado

Por seguranca, privacidade e limite de tamanho do GitHub, o `.gitignore` remove:

- ambientes virtuais (`venv`, `.venv`);
- caches Python;
- pesos de modelos (`.onnx`, `.pth`, `.safetensors`, etc.);
- outputs de sessoes;
- perfis biometricos;
- embeddings/encodings `.pkl` dos LABs;
- imagens, videos e PDFs locais;
- resultados morfologicos pessoais.

Os arquivos pequenos de configuracao dos detectores do monitor sao mantidos:

```text
MONITOR_DEEPFAKE_REUNIOES/models/detector.json
MONITOR_DEEPFAKE_REUNIOES/models/effort.json
```

Os pesos dos modelos precisam ser obtidos/configurados localmente conforme a
documentacao do monitor.

## Execucao do monitor

Ambiente usado durante os LABs:

```powershell
cd C:\Users\guisi\DeepFake\MONITOR_DEEPFAKE_REUNIOES
..\LAB_DEEPFAKE\teste_nivel_2\venv\Scripts\python.exe app.py
```

Testes:

```powershell
..\LAB_DEEPFAKE\teste_nivel_2\venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Observacao etica

Este projeto tem finalidade academica e defensiva. A ferramenta de monitoramento
serve como apoio a decisao humana e nao deve ser tratada como prova automatica
de fraude. Detectores de deepfake produzem evidencias probabilisticas, que
dependem de qualidade de imagem, compressao, iluminacao e calibracao.
