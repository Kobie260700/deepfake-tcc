# Roteiro do Monitor de Deepfakes em Videoconferencias

## 1. Objetivo

Desenvolver uma aplicacao desktop executada em paralelo a plataformas de
videoconferencia, como Google Meet, Zoom, Teams e Discord, para monitorar
continuamente a identidade facial de um participante e procurar indicios de
manipulacao por deepfake ou face swap.

A aplicacao sera uma ferramenta de apoio a decisao. Ela apresentara niveis de
risco e evidencias, sem afirmar de forma absoluta que uma pessoa cometeu fraude.

## 2. Pergunta de pesquisa

Uma arquitetura que combina verificacao continua de identidade, deteccao de
manipulacao facial e sinais de liveness consegue identificar ataques de face
swap em videoconferencias com baixa taxa de falsos alertas?

## 3. Escopo do MVP

O primeiro prototipo devera:

1. Cadastrar uma referencia facial previamente validada.
2. Permitir selecionar uma janela ou regiao da videoconferencia.
3. Detectar e acompanhar um rosto nessa regiao.
4. Recalcular a identidade facial periodicamente.
5. Executar um detector open-source de manipulacao facial.
6. Combinar os resultados em uma pontuacao de risco.
7. Exibir o estado atual e registrar as medicoes da sessao.

O MVP nao realizara pericia de documentos, KYC, integracao oficial com todas as
plataformas nem acusacao automatica de fraude.

## 4. Arquitetura planejada

```text
Referencia facial validada
           |
           v
Videoconferencia -> Captura da janela/regiao
                           |
                           v
                  Deteccao e rastreamento
                           |
              +------------+-------------+
              |                          |
              v                          v
     Verificacao de identidade   Detector de manipulacao
              |                          |
              +------------+-------------+
                           |
                           v
                 Analise temporal/liveness
                           |
                           v
                  Motor de pontuacao de risco
                           |
                           v
                   Interface, alertas e logs
```

## 5. Caminho de desenvolvimento

### Fase 1 - Definicao do cenario experimental

- Definir o Google Meet como primeira plataforma de teste.
- Usar o DeepLiveCam como ataque inicial de face swap.
- Definir quais participantes e imagens serao usados com consentimento.
- Definir resolucao, iluminacao, camera e duracao padrao das sessoes.
- Separar os cenarios real e manipulado.

**Criterio de conclusao:** protocolo experimental documentado e repetivel.

### Fase 2 - Captura da videoconferencia

- Criar captura de janela ou regiao selecionada pelo usuario.
- Manter taxa de captura controlada para nao sobrecarregar o computador.
- Registrar resolucao, FPS e perdas de captura.
- Permitir visualizar a regiao que esta sendo analisada.

**Caminho principal:** captura local da tela.

**Caminho futuro:** integracao oficial com APIs de plataformas que oferecam
acesso autorizado ao fluxo de midia.

**Criterio de conclusao:** capturar uma reuniao por pelo menos dez minutos sem
interromper a chamada.

### Fase 3 - Deteccao e rastreamento facial

- Detectar todos os rostos visiveis.
- Permitir que o operador selecione o participante monitorado.
- Manter o mesmo participante associado ao rastreamento.
- Detectar perda, troca ou reaparecimento do rosto.
- Avaliar qualidade minima: tamanho, nitidez e angulo da face.

**Criterio de conclusao:** acompanhar o participante mesmo com pequenas mudancas
de posicao e layout.

### Fase 4 - Verificacao continua de identidade

- Utilizar InsightFace para gerar embeddings faciais.
- Criar uma referencia com mais de uma amostra valida, quando possivel.
- Em periodos configurados, atualmente 30 segundos, coletar uma pequena janela de frames.
- Gerar um embedding representativo usando apenas frames de boa qualidade.
- Comparar o resultado com a referencia cadastrada.
- Registrar distancia media, dispersao e mudancas abruptas.

```text
A cada periodo configurado
        |
        v
Coletar varios frames validos
        |
        v
Calcular embeddings
        |
        v
Remover amostras ruins e obter valor representativo
        |
        v
Comparar com o cadastro e atualizar o risco
```

**Criterio de conclusao:** detectar troca de pessoa e alteracoes relevantes de
identidade sem reagir a um unico frame ruim.

### Fase 5 - Selecao do detector de deepfake

- Nao sera treinado um detector de deepfake do zero.
- O projeto utilizara um detector pre-treinado, open-source e validado na
  literatura, concentrando os esforcos na integracao, adaptacao e avaliacao em
  videoconferencias.
- Levantar detectores open-source consolidados.
- Usar, se conveniente, o DeepfakeBench como ambiente de comparacao.
- Nao definir antecipadamente um unico modelo.
- Avaliar modelos espaciais e temporais.
- Testar com imagens originais e com imagens submetidas a compressao,
  redimensionamento e baixa resolucao.

Os criterios de selecao serao:

- capacidade de detectar face swap em tempo real;
- generalizacao para videos nao vistos no treinamento;
- falsos positivos em pessoas reais;
- desempenho sob compressao de videoconferencia;
- tempo de inferencia;
- consumo de CPU, GPU e memoria;
- disponibilidade de codigo e pesos;
- licenca compativel com o projeto.

**Criterio de conclusao:** escolher o detector com melhor equilibrio entre
qualidade e custo computacional, justificando a decisao com resultados.

### Fase 6 - Adaptacao ao contexto de reunioes

- Processar sequencias de frames, e nao decisoes isoladas.
- Normalizar recortes faciais antes da inferencia.
- Avaliar o efeito da interface, compressao e baixa taxa de quadros.
- Definir agregacao temporal das probabilidades.
- Considerar ajuste fino apenas depois de medir o modelo original.
- Criar um conjunto de dados proprio de videoconferencias controladas.

**Criterio de conclusao:** o detector deve produzir uma saida estavel durante
uma sessao e identificar ataques sem depender de um unico artefato visual.

### Fase 7 - Liveness como evidencia complementar

- Reaproveitar conceitos dos LAB03 e LAB04.
- Medir movimento de landmarks, piscadas e orientacao da cabeca.
- Nao considerar esses sinais prova definitiva de autenticidade.
- Solicitar challenge-response somente quando necessario ou durante testes
  controlados.

**Criterio de conclusao:** integrar liveness ao risco sem permitir que ele
anule sozinho as demais evidencias.

### Fase 8 - Motor de risco

Manter tres saidas separadas:

1. **Risco de identidade:** incompatibilidade com a referencia cadastrada.
2. **Risco de manipulacao:** probabilidade produzida pelo detector.
3. **Risco de liveness:** ausencia ou inconsistencia dos sinais esperados.

Depois, calcular:

- risco instantaneo da janela atual;
- risco acumulado da sessao;
- confianca da analise, baseada na qualidade dos frames;
- quantidade de anomalias consecutivas.

O sistema devera exigir persistencia temporal antes de emitir alerta alto.

Exemplo de saida:

```text
Identidade: compativel
Manipulacao facial: 78%
Liveness: inconclusivo
Anomalias consecutivas: 4
Qualidade da captura: adequada
Risco acumulado: alto
```

**Criterio de conclusao:** regras de decisao documentadas, reproduziveis e
calibradas em dados separados dos usados para avaliacao final.

### Fase 9 - Interface e registros

- Mostrar participante monitorado e estado da captura.
- Exibir identidade, manipulacao, liveness, qualidade e risco.
- Apresentar historico do risco em uma linha do tempo.
- Permitir iniciar, pausar e encerrar uma sessao.
- Salvar metricas em CSV ou banco local.
- Evitar armazenar frames e documentos por padrao.

**Criterio de conclusao:** um operador consegue executar todo o experimento sem
alterar o codigo-fonte.

### Fase 10 - Avaliacao experimental

Testar inicialmente:

- pessoa real durante toda a chamada;
- DeepLiveCam ativo desde o inicio;
- DeepLiveCam ativado durante a chamada;
- troca de participante;
- ausencia temporaria de rosto;
- baixa iluminacao;
- baixa resolucao e compressao;
- movimentos rapidos;
- oclusao parcial;
- mais de um participante na tela.

Medir:

- precisao, recall e F1-score;
- taxa de falsos positivos e falsos negativos;
- ROC-AUC, quando aplicavel;
- tempo ate a deteccao;
- estabilidade do risco;
- FPS e latencia;
- uso de CPU, GPU e memoria;
- desempenho por pessoa e por condicao.

**Criterio de conclusao:** resultados obtidos com pessoas e sessoes que nao
foram usadas para calibrar thresholds ou ajustar o detector.

## 6. Ordem recomendada de implementacao

```text
1. Captura da tela
2. Deteccao e selecao do rosto
3. Cadastro e comparacao de identidade
4. Logs e visualizacao das metricas
5. Avaliacao de detectores open-source
6. Integracao do detector escolhido
7. Agregacao temporal e motor de risco
8. Liveness complementar
9. Interface final
10. Experimentos e documentacao
```

Nao iniciaremos pela interface completa nem pelo treinamento de uma rede. O
primeiro objetivo e construir um pipeline mensuravel e confirmar cada camada
isoladamente.

## 7. Estrutura prevista do projeto

```text
MONITOR_DEEPFAKE_REUNIOES/
|-- README.md
|-- ROTEIRO_PROJETO.md
|-- requirements.txt
|-- config/
|-- src/
|   |-- capture/
|   |-- face_tracking/
|   |-- identity/
|   |-- deepfake_detection/
|   |-- liveness/
|   |-- risk/
|   |-- ui/
|   `-- storage/
|-- scripts/
|-- tests/
|-- experiments/
|-- results/
`-- docs/
```

Essa estrutura sera criada gradualmente. Pastas sem implementacao real nao
precisam ser adicionadas antecipadamente.

## 8. Cuidados metodologicos

- Separar verificacao de identidade de deteccao de deepfake.
- Nao usar os mesmos dados para calibracao e avaliacao final.
- Nao concluir que liveness simples prova autenticidade.
- Documentar versoes, pesos, datasets e licencas dos modelos reutilizados.
- Manter explicito que o detector e reutilizado; a contribuicao do projeto esta
  na selecao experimental, integracao, adaptacao ao dominio de
  videoconferencias, autenticacao continua e avaliacao do sistema combinado.
- Comparar o sistema combinado com cada camada isolada.
- Reportar falhas e condicoes em que a analise fica inconclusiva.
- Obter consentimento para coleta e processamento de dados biometricos.
- Armazenar preferencialmente embeddings e metricas, nao videos brutos.

## 9. Marcos do projeto

### Marco 1 - Monitor basico

Captura uma regiao, acompanha um rosto e registra embeddings.

**Estado:** implementado em 14 de junho de 2026. A captura interativa ainda
precisa ser validada manualmente em uma sessao desktop, pois o ambiente de
execucao automatizada nao possui acesso a tela do Windows.

### Marco 2 - Autenticacao continua

Compara a identidade em periodos configurados e detecta troca de participante.

**Estado:** implementado. Utiliza janelas de embeddings e perfil supervisionado.

### Marco 3 - Detector integrado

Executa o detector selecionado em janelas temporais de videoconferencia.

**Estado:** implementado com o Effort CLIP-L/14 como detector principal e o
EfficientNet-B7 treinado no DFDC como fallback local. O resultado do Effort e
calibrado contra o video de referencia do candidato, reduzindo falsas leituras
causadas por compressao, camera e iluminacao.

### Marco 4 - Pontuacao de risco

Combina identidade, manipulacao, liveness passivo e qualidade da captura. O
challenge-response permanece como recurso experimental/sob demanda para testes
controlados, nao como exigencia padrao para o candidato durante a reuniao.

**Estado:** implementado com risco instantaneo, acumulado, persistencia de
anomalias e challenge-response sob demanda.

### Marco 5 - Validacao experimental

Executa o protocolo completo e produz tabelas, graficos e conclusoes do TCC.

**Estado:** infraestrutura implementada. O protocolo, a rotulagem e o resumo de
sessoes estao prontos; a coleta com participantes e DeepLiveCam deve ser
executada em sessoes reais de videoconferencia.

## 10. Primeira tarefa tecnica

Construir um prototipo pequeno que:

1. permita selecionar uma regiao da tela;
2. capture frames dessa regiao;
3. encontre o principal rosto;
4. desenhe o contorno facial;
5. mostre FPS e qualidade do recorte;
6. registre amostras periodicas conforme a configuracao do monitor.

Esse prototipo validara a base do monitor antes da escolha e integracao do
detector de deepfake.
