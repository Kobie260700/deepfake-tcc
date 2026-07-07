# Protocolo Experimental

## Objetivo

Avaliar separadamente e em conjunto:

1. verificacao continua de identidade;
2. detector neural de manipulacao facial;
3. liveness passivo;
4. challenge-response ativo;
5. motor de risco temporal.

## Participantes

- Utilizar participantes voluntarios com consentimento documentado.
- Separar participantes de calibracao e participantes de avaliacao final.
- Nao publicar imagens, documentos ou embeddings identificaveis.

## Cenarios minimos

| Codigo | Cenario |
|---|---|
| R1 | Pessoa real durante toda a reuniao |
| R2 | Pessoa real com baixa iluminacao |
| R3 | Pessoa real com baixa resolucao/compressao |
| R4 | Pessoa real com oclusao e movimentos rapidos |
| A1 | DeepLiveCam ativo desde o inicio |
| A2 | DeepLiveCam ativado no meio da reuniao |
| A3 | Troca fisica de participante |
| A4 | Reproducao de video previamente gravado |
| A5 | DeepLiveCam respondendo ao challenge |

## Plataformas

1. Google Meet como plataforma principal.
2. Zoom, Teams ou Discord como validacao de portabilidade.

Registrar plataforma, navegador/aplicativo, resolucao, FPS, camera, iluminacao,
hardware e versoes dos modelos.

## Repeticoes

- Minimo inicial: 10 sessoes por classe e condicao.
- Recomendado para avaliacao final: 20 ou mais sessoes por classe.
- Cada sessao deve durar pelo menos dois minutos.
- A ativacao tardia do ataque deve ocorrer em instante previamente definido.

## Separacao dos dados

- Calibracao: escolha de thresholds e pesos do risco.
- Validacao: verificacao intermediaria sem reajuste por tentativa.
- Teste final: participantes e sessoes nunca usados na calibracao.

## Metricas

### Classificacao

- precisao;
- recall/sensibilidade;
- especificidade;
- F1-score;
- ROC-AUC e PR-AUC;
- falso positivo por sessao;
- falso negativo por ataque.

### Operacao

- tempo ate detectar ataque ativado durante a reuniao;
- estabilidade do risco;
- FPS;
- latencia por janela de analise;
- CPU, GPU e memoria.

### Camadas isoladas

Executar ablation study comparando:

1. identidade apenas;
2. detector apenas;
3. liveness apenas;
4. identidade + detector;
5. identidade + detector + liveness;
6. sistema completo com risco temporal.

## Rotulagem

Cada sessao deve possuir um arquivo de metadados com:

- classe real ou manipulada;
- instante de ativacao do ataque;
- tipo de ataque;
- participante;
- condicao de captura;
- resultado esperado;
- observacoes de falhas operacionais.

## Criterio de alerta

Os thresholds nao devem ser escolhidos no conjunto final. O nivel alto deve
exigir persistencia temporal ou anomalias consecutivas, evitando que um unico
frame ruim produza acusacao.

## Interpretacao

O sistema apresenta risco e evidencias. Um alerta nao comprova fraude, e uma
classificacao baixa nao garante autenticidade. As limitacoes e casos
inconclusivos devem ser reportados junto com as metricas.

