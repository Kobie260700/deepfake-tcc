# Arquitetura Implementada

```text
Aplicacao principal (app.py)
  |-- Cadastro por video enviado pelo candidato
  |     |-- imagens de referencia
  |     |-- InsightFace buffalo_l
  |     `-- perfil local com embeddings
  |
  |-- Monitoramento
  |     |-- selecao de janela/regiao
  |     |-- captura de tela
  |     |-- deteccao e qualidade facial
  |     |-- janela temporal de amostras
  |     |-- verificacao continua de identidade
  |     |-- Effort CLIP-L/14 como detector principal
  |     |-- DFDC EfficientNet-B7 em ONNX como fallback
  |     |-- baseline individual do Effort
  |     |-- liveness passivo
  |     |-- observacao continua de movimento facial
  |     `-- risco instantaneo e acumulado
  |
  `-- Avaliacao
        |-- logs por sessao
        |-- rotulagem real/fake
        `-- resumo e matriz de classificacao
```

## Limites de responsabilidade

- O cadastro pressupoe que as imagens foram validadas por processo externo ou
  operador autorizado.
- O sistema nao valida juridicamente documentos.
- O detector apresenta probabilidade, nao prova de fraude.
- O liveness e evidencia complementar e pode ser reproduzido por face swap.
- A decisao final continua humana.

## Persistencia

- `profiles/`: referencias biometricas locais.
- `models/`: configuracoes dos detectores e pesos locais nao versionados.
- `outputs/`: metricas e avaliacoes de cada sessao.
- `results/`: resumos experimentais.

Frames e videos nao sao armazenados por padrao.
