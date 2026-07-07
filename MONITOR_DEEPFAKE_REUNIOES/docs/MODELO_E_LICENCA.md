# Detectores Reutilizados

## Detector principal atual

- Projeto: `YZY-stack/Effort-AIGI-Detection`
- Modelo integrado: Effort CLIP-L/14
- Uso no projeto: detector principal de manipulacao facial
- Licenca informada no projeto original: CC BY-NC 4.0
- Observacao: uso restrito ao contexto academico deste prototipo, salvo
  permissao comercial adicional.

Repositorio:

https://github.com/YZY-stack/Effort-AIGI-Detection

O Effort e interpretado de forma relativa ao candidato quando existe baseline
obtida do video real de referencia. A probabilidade absoluta e registrada para
pesquisa, mas nao deve decidir isoladamente.

## Detector auxiliar/fallback

- Projeto: `selimsef/dfdc_deepfake_challenge`
- Arquitetura: EfficientNet-B7
- Origem: solucao premiada do Deepfake Detection Challenge
- Licenca do codigo: MIT
- Entrada: recorte facial RGB 380 x 380
- Saida: probabilidade de manipulacao por frame

Repositorio:

https://github.com/selimsef/dfdc_deepfake_challenge

## Justificativa

O Effort foi escolhido como detector principal por apresentar melhor resposta
nos testes exploratorios com DeepLiveCam apos alinhamento facial. O DFDC
EfficientNet-B7 foi mantido como fallback por ser uma solucao open-source
consolidada, exportavel para ONNX e executavel no mesmo pipeline local.

## Limitacoes

- Ambos os detectores foram treinados em dominios diferentes de Meet, Zoom,
  Teams e Discord.
- O Effort pode apresentar probabilidades absolutas elevadas mesmo em videos
  reais, por isso o projeto usa baseline por candidato.
- O DFDC pode aprender artefatos especificos do dataset original.
- Nenhum detector foi desenvolvido especificamente para DeepLiveCam.
- As probabilidades nao devem ser interpretadas como certeza de fraude.
- A decisao deve combinar identidade, manipulacao, liveness, qualidade e
  persistencia temporal.

## Uso no projeto

O Effort roda em processo Python isolado por ser mais pesado e ter dependencias
proprias. O checkpoint do DFDC e convertido para ONNX em ambiente isolado; a
aplicacao principal usa ONNX Runtime para executar esse fallback.

O DeepfakeBench podera ser usado posteriormente como ambiente de comparacao,
mas nao e uma dependencia obrigatoria do prototipo.
