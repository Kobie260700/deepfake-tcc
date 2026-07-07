# Extracao de metricas morfologicas faciais

Esta ferramenta recebe uma ou mais imagens/videos de um unico rosto e gera uma
ficha de proporcoes para orientar a criacao de uma identidade sintetica
estruturalmente semelhante, mas biometricamente distinta.

Ela nao compara dois rostos e nao exporta embeddings biometricos.

## Execucao

```powershell
cd C:\Users\guisi\DeepFake\VALIDACAO_SEMELHANCA_MORFOLOGICA
& ..\LAB_DEEPFAKE\teste_nivel_2\venv\Scripts\python.exe app.py
```

Uma unica imagem frontal e aceita. Para maior estabilidade, use tambem um video
curto ou varias imagens com iluminacao uniforme e leves mudancas de posicao.

Pela linha de comando:

```powershell
& ..\LAB_DEEPFAKE\teste_nivel_2\venv\Scripts\python.exe analisar.py `
  --entrada "C:\caminho\foto.jpg" "C:\caminho\video.mp4" `
  --saida resultados
```

## Arquivos produzidos

- `FICHA_MORFOLOGICA.md`: medidas e interpretacao metodologica;
- `PROMPT_GERACAO.txt`: texto para orientar uma IA de geracao de imagem;
- `proporcoes_faciais.csv`: medias, medianas e desvios;
- `relatorio_completo.json`: landmarks normalizados e metadados;
- `amostras.csv`: qualidade das amostras processadas.

As proporcoes nao reproduzem a identidade por si mesmas. O prompt exige uma
pessoa ficticia e distinta, preservando apenas uma aproximacao estrutural para
o experimento academico.
