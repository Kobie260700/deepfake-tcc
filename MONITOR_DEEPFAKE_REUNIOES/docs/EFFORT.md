# Detector Effort

O monitor usa o Effort (ICML 2025) como detector principal e preserva o
EfficientNet-B7 DFDC como fallback. A probabilidade mais alta entre os modelos
disponiveis e usada de forma conservadora na avaliacao de risco.

## Arquivos externos

- Repositorio: `.detector-build/Effort-AIGI-Detection`
- Ambiente: `.detector-build/effort-venv`
- CLIP-L/14: `.detector-build/models--openai--clip-vit-large-patch14`
- Checkpoint facial: `models/effort_faceforensics.pth`

O checkpoint oficial e distribuido pelos autores em:
https://drive.google.com/file/d/1m4fyJecABU-Yk3bJ4b1WhUwQa0xCkMLI/view

O modelo e iniciado em segundo plano ao abrir o monitor. Em cada periodo ele
analisa quatro amostras distribuidas entre os oito recortes coletados, evitando
picos desnecessarios de GPU sem concentrar a decisao em um unico instante.

## Calibracao por candidato

O Effort recebe rostos alinhados pelos olhos, nariz e boca, seguindo o protocolo
do DeepfakeBench. Durante o cadastro por video, o resultado do arquivo real e
salvo como baseline do candidato. Na reuniao, o sistema registra:

- probabilidade absoluta do Effort;
- baseline do video de referencia;
- delta entre reuniao e referencia;
- probabilidade calibrada usada no risco.

O valor absoluto permanece disponivel para pesquisa, mas nao decide sozinho.
Sem baseline, o risco usa o DFDC como fallback. Perfis criados antes desta
mudanca devem receber novamente o video real de referencia.

O modelo e publicado sob CC BY-NC 4.0 e deve ser usado apenas no contexto
academico deste projeto, salvo obtencao de permissao comercial adicional.

Se qualquer artefato estiver ausente, o monitor continua funcionando com o
detector DFDC e registra o motivo da indisponibilidade do Effort.
