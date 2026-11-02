# Contramedidas para ataques de falsificação biométrica em sistemas críticos

Pesquisa de conclusão de curso em Engenharia de Computação na UNIFEI sobre ataques de personificação facial com deepfakes em tempo real e contramedidas baseadas em reconhecimento facial e prova de vida. O projeto reúne quatro laboratórios incrementais e uma prova de conceito para acompanhamento de videoconferências.

**Autor:** Guilherme Silva Teixeira · **Orientador:** Otávio de Souza Martins Gomes · **Data de defesa informada no artigo:** 09/07/2026

**[Leia o artigo completo — TCC 02](documentacao/TCC_02.pdf)**

## Problema e objetivo

A aparência de um participante em uma chamada pode ser manipulada em tempo real para apoiar uma tentativa de personificação. Uma verificação facial pontual não resolve, por si só, o acompanhamento de uma sessão inteira. O trabalho investiga como diferentes camadas de autenticação se comportam diante de um ataque controlado e propõe combinar evidências para apoiar a análise de sessões remotas.

Os objetivos incluem comparar embeddings faciais convencionais e profundos, avaliar liveness passivo e ativo, calibrar limiares no cenário experimental e desenvolver uma prova de conceito de monitoramento contínuo. A contribuição da aplicação está na **integração de técnicas existentes**, não no treinamento de um novo modelo de IA.

## Metodologia experimental

Foi cadastrada uma referência facial legítima e utilizado o DeepLiveCam para simular a personificação com uma identidade sintética escolhida por semelhança morfológica. A etapa preparatória analisou 27 amostras válidas e descartou um quadro de baixa qualidade; as proporções faciais foram usadas para orientar a seleção, não como equivalentes a embeddings de identidade.

Cada laboratório foi avaliado com **20 simulações controladas: 10 com o usuário cadastrado e 10 com a identidade sintética**. Os limiares foram definidos a partir das métricas observadas nessas amostras. Assim, os resultados e limiares descritos abaixo são específicos do experimento e não representam desempenho universal nem validação independente.

| Laboratório | Camada investigada | Resultado relatado no artigo |
| --- | --- | --- |
| LAB01 | `face_recognition` e distância entre embeddings convencionais | Distâncias de 0,207–0,237 para o usuário real e 0,662–0,688 para o ataque; limiar experimental de 0,45 separou os grupos observados. |
| LAB02 | InsightFace `buffalo_l` e distância L2 | Maior distância real de 7,893 e menor distância do ataque de 32,527; limiar experimental de 20,210 entre os grupos. |
| LAB03 | Identidade do LAB02 e variação temporal de landmarks | As faixas de movimento apresentaram sobreposição. A variação temporal foi considerada evidência complementar, sem limiar único definitivo. |
| LAB04 | Identidade e desafios ativos de movimento | A decisão combinou identidade compatível e sequência válida. O artigo observa que a deepfake reproduziu parte dos movimentos do operador, limitando o valor do desafio isolado. |

Os thresholds foram ajustados e avaliados no conjunto experimental descrito no artigo. A separação observada não estabelece taxas generalizáveis de erro nem assegura resistência a outras identidades, ferramentas ou condições de captura.

## Prova de conceito: monitoramento de reuniões

A aplicação desktop é executada em paralelo a chamadas, sem integração oficial com as plataformas. O responsável cadastra uma referência do participante e seleciona uma janela ou região da sessão. A solução avalia periodicamente as imagens e combina sinais para gerar indicadores de risco instantâneo e acumulado.

```text
Referência do participante → perfil local e baseline
Janela da chamada → captura → qualidade e detecção facial
                  → identidade + sinais de manipulação + liveness
                  → agregação temporal → resumo para análise humana
```

O artigo descreve verificações em ciclos de aproximadamente 30 segundos, com as seguintes camadas:

- **Identidade:** embeddings do InsightFace `buffalo_l` comparados à referência por distância L2.
- **Manipulação:** detector Effort baseado em CLIP-L/14 e detector auxiliar EfficientNet-B7 do DFDC, com execução local via ONNX.
- **Liveness e qualidade:** variação temporal e landmarks, associados a avaliações de nitidez, iluminação e qualidade da face capturada.
- **Evolução temporal:** combinação dos sinais ao longo da sessão para evitar conclusões baseadas apenas em um evento isolado.

O monitor é uma **prova de conceito de apoio à análise humana**: não apresenta um resultado como prova automática de fraude. O artigo descreve armazenamento local de métricas, embeddings e avaliações, sem gravação de frames ou vídeos por padrão.

Consulte a [documentação específica do monitor](MONITOR_DEEPFAKE_REUNIOES/README.md) para detalhes de configuração, dependências e execução.

## Tecnologias e estrutura

- **Python**, OpenCV, NumPy e Tkinter: processamento de imagens, cálculos e interface desktop.
- **face_recognition / dlib** e **InsightFace**: extração e comparação de representações faciais.
- **Effort AIGI Detection**, **EfficientNet-B7 / DFDC** e **ONNX Runtime**: análise complementar de sinais de manipulação.
- **DeepLiveCam:** simulação controlada de ataque de troca facial.

```text
LAB_DEEPFAKE/                    # Quatro laboratórios incrementais
MONITOR_DEEPFAKE_REUNIOES/       # Prova de conceito e documentação
VALIDACAO_SEMELHANCA_MORFOLOGICA/ # Seleção experimental da identidade sintética
Deep-Live-Cam/                   # Ferramenta utilizada no cenário experimental
documentacao/TCC_02.pdf          # Artigo do trabalho
```

Pesos de modelos, ambientes virtuais, perfis biométricos, resultados de sessões, imagens e vídeos locais não são incluídos por padrão. O artigo disponibilizado neste repositório é uma exceção explícita à regra geral de exclusão de PDFs. Consulte a documentação de cada componente antes de executar os experimentos.

## Limitações e próximos passos

O estudo utilizou um usuário cadastrado, uma identidade sintética e uma ferramenta de troca facial em condições controladas. A calibração foi realizada com amostras do próprio experimento; a validação em população e condições independentes permanece necessária. O trabalho também não reproduz necessariamente as arquiteturas proprietárias de sistemas comerciais.

Como continuidade, o artigo propõe ampliar participantes, identidades sintéticas, ferramentas, câmeras, iluminação e plataformas; avaliar sessões reais ou simuladas mais diversas; e investigar sinais adicionais, como áudio e voz. Os resultados devem ser interpretados em conjunto com essas limitações.

## Uso responsável

O repositório tem finalidade acadêmica e defensiva. Detectores de deepfake fornecem evidências probabilísticas sujeitas a erros, qualidade da imagem e condições de operação; suas saídas não devem ser usadas isoladamente para acusar alguém ou tomar decisões automatizadas sobre pessoas.
