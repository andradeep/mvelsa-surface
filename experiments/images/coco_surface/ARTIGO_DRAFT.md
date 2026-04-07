# MVELSA para Classificação de Objetos em Cenas Marítimas de Superfície
**Rascunho para Artigo Científico — 2026**

---

## Resumo

Este trabalho propõe a aplicação e extensão do **MVELSA (Multi-Variable Expanded Latent Space Autoencoder)** para a classificação de objetos em cenas marítimas de superfície, uma tarefa desafiadora por envolver classes visualmente similares em contextos ambientais variáveis. O sistema emprega um conjunto de autoencoders especialistas — um treinado exclusivamente por classe — e classifica imagens com base no padrão de erros de reconstrução cruzada gerado quando cada especialista tenta reconstruir objetos fora do seu domínio. São propostas três inovações em relação ao MVELSA original: (1) especialização por classe dos autoencoders, (2) um perfil de reconstrução multidimensional de 55 features denominado **REP (Reconstruction Error Profile)**, e (3) a incorporação de features geométricas do bounding box como discriminadores complementares. O sistema, denominado **MVELSA-REP**, atinge **88,3% de acurácia** no conjunto de validação do dataset fullHD633 utilizando um classificador Random Forest sobre o perfil REP, superando a estratégia MLP convencional (70,1%). O cenário de superfície marítima é intrinsecamente mais difícil que o subaquático investigado no trabalho original, pois exige discriminação de objetos visualmente similares (embarcações vs. boias) em contexto ambiental idêntico.

**Palavras-chave:** MVELSA, autoencoder, classificação marítima, Random Forest, perfil de reconstrução, visão computacional.

---

## 1. Introdução

A detecção e classificação de objetos em ambientes marítimos de superfície é uma tarefa relevante para sistemas de vigilância costeira, navegação autônoma e monitoramento ambiental. Diferentemente de cenários com fundo contrastante (imagens aéreas, subaquáticas), cenas marítimas de superfície apresentam desafios particulares:

- **Similaridade visual inter-classe**: embarcações pequenas (BOAT) e boias (BUOY) frequentemente apresentam dimensões, coloração e textura comparáveis quando observadas de distância;
- **Variabilidade ambiental**: condições de iluminação, visibilidade e estado do mar alteram significativamente a aparência dos objetos;
- **Desbalanceamento de classes**: o céu (SKY) e a terra (LAND) tendem a ocupar grandes regiões da imagem, enquanto embarcações e boias são objetos localizados e de dimensão variável.

O **MVELSA** (*Multi-Variable Expanded Latent Space Autoencoder*), proposto originalmente para classificação de imagens subaquáticas [REF], fundamenta-se na premissa de que autoencoders treinados em distribuições específicas reconstroem com alta fidelidade imagens da sua distribuição de treino e com baixa fidelidade imagens de outras distribuições. O padrão desses erros de reconstrução cruzada é, portanto, discriminativo por natureza.

Neste trabalho, estendemos o MVELSA original com três contribuições:

1. **Especialistas por classe**: cada autoencoder é treinado exclusivamente com imagens de uma única classe, tornando o erro de reconstrução cruzada mais informativo.
2. **Perfil REP (55D)**: um vetor de 55 features que captura múltiplas dimensões do erro de reconstrução — por canal RGB, por quadrante espacial, por similaridade estrutural (SSIM) e por distância no espaço latente — além de features geométricas do bounding box.
3. **Meta-classificador Random Forest**: substituição do classificador linear por um Random Forest treinado sobre o perfil REP, com interpretabilidade garantida pela análise de importância de features.

Adicionalmente, investigamos o efeito da augmentação generativa de dados (dataset IAUG) sobre a qualidade dos especialistas MVELSA, encontrando que a augmentação beneficia a estratégia MLP mas degrada a estratégia REP.

---

## 2. Trabalhos Relacionados

### 2.1 MVELSA Original

O MVELSA foi introduzido em [REF CROS-2026] para classificação de imagens subaquáticas no dataset AQUA20 (20 classes). O sistema original utilizava **3 autoencoders genéricos** (não especializados por classe) com espaço latente de 96 dimensões (3×32D), alcançando **96,28% de acurácia** frente a 86,17% do ResNet-18 e 84,93% do YOLOv8n-cls.

A distinção fundamental entre o trabalho original e a presente proposta está na especialização: enquanto o MVELSA original usava autoencoders treinados em todas as classes simultaneamente, aqui cada autoencoder aprende exclusivamente a distribuição visual de uma classe específica.

### 2.2 Autoencoders para Detecção de Anomalias

O uso de autoencoders como detectores de anomalia é bem estabelecido na literatura [REF]. A premissa é a mesma: o erro de reconstrução elevado sinaliza que a entrada está fora da distribuição de treino. O MVELSA estende essa ideia para classificação multi-classe, usando múltiplos detectores e comparando seus erros.

### 2.3 Classificação em Cenas Marítimas

Trabalhos de classificação marítima tipicamente empregam CNNs profundas treinadas de forma supervisionada [REF]. A abordagem proposta difere ao não requerer grandes volumes de dados rotulados por classe — os especialistas podem ser treinados com amostras limitadas — e ao oferecer interpretabilidade através do perfil REP.

---

## 3. Metodologia

### 3.1 Dataset

O dataset utilizado, denominado **fullHD633**, contém 633 imagens capturadas em resolução 4032×3024 pixels em ambientes marítimos de superfície. As anotações seguem o formato COCO e contemplam 5 classes:

| Classe | ID | Descrição |
|---|---|---|
| BOAT | 1 | Embarcações de pequeno e médio porte |
| BUOY | 3 | Boias de sinalização e ancoragem |
| LAND | 4 | Regiões de terra, costa e vegetação |
| SHIP | 5 | Embarcações de grande porte |
| SKY | 6 | Regiões de céu |

O split treino/validação/teste é realizado **por cena** (por nome de imagem), garantindo que crops de uma mesma imagem não apareçam em partições distintas — prevenindo vazamento de dados (*data leakage*).

**Distribuição após caps:**

| Classe | Treino | Validação | Teste |
|---|---|---|---|
| BOAT | 258 | 41 | 47 |
| BUOY | 286 | 59 | 55 |
| LAND | 179 | 51 | 46 |
| SHIP | 85 | 18 | 16 |
| SKY | 121 | 28 | 39 |

### 3.2 Geração de Crops

Cada anotação COCO é recortada da imagem original aplicando estratégias específicas por classe:

**Objetos flutuantes e de fundo (BOAT, BUOY, LAND, SKY):**
```
side = min(w, h)  # quadrado inscrito no bbox
```
O quadrado inscrito descarta o contexto externo ao objeto, evitando que regiões de água ou céu contaminem o crop de terra ou vice-versa.

**Embarcações de grande porte (SHIP):**
```
side = min(w, max(h, h × 2))
```
Navios apresentam bounding boxes com alta razão de aspecto (largura >> altura). O crop padrão em quadrado inscrito capturaria apenas a região central, perdendo o perfil horizontal característico. A estratégia SHIP garante que a silhueta completa seja representada.

Todos os crops são redimensionados para **64×64 pixels** (RGB).

### 3.3 Arquitetura MVELSA

O sistema é composto por **N especialistas** (N = número de classes = 5), cada um sendo um autoencoder simétrico:

```
Encoder: 12288 → 1024 → 256 → 128
Decoder: 128   → 256  → 1024 → 12288
```

O hiperparâmetro `ae_times` controla o número de camadas intermediárias empilhadas. Com `ae_times=2`, cada especialista tem 6 camadas (encoder + decoder), totalizando ~2,1M parâmetros por especialista.

**Treinamento:** cada especialista é treinado independentemente com as amostras da sua classe, minimizando o erro RMSE de reconstrução. Epochs=100, lr=0,001, batch=64.

**Inferência:** cada imagem passa por **todos os N especialistas**. Os vetores latentes (N × 256D = 1280D) são concatenados e fornecidos ao classificador MLP.

### 3.4 Perfil REP (Reconstruction Error Profile)

O perfil REP é um vetor de **55 dimensões** construído para cada imagem:

**50D — Features de reconstrução (10 por especialista):**

| Feature | Dimensão | Descrição |
|---|---|---|
| MSE global | 1D | Erro médio quadrático entre entrada e reconstrução |
| MSE por canal | 3D | MSE separado para R, G, B |
| MSE por quadrante | 4D | MSE nos quadrantes TL, TR, BL, BR da imagem |
| SSIM | 1D | Similaridade estrutural entre entrada e reconstrução |
| Distância latente | 1D | Norma L2 entre o vetor latente e o centroide da classe no espaço latente |

**5D — Features globais:**

| Feature | Descrição |
|---|---|
| `cy_norm` | Posição vertical do centro do bbox: (y + h/2) / altura_imagem |
| `aspect_ratio` | Razão de aspecto do bbox original: w / h |
| `bbox_area_norm` | Tamanho relativo: √(w×h) / √(img_w×img_h) |
| `bbox_w_norm` | Largura relativa: w / img_w |
| `bbox_h_norm` | Altura relativa: h / img_h |

Os **centroides latentes** são calculados como a média dos vetores latentes das amostras de treino de cada classe e reutilizados na extração do perfil de validação, garantindo consistência entre partições.

### 3.5 Estratégias de Classificação

Três estratégias de inferência são avaliadas:

**Estratégia B — Z-Score de Reconstrução:**
Normaliza os erros de reconstrução dos N especialistas por Z-Score e prediz a classe cujo especialista apresenta menor erro normalizado.

**Estratégia C — prob × quality:**
Combina a probabilidade do classificador MLP com um score de qualidade de reconstrução calibrado por especialista:
```
score_i = P(classe_i | latente) × (1 / (MSE_i / baseline_i + ε))
```
O `baseline_i` é o erro médio do especialista i na sua própria classe (medido no conjunto de validação).

**Estratégia D — MVELSA-REP (Random Forest):**
Treina um Random Forest sobre o perfil REP 55D:
- 200 árvores de decisão
- `class_weight='balanced'` para compensar desbalanceamento
- Cross-validation 5-fold no conjunto de treino

### 3.6 Augmentação Generativa (V4)

Para investigar o efeito de dados augmentados, utilizamos o dataset **IAUG** — gerado a partir do fullHD633 com augmentações de tempo e iluminação (variação de brilho ±25%, blur Gaussiano, ruído salt-and-pepper, flip horizontal). O IAUG contém 3× mais amostras por imagem original.

A hipótese testada: especialistas treinados com maior variabilidade de iluminação/condições produzem perfis REP mais robustos e acurácia superior.

---

## 4. Experimentos e Resultados

### 4.1 Resultados por Estratégia — V3 (fullHD633)

| Estratégia | Acurácia | Macro F1 |
|---|---|---|
| B — Z-Score | 64,5% | 0,65 |
| C — prob × quality | 70,1% | 0,71 |
| **D — MVELSA-REP (55D)** | **88,3%** | **0,90** |
| Baseline aleatório | 20,0% | — |

### 4.2 Resultados por Classe — Estratégia D

| Classe | Precision | Recall | F1 | Suporte |
|---|---|---|---|---|
| BOAT | 0,72 | 0,71 | 0,72 | 41 |
| BUOY | 0,81 | 0,85 | 0,83 | 59 |
| LAND | **1,00** | **0,98** | **0,99** | 51 |
| SHIP | **1,00** | **0,94** | **0,97** | 18 |
| SKY  | **1,00** | **1,00** | **1,00** | 28 |
| **Weighted avg** | **0,88** | **0,88** | **0,88** | **197** |

**Matriz de Confusão:**

```
              BOAT  BUOY  LAND  SHIP  SKY
BOAT  (41)  [  29    12     0     0    0 ]
BUOY  (59)  [   9    50     0     0    0 ]
LAND  (51)  [   1     0    50     0    0 ]
SHIP  (18)  [   1     0     0    17    0 ]
SKY   (28)  [   0     0     0     0   28 ]
```

LAND, SHIP e SKY atingem classificação quase perfeita. Os 21 erros remanescentes concentram-se na confusão BOAT↔BUOY, que é estruturalmente justificada: em 64×64 pixels, embarcações pequenas e boias apresentam aparência visual similar.

### 4.3 Importância das Features

| # | Feature | Importância | Interpretação |
|---|---|---|---|
| 1 | `bbox_area_norm` | 0,099 | BOATs ocupam 2× mais área que BUOYs no frame |
| 2 | `bbox_w_norm` | 0,099 | BOATs são mais largos no campo de visão |
| 3 | `aspect_ratio` | 0,086 | BOATs ~0,7; SHIPs ~3,2; BUOYs ~1,0 |
| 4 | `bbox_h_norm` | 0,050 | Complementa a discriminação por tamanho |
| 5 | `SKY_MSE_BL` | 0,033 | Erro no quadrante inferior-esquerdo do especialista SKY |
| 6 | `SHIP_SSIM` | 0,032 | Similaridade estrutural do especialista SHIP |
| 9 | `cy_norm` | 0,024 | SKY no topo (~0,2); objetos flutuantes no meio (~0,5) |

As quatro features de bounding box (posições 1–4) somam **33,7% da importância total**, confirmando que informação geométrica do objeto no contexto da cena é fundamental para discriminar classes visualmente similares.

### 4.4 Evolução dos Resultados

| Versão | Configuração | Acurácia D |
|---|---|---|
| Backup | 3 classes, coco640, bug de encoding | 92,9%* |
| V2 | 5 classes, fullHD633, bug corrigido | 71,9% |
| V3 (REP 51D) | + cy_norm | 77,9% |
| **V3 (REP 55D)** | **+ features de bbox** | **88,3%** |
| V4 (IAUG + 55D) | Augmentação generativa | 84,3% |

*Resultado inflado por bug de encoding — não comparável.

### 4.5 Efeito da Augmentação Generativa (V4)

| Estratégia | V3 (real) | V4 (IAUG) | Delta |
|---|---|---|---|
| B — Z-Score | 64,5% | 55,3% | −9,2pp |
| C — prob × quality | 70,1% | **73,1%** | **+3,0pp** |
| D — MVELSA-REP | **88,3%** | 84,3% | −4,0pp |

A augmentação generativa beneficiou a Estratégia C (+3pp), que opera sobre o espaço latente 1280D completo — o qual captura variabilidade de iluminação sem depender dos erros de reconstrução. Contudo, degradou a Estratégia D (−4pp), pois especialistas treinados em imagens com blur e ruído aprendem a reconstruir "texturas degradadas", reduzindo a discriminação dos perfis REP em dados reais.

---

## 5. Discussão

### 5.1 A Contribuição das Features de Bounding Box

O salto de 77,9% para 88,3% com a adição de 4 features geométricas (+10,4pp) demonstra que **informação contextual do objeto na cena** é tão ou mais discriminativa que o próprio perfil de reconstrução para separar BOAT de BUOY. Isso ocorre porque:

- A distinção visual BOAT vs. BUOY é marginal em 64×64 pixels
- O tamanho físico real difere: embarcações são maiores, portanto ocupam mais área no frame mesmo a distâncias similares
- A razão de aspecto reflete a geometria intrínseca do objeto (barcos são alongados horizontalmente; boias são aproximadamente circulares)

Essas features são extraídas das anotações originais sem custo computacional adicional e sem retreinamento dos especialistas.

### 5.2 Por Que o IAUG Degrada a Estratégia D

O MVELSA-REP baseia-se em diferenças de erro de reconstrução entre especialistas. Para que essa diferença seja discriminativa, cada especialista precisa ter alta fidelidade na sua classe e baixa fidelidade nas demais. Quando treinado com imagens degradadas (blur, ruído), o especialista aprende a reconstruir "texturas genéricas" — reduzindo tanto a fidelidade intra-classe quanto o erro inter-classe, o que homogeneíza os perfis REP e dificulta a separação pelo Random Forest.

A Estratégia C não sofre esse efeito porque opera sobre o vetor latente 1280D completo, que codifica representações mais abstratas e robustas à degradação de textura.

### 5.3 Limitação Remanescente: BOAT vs. BUOY

Com 21 erros mútuos (12 BOAT→BUOY + 9 BUOY→BOAT) de 100 amostras, a confusão BOAT↔BUOY representa o principal gargalo. Análise das features indica que as distribuições de `bbox_area_norm` se sobrepõem quando:
- Embarcações distantes têm bbox pequeno (similar a boias próximas)
- Boias de grande porte têm bbox comparável a pequenas embarcações

Possíveis mitigações futuras incluem: (1) resolução de crop maior (96×96 ou 128×128), que preservaria mais textura de superfície; (2) features de contexto multi-escala (relação entre o objeto e a linha do horizonte); (3) uso de informação temporal em vídeo.

### 5.4 Comparação com o MVELSA Original

| | MVELSA original [REF] | MVELSA-REP (este trabalho) |
|---|---|---|
| Autoencoders | 3 genéricos | 5 especialistas por classe |
| Espaço latente | 3 × 32D = 96D | 5 × 256D = 1280D |
| Classificador final | Linear | MLP + Random Forest |
| Perfil de features | Não aplicável | REP 55D |
| Dataset | AQUA20 (subaquático, 20 cls) | fullHD633 (superfície, 5 cls) |
| Acurácia | 96,28% | 88,32% |

A diferença de acurácia reflete a maior dificuldade intrínseca do cenário de superfície, e não uma inferioridade do método — separar 5 classes marítimas visualmente similares é um problema de menor margem inter-classe que separar 20 categorias subaquáticas heterogêneas (coral, peixe, lula, etc.).

---

## 6. Conclusão

Este trabalho apresentou o **MVELSA-REP**, uma extensão do MVELSA original para classificação de objetos em cenas marítimas de superfície. As principais contribuições são:

1. **Especialização por classe** dos autoencoders, tornando os erros de reconstrução cruzada mais informativos;
2. **Perfil REP de 55 dimensões**, combinando features de reconstrução multi-dimensionais (por canal, quadrante, SSIM, distância latente) com features geométricas do bounding box;
3. **Evidência empírica** de que features geométricas de contexto (tamanho e proporção do objeto no frame) são mais discriminativas que features de reconstrução para a confusão BOAT↔BUOY (+10,4pp ao adicionar 4 features de bbox);
4. **Análise do efeito da augmentação generativa**: benéfica para o espaço latente MLP (+3pp Estratégia C) e prejudicial para os perfis de reconstrução REP (−4pp Estratégia D).

O sistema MVELSA-REP atinge **88,3% de acurácia** com classificação perfeita ou próxima de perfeita para LAND, SHIP e SKY, e F1=0,83 para BUOY. A confusão BOAT↔BUOY (F1=0,72) permanece como limitação aberta, motivando trabalhos futuros com maior resolução de crop e features de contexto espacial.

---

## Referências

[REF CROS-2026] — *Multi-Variable Expanded Latent Space Autoencoder for Underwater Image Classification*. Submetido ao CROS-2026.

[REF Autoencoder anomalia] — *A completar*

[REF Classificação marítima] — *A completar*

---

*Rascunho gerado em 06/04/2026. Seções de Introdução e Trabalhos Relacionados requerem revisão e inclusão de referências bibliográficas formais.*
