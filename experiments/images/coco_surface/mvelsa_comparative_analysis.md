# Análise Comparativa: MVELSA Baseline vs. Otimização de Tese

Este documento detalha as mudanças técnicas e arquiteturais realizadas para transformar a implementação básica do MVELSA em uma solução robusta para pesquisa de doutorado, justificando o salto de **38% para 61% de Recall**.

---

## 1. Processamento de Dados e Visão
| Recurso | MVELSA Original (Pasta) | MVELSA Otimizado (Tese) | Impacto Acadêmico |
| :--- | :--- | :--- | :--- |
| **Formato de Cor** | Tons de Cinza (`as_gray=True`) | **RGB Nativo (3 canais)** | Preservação de atributos cromáticos vitais (ex: cor de bóias). |
| **Foco na Imagem** | Imagem Inteira (Ruidosa) | **Object-Centric (Cropped)** | Remoção de pixels irrelevantes de fundo; foco puramente no alvo. |
| **Augmentation** | Nenhuma (Dataset Estático) | **Geométrica & Ajuste de Cor** | Combate ao Overfitting; aumento virtual do dataset em 5x. |
| **Loader** | `skimage` (Lento) | **PIL + Pandas (Veloz)** | Eficiência no treinamento e compatibilidade com transforms. |

---

## 2. Arquitetura da Rede Neural (ELSA)
A arquitetura foi redesenhada para suportar a complexidade extra da cor sem perder informação no "gargalo" latente.

*   **Capacidade de Entrada:** Aumentada de **4.096** para **12.288** neurônios por autoencoder.
*   **Profundidade Latente:**
    *   *Baseline:* `[4096, 128, 64]` (Afunilamento agressivo).
    *   *Otimizado:* `[12288, 1024, 256, 128]` (Compressão suave em 4 níveis).
*   **Convergência:** Épocas aumentadas de 20 para **35**, permitindo ajuste fino dos gradientes RGB.

---

## 3. Estratégia de Combate ao Desbalanceamento
O problema clássico do "Paradoxo da Acurácia" foi mitigado através de punição matemática diferenciada.

### Class Weighting (Punição Categórica)
No **MVELSA Original**, errar um Navio (classe rara) tinha o mesmo custo para o gradiente de erro que errar uma Bóia.
No **MVELSA Otimizado**, injetamos pesos balanceados:
*   **SHIP (10):** Peso **8.75** (Punindo pesadamente o erro).
*   **BUOY (3):** Peso **0.49** (Classe dominante, punida levemente).
*   **BOAT (1):** Peso **1.14**.

---

## 4. Evolução das Métricas Finais

| Métrica | Original (Baseline) | Otimizado (Tese) | Ganho |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 89% | 92.9% | +3.9% |
| **Recall (Média)** | 38% | **61.3%** | **+23.3%** |
| **Precision** | 42% | **66.7%** | **+24.7%** |

---

## Conclusão
A versão otimizada não apenas alcançou métricas superiores, mas resolveu a **estagnação de gradiente** causada pelo desbalanceamento. O modelo agora é capaz de identificar navios e barcos com uma confiança significativamente maior, validando a metodologia proposta para a tese.
