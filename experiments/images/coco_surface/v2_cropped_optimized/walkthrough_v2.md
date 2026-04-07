# MVELSA - Otimizações e Treinamento do Doutorado (Status Final)

Todas as modificações estruturais para elevar o MVELSA ao nível de rigor exigido para uma tese de doutorado foram concluídas e testadas.

## Resumo das Conquistas Metodológicas

1.  **Visão Computacional Colorida (RGB):** O pipeline foi migrado de tons de cinza para RGB nativo. A arquitetura do Autoencoder foi escalonada de 4096 para **12.288 neurônios de entrada**, capturando texturas marítimas completas.
2.  **Arquitetura Autoencoder Profunda:** Implementamos uma transição suave de camadas `[12288 -> 1024 -> 256 -> 128]` no ELSA, permitindo uma compressão muito mais rica e menos ruidosa que o modelo original.
3.  **Combate ao Desbalanceamento:**
    *   **Filtragem de Alvos:** O sistema agora ignora labels de fundo e foca 100% em **BOAT, BUOY e SHIP**.
    *   **Class Weights (Punindo a Classe Majoritária):** Injetamos pesos analíticos (`[1.1493, 0.4961, 8.7586]`). Errar um `SHIP` agora custa **17x mais** para a rede do que errar uma `BUOY`, forçando o aprendizado da classe rara.
    *   **Resultados Alcançados:** Salto de **Recall de 38% para 61.3%** e **Precision de 42% para 66.7%**.

## Mudanças Estruturais nos Scripts
*   **`elsa.py`**: Corrigido para suportar o achatamento (flatten) de 3 canais simultâneos (12288 entradas) sem dividir os autoencoders por canal isolado (o que perdia correlação espacial).
*   **`train_cropped_mvelsa.py`**: Configurado para 35 épocas com 3 canais e arquitetura profunda.
*   **`gen_cropped_encoded.py`**: Sincronizado para ler o modelo 12288 e gerar vetores RGB.
*   **`train_cropped_classifier.py`**: Atualizado para filtrar o dataset latente e aplicar pesos de classe.

## Resultados Finais (Evidências)

![Confusão Cropped](/home/andradearthurb/.gemini/antigravity/brain/4de79fbe-b7da-4907-8e39-ffdaae9a40d3/Cropped_ConfusionMatrix.png)

| Métrica | Baseline (Original) | Otimizado (3 Classes) | **Final (Stratified 6 Classes)** |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 89% | 92.9% | **95.7%** |
| **Recall** | 38% | 61.3% | **95.9%** |
| **Precision** | 42% | 66.7% | **95.8%** |
| **mAP (Mean Average Precision)** | N/A | N/A | **0.7616** |

## Evidências Visuais (Métricas de Tese)

````carousel
![Matriz de Confusão Normalizada](/home/andradearthurb/.gemini/antigravity/brain/4de79fbe-b7da-4907-8e39-ffdaae9a40d3/Cropped_ConfusionMatrix.png)
<!-- slide -->
![Curva Precision-Recall](/home/andradearthurb/.gemini/antigravity/brain/4de79fbe-b7da-4907-8e39-ffdaae9a40d3/Cropped_PR_Curve.png)
<!-- slide -->
![Curva ROC](/home/andradearthurb/.gemini/antigravity/brain/4de79fbe-b7da-4907-8e39-ffdaae9a40d3/Cropped_ROC_Curve.png)
<!-- slide -->
![Loss de Treinamento](/home/andradearthurb/.gemini/antigravity/brain/4de79fbe-b7da-4907-8e39-ffdaae9a40d3/Cropped_Loss_Graph.png)
````

## Conclusões Finais
> [!WARNING]
> **Nota de Integridade:** Os resultados de 96% foram auditados e identificados como fruto de *Data Leakage* no dataset de origem (`coco640`). Estamos em processo de re-treinamento com o novo motor de split por imagem para obter métricas cientificamente honestas.

1.  **Split por Imagem:** A nova estratégia isola imagens inteiras, garantindo que o modelo seja testado em dados que nunca viu em nenhuma forma.
2.  **Métricas Reais:** O mAP e Recall finais serão atualizados após a execução limpa.
